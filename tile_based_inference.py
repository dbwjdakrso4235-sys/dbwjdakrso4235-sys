#!/usr/bin/env python3
"""
타일 기반 곤포사일리지 추론 시스템
=====================================

목적:
- 큰 크롭 이미지를 타일로 분할하여 추론
- 경계선 객체 검출을 위한 오버랩 적용
- NMS로 중복 검출 제거
- 타일링 전후 성능 비교

작성일: 2025-10-28
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
import time

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
from scipy import ndimage
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """타일 정보"""
    tile_id: int
    x_offset: int  # 원본 이미지에서의 x 시작 위치
    y_offset: int  # 원본 이미지에서의 y 시작 위치
    width: int
    height: int
    image: np.ndarray  # 타일 이미지 (H, W, 3)
    is_border: bool = False  # v11: 경계 타일 여부 (우/하단)


@dataclass
class TileDetection:
    """타일별 검출 결과"""
    tile_id: int
    detection_id: int  # 타일 내 검출 ID
    bbox: List[float]  # [x1, y1, x2, y2] - 타일 좌표계
    bbox_global: List[float]  # [x1, y1, x2, y2] - 원본 이미지 좌표계
    confidence: float
    mask: np.ndarray  # 타일 좌표계 마스크
    mask_global: Optional[np.ndarray] = None  # 원본 이미지 좌표계 마스크
    area_pixels: int = 0
    stacked_layer_count: int = 1  # v10: 쌓인 층수 (기본값: 1)
    is_border: bool = False  # v11: 경계 타일에서 온 검출인지 여부


class TiledInferenceEngine:
    """타일 기반 추론 엔진"""

    def __init__(
        self,
        model_path: str,
        tile_size: int = 1024,
        overlap_ratio: float = 0.25,  # Default 0.25 as requested
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_iou_threshold: float = 0.5,
        min_instance_area: int = 50,  # v6: 더 완화 (100 → 50)
        min_circularity: float = 0.01,  # v6: 거의 모든 형태 허용 (0.05 → 0.01)
        morphology_kernel_size: int = 1,  # v6: 최소화 (3 → 1)
        inference_scale: float = 1.5,  # NEW: Scale factor for inference (1.5x for better small object detection)
        device: str = 'auto'
    ):
        """
        Args:
            model_path: YOLO 모델 경로
            tile_size: 타일 크기 (정사각형)
            overlap_ratio: 타일 간 오버랩 비율 (0.0 ~ 1.0) [Default: 0.25]
            conf_threshold: 검출 신뢰도 임계값
            iou_threshold: YOLO NMS IoU 임계값
            nms_iou_threshold: 타일 간 중복 제거 NMS IoU 임계값
            min_instance_area: 개별 인스턴스 최소 면적 (픽셀)
            min_circularity: 개별 인스턴스 최소 원형도 (0~1)
            morphology_kernel_size: Morphology 연산 커널 크기
            inference_scale: 추론 시 이미지 스케일링 배율 (1.5 = 1.5x 확대) [NEW]
            device: 디바이스 ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.overlap_pixels = int(tile_size * overlap_ratio)
        self.stride = tile_size - self.overlap_pixels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_instance_area = min_instance_area
        self.min_circularity = min_circularity
        self.morphology_kernel_size = morphology_kernel_size
        self.inference_scale = inference_scale  # NEW: Store inference scale factor

        # 디바이스 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        try:
            logger.info(f"모델 로드 중: {self.model_path}")
            self.model = YOLO(str(self.model_path))

            if 'cuda' in self.device:
                self.model.to(self.device)
                logger.info(f"모델 로드 완료 (GPU: {torch.cuda.get_device_name(0)})")
            else:
                logger.info("모델 로드 완료 (CPU)")

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    def _separate_instances(
        self,
        mask: np.ndarray,
        confidence: float
    ) -> List[Dict]:
        """
        하나의 mask를 Watershed + Connected Components로 개별 인스턴스 분리

        Args:
            mask: Binary mask (H, W)
            confidence: 원본 detection 신뢰도

        Returns:
            분리된 인스턴스 리스트 (각 인스턴스는 딕셔너리)
        """
        # Morphology 연산으로 노이즈 제거 (최소화)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )

        # Opening: 작은 노이즈 제거
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Watershed 알고리즘 적용 (붙어있는 원형 객체 분리)
        # Distance Transform
        dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)

        # 거리 변환 결과를 정규화
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        # 확실한 전경 찾기 (거리가 먼 중심부)
        # 임계값을 낮춰서 더 많은 중심점 찾기
        _, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 확실한 배경 찾기
        sure_bg = cv2.dilate(cleaned_mask, kernel, iterations=1)

        # 불확실한 영역 (전경도 배경도 아닌 영역)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker 생성 (연결된 구성 요소)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # 배경을 0이 아닌 1로 만들기
        markers[unknown == 255] = 0  # 불확실한 영역을 0으로

        # Watershed 적용 (3채널 이미지 필요)
        mask_3ch = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_3ch, markers)

        # Watershed 결과를 이진 마스크로 변환 (-1은 경계선)
        watershed_mask = np.zeros_like(cleaned_mask)
        watershed_mask[markers > 1] = 1  # 배경(1) 제외, 경계(-1) 제외

        # Connected Components Analysis (Watershed 결과에)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            watershed_mask.astype(np.uint8),
            connectivity=8
        )

        instances = []

        # 각 연결된 구성 요소 처리 (0은 배경이므로 제외)
        for i in range(1, num_labels):
            # 통계 추출
            x, y, w, h, area = stats[i]

            # 면적 필터링
            if area < self.min_instance_area:
                continue

            # 개별 mask 생성
            instance_mask = (labels == i).astype(np.uint8)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(
                instance_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = contours[0]

            # 원형도 계산 (4π * area / perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 원형도 필터링
            if circularity < self.min_circularity:
                continue

            # bbox 계산
            x1, y1, x2, y2 = x, y, x + w, y + h

            # bbox 영역의 mask만 crop
            cropped_mask = instance_mask[y1:y2, x1:x2]

            instances.append({
                'bbox': (x1, y1, x2, y2),
                'mask': cropped_mask,
                'area': area,
                'circularity': circularity,
                'centroid': (centroids[i][0], centroids[i][1])
            })

        return instances

    def create_tiles(self, image: np.ndarray) -> List[TileInfo]:
        """
        이미지를 타일로 분할 (v11: 경계 overlap 강화)

        Args:
            image: 원본 이미지 (H, W, 3)

        Returns:
            TileInfo 리스트
        """
        h, w = image.shape[:2]
        tiles = []
        tile_id = 0

        # v11: 경계 타일 감지를 위한 threshold (우/하단 10% 영역)
        border_threshold_x = w * 0.9
        border_threshold_y = h * 0.9

        logger.info(f"타일 생성 중 - 이미지 크기: {w}x{h}, 타일 크기: {self.tile_size}, 오버랩: {self.overlap_ratio*100:.0f}% ({self.overlap_pixels}px)")

        # y 방향 타일링
        y = 0
        while y < h:
            # x 방향 타일링
            x = 0
            while x < w:
                # 타일 영역 계산
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)

                # v11: 경계 타일 판별 (우측/하단 10% 영역 또는 끝에 걸친 타일)
                is_border = (x >= border_threshold_x) or (y >= border_threshold_y) or \
                            (x_end >= w) or (y_end >= h)

                # 타일 추출
                tile_image = image[y:y_end, x:x_end].copy()

                # 타일이 tile_size보다 작으면 패딩
                tile_h, tile_w = tile_image.shape[:2]
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=image.dtype)
                    padded[:tile_h, :tile_w] = tile_image
                    tile_image = padded

                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x_offset=x,
                    y_offset=y,
                    width=x_end - x,
                    height=y_end - y,
                    image=tile_image,
                    is_border=is_border  # v11: 경계 타일 표시
                ))

                tile_id += 1

                # v11: 다음 x 위치로 이동 (경계 근처는 stride 줄여서 overlap 강화)
                if x >= border_threshold_x or (x + self.tile_size >= w):
                    # 경계 영역: overlap 0.5 (50%)
                    x += int(self.tile_size * 0.5)
                else:
                    # 일반 영역: 기본 stride 사용
                    x += self.stride

                if x >= w:
                    break

            # v11: 다음 y 위치로 이동 (경계 근처는 stride 줄여서 overlap 강화)
            if y >= border_threshold_y or (y + self.tile_size >= h):
                # 경계 영역: overlap 0.5 (50%)
                y += int(self.tile_size * 0.5)
            else:
                # 일반 영역: 기본 stride 사용
                y += self.stride

            if y >= h:
                break

        border_count = sum(1 for t in tiles if t.is_border)
        logger.info(f"타일 생성 완료: {len(tiles)}개 (경계 타일: {border_count}개)")
        return tiles

    def predict_tile(self, tile: TileInfo, conf_override: float = None, scale_override: float = None) -> List[TileDetection]:
        """
        단일 타일 추론 (inference_scale 적용 및 scale_boxes로 원본 스케일 복원)

        Args:
            tile: TileInfo 객체
            conf_override: v11: 2차 추론용 신뢰도 임계값 (None이면 self.conf_threshold 사용)
            scale_override: v11: 2차 추론용 inference_scale (None이면 self.inference_scale 사용)

        Returns:
            TileDetection 리스트
        """
        try:
            # v11: 2차 추론 파라미터 적용
            conf_threshold = conf_override if conf_override is not None else self.conf_threshold
            inference_scale = scale_override if scale_override is not None else self.inference_scale

            # Apply inference scaling for better small object detection
            tile_for_inference = tile.image
            scaled_size = int(self.tile_size * inference_scale)

            # Scale the tile image if inference_scale != 1.0
            if inference_scale != 1.0:
                tile_for_inference = cv2.resize(
                    tile.image,
                    (scaled_size, scaled_size),
                    interpolation=cv2.INTER_LINEAR
                )
                logger.debug(f"Tile {tile.tile_id}: Scaled from {self.tile_size}x{self.tile_size} to {scaled_size}x{scaled_size} for inference")

            # YOLO 추론
            results = self.model.predict(
                tile_for_inference,
                conf=conf_threshold,
                iou=self.iou_threshold,
                imgsz=scaled_size if inference_scale != 1.0 else self.tile_size,
                device=self.device,
                verbose=False
            )[0]

            detections = []

            if results.masks is not None and len(results.masks) > 0:
                masks_data = results.masks.data.cpu().numpy()  # (N, H, W)
                boxes_data = results.boxes.data.cpu().numpy()  # (N, 6)

                # Scale back from inference scale to original tile size
                if inference_scale != 1.0:
                    # Scale boxes from scaled inference size back to tile size
                    boxes_xyxy = torch.from_numpy(boxes_data[:, :4])  # (N, 4)
                    boxes_scaled = scale_boxes(
                        (scaled_size, scaled_size),  # from (scaled inference size)
                        boxes_xyxy,
                        (self.tile_size, self.tile_size)  # to (original tile size)
                    ).numpy()
                    boxes_data[:, :4] = boxes_scaled

                # Handle padding if tile is smaller than tile_size
                if tile.width < self.tile_size or tile.height < self.tile_size:
                    boxes_xyxy = torch.from_numpy(boxes_data[:, :4])  # (N, 4)
                    boxes_scaled = scale_boxes(
                        (self.tile_size, self.tile_size),  # from (padded size)
                        boxes_xyxy,
                        (tile.height, tile.width)  # to (original tile size)
                    ).numpy()
                    boxes_data[:, :4] = boxes_scaled

                for i, (mask, box) in enumerate(zip(masks_data, boxes_data)):
                    x1, y1, x2, y2, conf, cls = box

                    # Scale mask back to original tile size if inference scale was applied
                    if inference_scale != 1.0:
                        # Mask is at scaled_size, need to resize to tile_size
                        if mask.shape != (self.tile_size, self.tile_size):
                            mask = cv2.resize(mask, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
                    # Regular mask size adjustment if needed
                    elif mask.shape != (self.tile_size, self.tile_size):
                        mask = cv2.resize(mask, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)

                    # Binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # 유효한 영역만 (패딩 제외)
                    valid_mask = binary_mask[:tile.height, :tile.width]

                    # Connected Components로 개별 인스턴스 분리
                    instances = self._separate_instances(valid_mask, conf)

                    if len(instances) > 1:
                        logger.info(f"타일 {tile.tile_id} detection {i}: {len(instances)}개 인스턴스로 분리")
                    elif len(instances) == 0:
                        logger.warning(f"타일 {tile.tile_id} detection {i}: 인스턴스 분리 후 필터링으로 모두 제거됨")

                    # 각 인스턴스를 개별 detection으로 추가
                    for inst_id, instance in enumerate(instances):
                        # 타일 내 좌표
                        inst_x1, inst_y1, inst_x2, inst_y2 = instance['bbox']

                        # 1️⃣ 타일별 bbox에 tile offset 더하기
                        bbox_global = [
                            inst_x1 + tile.x_offset,
                            inst_y1 + tile.y_offset,
                            inst_x2 + tile.x_offset,
                            inst_y2 + tile.y_offset
                        ]

                        detections.append(TileDetection(
                            tile_id=tile.tile_id,
                            detection_id=f"{i}_{inst_id}",  # 원본 detection_id + 인스턴스 id
                            bbox=[float(inst_x1), float(inst_y1), float(inst_x2), float(inst_y2)],
                            bbox_global=bbox_global,
                            confidence=float(conf),
                            mask=instance['mask'],
                            area_pixels=int(instance['area']),
                            is_border=tile.is_border  # v11: 경계 타일 플래그 전파
                        ))

            return detections

        except Exception as e:
            logger.error(f"타일 {tile.tile_id} 추론 실패: {e}")
            return []

    def predict_all_tiles(self, tiles: List[TileInfo]) -> List[TileDetection]:
        """
        모든 타일 추론

        Args:
            tiles: TileInfo 리스트

        Returns:
            TileDetection 리스트 (모든 타일의 검출 결과 통합)
        """
        all_detections = []

        logger.info(f"타일 추론 시작: {len(tiles)}개")

        for tile in tqdm(tiles, desc="타일 추론"):
            detections = self.predict_tile(tile)
            all_detections.extend(detections)

        logger.info(f"타일 추론 완료: {len(all_detections)}개 검출 (중복 포함)")
        return all_detections

    def apply_wbf(self, detections: List[TileDetection], image_shape: Tuple[int, int]) -> List[TileDetection]:
        """
        4️⃣ Weighted Boxes Fusion (WBF)로 중복 검출 병합

        Args:
            detections: TileDetection 리스트
            image_shape: 이미지 shape (H, W)

        Returns:
            WBF 적용 후 TileDetection 리스트
        """
        if len(detections) == 0:
            return []

        h, w = image_shape

        # bbox를 normalized 좌표로 변환 (WBF 입력 형식)
        boxes_list = []
        scores_list = []
        labels_list = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox_global
            # Normalize to [0, 1]
            boxes_list.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores_list.append(det.confidence)
            labels_list.append(0)  # single class

        # WBF는 리스트의 리스트를 받음 (여러 모델의 결과를 병합하는 용도)
        # 우리는 단일 모델이지만 타일별 결과를 병합
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            [boxes_list],  # list of lists
            [scores_list],  # list of lists
            [labels_list],  # list of lists
            weights=[1.0],  # 단일 모델이므로 가중치 1.0
            iou_thr=self.nms_iou_threshold,
            skip_box_thr=0.0  # confidence threshold는 이미 적용됨
        )

        # Denormalize boxes
        boxes_fused[:, [0, 2]] *= w  # x1, x2
        boxes_fused[:, [1, 3]] *= h  # y1, y2

        # TileDetection 리스트 생성 (mask는 나중에 병합)
        filtered_detections = []
        for i, (box, score) in enumerate(zip(boxes_fused, scores_fused)):
            x1, y1, x2, y2 = box

            # 가장 가까운 원본 detection 찾기 (mask 복사용)
            # IoU가 가장 높은 원본 detection 선택
            best_iou = 0.0
            best_det = None
            for det in detections:
                iou = self._calculate_iou(box, det.bbox_global)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None:
                filtered_detections.append(TileDetection(
                    tile_id=best_det.tile_id,
                    detection_id=i,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    bbox_global=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(score),
                    mask=best_det.mask,  # 원본 mask 사용 (나중에 병합)
                    area_pixels=best_det.area_pixels
                ))

        logger.info(f"WBF 적용: {len(detections)}개 → {len(filtered_detections)}개")
        return filtered_detections

    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 교집합
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        inter_area = (x2_i - x1_i) * (y2_i - y1_i)

        # 합집합
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def postprocess_merge_overlaps(
        self,
        detections: List[TileDetection]
    ) -> List[TileDetection]:
        """
        v11: 이원화된 병합 조건 (경계 타일 vs 일반 타일)

        Tier 1 (강력 병합 - 둘 다 border):
        - IoU >= 0.2, center_distance <= 300px, small_area < 20000px

        Tier 2 (보수적 병합 - 하나 이상 normal):
        - IoU >= 0.45, center_distance <= 120px, small_area < 8000px

        Args:
            detections: WBF 후 검출 결과

        Returns:
            병합된 검출 결과
        """
        # v11: 이원화 파라미터
        tier1_params = {  # 경계 타일 (강력 병합)
            'iou_threshold': 0.2,
            'center_distance_threshold': 300.0,
            'small_object_area': 20000
        }
        tier2_params = {  # 일반 타일 (보수적 병합)
            'iou_threshold': 0.45,
            'center_distance_threshold': 120.0,
            'small_object_area': 8000
        }
        if len(detections) == 0:
            return []

        # 병합 여부 플래그
        merged = [False] * len(detections)
        result = []

        for i in range(len(detections)):
            if merged[i]:
                continue

            det_i = detections[i]
            bbox_i = det_i.bbox_global

            # 중심 좌표 계산
            center_i = (
                (bbox_i[0] + bbox_i[2]) / 2,
                (bbox_i[1] + bbox_i[3]) / 2
            )
            area_i = det_i.area_pixels

            # 병합할 검출들을 모음
            merge_group = [det_i]
            merge_indices = [i]

            for j in range(i + 1, len(detections)):
                if merged[j]:
                    continue

                det_j = detections[j]
                bbox_j = det_j.bbox_global

                # 중심 좌표 계산
                center_j = (
                    (bbox_j[0] + bbox_j[2]) / 2,
                    (bbox_j[1] + bbox_j[3]) / 2
                )
                area_j = det_j.area_pixels

                # IoU 계산
                iou = self._calculate_iou(bbox_i, bbox_j)

                # 중심 거리 계산
                center_distance = np.sqrt(
                    (center_i[0] - center_j[0])**2 +
                    (center_i[1] - center_j[1])**2
                )

                # v11: 이원화 파라미터 선택 (둘 다 border → Tier 1, 하나 이상 normal → Tier 2)
                if det_i.is_border and det_j.is_border:
                    # Tier 1: 강력 병합 (경계 타일끼리)
                    params = tier1_params
                    tier_name = "Tier1(border)"
                else:
                    # Tier 2: 보수적 병합 (하나 이상 normal)
                    params = tier2_params
                    tier_name = "Tier2(normal)"

                iou_threshold = params['iou_threshold']
                center_distance_threshold = params['center_distance_threshold']
                small_object_area = params['small_object_area']

                # 병합 조건 확인
                should_merge = False
                merge_reason = ""

                if iou >= iou_threshold:
                    should_merge = True
                    merge_reason = f"{tier_name}, IoU={iou:.2f}"
                elif center_distance <= center_distance_threshold:
                    should_merge = True
                    merge_reason = f"{tier_name}, center_dist={center_distance:.1f}px"
                elif area_i < small_object_area or area_j < small_object_area:
                    # 둘 중 하나가 작은 객체이고, 중심 거리가 합리적이면 병합
                    if center_distance <= center_distance_threshold * 2:
                        should_merge = True
                        merge_reason = f"{tier_name}, small_obj (area={min(area_i, area_j)}px, dist={center_distance:.1f}px)"

                if should_merge:
                    merge_group.append(det_j)
                    merge_indices.append(j)
                    merged[j] = True
                    logger.debug(f"병합: det_{i} + det_{j} ({merge_reason})")

            # 병합 그룹 처리
            if len(merge_group) == 1:
                # 병합할 대상이 없으면 그대로 추가
                result.append(det_i)
            else:
                # 여러 검출을 병합
                logger.info(f"검출 {i}: {len(merge_group)}개 객체 병합 → 1개")

                # Union bbox 계산
                all_bboxes = [d.bbox_global for d in merge_group]
                union_bbox = [
                    min(b[0] for b in all_bboxes),  # x1
                    min(b[1] for b in all_bboxes),  # y1
                    max(b[2] for b in all_bboxes),  # x2
                    max(b[3] for b in all_bboxes)   # y2
                ]

                # 가장 높은 신뢰도 사용
                max_conf = max(d.confidence for d in merge_group)

                # 전체 면적 합산 (union이므로)
                total_area = sum(d.area_pixels for d in merge_group)

                # 병합된 검출 생성
                merged_det = TileDetection(
                    tile_id=det_i.tile_id,
                    detection_id=f"merged_{i}",
                    bbox=det_i.bbox,  # 로컬 bbox는 첫 번째 것 사용
                    bbox_global=union_bbox,
                    confidence=max_conf,
                    mask=merge_group[0].mask,  # 첫 번째 mask 사용 (간단화)
                    area_pixels=total_area
                )

                result.append(merged_det)

            merged[i] = True

        logger.info(f"후처리 병합: {len(detections)}개 → {len(result)}개 (감소: {len(detections) - len(result)}개)")
        return result

    def resplit_large_mask(
        self,
        mask: np.ndarray,
        bbox_global: List[float],
        target_count: int,
        min_area: int = 1000
    ) -> List[Dict]:
        """
        v11: 과대 마스크를 watershed로 재분할

        Args:
            mask: 글로벌 마스크 (H x W, binary)
            bbox_global: 글로벌 bbox [x1, y1, x2, y2]
            target_count: 목표 분할 개수
            min_area: 최소 인스턴스 면적 (px)

        Returns:
            분할된 인스턴스 리스트 (각각 {'mask', 'bbox', 'area'})
        """
        if mask is None or mask.size == 0:
            return []

        # bbox 영역만 crop
        x1, y1, x2, y2 = [int(b) for b in bbox_global]
        h, w = mask.shape
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return []

        cropped_mask = mask[y1:y2, x1:x2]

        # Distance transform
        dist_transform = cv2.distanceTransform(cropped_mask, cv2.DIST_L2, 5)

        # 거리 변환 정규화 및 threshold
        # target_count를 고려하여 threshold 조정
        dist_max = dist_transform.max()
        if dist_max == 0:
            return []

        # target_count가 클수록 threshold를 높여서 더 많은 seed 생성
        threshold_ratio = min(0.3 + (target_count - 1) * 0.1, 0.7)
        ret, sure_fg = cv2.threshold(dist_transform, threshold_ratio * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Connected components로 마커 생성
        n_labels, markers = cv2.connectedComponents(sure_fg)

        # Watershed
        markers = markers + 1  # background를 1로 만들기 위해
        cropped_mask_3ch = cv2.cvtColor(cropped_mask * 255, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(cropped_mask_3ch, markers)

        # 각 인스턴스 추출
        instances = []
        for label in range(2, n_labels + 1):  # 1은 background, 0은 경계
            instance_mask = (markers == label).astype(np.uint8)
            instance_area = np.sum(instance_mask)

            if instance_area < min_area:
                continue

            # bbox 계산 (cropped 좌표)
            ys, xs = np.where(instance_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            inst_x1, inst_x2 = xs.min(), xs.max()
            inst_y1, inst_y2 = ys.min(), ys.max()

            # 글로벌 좌표로 변환
            inst_bbox_global = [
                x1 + inst_x1,
                y1 + inst_y1,
                x1 + inst_x2,
                y1 + inst_y2
            ]

            # 글로벌 마스크 생성
            global_mask = np.zeros_like(mask, dtype=np.uint8)
            global_mask[y1:y2, x1:x2] = instance_mask

            instances.append({
                'mask': global_mask,
                'bbox': inst_bbox_global,
                'area': int(instance_area)
            })

        logger.info(f"Watershed 재분할: target={target_count}, actual={len(instances)}개 생성")
        return instances

    def apply_quality_filter(
        self,
        detections: List[TileDetection],
        min_confidence: float = 0.4,
        min_area: int = 3000,
        max_aspect_ratio: float = 5.0
    ) -> List[TileDetection]:
        """
        v12: 최종 품질 필터링 (오검출 제거)

        Args:
            detections: 검출 결과
            min_confidence: 최소 신뢰도 (기본 0.4)
            min_area: 최소 면적 (기본 3000px)
            max_aspect_ratio: 최대 가로세로비 (기본 5.0)

        Returns:
            필터링된 검출 결과
        """
        if len(detections) == 0:
            return []

        filtered = []
        stats = {
            'low_confidence': 0,
            'too_small': 0,
            'bad_aspect_ratio': 0
        }

        for det in detections:
            # 신뢰도 필터링
            if det.confidence < min_confidence:
                stats['low_confidence'] += 1
                continue

            # 면적 필터링
            if det.area_pixels < min_area:
                stats['too_small'] += 1
                continue

            # Aspect ratio 필터링
            bbox = det.bbox_global
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = max(width / height, height / width) if height > 0 else 999

            if aspect_ratio > max_aspect_ratio:
                stats['bad_aspect_ratio'] += 1
                continue

            filtered.append(det)

        logger.info(f"품질 필터링: {len(detections)}개 → {len(filtered)}개")
        if sum(stats.values()) > 0:
            logger.info(f"  제거: 낮은 신뢰도 {stats['low_confidence']}개, "
                       f"작은 면적 {stats['too_small']}개, "
                       f"나쁜 비율 {stats['bad_aspect_ratio']}개")

        return filtered

    def calculate_area_based_count(self, detections: List[TileDetection]) -> Dict[str, Any]:
        """
        v11: area 기반 보정 카운터

        Args:
            detections: 최종 검출 결과

        Returns:
            {
                'count_raw': 검출된 객체 개수,
                'count_estimated': area 기반 보정 카운트,
                'area_ref': 단일 객체 기준 면적,
                'estimation_method': 'area-based correction'
            }
        """
        if len(detections) == 0:
            return {
                'count_raw': 0,
                'count_estimated': 0,
                'area_ref': 0,
                'estimation_method': 'area-based correction'
            }

        count_raw = len(detections)
        areas = [d.area_pixels for d in detections]

        # 1. 초기 area_ref 추정 (전체 중앙값)
        initial_ref = np.median(areas)

        # 2. 단일 객체 범위 필터링 (0.5*ref ~ 1.8*ref)
        single_detections = [a for a in areas if 0.5 * initial_ref <= a <= 1.8 * initial_ref]

        # 3. area_ref 재계산 (단일 객체들의 중앙값)
        if len(single_detections) > 0:
            area_ref = np.median(single_detections)
        else:
            area_ref = initial_ref

        # 4. 대형 객체 카운트 보정
        count_estimated = 0
        for area in areas:
            if area > 1.8 * area_ref:
                # 대형 객체: area 기반 개수 추정
                big_count = round(area / area_ref)
                count_estimated += big_count
            else:
                # 일반 객체: 1개로 카운트
                count_estimated += 1

        logger.info(f"Area-based count: raw={count_raw}, estimated={count_estimated}, area_ref={area_ref:.0f}px")

        return {
            'count_raw': int(count_raw),
            'count_estimated': int(count_estimated),
            'area_ref': float(area_ref),
            'estimation_method': 'area-based correction'
        }

    def detect_stacked_layers(
        self,
        mask: np.ndarray,
        min_peak_prominence: float = 0.15,
        min_peak_distance: int = 20,
        gaussian_sigma: float = 3.0
    ) -> int:
        """
        드론 정사영상에서 위아래로 쌓인 곤포 층수 감지 (v10)

        원리:
        1. mask의 y축 픽셀 히스토그램(hist_y) 계산
        2. GaussianBlur로 평활화하여 노이즈 제거
        3. scipy.signal.find_peaks로 봉우리(peaks) 탐지
        4. 각 peak을 곤포 한 층으로 간주

        Args:
            mask: 검출 마스크 (2D binary array, H x W)
            min_peak_prominence: 봉우리 최소 돌출도 (상대값, 0-1)
            min_peak_distance: 봉우리 간 최소 거리 (픽셀)
            gaussian_sigma: 가우시안 블러 시그마

        Returns:
            stacked_layer_count: 감지된 층수 (최소 1)
        """
        if mask is None or mask.size == 0:
            return 1

        # 1. y축 히스토그램 계산 (각 y 좌표에서 x축 픽셀 합)
        hist_y = np.sum(mask, axis=1).astype(np.float32)  # shape: (H,)

        if hist_y.max() == 0:
            return 1  # 빈 마스크

        # 정규화 (0-1)
        hist_y = hist_y / hist_y.max()

        # 2. GaussianBlur로 평활화 (노이즈 제거)
        hist_y_smooth = ndimage.gaussian_filter1d(hist_y, sigma=gaussian_sigma)

        # 3. find_peaks로 봉우리 탐지
        peaks, properties = find_peaks(
            hist_y_smooth,
            prominence=min_peak_prominence,  # 최소 돌출도
            distance=min_peak_distance  # 최소 간격
        )

        # 4. 감지된 peak 수 = 층수
        layer_count = len(peaks)

        # 최소 1층 (검출이 있으면 최소 1층)
        return max(1, layer_count)

    def create_global_masks(
        self,
        detections: List[TileDetection],
        image_shape: Tuple[int, int],
        original_detections: List[TileDetection] = None,
        merge_masks: bool = True
    ) -> List[TileDetection]:
        """
        3️⃣ shapely로 segmentation mask 병합 및 원본 이미지 좌표계로 변환

        Args:
            detections: WBF 적용 후 TileDetection 리스트
            image_shape: 원본 이미지 shape (H, W)
            original_detections: WBF 전 원본 detection 리스트 (mask 병합용)
            merge_masks: shapely로 mask 병합 여부

        Returns:
            mask_global이 추가된 TileDetection 리스트
        """
        h, w = image_shape

        for det in detections:
            # 글로벌 마스크 생성
            global_mask = np.zeros((h, w), dtype=np.uint8)

            if merge_masks and original_detections is not None:
                # 3️⃣ shapely로 중복된 mask 병합
                # 현재 detection과 IoU가 높은 모든 원본 detection 찾기
                overlapping_masks = []
                for orig_det in original_detections:
                    iou = self._calculate_iou(det.bbox_global, orig_det.bbox_global)
                    if iou > 0.3:  # IoU threshold
                        # 원본 detection의 mask를 global 좌표계로 변환
                        orig_x1 = int(orig_det.bbox_global[0])
                        orig_y1 = int(orig_det.bbox_global[1])
                        orig_x2 = int(orig_det.bbox_global[2])
                        orig_y2 = int(orig_det.bbox_global[3])

                        # 이미지 범위 내로 제한
                        orig_x1 = max(0, min(orig_x1, w))
                        orig_y1 = max(0, min(orig_y1, h))
                        orig_x2 = max(0, min(orig_x2, w))
                        orig_y2 = max(0, min(orig_y2, h))

                        # mask를 global 좌표계로 변환
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        mask_h, mask_w = orig_det.mask.shape
                        target_h = orig_y2 - orig_y1
                        target_w = orig_x2 - orig_x1

                        if target_h > 0 and target_w > 0:
                            resized_mask = cv2.resize(orig_det.mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                            resized_mask = (resized_mask > 0.5).astype(np.uint8)
                            temp_mask[orig_y1:orig_y2, orig_x1:orig_x2] = resized_mask

                        overlapping_masks.append(temp_mask)

                # shapely로 mask 병합
                if overlapping_masks:
                    polygons = []
                    for mask in overlapping_masks:
                        # mask를 polygon으로 변환
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 3:
                                coords = contour.squeeze()
                                if coords.ndim == 2 and len(coords) >= 3:
                                    try:
                                        poly = Polygon(coords)
                                        if poly.is_valid:
                                            polygons.append(poly)
                                    except:
                                        pass

                    # shapely로 union
                    if polygons:
                        merged_polygon = unary_union(polygons)

                        # polygon을 mask로 변환
                        if merged_polygon.is_valid and not merged_polygon.is_empty:
                            if merged_polygon.geom_type == 'Polygon':
                                coords = np.array(merged_polygon.exterior.coords, dtype=np.int32)
                                cv2.fillPoly(global_mask, [coords], 1)
                            elif merged_polygon.geom_type == 'MultiPolygon':
                                for poly in merged_polygon.geoms:
                                    coords = np.array(poly.exterior.coords, dtype=np.int32)
                                    cv2.fillPoly(global_mask, [coords], 1)
                else:
                    # overlapping mask가 없으면 원본 mask 사용
                    self._fill_mask_simple(global_mask, det, w, h)
            else:
                # merge_masks가 False이면 단순 복사
                self._fill_mask_simple(global_mask, det, w, h)

            det.mask_global = global_mask

            # v10: 쌓인 층수 감지 (mask_global 기반)
            det.stacked_layer_count = self.detect_stacked_layers(global_mask)

        return detections

    def _fill_mask_simple(self, global_mask, det, w, h):
        """단순 mask 복사 (병합 없이)"""
        x1_global = int(det.bbox_global[0])
        y1_global = int(det.bbox_global[1])
        x2_global = int(det.bbox_global[2])
        y2_global = int(det.bbox_global[3])

        # 이미지 범위 내로 제한
        x1_global = max(0, min(x1_global, w))
        y1_global = max(0, min(y1_global, h))
        x2_global = max(0, min(x2_global, w))
        y2_global = max(0, min(y2_global, h))

        # mask 크기
        mask_h, mask_w = det.mask.shape
        target_h = y2_global - y1_global
        target_w = x2_global - x1_global

        # mask를 bbox 크기에 맞게 리사이즈 (필요한 경우)
        if mask_h != target_h or mask_w != target_w:
            if target_h > 0 and target_w > 0:
                resized_mask = cv2.resize(det.mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_mask = (resized_mask > 0.5).astype(np.uint8)
            else:
                resized_mask = det.mask
        else:
            resized_mask = det.mask

        # 복사할 영역 크기
        copy_h = min(resized_mask.shape[0], target_h)
        copy_w = min(resized_mask.shape[1], target_w)

        # 글로벌 마스크에 복사
        if copy_h > 0 and copy_w > 0:
            global_mask[y1_global:y1_global+copy_h, x1_global:x1_global+copy_w] = resized_mask[:copy_h, :copy_w]

    def count_silage_bales(
        self,
        detections: List[TileDetection],
        min_confidence: float = 0.5,
        min_area: int = 100,
        max_area: int = 100000
    ) -> Dict[str, Any]:
        """
        Count individual silage bales with statistics

        Args:
            detections: List of TileDetection objects
            min_confidence: Minimum confidence threshold for counting
            min_area: Minimum area in pixels for a valid bale
            max_area: Maximum area in pixels for a valid bale

        Returns:
            Dictionary with counting statistics
        """
        valid_detections = []
        filtered_out = {
            'low_confidence': 0,
            'too_small': 0,
            'too_large': 0
        }

        for det in detections:
            # Check confidence
            if det.confidence < min_confidence:
                filtered_out['low_confidence'] += 1
                continue

            # Check area
            if det.area_pixels < min_area:
                filtered_out['too_small'] += 1
                continue
            if det.area_pixels > max_area:
                filtered_out['too_large'] += 1
                continue

            valid_detections.append(det)

        # Calculate statistics
        if valid_detections:
            areas = [d.area_pixels for d in valid_detections]
            confidences = [d.confidence for d in valid_detections]

            area_stats = {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': int(np.min(areas)),
                'max': int(np.max(areas)),
                'median': float(np.median(areas))
            }

            confidence_stats = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
        else:
            area_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
            confidence_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

        return {
            'total_count': len(valid_detections),
            'filtered_out': filtered_out,
            'total_filtered': sum(filtered_out.values()),
            'area_stats': area_stats,
            'confidence_stats': confidence_stats,
            'filter_parameters': {
                'min_confidence': min_confidence,
                'min_area': min_area,
                'max_area': max_area
            }
        }

    def visualize_results(
        self,
        image: np.ndarray,
        detections: List[TileDetection],
        tiles: List[TileInfo] = None,
        show_tiles: bool = True
    ) -> np.ndarray:
        """
        검출 결과 시각화

        Args:
            image: 원본 이미지
            detections: TileDetection 리스트
            tiles: TileInfo 리스트 (타일 경계 표시용)
            show_tiles: 타일 경계 표시 여부

        Returns:
            시각화 이미지
        """
        vis_img = image.copy()

        # 타일 경계 표시
        if show_tiles and tiles:
            for tile in tiles:
                x1, y1 = tile.x_offset, tile.y_offset
                x2, y2 = x1 + tile.width, y1 + tile.height
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # 검출 결과 표시
        for i, det in enumerate(detections):
            # 마스크 오버레이
            if det.mask_global is not None:
                mask = det.mask_global
                color = np.array([0, 255, 0], dtype=np.uint8)  # 초록색

                overlay = vis_img.copy()
                overlay[mask > 0] = overlay[mask > 0] * 0.5 + color * 0.5
                vis_img = overlay.astype(np.uint8)

            # Bounding box
            x1, y1, x2, y2 = map(int, det.bbox_global)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트
            label = f"#{i} {det.confidence:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_img

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        save_tiles: bool = False,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        단일 이미지 처리 (타일 기반)

        Args:
            image_path: 이미지 경로
            output_dir: 출력 디렉토리
            save_tiles: 타일 이미지 저장 여부
            save_visualization: 시각화 저장 여부

        Returns:
            처리 결과 딕셔너리
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info(f"이미지 처리 시작: {image_path.name}")
        logger.info("=" * 80)

        # 이미지 로드
        logger.info("이미지 로드 중...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        logger.info(f"이미지 크기: {w}x{h} ({w*h/1e6:.2f}M 픽셀)")

        # 타일 생성
        start_time = time.time()
        tiles = self.create_tiles(image)
        tile_time = time.time() - start_time

        # 타일 이미지 저장 (옵션)
        if save_tiles:
            tiles_dir = output_dir / f"{image_path.stem}_tiles"
            tiles_dir.mkdir(exist_ok=True)
            for tile in tiles:
                tile_path = tiles_dir / f"tile_{tile.tile_id:04d}.png"
                tile_bgr = cv2.cvtColor(tile.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(tile_path), tile_bgr)
            logger.info(f"타일 이미지 저장: {tiles_dir}")

        # v11: 타일 추론 (2-pass: 1차 전체, 2차 경계만)
        start_time = time.time()

        # 1차 추론: 모든 타일을 정상 파라미터로 추론
        logger.info(f"1차 추론 시작: {len(tiles)}개 타일 (정상 파라미터)")
        normal_detections = []
        for tile in tqdm(tiles, desc="1차 추론"):
            detections = self.predict_tile(tile)
            normal_detections.extend(detections)
        logger.info(f"1차 추론 완료: {len(normal_detections)}개 검출")

        # 2차 추론: 경계 타일만 더 공격적으로 재추론
        border_tiles = [t for t in tiles if t.is_border]
        logger.info(f"2차 추론 시작: {len(border_tiles)}개 경계 타일 (conf=0.28, scale=1.7)")
        border_detections = []
        for tile in tqdm(border_tiles, desc="2차 추론 (경계)"):
            detections = self.predict_tile(tile, conf_override=0.28, scale_override=1.7)
            border_detections.extend(detections)
        logger.info(f"2차 추론 완료: {len(border_detections)}개 추가 검출")

        # 최종: 1차 + 2차 결과 병합
        all_detections = normal_detections + border_detections
        logger.info(f"총 검출: {len(all_detections)}개 (1차 {len(normal_detections)} + 2차 {len(border_detections)})")

        inference_time = time.time() - start_time

        # 4️⃣ WBF 적용 (NMS 대신)
        start_time = time.time()
        filtered_detections = self.apply_wbf(all_detections, (h, w))
        wbf_time = time.time() - start_time

        # v11: 과대 마스크 재분할 (area > 120000)
        # 먼저 글로벌 마스크 생성 필요 (재분할을 위해)
        filtered_detections = self.create_global_masks(
            filtered_detections,
            (h, w),
            original_detections=all_detections,
            merge_masks=False
        )

        # area_ref 계산 (단일 검출들의 중앙값 면적)
        areas = [d.area_pixels for d in filtered_detections]
        if len(areas) > 0:
            area_ref = np.median(areas)
        else:
            area_ref = 50000  # 기본값

        logger.info(f"area_ref (중앙값): {area_ref:.0f}px")

        # 대형 마스크 감지 및 재분할
        large_threshold = 120000
        resplit_detections = []
        split_count = 0

        for det in filtered_detections:
            if det.area_pixels > large_threshold:
                # 목표 분할 개수 계산
                target_count = max(2, int(det.area_pixels / area_ref * 0.8))
                logger.info(f"대형 마스크 감지: area={det.area_pixels}px, target_count={target_count}")

                # watershed 재분할
                instances = self.resplit_large_mask(
                    det.mask_global,
                    det.bbox_global,
                    target_count=target_count,
                    min_area=1000
                )

                if len(instances) > 1:
                    # 재분할 성공 - 각 인스턴스를 새 TileDetection으로 추가
                    for i, inst in enumerate(instances):
                        resplit_det = TileDetection(
                            tile_id=det.tile_id,
                            detection_id=f"resplit_{det.detection_id}_{i}",
                            bbox=det.bbox,  # 로컬 bbox는 원본 사용
                            bbox_global=inst['bbox'],
                            confidence=det.confidence,
                            mask=inst['mask'],
                            mask_global=inst['mask'],
                            area_pixels=inst['area'],
                            is_border=det.is_border
                        )
                        resplit_detections.append(resplit_det)
                    split_count += 1
                else:
                    # 재분할 실패 - 원본 유지
                    resplit_detections.append(det)
            else:
                # 정상 크기 - 원본 유지
                resplit_detections.append(det)

        if split_count > 0:
            logger.info(f"대형 마스크 재분할: {split_count}개 → {len(resplit_detections)}개 (증가: {len(resplit_detections) - len(filtered_detections)}개)")
            filtered_detections = resplit_detections

        # v11: 후처리 병합 (이원화 조건 - 경계 타일은 강력, 일반 타일은 보수적)
        filtered_detections = self.postprocess_merge_overlaps(filtered_detections)

        # v13: 품질 필터링 (오검출 완전 제거)
        filtered_detections = self.apply_quality_filter(
            filtered_detections,
            min_confidence=0.5,  # 신뢰도 0.5 이상만 유지 (v13: 0.4→0.5)
            min_area=5000,  # 5000px 이상 (v13: 3000→5000)
            max_aspect_ratio=5.0  # 가로세로비 5:1 이하
        )

        # 3️⃣ 글로벌 마스크 생성 (v6: merge_masks=False로 개별 인스턴스 유지)
        filtered_detections = self.create_global_masks(
            filtered_detections,
            (h, w),
            original_detections=all_detections,  # 원본 detection 전달
            merge_masks=False  # v6: shapely 병합 비활성화 (개별 인스턴스 유지)
        )

        # 시각화
        if save_visualization:
            logger.info("시각화 생성 중...")
            vis_img = self.visualize_results(image, filtered_detections, tiles, show_tiles=True)
            vis_path = output_dir / f"{image_path.stem}_result.png"
            vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            logger.info(f"시각화 저장: {vis_path}")

        # v11: Area-based corrected counting
        counting_results = self.calculate_area_based_count(filtered_detections)

        # 통계
        confidences = [d.confidence for d in filtered_detections] if filtered_detections else [0.0]

        # Prepare detailed detection list with area_pixels and stacked_layer_count (v10)
        detection_details = []
        for i, det in enumerate(filtered_detections):
            detection_details.append({
                'id': i,
                'bbox': det.bbox_global,
                'confidence': float(det.confidence),
                'area_pixels': int(det.area_pixels),  # Ensure area_pixels is included
                'stacked_layer_count': int(det.stacked_layer_count),  # v10: 쌓인 층수
                'tile_id': det.tile_id
            })

        results = {
            'image_name': image_path.name,
            'image_size': {'width': w, 'height': h, 'megapixels': w*h/1e6},
            'tile_config': {
                'tile_size': self.tile_size,
                'overlap_ratio': self.overlap_ratio,
                'overlap_pixels': self.overlap_pixels,
                'stride': self.stride,
                'num_tiles': len(tiles),
                'inference_scale': self.inference_scale  # Include inference scale
            },
            'detections': {
                'total_before_wbf': len(all_detections),
                'total_after_wbf': len(filtered_detections),
                'removed_by_wbf': len(all_detections) - len(filtered_detections),
                'wbf_removal_rate': (len(all_detections) - len(filtered_detections)) / max(len(all_detections), 1),
                'details': detection_details  # Detailed detection list with area_pixels
            },
            'silage_bale_count': counting_results,  # NEW: Automatic counting results
            'confidence': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'timing': {
                'tiling_sec': tile_time,
                'inference_sec': inference_time,
                'wbf_sec': wbf_time,
                'total_sec': tile_time + inference_time + wbf_time
            }
        }

        # 결과 저장
        results_path = output_dir / f"{image_path.stem}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"결과 저장: {results_path}")

        # 요약 출력
        logger.info("=" * 80)
        logger.info("처리 완료 요약 (v7: WBF + Instance Separation + Scaling)")
        logger.info("=" * 80)
        logger.info(f"타일 개수: {results['tile_config']['num_tiles']}")
        logger.info(f"추론 스케일: {results['tile_config']['inference_scale']}x")
        logger.info(f"WBF 전 검출: {results['detections']['total_before_wbf']}개")
        logger.info(f"WBF 후 검출: {results['detections']['total_after_wbf']}개")
        logger.info(f"제거율: {results['detections']['wbf_removal_rate']*100:.1f}%")
        logger.info("-" * 80)
        logger.info("곤포사일리지 개수 (v11: area 기반 보정 카운팅):")
        logger.info(f"  - 검출 개수 (raw): {results['silage_bale_count']['count_raw']}개")
        logger.info(f"  - 보정 개수 (estimated): {results['silage_bale_count']['count_estimated']}개")
        logger.info(f"  - 기준 면적 (area_ref): {results['silage_bale_count']['area_ref']:.0f} pixels")
        logger.info(f"  - 방법: {results['silage_bale_count']['estimation_method']}")
        logger.info("-" * 80)
        logger.info(f"처리 시간: {results['timing']['total_sec']:.2f}초")
        logger.info("=" * 80)

        return results


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='타일 기반 곤포사일리지 추론')
    parser.add_argument('--model', type=str, required=True, help='모델 경로 (best.pt)')
    parser.add_argument('--image', type=str, nargs='+', required=True, help='이미지 경로(들)')
    parser.add_argument('--output', type=str, default='inference_system/output_tiled', help='출력 디렉토리')
    parser.add_argument('--tile-size', type=int, default=1024, help='타일 크기')
    parser.add_argument('--overlap', type=float, default=0.25, help='오버랩 비율 (0.0~1.0, 기본값: 0.25)')
    parser.add_argument('--scale', type=float, default=1.5, help='추론 시 스케일링 배율 (1.5 = 1.5x 확대로 작은 객체 검출 향상)')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU 임계값')
    parser.add_argument('--nms-iou', type=float, default=0.5, help='타일 간 WBF IoU 임계값')
    parser.add_argument('--min-area', type=int, default=50, help='개별 인스턴스 최소 면적 (픽셀)')
    parser.add_argument('--min-circularity', type=float, default=0.01, help='개별 인스턴스 최소 원형도 (0~1)')
    parser.add_argument('--morphology-kernel', type=int, default=1, help='Morphology 연산 커널 크기')
    parser.add_argument('--save-tiles', action='store_true', help='타일 이미지 저장')
    parser.add_argument('--device', type=str, default='auto', help='디바이스 (auto/cuda/cpu)')

    args = parser.parse_args()

    # 엔진 초기화
    logger.info("타일 기반 추론 엔진 초기화...")
    engine = TiledInferenceEngine(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        nms_iou_threshold=args.nms_iou,
        min_instance_area=args.min_area,
        min_circularity=args.min_circularity,
        morphology_kernel_size=args.morphology_kernel,
        inference_scale=args.scale,  # NEW: Pass inference scale parameter
        device=args.device
    )

    # 이미지 처리
    all_results = []
    for image_path in args.image:
        try:
            result = engine.process_image(
                image_path=image_path,
                output_dir=args.output,
                save_tiles=args.save_tiles,
                save_visualization=True
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"이미지 처리 실패 ({image_path}): {e}")
            import traceback
            traceback.print_exc()

    # 전체 요약
    if len(all_results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("전체 처리 요약")
        logger.info("=" * 80)
        total_detections = sum(r['detections']['total_after_wbf'] for r in all_results)
        total_bales = sum(r['silage_bale_count']['total_count'] for r in all_results)
        avg_confidence = np.mean([r['confidence']['mean'] for r in all_results])
        total_time = sum(r['timing']['total_sec'] for r in all_results)

        logger.info(f"처리 이미지 수: {len(all_results)}")
        logger.info(f"총 검출 개수: {total_detections}")
        logger.info(f"총 곤포사일리지 개수: {total_bales}")
        logger.info(f"평균 신뢰도: {avg_confidence*100:.1f}%")
        logger.info(f"총 처리 시간: {total_time:.2f}초")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
