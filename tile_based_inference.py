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
from tqdm import tqdm

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


class TiledInferenceEngine:
    """타일 기반 추론 엔진"""

    def __init__(
        self,
        model_path: str,
        tile_size: int = 1024,
        overlap_ratio: float = 0.25,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_iou_threshold: float = 0.5,
        device: str = 'auto'
    ):
        """
        Args:
            model_path: YOLO 모델 경로
            tile_size: 타일 크기 (정사각형)
            overlap_ratio: 타일 간 오버랩 비율 (0.0 ~ 1.0)
            conf_threshold: 검출 신뢰도 임계값
            iou_threshold: YOLO NMS IoU 임계값
            nms_iou_threshold: 타일 간 중복 제거 NMS IoU 임계값
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

    def create_tiles(self, image: np.ndarray) -> List[TileInfo]:
        """
        이미지를 타일로 분할

        Args:
            image: 원본 이미지 (H, W, 3)

        Returns:
            TileInfo 리스트
        """
        h, w = image.shape[:2]
        tiles = []
        tile_id = 0

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
                    image=tile_image
                ))

                tile_id += 1

                # 다음 x 위치로 이동
                x += self.stride
                if x >= w:
                    break

            # 다음 y 위치로 이동
            y += self.stride
            if y >= h:
                break

        logger.info(f"타일 생성 완료: {len(tiles)}개")
        return tiles

    def predict_tile(self, tile: TileInfo) -> List[TileDetection]:
        """
        단일 타일 추론

        Args:
            tile: TileInfo 객체

        Returns:
            TileDetection 리스트
        """
        try:
            # YOLO 추론
            results = self.model.predict(
                tile.image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.tile_size,
                device=self.device,
                verbose=False
            )[0]

            detections = []

            if results.masks is not None and len(results.masks) > 0:
                masks_data = results.masks.data.cpu().numpy()  # (N, H, W)
                boxes_data = results.boxes.data.cpu().numpy()  # (N, 6)

                for i, (mask, box) in enumerate(zip(masks_data, boxes_data)):
                    x1, y1, x2, y2, conf, cls = box

                    # 마스크 크기 조정
                    if mask.shape != (self.tile_size, self.tile_size):
                        mask = cv2.resize(mask, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)

                    # Binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # 유효한 영역만 (패딩 제외)
                    valid_mask = binary_mask[:tile.height, :tile.width]

                    # bbox 영역에 해당하는 mask만 추출
                    x1_int, y1_int = int(x1), int(y1)
                    x2_int, y2_int = int(x2), int(y2)

                    # bbox 범위를 타일 크기 내로 제한
                    x1_int = max(0, min(x1_int, tile.width))
                    y1_int = max(0, min(y1_int, tile.height))
                    x2_int = max(0, min(x2_int, tile.width))
                    y2_int = max(0, min(y2_int, tile.height))

                    # bbox 영역의 mask만 crop
                    cropped_mask = valid_mask[y1_int:y2_int, x1_int:x2_int]

                    # 글로벌 좌표 계산
                    bbox_global = [
                        x1 + tile.x_offset,
                        y1 + tile.y_offset,
                        x2 + tile.x_offset,
                        y2 + tile.y_offset
                    ]

                    detections.append(TileDetection(
                        tile_id=tile.tile_id,
                        detection_id=i,
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        bbox_global=bbox_global,
                        confidence=float(conf),
                        mask=cropped_mask,  # bbox 영역만 crop된 mask 사용
                        area_pixels=int(cropped_mask.sum())
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

    def apply_nms(self, detections: List[TileDetection]) -> List[TileDetection]:
        """
        NMS로 중복 검출 제거

        Args:
            detections: TileDetection 리스트

        Returns:
            NMS 적용 후 TileDetection 리스트
        """
        if len(detections) == 0:
            return []

        # 글로벌 좌표계 bbox와 confidence 추출
        boxes = np.array([d.bbox_global for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        # NMS 적용 (torchvision 사용)
        import torchvision
        boxes_tensor = torch.from_numpy(boxes)
        scores_tensor = torch.from_numpy(scores)

        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, self.nms_iou_threshold)
        keep_indices = keep_indices.cpu().numpy()

        # 유지할 검출 결과만 선택
        filtered_detections = [detections[i] for i in keep_indices]

        logger.info(f"NMS 적용 중: {len(detections)}개 → {len(filtered_detections)}개")
        return filtered_detections

    def create_global_masks(
        self,
        detections: List[TileDetection],
        image_shape: Tuple[int, int]
    ) -> List[TileDetection]:
        """
        타일 마스크를 원본 이미지 좌표계로 변환

        Args:
            detections: TileDetection 리스트
            image_shape: 원본 이미지 shape (H, W)

        Returns:
            mask_global이 추가된 TileDetection 리스트
        """
        h, w = image_shape

        for det in detections:
            # 글로벌 마스크 생성
            global_mask = np.zeros((h, w), dtype=np.uint8)

            # bbox_global 좌표 가져오기
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

            det.mask_global = global_mask

        return detections

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

        # 타일 추론
        start_time = time.time()
        all_detections = self.predict_all_tiles(tiles)
        inference_time = time.time() - start_time

        # NMS 적용
        start_time = time.time()
        filtered_detections = self.apply_nms(all_detections)
        nms_time = time.time() - start_time

        # 글로벌 마스크 생성
        filtered_detections = self.create_global_masks(filtered_detections, (h, w))

        # 시각화
        if save_visualization:
            logger.info("시각화 생성 중...")
            vis_img = self.visualize_results(image, filtered_detections, tiles, show_tiles=True)
            vis_path = output_dir / f"{image_path.stem}_result.png"
            vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            logger.info(f"시각화 저장: {vis_path}")

        # 통계
        confidences = [d.confidence for d in filtered_detections] if filtered_detections else [0.0]

        results = {
            'image_name': image_path.name,
            'image_size': {'width': w, 'height': h, 'megapixels': w*h/1e6},
            'tile_config': {
                'tile_size': self.tile_size,
                'overlap_ratio': self.overlap_ratio,
                'overlap_pixels': self.overlap_pixels,
                'stride': self.stride,
                'num_tiles': len(tiles)
            },
            'detections': {
                'total_before_nms': len(all_detections),
                'total_after_nms': len(filtered_detections),
                'removed_by_nms': len(all_detections) - len(filtered_detections),
                'nms_removal_rate': (len(all_detections) - len(filtered_detections)) / max(len(all_detections), 1)
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'timing': {
                'tiling_sec': tile_time,
                'inference_sec': inference_time,
                'nms_sec': nms_time,
                'total_sec': tile_time + inference_time + nms_time
            }
        }

        # 결과 저장
        results_path = output_dir / f"{image_path.stem}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"결과 저장: {results_path}")

        # 요약 출력
        logger.info("=" * 80)
        logger.info("처리 완료 요약")
        logger.info("=" * 80)
        logger.info(f"타일 개수: {results['tile_config']['num_tiles']}")
        logger.info(f"NMS 전 검출: {results['detections']['total_before_nms']}개")
        logger.info(f"NMS 후 검출: {results['detections']['total_after_nms']}개")
        logger.info(f"제거율: {results['detections']['nms_removal_rate']*100:.1f}%")
        logger.info(f"평균 신뢰도: {results['confidence']['mean']*100:.1f}%")
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
    parser.add_argument('--overlap', type=float, default=0.25, help='오버랩 비율 (0.0~1.0)')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU 임계값')
    parser.add_argument('--nms-iou', type=float, default=0.5, help='타일 간 NMS IoU 임계값')
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
        total_detections = sum(r['detections']['total_after_nms'] for r in all_results)
        avg_confidence = np.mean([r['confidence']['mean'] for r in all_results])
        total_time = sum(r['timing']['total_sec'] for r in all_results)

        logger.info(f"처리 이미지 수: {len(all_results)}")
        logger.info(f"총 검출 개수: {total_detections}")
        logger.info(f"평균 신뢰도: {avg_confidence*100:.1f}%")
        logger.info(f"총 처리 시간: {total_time:.2f}초")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
