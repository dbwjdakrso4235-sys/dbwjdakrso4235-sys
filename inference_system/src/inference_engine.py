#!/usr/bin/env python3
"""
곤포사일리지 추론 엔진

기능:
- 크롭된 영역에서 곤포사일리지 검출
- 우리가 학습한 YOLOv11n-seg 모델 활용 (mAP50: 92.2%)
- 대용량 이미지를 타일 기반으로 처리
- GeoPackage 형식 결과 출력

작성일: 2025-10-23
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

import numpy as np
import cv2
import torch
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from rasterio.features import shapes
from tqdm import tqdm

# 크롭 모듈 import
sys.path.append(str(Path(__file__).parent))
from crop_processor import CroppedRegion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """검출 결과 데이터"""
    polygon_id: str
    detections: List[Dict[str, Any]]  # 개별 검출 객체 리스트
    count: int  # 총 검출 개수
    confidence_mean: float  # 평균 신뢰도
    masks: Optional[np.ndarray] = None  # 통합 마스크 (H, W)
    visualization: Optional[np.ndarray] = None  # 시각화 이미지


class InferenceEngine:
    """곤포사일리지 추론 엔진"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto',
        imgsz: int = 1024
    ):
        """
        Args:
            model_path: 학습된 모델 경로 (우리가 학습한 best.pt)
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
            device: 디바이스 ('auto', 'cuda', 'cpu')
            imgsz: 추론 이미지 크기
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

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

    def predict_image(
        self,
        image: np.ndarray,
        visualize: bool = True
    ) -> DetectionResult:
        """
        단일 이미지 추론

        Args:
            image: RGB 이미지 (H, W, 3), dtype=uint8
            visualize: 시각화 이미지 생성 여부

        Returns:
            DetectionResult 객체
        """
        try:
            # YOLO 추론
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )[0]

            # 결과 파싱
            detections = []
            masks_list = []

            if results.masks is not None and len(results.masks) > 0:
                masks_data = results.masks.data.cpu().numpy()  # (N, H, W)
                boxes_data = results.boxes.data.cpu().numpy()  # (N, 6) [x1,y1,x2,y2,conf,cls]

                for i, (mask, box) in enumerate(zip(masks_data, boxes_data)):
                    x1, y1, x2, y2, conf, cls = box

                    # 마스크 크기 조정 (원본 이미지 크기에 맞춤)
                    h, w = image.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

                    # Binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    masks_list.append(binary_mask)

                    # 검출 정보
                    detection = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': '곤포사일리지',
                        'mask': binary_mask,
                        'area_pixels': int(binary_mask.sum())
                    }
                    detections.append(detection)

            # 통합 마스크 생성
            if masks_list:
                integrated_mask = np.max(np.stack(masks_list, axis=0), axis=0)
            else:
                integrated_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # 시각화
            vis_image = None
            if visualize and detections:
                vis_image = self._visualize_results(image, detections)

            # DetectionResult 생성
            result = DetectionResult(
                polygon_id='',  # 외부에서 설정
                detections=detections,
                count=len(detections),
                confidence_mean=float(np.mean([d['confidence'] for d in detections])) if detections else 0.0,
                masks=integrated_mask,
                visualization=vis_image
            )

            return result

        except Exception as e:
            logger.error(f"추론 실패: {e}")
            return DetectionResult(
                polygon_id='',
                detections=[],
                count=0,
                confidence_mean=0.0
            )

    def _visualize_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        검출 결과 시각화

        Args:
            image: 원본 이미지
            detections: 검출 결과 리스트

        Returns:
            시각화 이미지 (RGB)
        """
        vis_img = image.copy()

        for det in detections:
            # 마스크 오버레이
            mask = det['mask']
            color = np.array([0, 255, 0], dtype=np.uint8)  # 초록색

            # 투명도 적용
            overlay = vis_img.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + color * 0.5
            vis_img = overlay.astype(np.uint8)

            # Bounding box
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트 (ID, 신뢰도)
            label = f"#{det['id']} {det['confidence']:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_img

    def process_cropped_region(
        self,
        cropped_region: CroppedRegion,
        visualize: bool = True,
        save_visualization: bool = False,
        output_dir: Optional[Path] = None
    ) -> DetectionResult:
        """
        크롭된 영역 처리

        Args:
            cropped_region: CroppedRegion 객체
            visualize: 시각화 여부
            save_visualization: 시각화 저장 여부
            output_dir: 시각화 저장 디렉토리

        Returns:
            DetectionResult
        """
        logger.info(f"추론 시작: {cropped_region.polygon_id}")

        # 추론
        result = self.predict_image(cropped_region.image, visualize=visualize)
        result.polygon_id = cropped_region.polygon_id

        logger.info(f"추론 완료: {result.count}개 검출 (평균 신뢰도: {result.confidence_mean:.2%})")

        # 시각화 저장
        if save_visualization and result.visualization is not None:
            if output_dir is None:
                output_dir = Path("inference_system/output/visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)

            vis_path = output_dir / f"{result.polygon_id}_result.png"
            vis_bgr = cv2.cvtColor(result.visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_path), vis_bgr)
            logger.debug(f"시각화 저장: {vis_path}")

        return result

    def batch_process(
        self,
        cropped_regions: List[CroppedRegion],
        visualize: bool = True,
        save_visualization: bool = False,
        output_dir: Optional[Path] = None
    ) -> List[DetectionResult]:
        """
        여러 크롭 영역 배치 처리

        Args:
            cropped_regions: CroppedRegion 리스트
            visualize: 시각화 여부
            save_visualization: 시각화 저장 여부
            output_dir: 저장 디렉토리

        Returns:
            DetectionResult 리스트
        """
        logger.info(f"배치 추론 시작: {len(cropped_regions)}개 영역")

        results = []
        for cropped in tqdm(cropped_regions, desc="추론 진행"):
            result = self.process_cropped_region(
                cropped,
                visualize=visualize,
                save_visualization=save_visualization,
                output_dir=output_dir
            )
            results.append(result)

        # 통계
        total_detections = sum(r.count for r in results)
        avg_confidence = np.mean([r.confidence_mean for r in results if r.count > 0])

        logger.info("=" * 80)
        logger.info(f"배치 추론 완료")
        logger.info(f"총 검출: {total_detections}개")
        logger.info(f"평균 신뢰도: {avg_confidence:.2%}")
        logger.info(f"검출 성공률: {sum(1 for r in results if r.count > 0)}/{len(results)} ({sum(1 for r in results if r.count > 0)/len(results)*100:.1f}%)")
        logger.info("=" * 80)

        return results

    def save_to_geopackage(
        self,
        results: List[DetectionResult],
        cropped_regions: List[CroppedRegion],
        output_path: Path
    ):
        """
        결과를 GeoPackage로 저장

        Args:
            results: DetectionResult 리스트
            cropped_regions: CroppedRegion 리스트 (좌표 변환용)
            output_path: 출력 GeoPackage 경로
        """
        geometries = []

        for result, cropped in zip(results, cropped_regions):
            if result.count == 0:
                continue

            # 각 검출 객체를 폴리곤으로 변환
            for det in result.detections:
                mask = det['mask']

                # 마스크 → 폴리곤 변환
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    # 픽셀 좌표 → 지리 좌표 변환
                    coords = []
                    for point in contour[:, 0, :]:
                        x, y = point
                        # Affine transform 적용
                        geo_x, geo_y = cropped.transform * (x, y)
                        coords.append((geo_x, geo_y))

                    if len(coords) >= 3:
                        polygon = Polygon(coords)

                        geometries.append({
                            'geometry': polygon,
                            'polygon_id': result.polygon_id,
                            'detection_id': det['id'],
                            'confidence': det['confidence'],
                            'class_name': det['class_name'],
                            'area_pixels': det['area_pixels'],
                            'area_m2': polygon.area
                        })

        if not geometries:
            logger.warning("저장할 검출 결과가 없습니다.")
            return

        # GeoDataFrame 생성
        gdf = gpd.GeoDataFrame(geometries, crs=cropped_regions[0].crs)

        # 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver='GPKG')

        logger.info(f"GeoPackage 저장 완료: {output_path}")
        logger.info(f"총 {len(geometries)}개 객체 저장")


def main():
    """테스트 실행"""
    import argparse
    from crop_processor import CropProcessor

    parser = argparse.ArgumentParser(description='곤포사일리지 추론')
    parser.add_argument('--model', type=str, required=True, help='모델 경로 (best.pt)')
    parser.add_argument('--tif', type=str, required=True, help='TIF 파일')
    parser.add_argument('--shp', type=str, required=True, help='Shapefile')
    parser.add_argument('--output', type=str, default='inference_system/output', help='출력 디렉토리')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--limit', type=int, default=None, help='처리 제한')
    parser.add_argument('--save-vis', action='store_true', help='시각화 저장')

    args = parser.parse_args()

    # 1. 크롭 처리
    logger.info("Step 1: SHP 기반 TIF 크롭")
    crop_processor = CropProcessor(args.tif, args.shp)

    polygon_ids = list(range(len(crop_processor.gdf)))
    if args.limit:
        polygon_ids = polygon_ids[:args.limit]

    cropped_regions = crop_processor.batch_crop(
        polygon_ids=polygon_ids,
        save_images=False
    )

    # 2. 추론
    logger.info("Step 2: 곤포사일리지 검출")
    engine = InferenceEngine(
        model_path=args.model,
        conf_threshold=args.conf
    )

    results = engine.batch_process(
        cropped_regions=cropped_regions,
        save_visualization=args.save_vis,
        output_dir=Path(args.output) / "visualizations"
    )

    # 3. GeoPackage 저장
    logger.info("Step 3: GeoPackage 저장")
    output_gpkg = Path(args.output) / "silage_bale_detections.gpkg"
    engine.save_to_geopackage(results, cropped_regions, output_gpkg)

    logger.info("=" * 80)
    logger.info("처리 완료!")
    logger.info(f"결과: {output_gpkg}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
