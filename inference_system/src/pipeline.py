#!/usr/bin/env python3
"""
곤포사일리지 통합 추론 파이프라인

전체 워크플로우:
1. SHP 파일에서 폴리곤 로드
2. 각 폴리곤별로 대용량 TIF 크롭
3. 크롭된 영역에서 곤포사일리지 검출 (YOLOv11)
4. 결과를 GeoPackage로 저장
5. 통계 보고서 생성

작성일: 2025-10-23
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

# 로컬 모듈 import
sys.path.append(str(Path(__file__).parent))
from crop_processor import CropProcessor, CroppedRegion
from inference_engine import InferenceEngine, DetectionResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SilageBaleDetectionPipeline:
    """곤포사일리지 통합 검출 파이프라인"""

    def __init__(
        self,
        tif_path: str,
        shp_path: str,
        model_path: str,
        output_dir: str = "inference_system/output",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Args:
            tif_path: 대용량 TIF 파일 경로
            shp_path: Shapefile 경로
            model_path: 학습된 YOLO 모델 경로
            output_dir: 결과 저장 디렉토리
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
            device: 디바이스 설정
        """
        self.tif_path = Path(tif_path)
        self.shp_path = Path(shp_path)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "cropped_images").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # 컴포넌트 초기화
        self.crop_processor = None
        self.inference_engine = None

        self.start_time = None
        self.stats = {}

    def initialize(self):
        """컴포넌트 초기화"""
        logger.info("=" * 80)
        logger.info("곤포사일리지 검출 파이프라인 초기화")
        logger.info("=" * 80)

        # CropProcessor 초기화
        logger.info("Step 1: CropProcessor 초기화")
        self.crop_processor = CropProcessor(str(self.tif_path), str(self.shp_path))

        # Shapefile 통계
        shp_stats = self.crop_processor.get_statistics()
        logger.info(f"총 폴리곤: {shp_stats['total_polygons']}개")
        logger.info(f"평균 면적: {shp_stats['area_stats']['mean']:.0f} m²")

        # InferenceEngine 초기화
        logger.info("Step 2: InferenceEngine 초기화")
        self.inference_engine = InferenceEngine(
            model_path=str(self.model_path),
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device
        )

        logger.info("초기화 완료")
        logger.info("=" * 80)

    def run(
        self,
        polygon_ids: Optional[List[int]] = None,
        min_area: float = 0,
        max_area: float = float('inf'),
        save_cropped: bool = False,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        파이프라인 실행

        Args:
            polygon_ids: 처리할 폴리곤 ID 리스트 (None이면 전체)
            min_area: 최소 면적 필터 (m²)
            max_area: 최대 면적 필터 (m²)
            save_cropped: 크롭 이미지 저장 여부
            save_visualization: 시각화 저장 여부

        Returns:
            처리 결과 통계
        """
        self.start_time = time.time()

        # 초기화
        if self.crop_processor is None or self.inference_engine is None:
            self.initialize()

        logger.info("=" * 80)
        logger.info("파이프라인 실행 시작")
        logger.info("=" * 80)

        # Step 1: 폴리곤 선택
        logger.info("Step 1: 폴리곤 선택")
        if polygon_ids is None:
            # 면적 필터링
            polygon_ids = self.crop_processor.filter_by_area(min_area, max_area)
        logger.info(f"선택된 폴리곤: {len(polygon_ids)}개")

        # Step 2: 크롭 처리
        logger.info("Step 2: TIF 크롭 처리")
        cropped_regions = self.crop_processor.batch_crop(
            polygon_ids=polygon_ids,
            save_images=save_cropped,
            output_dir=self.output_dir / "cropped_images" if save_cropped else None
        )

        if not cropped_regions:
            logger.error("크롭된 영역이 없습니다. 파이프라인 종료.")
            return {}

        # Step 3: 추론
        logger.info("Step 3: 곤포사일리지 검출")
        results = self.inference_engine.batch_process(
            cropped_regions=cropped_regions,
            save_visualization=save_visualization,
            output_dir=self.output_dir / "visualizations" if save_visualization else None
        )

        # Step 4: GeoPackage 저장
        logger.info("Step 4: GeoPackage 저장")
        output_gpkg = self.output_dir / "silage_bale_detections.gpkg"
        self.inference_engine.save_to_geopackage(results, cropped_regions, output_gpkg)

        # Step 5: 통계 보고서 생성
        logger.info("Step 5: 통계 보고서 생성")
        stats = self._generate_statistics(results, cropped_regions)
        self._save_statistics(stats)

        # 실행 시간
        elapsed_time = time.time() - self.start_time
        stats['elapsed_time_seconds'] = elapsed_time
        stats['elapsed_time_formatted'] = self._format_time(elapsed_time)

        logger.info("=" * 80)
        logger.info("파이프라인 완료!")
        logger.info(f"총 처리 시간: {stats['elapsed_time_formatted']}")
        logger.info(f"총 검출: {stats['total_detections']}개")
        logger.info(f"평균 신뢰도: {stats['avg_confidence']:.2%}")
        logger.info(f"결과: {output_gpkg}")
        logger.info("=" * 80)

        return stats

    def _generate_statistics(
        self,
        results: List[DetectionResult],
        cropped_regions: List[CroppedRegion]
    ) -> Dict[str, Any]:
        """통계 생성"""

        # 기본 통계
        total_polygons = len(results)
        total_detections = sum(r.count for r in results)
        successful_polygons = sum(1 for r in results if r.count > 0)

        # 신뢰도 통계
        all_confidences = []
        for r in results:
            if r.count > 0:
                all_confidences.extend([d['confidence'] for d in r.detections])

        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        min_confidence = np.min(all_confidences) if all_confidences else 0.0
        max_confidence = np.max(all_confidences) if all_confidences else 0.0

        # 검출 개수 분포
        detection_counts = [r.count for r in results]

        # 폴리곤별 상세 정보
        polygon_details = []
        for result, cropped in zip(results, cropped_regions):
            detail = {
                'polygon_id': result.polygon_id,
                'detection_count': result.count,
                'avg_confidence': result.confidence_mean,
                'area_m2': cropped.polygon.area,
                'metadata': cropped.metadata
            }
            polygon_details.append(detail)

        stats = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'tif_path': str(self.tif_path),
                'shp_path': str(self.shp_path),
                'model_path': str(self.model_path)
            },
            'processing': {
                'total_polygons': total_polygons,
                'successful_polygons': successful_polygons,
                'success_rate': successful_polygons / total_polygons if total_polygons > 0 else 0.0
            },
            'detections': {
                'total_detections': total_detections,
                'avg_detections_per_polygon': total_detections / total_polygons if total_polygons > 0 else 0.0,
                'min_detections': int(np.min(detection_counts)) if detection_counts else 0,
                'max_detections': int(np.max(detection_counts)) if detection_counts else 0,
                'median_detections': float(np.median(detection_counts)) if detection_counts else 0.0
            },
            'confidence': {
                'avg_confidence': float(avg_confidence),
                'min_confidence': float(min_confidence),
                'max_confidence': float(max_confidence)
            },
            'polygon_details': polygon_details
        }

        return stats

    def _save_statistics(self, stats: Dict[str, Any]):
        """통계를 파일로 저장"""

        # JSON 저장
        json_path = self.output_dir / "reports" / "statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"통계 JSON 저장: {json_path}")

        # CSV 저장 (폴리곤별 상세)
        csv_path = self.output_dir / "reports" / "polygon_details.csv"
        df = pd.DataFrame(stats['polygon_details'])
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"폴리곤 상세 CSV 저장: {csv_path}")

        # 요약 텍스트 저장
        summary_path = self.output_dir / "reports" / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("곤포사일리지 검출 결과 요약\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"처리 시간: {stats.get('timestamp', 'N/A')}\n")
            f.write(f"실행 시간: {stats.get('elapsed_time_formatted', 'N/A')}\n\n")

            f.write("입력 파일:\n")
            f.write(f"  TIF: {stats['input']['tif_path']}\n")
            f.write(f"  SHP: {stats['input']['shp_path']}\n")
            f.write(f"  Model: {stats['input']['model_path']}\n\n")

            f.write("처리 결과:\n")
            f.write(f"  총 폴리곤: {stats['processing']['total_polygons']}개\n")
            f.write(f"  성공 폴리곤: {stats['processing']['successful_polygons']}개\n")
            f.write(f"  성공률: {stats['processing']['success_rate']:.1%}\n\n")

            f.write("검출 통계:\n")
            f.write(f"  총 검출: {stats['detections']['total_detections']}개\n")
            f.write(f"  평균 검출: {stats['detections']['avg_detections_per_polygon']:.2f}개/폴리곤\n")
            f.write(f"  최소 검출: {stats['detections']['min_detections']}개\n")
            f.write(f"  최대 검출: {stats['detections']['max_detections']}개\n")
            f.write(f"  중앙값: {stats['detections']['median_detections']:.0f}개\n\n")

            f.write("신뢰도 통계:\n")
            f.write(f"  평균 신뢰도: {stats['confidence']['avg_confidence']:.2%}\n")
            f.write(f"  최소 신뢰도: {stats['confidence']['min_confidence']:.2%}\n")
            f.write(f"  최대 신뢰도: {stats['confidence']['max_confidence']:.2%}\n\n")

            f.write("=" * 80 + "\n")

        logger.info(f"요약 보고서 저장: {summary_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """시간 포맷팅"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='곤포사일리지 통합 추론 파이프라인')
    parser.add_argument('--tif', type=str, required=True, help='TIF 파일 경로')
    parser.add_argument('--shp', type=str, required=True, help='Shapefile 경로')
    parser.add_argument('--model', type=str, required=True, help='YOLO 모델 경로')
    parser.add_argument('--output', type=str, default='inference_system/output', help='출력 디렉토리')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU 임계값')
    parser.add_argument('--device', type=str, default='auto', help='디바이스 (auto/cuda/cpu)')
    parser.add_argument('--limit', type=int, default=None, help='처리 폴리곤 개수 제한')
    parser.add_argument('--min-area', type=float, default=0, help='최소 면적 (m²)')
    parser.add_argument('--max-area', type=float, default=float('inf'), help='최대 면적 (m²)')
    parser.add_argument('--save-cropped', action='store_true', help='크롭 이미지 저장')
    parser.add_argument('--no-vis', action='store_true', help='시각화 저장 안함')

    args = parser.parse_args()

    # 파이프라인 생성
    pipeline = SilageBaleDetectionPipeline(
        tif_path=args.tif,
        shp_path=args.shp,
        model_path=args.model,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

    # 폴리곤 ID 설정
    polygon_ids = None
    if args.limit:
        polygon_ids = list(range(args.limit))

    # 실행
    stats = pipeline.run(
        polygon_ids=polygon_ids,
        min_area=args.min_area,
        max_area=args.max_area,
        save_cropped=args.save_cropped,
        save_visualization=not args.no_vis
    )

    print("\n" + "=" * 80)
    print("처리 완료!")
    print(f"결과 디렉토리: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
