#!/usr/bin/env python3
r"""
251102 곤포사일리지 타일 기반 추론 파이프라인 v11
=================================================

v11 경계 강화 + 2차 추론 + 이원화 병합 + 마스크 재분할 + 보정 카운팅 (v10 개선):
① make_tiles 경계 보강:
   - overlap=0.40 (40% 기본 오버랩)
   - 경계 타일 (우/하단): overlap=0.50 (50% 강화)
   - is_border 필드 추가 (경계 타일 표시)

② 2차 추론 (경계 타일만):
   - 1차 추론: 모든 타일 정상 파라미터 (conf=0.18, scale=1.5)
   - 2차 추론: 경계 타일만 공격적 재추론 (conf=0.15, scale=1.7)
   - 검출 병합: all_detections = normal + border

③ 이원화 병합 조건 (postprocess_merge_overlaps):
   - Tier 1 (강력 병합 - 둘 다 border): IoU≥0.2, dist≤300px, small<20000px
   - Tier 2 (보수적 병합 - 하나 이상 normal): IoU≥0.45, dist≤120px, small<8000px

④ 과대 마스크 재분할 (area > 120000):
   - resplit_large_mask(): watershed + distance transform
   - target_count = int(area / area_ref * 0.8)
   - 재분할된 인스턴스를 개별 TileDetection으로 추가

⑤ area 기반 보정 카운터:
   - count_raw: 검출 개수
   - area_ref: 단일 검출 중앙값 (0.5*ref ~ 1.8*ref 범위)
   - count_estimated: area 기반 보정 (big_count = round(area/ref))

⑥ 기존 유지:
   - scale=1.5 (추론 시 1.5x 스케일링)
   - Global WBF 1회
   - detect_stacked_layers (층수 감지)
   - conf_threshold=0.18

v11의 핵심 개선 (v10 대비):
- 경계 검출 향상 (50% overlap + 2차 추론)
- 중복 최소화 (이원화 병합)
- 과대 병합 해결 (재분할)
- 정확한 카운팅 (area 기반 보정)

작업:
1. SHP 기반 TIF 크롭 (2.0m 버퍼)
2. 타일 생성 (40% overlap, 경계 50%)
3. 2차 추론 (1차 전체 + 2차 경계만)
4. Global WBF
5. 과대 마스크 재분할 (area > 120000)
6. 이원화 병합
7. 층수 감지 + area 기반 카운팅

입력:
- 모델: C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt
- TIF: E:\namwon_ai\input_tif\금지면_1차.tif
- SHP: E:\namwon_ai\gonpo\gonpo_251028_fixed.shp

출력:
- inference_system/output_251102_v11/

작성일: 2025-11-02 (v11 - Border Enhancement + Dual-Tier Merge + Resplit + Area-based Count)
작성자: Claude Sonnet 4.5
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import json
import time
from datetime import datetime

import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm

# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))  # tile_based_inference.py는 루트에 있음

from tile_based_inference import TiledInferenceEngine

from dataclasses import dataclass
from shapely.geometry import Polygon
from rasterio.windows import from_bounds
from rasterio.mask import mask as rasterio_mask

@dataclass
class CroppedRegion:
    """크롭된 영역 데이터"""
    polygon_id: int
    image: np.ndarray  # RGB, shape=(H, W, 3), dtype=uint8
    bounds: tuple  # (minx, miny, maxx, maxy)
    polygon: Any  # 원본 폴리곤

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('251102_v11_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class GonpoInferencePipeline:
    """곤포사일리지 추론 파이프라인 v11"""

    def __init__(
        self,
        model_path: str,
        tif_path: str,
        shp_path: str,
        output_dir: str,
        tile_size: int = 1024,
        overlap_ratio: float = 0.40,  # v11: 0.35 → 0.40
        conf_threshold: float = 0.18,
        nms_iou_threshold: float = 0.60,
        inference_scale: float = 1.5,
        file_prefix: str = "251102_v11"
    ):
        """
        Args:
            model_path: YOLO 모델 경로
            tif_path: TIF 파일 경로
            shp_path: Shapefile 경로
            output_dir: 출력 디렉토리
            tile_size: 타일 크기
            overlap_ratio: 타일 간 오버랩 비율 (v11: 0.40, 경계는 0.50)
            conf_threshold: 검출 신뢰도 임계값
            nms_iou_threshold: WBF IoU 임계값
            inference_scale: 추론 시 이미지 스케일링 배율
            file_prefix: 출력 파일명 접두사
        """
        self.model_path = Path(model_path)
        self.tif_path = Path(tif_path)
        self.shp_path = Path(shp_path)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.inference_scale = inference_scale
        self.file_prefix = file_prefix

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 처리 시작 시간
        self.start_time = time.time()

        # 결과 저장용
        self.results = {
            "pipeline_info": {
                "version": "v11",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": str(self.model_path),
                "tif": str(self.tif_path),
                "shp": str(self.shp_path),
                "parameters": {
                    "tile_size": tile_size,
                    "overlap_ratio": overlap_ratio,
                    "conf_threshold": conf_threshold,
                    "nms_iou_threshold": nms_iou_threshold,
                    "inference_scale": inference_scale
                }
            },
            "polygons": []
        }

        logger.info("=" * 80)
        logger.info("251102 곤포사일리지 추론 파이프라인 v11 시작")
        logger.info("=" * 80)
        logger.info(f"모델: {self.model_path}")
        logger.info(f"TIF: {self.tif_path}")
        logger.info(f"SHP: {self.shp_path}")
        logger.info(f"출력: {self.output_dir}")
        logger.info(f"타일 크기: {tile_size}x{tile_size}, 오버랩: {overlap_ratio*100}% (경계: 50%)")
        logger.info(f"추론 스케일: {inference_scale}x")
        logger.info(f"신뢰도 임계값: {conf_threshold}, WBF IoU: {nms_iou_threshold}")

    def run(self):
        """전체 파이프라인 실행"""
        try:
            # Step 1: SHP 기반 TIF 크롭
            logger.info("\n" + "=" * 80)
            logger.info("Step 1: SHP 기반 TIF 크롭")
            logger.info("=" * 80)
            cropped_regions = self._crop_tif_by_shp()

            # Step 2: 타일 기반 추론
            logger.info("\n" + "=" * 80)
            logger.info("Step 2: 타일 기반 추론 (v11: 2-pass + 이원화 병합)")
            logger.info("=" * 80)
            self._run_tiled_inference(cropped_regions)

            # Step 3: 결과 저장
            logger.info("\n" + "=" * 80)
            logger.info("Step 3: 결과 저장")
            logger.info("=" * 80)
            self._save_summary()

            # 완료
            elapsed = time.time() - self.start_time
            logger.info("\n" + "=" * 80)
            logger.info(f"파이프라인 완료! (소요 시간: {elapsed:.2f}초)")
            logger.info("=" * 80)

            return self.results

        except Exception as e:
            logger.error(f"파이프라인 실패: {e}", exc_info=True)
            raise

    def _crop_tif_by_shp(self) -> List[CroppedRegion]:
        """SHP 기반 TIF 크롭 (2.0m 버퍼 적용)"""
        # Shapefile 로드
        gdf = gpd.read_file(str(self.shp_path))
        logger.info(f"Shapefile 로드: {len(gdf)}개 폴리곤, CRS: {gdf.crs}")

        cropped_regions = []

        # TIF 파일 열기
        with rasterio.open(str(self.tif_path)) as src:
            logger.info(f"TIF 파일: {src.width}x{src.height}, CRS: {src.crs}, Bands: {src.count}")

            # 각 폴리곤별로 크롭
            for idx, row in gdf.iterrows():
                geom = row.geometry

                # 폴리곤을 TIF CRS로 변환 (필요시)
                if gdf.crs != src.crs:
                    gdf_reprojected = gdf.to_crs(src.crs)
                    geom = gdf_reprojected.iloc[idx].geometry

                # 2.0m 버퍼 적용 (EPSG:5186은 미터 단위)
                geom_buffered = geom.buffer(2.0)
                logger.info(f"  Polygon {idx}: 2.0m 버퍼 적용")

                # 마스크를 사용해 크롭
                try:
                    out_image, out_transform = rasterio_mask(src, [mapping(geom_buffered)], crop=True, all_touched=True)

                    # 4-band → RGB 변환 (필요시)
                    if out_image.shape[0] > 3:
                        rgb_image = out_image[:3]  # RGB 밴드만
                    else:
                        rgb_image = out_image

                    # (C, H, W) → (H, W, C) 변환
                    rgb_image = np.transpose(rgb_image, (1, 2, 0))

                    # uint8 변환
                    if rgb_image.dtype != np.uint8:
                        # 값 범위 확인 후 스케일링
                        if rgb_image.max() > 255:
                            rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)
                        else:
                            rgb_image = rgb_image.astype(np.uint8)

                    # CroppedRegion 생성
                    region = CroppedRegion(
                        polygon_id=idx,
                        image=rgb_image,
                        bounds=geom.bounds,
                        polygon=geom
                    )

                    cropped_regions.append(region)
                    logger.info(f"  Polygon {idx}: {rgb_image.shape} cropped")

                except Exception as e:
                    logger.error(f"  Polygon {idx} 크롭 실패: {e}")
                    continue

        logger.info(f"크롭 완료: {len(cropped_regions)}개 폴리곤")

        # 크롭된 이미지 저장
        for region in cropped_regions:
            output_path = self.output_dir / f"{self.file_prefix}_polygon_{region.polygon_id}_cropped.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(region.image, cv2.COLOR_RGB2BGR))
            logger.info(f"  저장: {output_path.name}")

        return cropped_regions

    def _run_tiled_inference(self, cropped_regions: List[CroppedRegion]):
        """타일 기반 추론 (v11: 2-pass inference)"""
        # 추론 엔진 초기화
        engine = TiledInferenceEngine(
            model_path=str(self.model_path),
            tile_size=self.tile_size,
            overlap_ratio=self.overlap_ratio,
            conf_threshold=self.conf_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            inference_scale=self.inference_scale
        )

        for i, region in enumerate(cropped_regions):
            logger.info(f"\n[Polygon {i}] 추론 시작 (크기: {region.image.shape[:2]})")

            # 크롭 이미지 경로
            cropped_image_path = self.output_dir / f"{self.file_prefix}_polygon_{i}_cropped.png"

            # 타일 생성 및 추론
            tile_start = time.time()
            result = engine.process_image(
                image_path=str(cropped_image_path),
                output_dir=str(self.output_dir),
                save_tiles=False,
                save_visualization=True
            )
            tile_elapsed = time.time() - tile_start

            # 결과 통계
            det_stats = result.get('detections', {})
            conf_stats = result.get('confidence', {})
            count_stats = result.get('silage_bale_count', {})  # v11: area-based count

            num_detections = det_stats.get('total_after_wbf', 0)
            avg_conf = conf_stats.get('mean', 0.0)

            logger.info(f"  검출 완료: {num_detections}개, 평균 신뢰도: {avg_conf:.1%}, 소요: {tile_elapsed:.2f}초")
            logger.info(f"  카운팅: raw={count_stats.get('count_raw', 0)}, estimated={count_stats.get('count_estimated', 0)}")

            # 결과 저장
            polygon_result = {
                "polygon_id": i,
                "image_shape": region.image.shape[:2],
                "num_detections": num_detections,
                "avg_confidence": float(avg_conf),
                "processing_time_sec": tile_elapsed,
                "detection_stats": det_stats,
                "confidence_stats": conf_stats,
                "counting_stats": count_stats,  # v11
                "timing": result.get('timing', {})
            }

            self.results["polygons"].append(polygon_result)

            # 시각화 파일 이름 변경
            original_vis = self.output_dir / f"{cropped_image_path.stem}_result.png"
            new_vis = self.output_dir / f"{self.file_prefix}_polygon_{i}_result.png"
            if original_vis.exists():
                original_vis.rename(new_vis)
                logger.info(f"  시각화 저장: {new_vis.name}")

            # JSON 파일 이름 변경
            original_json = self.output_dir / f"{cropped_image_path.stem}_results.json"
            new_json = self.output_dir / f"{self.file_prefix}_polygon_{i}_results.json"
            if original_json.exists():
                original_json.rename(new_json)
                logger.info(f"  JSON 저장: {new_json.name}")

    def _save_summary(self):
        """전체 요약 저장"""
        # 전체 통계 계산
        total_detections = sum(p["num_detections"] for p in self.results["polygons"])
        total_estimated = sum(p.get("counting_stats", {}).get("count_estimated", 0) for p in self.results["polygons"])
        total_time = time.time() - self.start_time

        self.results["summary"] = {
            "total_polygons": len(self.results["polygons"]),
            "total_detections_raw": total_detections,
            "total_detections_estimated": total_estimated,  # v11
            "total_processing_time_sec": total_time,
            "avg_detections_per_polygon": total_detections / len(self.results["polygons"]) if self.results["polygons"] else 0
        }

        # JSON 저장
        summary_path = self.output_dir / f"{self.file_prefix}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"전체 요약 저장: {summary_path}")

        # 텍스트 로그 저장
        log_path = self.output_dir / f"{self.file_prefix}_processing_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("251102 곤포사일리지 추론 결과 (v11)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"처리 일시: {self.results['pipeline_info']['date']}\n")
            f.write(f"모델: {self.results['pipeline_info']['model']}\n")
            f.write(f"TIF: {self.results['pipeline_info']['tif']}\n")
            f.write(f"SHP: {self.results['pipeline_info']['shp']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("전체 요약\n")
            f.write("=" * 80 + "\n")
            f.write(f"총 폴리곤 수: {self.results['summary']['total_polygons']}개\n")
            f.write(f"총 검출 개수 (raw): {self.results['summary']['total_detections_raw']}개\n")
            f.write(f"총 검출 개수 (estimated): {self.results['summary']['total_detections_estimated']}개\n")
            f.write(f"폴리곤당 평균: {self.results['summary']['avg_detections_per_polygon']:.1f}개\n")
            f.write(f"총 처리 시간: {self.results['summary']['total_processing_time_sec']:.2f}초\n\n")

            f.write("=" * 80 + "\n")
            f.write("폴리곤별 상세\n")
            f.write("=" * 80 + "\n")
            for p in self.results["polygons"]:
                f.write(f"\nPolygon {p['polygon_id']}:\n")
                f.write(f"  이미지 크기: {p['image_shape'][0]}x{p['image_shape'][1]}\n")
                f.write(f"  검출 개수: {p['num_detections']}개\n")
                count_stats = p.get('counting_stats', {})
                f.write(f"  보정 카운트: {count_stats.get('count_estimated', 0)}개\n")
                f.write(f"  평균 신뢰도: {p['avg_confidence']:.1%}\n")
                f.write(f"  처리 시간: {p['processing_time_sec']:.2f}초\n")

        logger.info(f"처리 로그 저장: {log_path}")


def main():
    """
    메인 실행 - v11

    v11 특징 (v10 개선):
    ① 경계 타일 강화: overlap 40% (경계 50%)
    ② 2차 추론: 경계 타일만 재추론 (conf=0.15, scale=1.7)
    ③ 이원화 병합: border vs normal 다른 threshold
    ④ 과대 마스크 재분할: area > 120000 → watershed
    ⑤ area 기반 보정 카운터: count_raw + count_estimated
    """
    # 경로 설정
    MODEL_PATH = r"C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt"
    TIF_PATH = r"E:\namwon_ai\input_tif\금지면_1차.tif"
    SHP_PATH = r"E:\namwon_ai\gonpo\gonpo_251028_fixed.shp"
    OUTPUT_DIR = "inference_system/output_251102_v11"

    # 파이프라인 실행 (v11)
    pipeline = GonpoInferencePipeline(
        model_path=MODEL_PATH,
        tif_path=TIF_PATH,
        shp_path=SHP_PATH,
        output_dir=OUTPUT_DIR,
        tile_size=1024,
        overlap_ratio=0.40,  # v11: 40% (경계 50%)
        conf_threshold=0.18,
        nms_iou_threshold=0.60,
        inference_scale=1.5,
        file_prefix="251102_v11"
    )

    results = pipeline.run()

    # 최종 요약 출력
    print("\n" + "=" * 80)
    print("v11 처리 완료! (경계 강화 + 2차 추론 + 이원화 병합 + 재분할 + 보정 카운팅)")
    print("=" * 80)
    print(f"총 검출 (raw): {results['summary']['total_detections_raw']}개")
    print(f"총 검출 (estimated): {results['summary']['total_detections_estimated']}개")
    print(f"출력 위치: {OUTPUT_DIR}")
    print("=" * 80)
    print("\nv11 개선사항:")
    print("  1. 경계 타일: overlap 50% (일반 40%)")
    print("  2. 2차 추론: 경계 타일만 conf=0.15, scale=1.7로 재추론")
    print("  3. 이원화 병합:")
    print("     - Tier 1 (border): IoU≥0.2, dist≤300px, small<20000px")
    print("     - Tier 2 (normal): IoU≥0.45, dist≤120px, small<8000px")
    print("  4. 재분할: area > 120000 → watershed 재분할")
    print("  5. 보정 카운팅: area_ref 기준 count_estimated 계산")
    print("=" * 80)


if __name__ == "__main__":
    main()
