#!/usr/bin/env python3
"""
전체 파이프라인 테스트: 크롭 → 추론 → GeoPackage 저장

작성일: 2025-10-23
"""

import sys
from pathlib import Path

# 상위 디렉토리의 src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SilageBaleDetectionPipeline
import geopandas as gpd
import rasterio
from shapely.geometry import box


def main():
    # 경로 설정
    TIF_PATH = "F:/namwon_ai/input_tif/금지면_1차.tif"
    SHP_PATH = "F:/namwon_ai/saryo_jeongbo/saryo_parcel.shp"
    MODEL_PATH = "runs/segment/silage_optimized/weights/best.pt"
    OUTPUT_DIR = "inference_system/output/full_pipeline_test"

    print("=" * 80)
    print("전체 파이프라인 테스트")
    print("곤포사일리지 자동 검출 시스템")
    print("=" * 80)
    print()

    # 1. TIF와 중첩되는 폴리곤 찾기
    print("Step 1: 유효한 폴리곤 찾기...")
    gdf = gpd.read_file(SHP_PATH)

    with rasterio.open(TIF_PATH) as src:
        tif_bounds = src.bounds

    tif_box = box(*tif_bounds)
    intersecting = gdf[gdf.intersects(tif_box)]

    # 테스트용으로 처음 5개만
    polygon_ids = intersecting.index.tolist()[:5]

    print(f"총 폴리곤: {len(gdf)}개")
    print(f"TIF와 교차하는 폴리곤: {len(intersecting)}개")
    print(f"테스트할 폴리곤: {len(polygon_ids)}개")
    print(f"폴리곤 인덱스: {polygon_ids}")
    print()

    # 2. 파이프라인 실행
    print("Step 2: 파이프라인 초기화...")
    pipeline = SilageBaleDetectionPipeline(
        tif_path=TIF_PATH,
        shp_path=SHP_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='auto'
    )

    print("Step 3: 파이프라인 실행 (크롭 → 추론 → 저장)...")
    print()

    stats = pipeline.run(
        polygon_ids=polygon_ids,
        save_visualization=True,
        save_cropped=False
    )

    # 3. 결과 출력
    print()
    print("=" * 80)
    print("최종 결과")
    print("=" * 80)
    print()
    print(f"처리 시간: {stats.get('elapsed_time_formatted', 'N/A')}")
    print(f"성공 폴리곤: {stats['processing']['successful_polygons']}/{stats['processing']['total_polygons']}")
    print(f"총 검출: {stats['detections']['total_detections']}개")
    print(f"평균 신뢰도: {stats['confidence']['avg_confidence']:.2%}")
    print()
    print("출력 파일:")
    print(f"  - GeoPackage: {OUTPUT_DIR}/silage_bale_detections.gpkg")
    print(f"  - 시각화: {OUTPUT_DIR}/visualizations/")
    print(f"  - 통계: {OUTPUT_DIR}/reports/")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
