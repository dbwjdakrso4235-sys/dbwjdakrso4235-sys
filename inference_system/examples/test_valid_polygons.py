#!/usr/bin/env python3
"""
TIF 범위와 중첩되는 폴리곤만 테스트

작성일: 2025-10-23
"""

import sys
from pathlib import Path

# 상위 디렉토리의 src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crop_processor import CropProcessor
import geopandas as gpd
import rasterio
from shapely.geometry import box


def main():
    # 경로 설정
    TIF_PATH = "F:/namwon_ai/input_tif/금지면_1차.tif"
    SHP_PATH = "F:/namwon_ai/saryo_jeongbo/saryo_parcel.shp"
    OUTPUT_DIR = "inference_system/output/test_valid"

    print("=" * 80)
    print("TIF 범위 내 유효한 폴리곤 테스트")
    print("=" * 80)

    # 1. TIF와 중첩되는 폴리곤 찾기
    print("\nStep 1: TIF 범위와 중첩되는 폴리곤 찾기...")
    gdf = gpd.read_file(SHP_PATH)

    with rasterio.open(TIF_PATH) as src:
        tif_bounds = src.bounds
        print(f"TIF 범위: {tif_bounds}")

    tif_box = box(*tif_bounds)
    intersecting = gdf[gdf.intersects(tif_box)]

    print(f"총 폴리곤: {len(gdf)}개")
    print(f"TIF 범위와 교차하는 폴리곤: {len(intersecting)}개")

    if len(intersecting) == 0:
        print("교차하는 폴리곤이 없습니다!")
        return

    # 처음 10개 폴리곤 인덱스
    polygon_ids = intersecting.index.tolist()[:10]
    print(f"테스트할 폴리곤 인덱스: {polygon_ids}")

    # 2. CropProcessor로 크롭
    print("\nStep 2: 폴리곤 크롭...")
    processor = CropProcessor(TIF_PATH, SHP_PATH)

    results = processor.batch_crop(
        polygon_ids=polygon_ids,
        save_images=True,
        output_dir=Path(OUTPUT_DIR)
    )

    # 3. 결과 확인
    print("\n" + "=" * 80)
    print("결과")
    print("=" * 80)
    print(f"성공한 크롭: {len(results)}/{len(polygon_ids)}개")
    print(f"저장 위치: {OUTPUT_DIR}")

    if len(results) > 0:
        print("\n크롭 성공한 폴리곤:")
        for r in results:
            print(f"  - {r.polygon_id}: {r.image.shape[1]}x{r.image.shape[0]} 픽셀")

    print("=" * 80)


if __name__ == "__main__":
    main()
