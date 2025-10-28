r"""
곤포사일리지 추론 테스트 - gonpo 데이터셋
==========================================

TIF: E:\namwon_ai\input_tif\금지면_1차.tif (24.09 GB)
SHP: E:\namwon_ai\gonpo\gonpo_251028.shp (2개 폴리곤)
Model: C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt
"""

import sys
import os
from pathlib import Path
import time
import json

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from inference_system.src.pipeline import SilageBaleDetectionPipeline


def main():
    print("=" * 80)
    print("곤포사일리지 추론 시스템 - gonpo 데이터셋 테스트")
    print("=" * 80)
    print()

    # 경로 설정
    tif_path = r"E:\namwon_ai\input_tif\금지면_1차.tif"
    shp_path = r"E:\namwon_ai\gonpo\gonpo_251028_fixed.shp"  # CRS 수정된 파일 사용
    model_path = r"C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt"
    output_dir = "inference_system/output_gonpo_fixed"  # 새 출력 디렉토리

    print("📁 입력 파일:")
    print(f"  TIF: {tif_path}")
    print(f"  SHP: {shp_path}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_dir}")
    print()

    # 파일 존재 확인
    print("🔍 파일 존재 확인...")
    if not os.path.exists(tif_path):
        print(f"❌ TIF 파일을 찾을 수 없습니다: {tif_path}")
        return
    if not os.path.exists(shp_path):
        print(f"❌ SHP 파일을 찾을 수 없습니다: {shp_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    print("✅ 모든 파일 확인 완료")
    print()

    # 파이프라인 초기화
    print("🔧 파이프라인 초기화 중...")
    try:
        pipeline = SilageBaleDetectionPipeline(
            tif_path=tif_path,
            shp_path=shp_path,
            model_path=model_path,
            output_dir=output_dir,
            conf_threshold=0.25,
            iou_threshold=0.45,
            device='auto'
        )
        print("✅ 파이프라인 초기화 완료")
        print()
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 데이터 정보 출력
    print("📊 데이터 통계:")
    try:
        stats = pipeline.crop_processor.get_statistics()
        print(f"  총 폴리곤 수: {stats.get('total_polygons', 'N/A')}")
        print(f"  TIF 범위와 교차하는 폴리곤: {stats.get('intersecting_polygons', 'N/A')}")
        print()
    except Exception as e:
        print(f"⚠️ 통계 정보 가져오기 실패: {e}")
        print()

    # 추론 실행
    print("🚀 추론 시작...")
    print("=" * 80)
    start_time = time.time()

    try:
        # 전체 폴리곤 처리 (2개)
        results = pipeline.run(
            polygon_ids=None,  # None = 전체 처리
            min_area=0,
            max_area=float('inf'),
            save_cropped=True,  # 크롭 이미지 저장
            save_visualization=True  # 시각화 저장
        )

        elapsed_time = time.time() - start_time

        print("=" * 80)
        print("✅ 추론 완료!")
        print()

        # 결과 출력
        print("📊 처리 결과:")
        print(f"  처리 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
        print(f"  처리 폴리곤 수: {results.get('processing', {}).get('total_polygons', 0)}")
        print(f"  성공 폴리곤 수: {results.get('processing', {}).get('successful_polygons', 0)}")
        print(f"  성공률: {results.get('processing', {}).get('success_rate', 0)*100:.1f}%")
        print()

        print("🎯 검출 결과:")
        print(f"  총 검출 개수: {results.get('detections', {}).get('total_detections', 0)}")
        print(f"  폴리곤당 평균: {results.get('detections', {}).get('avg_detections_per_polygon', 0):.1f}")
        print(f"  최소/최대: {results.get('detections', {}).get('min_detections', 0)} / {results.get('detections', {}).get('max_detections', 0)}")
        print()

        print("💯 신뢰도:")
        print(f"  평균: {results.get('confidence', {}).get('avg_confidence', 0)*100:.1f}%")
        print(f"  최소/최대: {results.get('confidence', {}).get('min_confidence', 0)*100:.1f}% / {results.get('confidence', {}).get('max_confidence', 0)*100:.1f}%")
        print()

        print("📁 출력 파일:")
        output_path = Path(output_dir)
        if (output_path / "silage_bale_detections.gpkg").exists():
            print(f"  ✅ GeoPackage: {output_path / 'silage_bale_detections.gpkg'}")
        if (output_path / "reports" / "statistics.json").exists():
            print(f"  ✅ 통계 JSON: {output_path / 'reports' / 'statistics.json'}")
        if (output_path / "reports" / "polygon_details.csv").exists():
            print(f"  ✅ 상세 CSV: {output_path / 'reports' / 'polygon_details.csv'}")
        if (output_path / "reports" / "summary.txt").exists():
            print(f"  ✅ 요약 TXT: {output_path / 'reports' / 'summary.txt'}")

        vis_dir = output_path / "visualizations"
        if vis_dir.exists():
            vis_files = list(vis_dir.glob("*.png"))
            print(f"  ✅ 시각화 이미지: {len(vis_files)}개")

        cropped_dir = output_path / "cropped_images"
        if cropped_dir.exists():
            cropped_files = list(cropped_dir.glob("*.png"))
            print(f"  ✅ 크롭 이미지: {len(cropped_files)}개")
        print()

        print("=" * 80)
        print("🎉 모든 작업이 완료되었습니다!")
        print("=" * 80)

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print(f"❌ 추론 실패 (경과 시간: {elapsed_time:.2f}초)")
        print(f"에러: {e}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    main()
