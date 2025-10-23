#!/usr/bin/env python3
"""
곤포사일리지 추론 실행 예제

사용 예:
  # 전체 폴리곤 처리
  python run_inference.py

  # 처음 10개 폴리곤만 처리 (테스트)
  python run_inference.py --limit 10

  # 신뢰도 임계값 조정
  python run_inference.py --conf 0.35

작성일: 2025-10-23
"""

import sys
from pathlib import Path

# 상위 디렉토리의 src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SilageBaleDetectionPipeline


def main():
    """간단한 실행 예제"""

    # ==============================================
    # 설정
    # ==============================================

    # 입력 파일 경로
    TIF_PATH = "E:/namwon_ai/input_tif/금지면_1차.tif"
    SHP_PATH = "E:/namwon_ai/saryo_jeongbo/saryo_4m.shp"

    # 학습된 모델 경로
    MODEL_PATH = "runs/segment/silage_optimized/weights/best.pt"

    # 출력 디렉토리
    OUTPUT_DIR = "inference_system/output"

    # 추론 설정
    CONF_THRESHOLD = 0.25  # 신뢰도 임계값
    IOU_THRESHOLD = 0.45   # IoU 임계값
    DEVICE = 'auto'        # 'auto', 'cuda', 'cpu'

    # 처리 옵션
    LIMIT = 10             # 처리할 폴리곤 개수 제한 (None이면 전체)
    MIN_AREA = 0           # 최소 면적 (m²)
    MAX_AREA = float('inf')  # 최대 면적 (m²)

    SAVE_CROPPED = False   # 크롭 이미지 저장 여부
    SAVE_VIS = True        # 시각화 저장 여부

    # ==============================================
    # 파이프라인 실행
    # ==============================================

    print("=" * 80)
    print("곤포사일리지 자동 검출 시스템")
    print("=" * 80)
    print()
    print("설정:")
    print(f"  TIF: {TIF_PATH}")
    print(f"  SHP: {SHP_PATH}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Confidence: {CONF_THRESHOLD}")
    print(f"  Limit: {LIMIT if LIMIT else '전체'}")
    print()

    # 파이프라인 생성
    pipeline = SilageBaleDetectionPipeline(
        tif_path=TIF_PATH,
        shp_path=SHP_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE
    )

    # 폴리곤 ID 설정
    polygon_ids = None
    if LIMIT:
        polygon_ids = list(range(LIMIT))

    # 실행
    stats = pipeline.run(
        polygon_ids=polygon_ids,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        save_cropped=SAVE_CROPPED,
        save_visualization=SAVE_VIS
    )

    # 결과 출력
    print()
    print("=" * 80)
    print("처리 완료!")
    print("=" * 80)
    print()
    print("결과:")
    print(f"  GeoPackage: {OUTPUT_DIR}/silage_bale_detections.gpkg")
    print(f"  통계: {OUTPUT_DIR}/reports/")
    if SAVE_VIS:
        print(f"  시각화: {OUTPUT_DIR}/visualizations/")
    print()
    print("통계:")
    print(f"  총 검출: {stats['detections']['total_detections']}개")
    print(f"  평균 신뢰도: {stats['confidence']['avg_confidence']:.2%}")
    print(f"  성공률: {stats['processing']['success_rate']:.1%}")
    print(f"  처리 시간: {stats['elapsed_time_formatted']}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
