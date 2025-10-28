# 타일 기반 추론 시스템 - 빠른 시작 가이드

**목적**: 큰 크롭 이미지에서 곤포사일리지 검출

---

## 1. 빠른 실행

### 단일 이미지 처리
```bash
python tile_based_inference.py \
  --model "C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt" \
  --image "polygon_0.png" \
  --output "inference_system/output_tiled" \
  --conf 0.25
```

### 여러 이미지 처리
```bash
python tile_based_inference.py \
  --model "C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt" \
  --image "polygon_0.png" "polygon_1.png" \
  --output "inference_system/output_tiled" \
  --conf 0.25
```

---

## 2. 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | 필수 | 모델 경로 (best.pt) |
| `--image` | 필수 | 이미지 경로 (여러 개 가능) |
| `--output` | `output_tiled` | 출력 디렉토리 |
| `--tile-size` | 1024 | 타일 크기 (픽셀) |
| `--overlap` | 0.25 | 오버랩 비율 (0.0~1.0) |
| `--conf` | 0.25 | 신뢰도 임계값 |
| `--iou` | 0.45 | YOLO NMS IoU 임계값 |
| `--nms-iou` | 0.5 | 타일 간 NMS IoU 임계값 |
| `--save-tiles` | False | 타일 이미지 저장 여부 |

---

## 3. 출력 파일

실행 후 다음 파일들이 생성됩니다:

```
output_tiled/
├── polygon_0_result.png       # 시각화 (타일 경계 + 검출 결과)
├── polygon_0_results.json     # 상세 통계 (JSON)
├── polygon_1_result.png
└── polygon_1_results.json
```

---

## 4. 결과 확인

### 4.1. 시각화 이미지
- `*_result.png`: 검출 결과를 시각화한 이미지
  - 회색 선: 타일 경계
  - 초록색 마스크: 검출된 곤포사일리지
  - 초록색 박스: Bounding box
  - 텍스트: 검출 ID와 신뢰도

### 4.2. 통계 JSON
```json
{
  "image_name": "polygon_0.png",
  "image_size": {
    "width": 2953,
    "height": 5721,
    "megapixels": 16.89
  },
  "tile_config": {
    "tile_size": 1024,
    "overlap_ratio": 0.25,
    "num_tiles": 32
  },
  "detections": {
    "total_before_nms": 5,
    "total_after_nms": 3,
    "nms_removal_rate": 0.4
  },
  "confidence": {
    "mean": 0.764,
    "min": 0.575,
    "max": 0.878
  },
  "timing": {
    "total_sec": 4.55
  }
}
```

---

## 5. 오버랩 비율 선택 가이드

| 오버랩 | 장점 | 단점 | 추천 용도 |
|--------|------|------|-----------|
| 10% | - 빠른 처리<br>- 높은 신뢰도 | - 경계 객체 누락 가능 | 객체가 타일 중앙에 있을 때 |
| **25%** | - 균형잡힌 성능<br>- 적절한 처리 시간 | - | **일반적인 경우 (권장)** |
| 50% | - 경계 객체 검출 우수 | - 느린 처리<br>- 높은 중복률 | 객체 밀도가 높을 때 |

**권장**: **25% 오버랩** (검출 개수와 처리 속도 균형)

---

## 6. 실험 결과 요약

### polygon_0.png (2953 x 5721)
- **타일링 전**: 0개 검출
- **타일링 후 (25% 오버랩)**: 3개 검출 (평균 신뢰도 76.4%)

### polygon_1.png (3381 x 6278)
- **타일링 전**: 0개 검출
- **타일링 후 (25% 오버랩)**: 12개 검출 (평균 신뢰도 69.6%)

**결론**: 타일 기반 추론으로 큰 이미지 검출 문제 해결!

---

## 7. 문제 해결

### Q1. 검출 개수가 0개입니다.
**A**: 신뢰도 임계값을 낮춰보세요.
```bash
python tile_based_inference.py ... --conf 0.1
```

### Q2. 중복 검출이 많습니다.
**A**: NMS IoU 임계값을 높이거나 오버랩 비율을 낮추세요.
```bash
python tile_based_inference.py ... --nms-iou 0.7 --overlap 0.1
```

### Q3. 처리 시간이 너무 깁니다.
**A**: 오버랩 비율을 낮추세요.
```bash
python tile_based_inference.py ... --overlap 0.1
```

### Q4. 경계 영역 객체가 누락됩니다.
**A**: 오버랩 비율을 높이세요.
```bash
python tile_based_inference.py ... --overlap 0.5
```

---

## 8. Python API 사용

```python
from tile_based_inference import TiledInferenceEngine

# 엔진 초기화
engine = TiledInferenceEngine(
    model_path="best.pt",
    tile_size=1024,
    overlap_ratio=0.25,
    conf_threshold=0.25,
    nms_iou_threshold=0.5
)

# 이미지 처리
result = engine.process_image(
    image_path="polygon_0.png",
    output_dir="output_tiled",
    save_visualization=True
)

# 결과 확인
print(f"검출 개수: {result['detections']['total_after_nms']}")
print(f"평균 신뢰도: {result['confidence']['mean']:.1%}")
print(f"처리 시간: {result['timing']['total_sec']:.2f}초")
```

---

## 9. 성능 벤치마크

| 이미지 크기 | 타일 개수 | 검출 개수 | 처리 시간 |
|------------|----------|----------|----------|
| 2953x5721 (16.89M) | 32 | 3개 | 4.55초 |
| 3381x6278 (21.23M) | 45 | 12개 | 1.24초 |

**하드웨어**: NVIDIA RTX A6000

---

## 10. 추가 자료

- **상세 보고서**: `TILE_BASED_INFERENCE_SUMMARY.md`
- **분석 리포트**: `inference_system/output_tiled/analysis_report.txt`
- **소스 코드**: `tile_based_inference.py`

---

**작성일**: 2025-10-28
