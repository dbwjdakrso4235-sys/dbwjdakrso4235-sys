# 곤포사일리지 추론 시스템

**SHP 기반 대용량 TIF 이미지 크롭 및 곤포사일리지 자동 검출**

## 📋 개요

이 시스템은 Shapefile로 정의된 영역만 대용량 TIF 이미지에서 추출하여 곤포사일리지를 자동으로 검출합니다.

### 주요 기능
- ✅ **SHP 기반 크롭**: 필요한 영역만 선택적 처리
- ✅ **대용량 TIF 처리**: 25GB+ 파일 안정적 처리
- ✅ **고정밀 검출**: 학습된 YOLOv11n-seg 모델 활용 (mAP50: 92.2%)
- ✅ **GeoPackage 출력**: 좌표계 보존된 검출 결과
- ✅ **자동화 파이프라인**: 크롭 → 추론 → 저장 원스톱

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 루트로 이동
cd dbwjdakrso4235-sys

# 의존성 설치 (이미 설치되어 있음)
pip install rasterio geopandas shapely ultralytics opencv-python
```

### 2. 데이터 준비

필요한 파일:
```
E:/namwon_ai/input_tif/금지면_1차.tif      # 대용량 TIF (25.8GB)
E:/namwon_ai/saryo_jeongbo/saryo_4m.shp    # Shapefile
runs/segment/silage_optimized/weights/best.pt  # 학습된 모델
```

### 3. 실행

#### 방법 1: 간단한 실행 (권장)
```bash
cd inference_system/examples
python run_inference.py
```

#### 방법 2: 커맨드라인 실행
```bash
python inference_system/src/pipeline.py \
    --tif "E:/namwon_ai/input_tif/금지면_1차.tif" \
    --shp "E:/namwon_ai/saryo_jeongbo/saryo_4m.shp" \
    --model "runs/segment/silage_optimized/weights/best.pt" \
    --output "inference_system/output" \
    --conf 0.25 \
    --limit 10
```

### 4. 결과 확인

```
inference_system/output/
├── silage_bale_detections.gpkg    # GeoPackage 결과
├── visualizations/                # 시각화 이미지
│   ├── polygon_0_result.png
│   ├── polygon_1_result.png
│   └── ...
└── reports/                       # 통계 보고서
    ├── statistics.json
    ├── polygon_details.csv
    └── summary.txt
```

---

## 📁 프로젝트 구조

```
inference_system/
├── src/
│   ├── crop_processor.py         # SHP 기반 TIF 크롭
│   ├── inference_engine.py       # YOLO 추론 엔진
│   └── pipeline.py               # 통합 파이프라인
├── examples/
│   └── run_inference.py          # 실행 예제
├── configs/
│   └── (설정 파일)
├── output/                        # 결과 출력
│   ├── silage_bale_detections.gpkg
│   ├── visualizations/
│   ├── cropped_images/
│   └── reports/
└── README.md                      # 본 문서
```

---

## 🔧 모듈 설명

### 1. CropProcessor (crop_processor.py)

Shapefile 기반 TIF 크롭 처리

**주요 기능**:
- Shapefile 로드 및 검증
- 좌표계 자동 감지 및 변환
- 폴리곤별 TIF 크롭
- 4-band → RGB 자동 변환

**사용 예**:
```python
from crop_processor import CropProcessor

processor = CropProcessor(
    tif_path="E:/namwon_ai/input_tif/금지면_1차.tif",
    shp_path="E:/namwon_ai/saryo_jeongbo/saryo_4m.shp"
)

# 단일 폴리곤 크롭
cropped = processor.crop_by_polygon(polygon_id=0)

# 배치 크롭
results = processor.batch_crop(polygon_ids=[0, 1, 2, 3, 4])
```

### 2. InferenceEngine (inference_engine.py)

곤포사일리지 검출 엔진

**주요 기능**:
- YOLOv11n-seg 모델 추론
- 마스크 → 폴리곤 변환
- 시각화 생성
- GeoPackage 저장

**사용 예**:
```python
from inference_engine import InferenceEngine

engine = InferenceEngine(
    model_path="runs/segment/silage_optimized/weights/best.pt",
    conf_threshold=0.25
)

# 단일 이미지 추론
result = engine.predict_image(image)

# 배치 처리
results = engine.batch_process(cropped_regions)
```

### 3. SilageBaleDetectionPipeline (pipeline.py)

통합 파이프라인

**주요 기능**:
- CropProcessor + InferenceEngine 통합
- 자동 워크플로우 관리
- 통계 보고서 자동 생성

**사용 예**:
```python
from pipeline import SilageBaleDetectionPipeline

pipeline = SilageBaleDetectionPipeline(
    tif_path="E:/namwon_ai/input_tif/금지면_1차.tif",
    shp_path="E:/namwon_ai/saryo_jeongbo/saryo_4m.shp",
    model_path="runs/segment/silage_optimized/weights/best.pt",
    output_dir="inference_system/output"
)

stats = pipeline.run(polygon_ids=[0, 1, 2, 3, 4])
```

---

## ⚙️ 설정 옵션

### 명령줄 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--tif` | TIF 파일 경로 | (필수) |
| `--shp` | Shapefile 경로 | (필수) |
| `--model` | YOLO 모델 경로 | (필수) |
| `--output` | 출력 디렉토리 | `inference_system/output` |
| `--conf` | 신뢰도 임계값 | `0.25` |
| `--iou` | IoU 임계값 | `0.45` |
| `--device` | 디바이스 (auto/cuda/cpu) | `auto` |
| `--limit` | 처리 폴리곤 개수 제한 | `None` (전체) |
| `--min-area` | 최소 면적 필터 (m²) | `0` |
| `--max-area` | 최대 면적 필터 (m²) | `∞` |
| `--save-cropped` | 크롭 이미지 저장 | `False` |
| `--no-vis` | 시각화 저장 안함 | `False` |

### Python API

```python
pipeline = SilageBaleDetectionPipeline(
    tif_path="...",
    shp_path="...",
    model_path="...",
    output_dir="inference_system/output",
    conf_threshold=0.25,     # 신뢰도 임계값
    iou_threshold=0.45,      # IoU 임계값
    device='auto'            # 디바이스 설정
)

stats = pipeline.run(
    polygon_ids=None,        # None이면 전체
    min_area=0,              # 최소 면적 (m²)
    max_area=float('inf'),   # 최대 면적 (m²)
    save_cropped=False,      # 크롭 이미지 저장
    save_visualization=True  # 시각화 저장
)
```

---

## 📊 출력 형식

### 1. GeoPackage (*.gpkg)

검출된 곤포사일리지의 폴리곤 정보

**컬럼**:
- `geometry`: 폴리곤 (좌표계 보존)
- `polygon_id`: 원본 폴리곤 ID
- `detection_id`: 검출 객체 ID
- `confidence`: 신뢰도 (0-1)
- `class_name`: 클래스명 ("곤포사일리지")
- `area_pixels`: 픽셀 면적
- `area_m2`: 실제 면적 (m²)

### 2. 통계 보고서

#### statistics.json
```json
{
  "timestamp": "2025-10-23T...",
  "processing": {
    "total_polygons": 10,
    "successful_polygons": 9,
    "success_rate": 0.9
  },
  "detections": {
    "total_detections": 45,
    "avg_detections_per_polygon": 4.5,
    "min_detections": 0,
    "max_detections": 12
  },
  "confidence": {
    "avg_confidence": 0.844,
    "min_confidence": 0.32,
    "max_confidence": 0.98
  }
}
```

#### polygon_details.csv
```csv
polygon_id,detection_count,avg_confidence,area_m2
polygon_0,5,0.85,1250.5
polygon_1,3,0.78,980.2
...
```

---

## 🎯 성능 지표

### 모델 성능 (테스트 세트)
- **mAP50 (Mask)**: 92.2%
- **Precision**: 96.5%
- **Recall**: 86.3%
- **추론 속도**: 11.3ms/이미지 (~36 FPS)

### 시스템 성능 (예상)
- **크롭 속도**: ~50개/분
- **추론 속도**: ~11.3ms/영역
- **전체 처리**: 1,000개 폴리곤 < 2시간

---

## 🐛 문제 해결

### 1. 메모리 부족 에러
```
MemoryError: Unable to allocate ...
```

**해결책**:
- `--limit` 옵션으로 처리 개수 제한
- `--min-area`, `--max-area`로 큰 폴리곤 필터링
- 배치 크기 조정 (코드 수정 필요)

### 2. 좌표계 불일치 경고
```
WARNING: 좌표계 불일치 감지!
```

**해결책**:
- 자동으로 변환됩니다 (걱정 안해도 됨)
- 수동 변환 필요 시: QGIS에서 SHP 재투영

### 3. GPU 메모리 부족
```
CUDA out of memory
```

**해결책**:
```bash
# CPU 사용
python pipeline.py --device cpu ...

# 또는 배치 크기 줄이기 (코드 수정)
```

---

## 📚 참조 문서

- [개발 계획서](../../Dev_md/09_곤포사일리지_추론시스템_개발계획.md)
- [YOLOv11 세그멘테이션 프로젝트](../../README.md)
- [saryo4model 참조 시스템](E:/namwon_ai/saryo4model/)

---

## 🔄 업데이트 이력

- **v1.0.0** (2025-10-23): 초기 릴리즈
  - SHP 기반 크롭 기능
  - 곤포사일리지 검출 엔진
  - 통합 파이프라인
  - GeoPackage 출력

---

## 👥 개발팀

- **시스템 설계**: Claude Sonnet 4.5
- **프로젝트 관리**: LX
- **모델 학습**: YOLOv11n-seg (mAP50: 92.2%)

---

**Made with ❤️ by AI Development Team**
