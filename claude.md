# Claude 개발 로그 - YOLOv11 Silage Bale Detection

**프로젝트**: 곤포사일리지 자동 검출 시스템
**모델**: Claude Sonnet 4.5 & Claude Opus
**시작일**: 2025-10-22
**최종 업데이트**: 2025-10-28

---

## 목차

1. [Day 1-2: 모델 학습 및 최적화 (2025-10-22~23)](#day-1-2-모델-학습-및-최적화)
2. [Day 3: 타일 기반 추론 시스템 구축 (2025-10-28)](#day-3-타일-기반-추론-시스템-구축)
3. [개발 성과 요약](#개발-성과-요약)
4. [향후 계획](#향후-계획)

---

## Day 1-2: 모델 학습 및 최적화

### 프로젝트 설정 및 데이터 준비

**목표**: 곤포사일리지 자동 검출을 위한 YOLOv11 Segmentation 모델 학습

**데이터셋**:
- 총 이미지: 324개 (Train: 259, Val: 32, Test: 33)
- 총 객체: 970개
- 클래스: 1 (silage_bale)
- 해상도: 1024 x 1024 pixels
- 포맷: 4-band TIF (R,G,B,NIR) → RGB PNG 변환

### 하이퍼파라미터 최적화

**최종 설정 (Optimized)**:
```yaml
model: yolo11n-seg
epochs: 150
batch: 8
imgsz: 1024
lr0: 0.001
optimizer: AdamW
weight_decay: 0.001

# Augmentation
mixup: 0.15
copy_paste: 0.3
mosaic: 1.0
degrees: 10.0
```

### 학습 결과

**성능 지표**:
| 메트릭 | 값 | 평가 |
|--------|-----|------|
| mAP50 (Mask) | 92.2% | ⭐⭐⭐ 우수 |
| mAP50-95 (Mask) | 85.3% | ⭐⭐⭐ 우수 |
| Precision | 96.5% | ⭐⭐⭐ 매우 높음 |
| Recall | 86.3% | ⭐⭐ 양호 |
| 추론 속도 | 11.3ms | ⭐⭐⭐ 실시간 |
| 검출률 | 97% | ⭐⭐⭐ 우수 |

**Loss 수렴**:
- box_loss: 1.188 → 0.372 (68.7% 감소)
- seg_loss: 2.273 → 0.509 (77.6% 감소)
- cls_loss: 3.411 → 0.347 (89.8% 감소)

### 추론 시스템 구축

**SHP-TIF 기반 추론 파이프라인**:
- SHP 기반 선택적 처리 (지정된 영역만 크롭)
- 대용량 TIF 파일 효율적 처리 (25.8GB → 부분 읽기)
- GeoPackage 출력 (좌표계 보존)
- 자동 통계 생성 (JSON, CSV, TXT)

**성능**:
- 크롭 속도: 100개/분
- 추론 속도: 0.6초/폴리곤
- 처리 성공률: 100% (15/15 폴리곤)

---

## Day 3: 타일 기반 추론 시스템 구축

**날짜**: 2025-10-28
**주요 문제**: Gonpo 데이터에서 검출 실패 (0개 검출)
**원인 분석**: 대형 크롭 이미지(2953x5721, 3381x6278)를 1024x1024로 리사이즈하면서 작은 객체 손실

### 문제 해결 과정

#### 1단계: SHP CRS 문제 진단 및 수정

**문제**:
- gonpo_251028.shp의 CRS 메타데이터 오류
- EPSG:4326으로 표시되었으나 실제 좌표는 EPSG:5186
- TIF와 SHP 중첩 실패로 크롭 불가

**해결**:
```python
# fix_shp_crs.py
import geopandas as gpd

shp = gpd.read_file("gonpo_251028.shp")
shp_fixed = shp.set_crs("EPSG:5186", allow_override=True)
shp_fixed.to_file("gonpo_251028_fixed.shp")
```

**결과**:
- 공간 중첩 100% 확인
- 2개 폴리곤 모두 정상 크롭 완료

#### 2단계: 검출 실패 원인 분석

**테스트 결과**:
- 신뢰도 0.25: 0개 검출
- 신뢰도 0.1: 0개 검출
- 육안 확인: 크롭된 이미지에 곤포사일리지 명확히 보임

**원인 분석** (GONPO_검출실패_최종분석.md):
1. **모델-데이터 불일치** (가능성: ⭐⭐⭐⭐⭐)
   - 학습 데이터와 테스트 데이터 특성 차이

2. **전처리 문제** (가능성: ⭐⭐⭐)
   - 4-band → RGB 변환 과정의 정보 손실

3. **이미지 해상도 문제** (가능성: ⭐⭐⭐⭐⭐) ← **실제 원인**
   - 대형 이미지를 1024x1024로 리사이즈하면서 객체 크기 축소
   - 작은 곤포사일리지가 검출 불가능한 크기로 축소됨

#### 3단계: 타일 기반 추론 시스템 개발

**설계 원칙**:
- 이미지를 원본 해상도로 처리
- 1024x1024 타일로 분할 (모델 입력 크기)
- 타일 간 overlap으로 경계 객체 검출 보장
- NMS로 중복 검출 제거

**구현** (tile_based_inference.py, 572 lines):

```python
# 핵심 클래스
class TileInfo:
    """타일 정보 (위치, 크기)"""

class TileDetection:
    """타일별 검출 결과 (bbox, mask, confidence)"""

class TiledInferenceEngine:
    """타일 기반 추론 엔진"""

    def create_tiles(self, image_shape, tile_size=1024, overlap=0.25):
        """이미지를 타일로 분할"""

    def process_tile(self, tile_image, tile_info):
        """개별 타일 추론"""

    def merge_detections(self, detections, iou_threshold=0.45):
        """NMS로 중복 제거"""

    def visualize_results(self, image, detections):
        """결과 시각화 (bbox + mask)"""
```

**주요 파라미터**:
- `tile_size`: 1024x1024 (모델 입력 크기)
- `overlap`: 0.25 (25% 중첩)
- `conf_threshold`: 0.25 (신뢰도 임계값)
- `iou_threshold`: 0.45 (NMS IoU 임계값)

#### 4단계: 좌표 인덱싱 문제 수정

**문제**:
- bbox와 segmentation mask 위치 불일치
- mask가 bbox 영역보다 크게 표시됨

**원인**:
```python
# 기존 코드 (문제)
valid_mask = binary_mask[:tile.height, :tile.width]
# 타일 전체 영역 사용 → bbox와 불일치
```

**해결** (2차례 수정):

**수정 1** (라인 218-247): Mask를 bbox 영역으로 crop
```python
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

detections.append(TileDetection(
    ...
    mask=cropped_mask,  # bbox 영역만 crop된 mask 사용
    ...
))
```

**수정 2** (라인 324-365): 글로벌 좌표 변환 시 정렬
```python
# bbox_global 좌표 가져오기
x1_global = int(det.bbox_global[0])
y1_global = int(det.bbox_global[1])
x2_global = int(det.bbox_global[2])
y2_global = int(det.bbox_global[3])

# mask를 bbox 크기에 맞게 리사이즈
if mask_h != target_h or mask_w != target_w:
    if target_h > 0 and target_w > 0:
        resized_mask = cv2.resize(det.mask, (target_w, target_h),
                                  interpolation=cv2.INTER_LINEAR)
        resized_mask = (resized_mask > 0.5).astype(np.uint8)

# 글로벌 마스크에 복사 (bbox 위치에 정확히 배치)
if copy_h > 0 and copy_w > 0:
    global_mask[y1_global:y1_global+copy_h,
                x1_global:x1_global+copy_w] = resized_mask[:copy_h, :copy_w]
```

### 최종 결과

**검출 성능**:
| Polygon | 크기 (픽셀) | 타일 수 | Raw 검출 | NMS 후 | 평균 신뢰도 | 처리 시간 |
|---------|------------|---------|---------|---------|------------|----------|
| 0 | 2953x5721 | 32 | 5 | 3 | 76.4% | 4.27초 |
| 1 | 3381x6278 | 45 | 12 | 12 | 69.6% | 1.26초 |
| **총합** | - | 77 | 17 | **15** | **73.0%** | 5.53초 |

**출력 파일**:
```
inference_system/output_tiled_fixed/
├── polygon_0_result.png          # 시각화 결과 (bbox + mask)
├── polygon_0_results.json        # 검출 상세 정보
├── polygon_1_result.png
└── polygon_1_results.json
```

**성공 요인**:
1. 원본 해상도 유지 (리사이즈 없음)
2. 적절한 타일 크기 (1024x1024)
3. 25% overlap으로 경계 객체 검출 보장
4. NMS로 중복 효과적 제거
5. 정확한 좌표 변환 및 정렬

---

## 개발 성과 요약

### 기술적 성과

**1. 고성능 모델 개발**
- mAP50: 92.2% (목표 75-85% 초과 달성)
- 추론 속도: 11.3ms/이미지 (~36 FPS)
- 검출률: 97% (33개 중 32개 성공)

**2. 지리공간 추론 시스템**
- SHP 기반 선택적 처리
- 대용량 TIF 효율적 처리
- GeoPackage 출력 (좌표계 보존)
- 자동 통계 생성

**3. 타일 기반 추론 시스템** ⭐ NEW
- 대형 이미지 처리 (2953x5721, 3381x6278)
- 1024x1024 타일 분할 (25% overlap)
- NMS 중복 제거
- 정확한 좌표 변환 및 정렬
- 15개 곤포사일리지 검출 성공 (평균 신뢰도 73.0%)

### 문제 해결 능력

**진단한 문제**:
1. SHP CRS 메타데이터 오류 (EPSG:4326 vs 5186)
2. 대형 이미지 리사이즈로 인한 객체 손실
3. bbox와 segmentation mask 좌표 불일치

**해결 전략**:
1. CRS 수정 도구 개발 (fix_shp_crs.py)
2. 타일 기반 추론 시스템 구축 (tile_based_inference.py)
3. 좌표 인덱싱 정렬 알고리즘 개발

### 문서화

**작성한 문서**:
1. README.md - 프로젝트 전체 가이드
2. claude.md - 개발 로그 (본 문서)
3. 곤포사일리지_추론_작업분담.md - 작업 분담 계획
4. GONPO_검출실패_최종분석.md - 문제 분석 보고서
5. START_HERE.md - 빠른 시작 가이드
6. Dev_md/ - 각종 개발 문서

---

## 향후 계획

### 단기 (1-2주)

**1. 모델 개선**
- [ ] Gonpo 데이터로 Fine-tuning
- [ ] Advanced 모델 (yolo11s-seg) 학습
- [ ] 신뢰도 임계값 최적화

**2. 시스템 최적화**
- [ ] 타일 기반 추론 속도 향상 (배치 처리)
- [ ] 메모리 사용량 최적화
- [ ] GPU 활용률 개선

**3. 기능 추가**
- [ ] 전체 파이프라인에 타일 기반 추론 통합
- [ ] 대형 이미지 자동 감지 및 처리 방식 선택
- [ ] 실시간 진행 상황 모니터링

### 중기 (1-2개월)

**1. API 서버 구축**
- [ ] FastAPI 기반 REST API
- [ ] 비동기 작업 큐 (Celery)
- [ ] 웹 인터페이스

**2. 모델 최적화**
- [ ] ONNX 변환
- [ ] TensorRT 최적화
- [ ] 모바일 배포 (ONNX Runtime Mobile)

**3. NIR 밴드 활용**
- [ ] 4채널 입력 모델 연구
- [ ] RGB+NIR fusion 방법 개발

### 장기 (3개월+)

**1. 다중 작물 지원**
- [ ] 다른 작물 데이터셋 수집
- [ ] Multi-class 모델 학습
- [ ] 작물별 특화 모델 개발

**2. 시계열 분석**
- [ ] 시간에 따른 재고 변화 추적
- [ ] 이상 패턴 감지
- [ ] 예측 모델 개발

**3. 프로덕션 배포**
- [ ] Docker 컨테이너화
- [ ] Kubernetes 배포
- [ ] 모니터링 시스템 구축
- [ ] CI/CD 파이프라인

---

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **Deep Learning** | PyTorch 2.7.1, Ultralytics YOLOv11 |
| **Computer Vision** | OpenCV 4.12.0, rasterio 1.4.3 |
| **Geospatial** | geopandas 1.1.1, shapely 2.0.6 |
| **Scientific Computing** | NumPy 2.1.2, Matplotlib 3.10.7 |
| **GPU** | CUDA 11.8, NVIDIA RTX A6000 |

---

## 협업 모델

**Claude Sonnet 4.5** (Implementation):
- 코드 구현 및 디버깅
- 시스템 설정 및 환경 구축
- 실험 실행 및 결과 정리
- 문서화 및 리포트 작성

**Claude Opus** (Strategy & Analysis):
- 문제 원인 심층 분석
- 최적화 전략 수립
- 성능 메트릭 분석
- 아키텍처 설계

---

## 참고 자료

**코드 저장소**:
- GitHub: [dbwjdakrso4235-sys-1](https://github.com/YOUR_USERNAME/dbwjdakrso4235-sys-1)

**주요 파일**:
- `tile_based_inference.py` - 타일 기반 추론 시스템 (572 lines)
- `fix_shp_crs.py` - SHP CRS 수정 도구
- `check_spatial_overlap.py` - 공간 중첩 검증
- `inference_system/` - 기본 추론 시스템
- `scripts/train.py` - 학습 스크립트
- `configs/train_optimized.yaml` - 최적화된 학습 설정

**문서**:
- `README.md` - 프로젝트 메인 가이드
- `Dev_md/` - 개발 문서 모음
- `GONPO_검출실패_최종분석.md` - 문제 분석
- `곤포사일리지_추론_작업분담.md` - 작업 분담

---

**Last Updated**: 2025-10-28
**Author**: Claude Sonnet 4.5
**Project**: YOLOv11 Segmentation - Silage Bale Detection
**License**: MIT
