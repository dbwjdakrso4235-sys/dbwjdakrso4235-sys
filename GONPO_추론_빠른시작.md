# 곤포사일리지 추론 - 빠른 시작 가이드

## 📋 현재 상태

### ✅ 준비 완료
- **모델**: `C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt` (5.8 MB, mAP50: 92.2%)
- **TIF**: `E:\namwon_ai\input_tif\금지면_1차.tif` (24.09 GB)
- **SHP**: `E:\namwon_ai\gonpo\gonpo_251028.shp` (2개 폴리곤)
- **추론 시스템**: 완전 구현됨

---

## 🚀 실행 방법

### 방법 1: 배치 파일 실행 (가장 간단)

```cmd
run_gonpo_inference.bat
```

더블클릭 또는 터미널에서 실행하면 됩니다.

### 방법 2: Python 직접 실행

```bash
python inference_system\examples\test_gonpo_inference.py
```

### 방법 3: 명령줄 옵션 사용

```bash
python inference_system\src\pipeline.py ^
    --tif "E:\namwon_ai\input_tif\금지면_1차.tif" ^
    --shp "E:\namwon_ai\gonpo\gonpo_251028.shp" ^
    --model "C:\Users\LX\dbwjdakrso4235-sys\runs\segment\silage_optimized\weights\best.pt" ^
    --output "inference_system\output_gonpo" ^
    --conf 0.25 ^
    --save-cropped
```

---

## 📊 예상 결과

### 처리 정보
- **폴리곤 수**: 2개
- **예상 처리 시간**: 1-3분 (폴리곤 크기에 따라 다름)
- **메모리 사용량**: 2-4GB (GPU 메모리)

### 출력 파일
```
inference_system/output_gonpo/
├── silage_bale_detections.gpkg    # GeoPackage 결과
├── visualizations/                # 검출 결과 시각화
│   ├── polygon_0_result.png
│   └── polygon_1_result.png
├── cropped_images/                # 크롭된 원본 이미지
│   ├── polygon_0_cropped.png
│   └── polygon_1_cropped.png
└── reports/                       # 통계 보고서
    ├── statistics.json            # 전체 통계
    ├── polygon_details.csv        # 폴리곤별 상세
    └── summary.txt                # 요약 보고서
```

---

## 🔍 결과 확인

### 1. 시각화 이미지 확인
```
inference_system\output_gonpo\visualizations\
```
폴더에서 PNG 이미지 확인

### 2. 통계 확인
```
inference_system\output_gonpo\reports\summary.txt
```
텍스트 에디터로 열어서 요약 확인

### 3. GeoPackage 확인
QGIS 또는 ArcGIS에서 `silage_bale_detections.gpkg` 열기

---

## ⚠️ 주의사항

### 메모리 부족 시
```bash
# CPU 모드로 실행 (느리지만 안전)
python inference_system\examples\test_gonpo_inference.py --device cpu
```

### 큰 폴리곤 처리 시
- 첫 번째 폴리곤만 테스트하고 싶다면 스크립트 수정 필요
- `pipeline.run(polygon_ids=[0])` 으로 변경

---

## 📞 문제 발생 시

### 1. 파일을 찾을 수 없음
- TIF, SHP, 모델 파일 경로를 다시 확인
- 경로에 한글이 있는 경우 문제가 될 수 있음

### 2. CUDA out of memory
```python
# test_gonpo_inference.py 수정
device='cpu'  # GPU 대신 CPU 사용
```

### 3. 좌표계 불일치 경고
- 자동으로 변환되므로 걱정하지 않아도 됨
- 결과에는 영향 없음

---

## 📈 다음 단계

추론 완료 후:
1. **결과 검토**: 시각화 이미지로 검출 품질 확인
2. **통계 분석**: `statistics.json` 확인
3. **Opus 분석**: 성능 메트릭 심층 분석
4. **최적화**: 필요 시 파라미터 조정

---

## 🎯 작업 체크리스트

### Sonnet (실무)
- [x] 모델 파일 위치 확인 ✅
- [x] 추론 스크립트 생성 ✅
- [ ] 추론 실행
- [ ] 결과 정리
- [ ] 기본 분석

### Opus (전략)
- [ ] 결과 심층 분석 (Sonnet 완료 후)
- [ ] 성능 최적화 전략
- [ ] 대규모 처리 계획

---

**준비 완료! 이제 `run_gonpo_inference.bat`을 실행하세요!**
