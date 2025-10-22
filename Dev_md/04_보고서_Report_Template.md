# YOLOv11 Segmentation 학습 보고서

**보고 일자**: YYYY-MM-DD
**프로젝트**: 곤포사일리지(Silage Bale) 객체 탐지 및 세그멘테이션
**모델**: YOLOv11-seg

---

## 1. 개요 (Executive Summary)

### 1.1 프로젝트 목표
- 곤포사일리지 객체의 자동 탐지 및 세그멘테이션
- YOLOv11 segmentation 모델 학습
- mAP50 80% 이상 달성

### 1.2 주요 결과
- **최종 mAP50**: XX.XX%
- **최종 mAP50-95**: XX.XX%
- **Precision**: XX.XX%
- **Recall**: XX.XX%
- **F1-Score**: XX.XX%

### 1.3 결론
[학습 성공 여부 및 종합 평가]

---

## 2. 데이터셋 정보

### 2.1 데이터셋 구성
| 항목 | 내용 |
|------|------|
| 데이터셋명 | Silage Bale Dataset |
| 클래스 수 | 1 (곤포사일리지) |
| 총 이미지 수 | 324장 |
| 이미지 형식 | TIF (4-band → RGB 변환) |
| 해상도 | [width x height] |

### 2.2 데이터 분할
| Split | 이미지 수 | 객체 수 | 비율 |
|-------|----------|---------|------|
| Train | 259 | ~776 | 79.9% |
| Val   | 32  | ~97  | 9.9%  |
| Test  | 33  | ~97  | 10.2% |

### 2.3 데이터 전처리
- 4밴드 TIF 이미지를 RGB 3밴드로 변환
- 이미지 크기: 640x640으로 리사이즈
- 정규화: [0, 255] → [0, 1]
- 데이터 증강:
  - Horizontal Flip: 50%
  - HSV 변환: H(0.015), S(0.7), V(0.4)
  - Translation: 10%
  - Scale: 50%

---

## 3. 모델 설정

### 3.1 모델 아키텍처
- **Base Model**: YOLOv11n-seg / YOLOv11s-seg / YOLOv11m-seg
- **Input Size**: 640x640
- **Output**: Segmentation masks + Bounding boxes

### 3.2 학습 하이퍼파라미터
```yaml
# 기본 설정
epochs: 100
batch_size: 16
imgsz: 640
device: GPU 0

# Optimizer
optimizer: AdamW / SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# Early Stopping
patience: 50

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
```

### 3.3 학습 환경
- **OS**: Windows / Linux
- **GPU**: [GPU 모델명]
- **CUDA**: [버전]
- **PyTorch**: [버전]
- **Ultralytics**: [버전]

---

## 4. 학습 결과

### 4.1 성능 메트릭

#### 최종 성능 (Best Epoch: XX)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| mAP50 | XX.XX% | XX.XX% | XX.XX% |
| mAP50-95 | XX.XX% | XX.XX% | XX.XX% |
| Precision | XX.XX% | XX.XX% | XX.XX% |
| Recall | XX.XX% | XX.XX% | XX.XX% |
| F1-Score | XX.XX% | XX.XX% | XX.XX% |

#### Loss 변화
| Loss Type | 초기값 | 최종값 |
|-----------|--------|--------|
| Box Loss | X.XXX | X.XXX |
| Seg Loss | X.XXX | X.XXX |
| Cls Loss | X.XXX | X.XXX |
| Total Loss | X.XXX | X.XXX |

### 4.2 학습 곡선
[학습 곡선 그래프 이미지 첨부]

- `loss_curve.png`: Loss 변화 그래프
- `metrics_curve.png`: mAP, Precision, Recall 변화
- `lr_curve.png`: Learning rate schedule

### 4.3 혼동 행렬 (Confusion Matrix)
[Confusion Matrix 이미지 첨부]

```
              Predicted
              BG    곤포사일리지
Actual BG      XX      XX
       곤포     XX      XX
```

---

## 5. 검증 결과 분석

### 5.1 클래스별 성능
| 클래스 | AP50 | AP75 | AP50-95 | Precision | Recall |
|--------|------|------|---------|-----------|--------|
| 곤포사일리지 | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

### 5.2 IoU 분포
[IoU 분포 히스토그램]

### 5.3 Confidence 분포
[Confidence score 분포 그래프]

---

## 6. 추론 결과

### 6.1 정성적 평가
[샘플 이미지 결과 첨부]

#### 우수 사례 (Good Cases)
![Good Case 1](path/to/good1.png)
![Good Case 2](path/to/good2.png)

**분석**:
- 명확한 경계 검출
- 높은 confidence score
- 정확한 segmentation

#### 실패 사례 (Failure Cases)
![Failure Case 1](path/to/fail1.png)
![Failure Case 2](path/to/fail2.png)

**분석**:
- 겹쳐진 객체 분리 실패
- 작은 객체 미검출
- 그림자/조명 영향

### 6.2 정량적 평가
| 구분 | 이미지 수 | 검출 객체 수 | 평균 Confidence |
|------|----------|-------------|----------------|
| Test Set | 33 | XX | X.XXX |

---

## 7. 오류 분석 (Error Analysis)

### 7.1 False Positive 분석
- **원인**: [배경을 객체로 오검출하는 경우]
- **빈도**: XX cases
- **대응 방안**: [Confidence threshold 조정, Hard Negative Mining]

### 7.2 False Negative 분석
- **원인**: [작은 객체, 가려진 객체 미검출]
- **빈도**: XX cases
- **대응 방안**: [Multi-scale training, Data augmentation 강화]

### 7.3 Segmentation 품질
- **Over-segmentation**: XX cases
- **Under-segmentation**: XX cases
- **경계 정확도**: [평가 결과]

---

## 8. 비교 실험 (Ablation Study)

### 8.1 모델 크기 비교
| Model | mAP50 | mAP50-95 | Speed (ms) | Params (M) |
|-------|-------|----------|------------|------------|
| YOLOv11n-seg | XX.XX% | XX.XX% | XX.X | XX.X |
| YOLOv11s-seg | XX.XX% | XX.XX% | XX.X | XX.X |
| YOLOv11m-seg | XX.XX% | XX.XX% | XX.X | XX.X |

### 8.2 하이퍼파라미터 영향
| 파라미터 | 값 | mAP50 | 비고 |
|----------|-----|-------|------|
| Image Size | 416 | XX.XX% | Baseline |
| Image Size | 640 | XX.XX% | 선택 |
| Batch Size | 8 | XX.XX% | |
| Batch Size | 16 | XX.XX% | 선택 |

### 8.3 데이터 증강 영향
| Augmentation | mAP50 | 변화 |
|--------------|-------|------|
| No Aug | XX.XX% | Baseline |
| Flip only | XX.XX% | +X.XX% |
| Full Aug | XX.XX% | +X.XX% |

---

## 9. 결론 및 제언

### 9.1 달성 사항
- ✅ YOLOv11 segmentation 모델 성공적 학습
- ✅ 4밴드 TIF 이미지 전처리 파이프라인 구축
- ✅ mAP50 목표(80%) 달성 여부: [YES/NO]

### 9.2 한계점
- 작은 객체 검출 정확도 낮음
- 겹쳐진 객체 분리 어려움
- 조명 변화에 민감

### 9.3 개선 방안
1. **데이터 확보**:
   - 학습 데이터 추가 수집
   - 어려운 케이스(작은 객체, 겹침) 보강

2. **모델 개선**:
   - 더 큰 모델 시도 (YOLOv11l-seg, YOLOv11x-seg)
   - Multi-scale training 적용
   - Attention mechanism 추가

3. **후처리**:
   - NMS threshold 최적화
   - Confidence threshold 조정
   - Post-processing filter 적용

4. **NIR 밴드 활용**:
   - 4채널 입력 모델 실험
   - RGB + NIR fusion 연구

### 9.4 향후 계획
- [ ] 추가 데이터 수집 및 라벨링
- [ ] 앙상블 모델 실험
- [ ] 실시간 추론 최적화
- [ ] 배포 환경 구축

---

## 10. 참고 자료

### 10.1 관련 문서
- [01_규칙_Rules.md](./01_규칙_Rules.md)
- [02_가이드_Guide.md](./02_가이드_Guide.md)
- [03_개발일지_DevLog_20251022.md](./03_개발일지_DevLog_20251022.md)

### 10.2 외부 링크
- [Ultralytics YOLOv11 문서](https://docs.ultralytics.com/)
- [YOLO Segmentation](https://docs.ultralytics.com/tasks/segment/)

### 10.3 코드 저장소
- GitHub: [Repository URL]
- Models: `runs/segment/silage_bale_seg/`

---

## 부록 (Appendix)

### A. 학습 로그
```
Epoch 1/100: loss=X.XXX, mAP50=XX.XX%
Epoch 10/100: loss=X.XXX, mAP50=XX.XX%
...
Epoch 100/100: loss=X.XXX, mAP50=XX.XX%
```

### B. 환경 설정
```bash
# requirements.txt
torch==2.x.x
ultralytics==8.x.x
opencv-python==4.x.x
rasterio==1.x.x
...
```

### C. 실험 파라미터
[전체 실험 설정 YAML 파일]

---

**작성자**: [이름]
**검토자**: [이름]
**승인자**: [이름]
**최종 수정일**: YYYY-MM-DD
