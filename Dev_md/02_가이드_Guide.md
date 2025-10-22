# YOLOv11 Segmentation 학습 가이드

**작성일**: 2025-10-22
**프로젝트**: 곤포사일리지(Silage Bale) 객체 탐지 및 세그멘테이션

---

## 목차
1. [환경 설정](#1-환경-설정)
2. [데이터 준비](#2-데이터-준비)
3. [모델 학습](#3-모델-학습)
4. [모델 검증](#4-모델-검증)
5. [추론 실행](#5-추론-실행)
6. [결과 분석](#6-결과-분석)

---

## 1. 환경 설정

### 1.1 필수 라이브러리 설치

```bash
# PyTorch 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ultralytics YOLOv11 설치
pip install ultralytics

# 추가 라이브러리
pip install opencv-python pillow numpy matplotlib
pip install rasterio  # TIF 파일 처리용
pip install tensorboard  # 학습 모니터링
```

### 1.2 환경 확인

```python
import torch
import ultralytics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics version: {ultralytics.__version__}")
```

---

## 2. 데이터 준비

### 2.1 데이터셋 구조

```
E:\namwon_ai\dataset_silage_bale/
├── images/
│   ├── train/  (259 TIF images)
│   ├── val/    (32 TIF images)
│   └── test/   (33 TIF images)
├── labels/
│   ├── train/  (259 .txt files)
│   ├── val/    (32 .txt files)
│   └── test/   (33 .txt files)
└── dataset.yaml
```

### 2.2 4-Band to 3-Band RGB 변환

TIF 이미지가 4밴드(R, G, B, NIR)인 경우 RGB 3밴드로 변환:

```python
import rasterio
import numpy as np
from pathlib import Path

def convert_4band_to_3band(input_path, output_path):
    """
    4밴드 TIF 이미지를 RGB 3밴드로 변환

    Args:
        input_path: 입력 4밴드 TIF 경로
        output_path: 출력 RGB 이미지 경로
    """
    with rasterio.open(input_path) as src:
        # 첫 3개 밴드(RGB)만 읽기
        rgb = src.read([1, 2, 3])  # R, G, B

        # (C, H, W) -> (H, W, C)로 변환
        rgb = np.transpose(rgb, (1, 2, 0))

        # 정규화 및 저장
        # ... (세부 구현)

    return rgb

# 전체 데이터셋 변환
def preprocess_dataset(dataset_path):
    """데이터셋 전체를 3밴드로 변환"""
    for split in ['train', 'val', 'test']:
        input_dir = Path(dataset_path) / 'images' / split
        output_dir = Path(dataset_path) / 'images_rgb' / split
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in input_dir.glob('*.tif'):
            output_path = output_dir / img_path.name
            convert_4band_to_3band(img_path, output_path)
```

### 2.3 dataset.yaml 확인

```yaml
# E:\namwon_ai\dataset_silage_bale\dataset.yaml
path: E:\namwon_ai\dataset_silage_bale
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names:
  - 곤포사일리지

task: segment
```

---

## 3. 모델 학습

### 3.1 기본 학습 코드

```python
from ultralytics import YOLO

# YOLOv11 segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # nano 모델
# 또는: yolo11s-seg.pt (small), yolo11m-seg.pt (medium), yolo11l-seg.pt (large)

# 학습 시작
results = model.train(
    data='E:/namwon_ai/dataset_silage_bale/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='silage_bale_seg',
    project='runs/segment',
    device=0,  # GPU 0번 사용 (CPU는 'cpu')

    # 추가 파라미터
    patience=50,  # Early stopping
    save=True,
    save_period=10,  # 10 epoch마다 저장

    # 하이퍼파라미터
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
```

### 3.2 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir runs/segment/silage_bale_seg
```

브라우저에서 `http://localhost:6006` 접속

### 3.3 학습 재개

```python
# 중단된 학습 재개
model = YOLO('runs/segment/silage_bale_seg/weights/last.pt')
results = model.train(resume=True)
```

---

## 4. 모델 검증

### 4.1 검증 실행

```python
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('runs/segment/silage_bale_seg/weights/best.pt')

# 검증
metrics = model.val(
    data='E:/namwon_ai/dataset_silage_bale/dataset.yaml',
    split='val',
    imgsz=640,
    batch=16,
    save_json=True,  # COCO 포맷으로 결과 저장
    save_hybrid=True,  # 하이브리드 라벨 저장
)

# 결과 출력
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

### 4.2 클래스별 성능 분석

```python
# 클래스별 AP
class_ap = metrics.box.maps
print(f"곤포사일리지 AP: {class_ap[0]}")

# Confusion Matrix
metrics.confusion_matrix.plot(save_dir='runs/segment/analysis')
```

---

## 5. 추론 실행

### 5.1 단일 이미지 추론

```python
from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/segment/silage_bale_seg/weights/best.pt')

# 추론
results = model.predict(
    source='E:/namwon_ai/dataset_silage_bale/images/test/1F001D60362.tif',
    imgsz=640,
    conf=0.25,  # Confidence threshold
    iou=0.7,    # NMS IoU threshold
    save=True,  # 결과 이미지 저장
    save_txt=True,  # 라벨 저장
    save_conf=True,  # Confidence 저장
)

# 결과 확인
for r in results:
    print(f"Boxes: {r.boxes}")
    print(f"Masks: {r.masks}")
    print(f"Confidence: {r.boxes.conf}")
```

### 5.2 배치 추론

```python
# 폴더 내 모든 이미지 추론
results = model.predict(
    source='E:/namwon_ai/dataset_silage_bale/images/test',
    imgsz=640,
    conf=0.25,
    save=True,
    project='runs/segment',
    name='inference_test'
)
```

### 5.3 결과 시각화

```python
import cv2
import matplotlib.pyplot as plt

# 결과 시각화
for r in results:
    img = r.plot()  # 마스크 오버레이된 이미지

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Segmentation Result')
    plt.savefig('result_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 6. 결과 분석

### 6.1 성능 메트릭 해석

| Metric | 설명 | 목표값 |
|--------|------|--------|
| mAP50 | IoU 0.5에서의 평균 정밀도 | > 0.80 |
| mAP50-95 | IoU 0.5~0.95의 평균 정밀도 | > 0.60 |
| Precision | 정밀도 (TP / (TP + FP)) | > 0.85 |
| Recall | 재현율 (TP / (TP + FN)) | > 0.80 |
| F1-Score | 정밀도와 재현율의 조화 평균 | > 0.82 |

### 6.2 학습 곡선 분석

- **Loss 감소**: 꾸준히 감소해야 함
- **Overfitting 체크**: Train/Val loss 차이 확인
- **수렴 여부**: Loss가 안정화되었는지 확인

### 6.3 실패 사례 분석

```python
# Confidence가 낮은 샘플 추출
low_conf_samples = [r for r in results if r.boxes.conf.max() < 0.5]

# False Positive/Negative 분석
# ... (세부 분석 코드)
```

---

## 7. 모델 최적화

### 7.1 하이퍼파라미터 튜닝

```python
# Optuna를 사용한 자동 튜닝
model.tune(
    data='E:/namwon_ai/dataset_silage_bale/dataset.yaml',
    epochs=30,
    iterations=100,
    optimizer='AdamW',
    plots=True,
    save=True,
)
```

### 7.2 모델 경량화

```python
# 모델 export (ONNX, TensorRT 등)
model.export(format='onnx')  # ONNX
model.export(format='engine')  # TensorRT
model.export(format='torchscript')  # TorchScript
```

---

## 8. 트러블슈팅

### 8.1 일반적인 문제

**Q: CUDA out of memory 에러**
```python
# 해결: batch size 줄이기
batch=8  # 16 -> 8
```

**Q: 학습이 수렴하지 않음**
```python
# 해결: Learning rate 조정
lr0=0.001  # 0.01 -> 0.001
```

**Q: mAP가 낮음**
- 데이터 증강 강화
- 학습 epoch 증가
- 더 큰 모델 사용 (n -> s -> m)

### 8.2 4-Band 이미지 처리 문제

**Q: YOLO가 4밴드 이미지를 읽지 못함**
- 반드시 3밴드 RGB로 변환 필요
- 전처리 스크립트 사용

---

## 참고 자료

- [Ultralytics YOLOv11 문서](https://docs.ultralytics.com/)
- [YOLO Segmentation 가이드](https://docs.ultralytics.com/tasks/segment/)
- [하이퍼파라미터 튜닝](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

---

**최종 수정**: 2025-10-22
**담당자**: AI Development Team
