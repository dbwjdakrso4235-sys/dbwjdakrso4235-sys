# 자주 사용하는 명령어 모음 (Command Reference)

**작성일**: 2025-10-22
**프로젝트**: YOLOv11 Segmentation - Silage Bale Detection

---

## 1. 환경 설정 (Environment Setup)

### 1.1 Python 가상환경
```bash
# 가상환경 생성
python -m venv yolo_env

# 가상환경 활성화 (Windows)
yolo_env\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source yolo_env/bin/activate

# 가상환경 비활성화
deactivate
```

### 1.2 라이브러리 설치
```bash
# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch 설치 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio

# Ultralytics YOLO
pip install ultralytics

# 추가 라이브러리
pip install opencv-python pillow numpy matplotlib rasterio tensorboard

# requirements.txt로 일괄 설치
pip install -r requirements.txt

# requirements.txt 생성
pip freeze > requirements.txt
```

### 1.3 환경 확인
```bash
# Python 버전
python --version

# CUDA 버전
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Ultralytics 버전
yolo version
```

---

## 2. 데이터 전처리 (Data Preprocessing)

### 2.1 4-Band to RGB 변환
```python
# Python 스크립트 실행
python scripts/convert_4band_to_rgb.py \
    --input E:/namwon_ai/dataset_silage_bale/images \
    --output E:/namwon_ai/dataset_silage_bale_rgb/images

# 배치 변환
python scripts/batch_preprocess.py \
    --dataset_path E:/namwon_ai/dataset_silage_bale
```

### 2.2 데이터 검증
```bash
# 이미지 개수 확인
python scripts/check_dataset.py --path E:/namwon_ai/dataset_silage_bale

# 라벨 검증
python scripts/validate_labels.py --labels_path E:/namwon_ai/dataset_silage_bale/labels

# 데이터셋 시각화
python scripts/visualize_dataset.py --num_samples 10
```

---

## 3. 모델 학습 (Training)

### 3.1 기본 학습
```bash
# YOLOv11n-seg (nano) 학습
yolo segment train \
    data=E:/namwon_ai/dataset_silage_bale/dataset.yaml \
    model=yolo11n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=silage_bale_seg

# YOLOv11s-seg (small) 학습
yolo segment train \
    data=E:/namwon_ai/dataset_silage_bale/dataset.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16

# Python 스크립트로 학습
python scripts/train.py --config configs/yolo11_seg.yaml
```

### 3.2 고급 학습 옵션
```bash
# 커스텀 하이퍼파라미터
yolo segment train \
    data=dataset.yaml \
    model=yolo11n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    lr0=0.01 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    patience=50 \
    save_period=10 \
    device=0

# Multi-GPU 학습
yolo segment train data=dataset.yaml model=yolo11n-seg.pt device=0,1,2,3

# 학습 재개
yolo segment train resume model=runs/segment/train/weights/last.pt

# Transfer Learning (사전학습 모델 사용)
yolo segment train \
    data=dataset.yaml \
    model=runs/segment/pretrained/best.pt \
    epochs=50
```

### 3.3 하이퍼파라미터 튜닝
```bash
# Optuna 자동 튜닝
yolo segment tune \
    data=dataset.yaml \
    model=yolo11n-seg.pt \
    epochs=30 \
    iterations=100 \
    optimizer=AdamW
```

---

## 4. 모델 검증 (Validation)

### 4.1 검증 실행
```bash
# Best 모델 검증
yolo segment val \
    model=runs/segment/train/weights/best.pt \
    data=dataset.yaml \
    imgsz=640 \
    batch=16

# 특정 split 검증
yolo segment val \
    model=best.pt \
    data=dataset.yaml \
    split=test

# 결과 저장
yolo segment val \
    model=best.pt \
    data=dataset.yaml \
    save_json=True \
    save_hybrid=True
```

### 4.2 성능 분석
```python
# Python 스크립트로 상세 분석
python scripts/analyze_metrics.py \
    --model runs/segment/train/weights/best.pt \
    --data dataset.yaml
```

---

## 5. 추론 (Inference)

### 5.1 단일 이미지 추론
```bash
# 기본 추론
yolo segment predict \
    model=runs/segment/train/weights/best.pt \
    source=path/to/image.tif \
    imgsz=640 \
    conf=0.25

# 결과 저장
yolo segment predict \
    model=best.pt \
    source=image.tif \
    save=True \
    save_txt=True \
    save_conf=True \
    project=runs/predict \
    name=test_results
```

### 5.2 배치 추론
```bash
# 폴더 내 모든 이미지
yolo segment predict \
    model=best.pt \
    source=E:/namwon_ai/dataset_silage_bale/images/test \
    imgsz=640 \
    conf=0.25

# 비디오 추론
yolo segment predict \
    model=best.pt \
    source=video.mp4 \
    save=True

# 웹캠 실시간 추론
yolo segment predict \
    model=best.pt \
    source=0 \
    show=True
```

### 5.3 추론 스크립트
```python
# Python 스크립트
python scripts/inference.py \
    --model runs/segment/train/weights/best.pt \
    --source test_images/ \
    --output results/ \
    --conf 0.25 \
    --iou 0.7
```

---

## 6. 모델 Export

### 6.1 다양한 포맷으로 변환
```bash
# ONNX
yolo export model=best.pt format=onnx

# TensorRT
yolo export model=best.pt format=engine device=0

# TorchScript
yolo export model=best.pt format=torchscript

# CoreML (iOS)
yolo export model=best.pt format=coreml

# TFLite (Android)
yolo export model=best.pt format=tflite

# 모든 포맷 export
python scripts/export_all.py --model best.pt
```

### 6.2 최적화 옵션
```bash
# INT8 양자화
yolo export model=best.pt format=engine int8=True

# 반정밀도 (FP16)
yolo export model=best.pt format=onnx half=True

# 동적 배치 크기
yolo export model=best.pt format=onnx dynamic=True
```

---

## 7. 시각화 및 모니터링

### 7.1 TensorBoard
```bash
# TensorBoard 시작
tensorboard --logdir runs/segment/train

# 특정 포트
tensorboard --logdir runs/segment --port 6006

# 원격 접속 허용
tensorboard --logdir runs/segment --host 0.0.0.0
```

### 7.2 결과 시각화
```bash
# 학습 곡선 플롯
python scripts/plot_results.py --dir runs/segment/train

# Confusion Matrix
python scripts/plot_confusion_matrix.py --model best.pt --data dataset.yaml

# 예측 결과 시각화
python scripts/visualize_predictions.py \
    --model best.pt \
    --images test_images/ \
    --output visualizations/
```

---

## 8. 유틸리티 명령어

### 8.1 파일 관리
```bash
# 디렉토리 생성
mkdir -p runs/segment/experiments

# 파일 복사
cp runs/segment/train/weights/best.pt models/production/

# 파일 이동
mv runs/segment/exp1 runs/segment/silage_bale_v1

# 압축
tar -czf models_backup.tar.gz runs/segment/train/weights/

# 압축 해제
tar -xzf models_backup.tar.gz
```

### 8.2 Git 명령어
```bash
# 초기화
git init
git add .
git commit -m "Initial commit: YOLOv11 segmentation setup"

# 변경사항 확인
git status
git diff

# 커밋
git add .
git commit -m "[feat] Add 4-band to RGB preprocessing"

# 푸시
git push origin main

# 브랜치 생성 및 전환
git checkout -b feature/data-augmentation

# 브랜치 병합
git checkout main
git merge feature/data-augmentation
```

### 8.3 시스템 모니터링
```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# GPU 사용량 로그
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_log.csv

# 디스크 사용량
df -h

# 프로세스 확인
ps aux | grep python

# 프로세스 종료
kill -9 [PID]
```

---

## 9. 데이터셋 관리

### 9.1 데이터셋 분할
```python
# 데이터셋 재분할
python scripts/split_dataset.py \
    --input dataset/ \
    --output dataset_split/ \
    --train 0.8 \
    --val 0.1 \
    --test 0.1
```

### 9.2 데이터 증강
```bash
# Albumentations 사용
python scripts/augment_dataset.py \
    --input images/train \
    --output images/train_aug \
    --augment_factor 3
```

---

## 10. 프로젝트 관리

### 10.1 실험 관리
```bash
# 실험 로그 생성
python scripts/log_experiment.py \
    --name exp1 \
    --model yolo11n-seg \
    --epochs 100 \
    --batch 16

# 실험 비교
python scripts/compare_experiments.py \
    --exp1 runs/segment/exp1 \
    --exp2 runs/segment/exp2
```

### 10.2 문서 생성
```bash
# 자동 보고서 생성
python scripts/generate_report.py \
    --model best.pt \
    --results runs/segment/train \
    --output Dev_md/04_보고서_Report_$(date +%Y%m%d).md

# README 업데이트
python scripts/update_readme.py --metrics runs/segment/train/results.csv
```

---

## 11. 자주 사용하는 Python 스니펫

### 11.1 모델 로드 및 추론
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/segment/train/weights/best.pt')

# 추론
results = model.predict('image.tif', conf=0.25)

# 결과 확인
for r in results:
    print(f"Boxes: {r.boxes}")
    print(f"Masks: {r.masks}")
```

### 11.2 성능 평가
```python
# 검증
metrics = model.val(data='dataset.yaml')
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

---

## 12. 트러블슈팅 명령어

### 12.1 CUDA/GPU 문제
```bash
# CUDA 재설정
export CUDA_VISIBLE_DEVICES=0

# GPU 메모리 정리
nvidia-smi --gpu-reset

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 12.2 패키지 문제
```bash
# 패키지 재설치
pip uninstall ultralytics
pip install ultralytics --no-cache-dir

# 캐시 정리
pip cache purge

# 의존성 문제 해결
pip install --upgrade pip setuptools wheel
```

---

## 참고 사항

### 단축 명령어 (Aliases)
```bash
# .bashrc 또는 .zshrc에 추가
alias yolo-train="yolo segment train data=dataset.yaml"
alias yolo-val="yolo segment val data=dataset.yaml"
alias yolo-pred="yolo segment predict"
alias gpu-watch="watch -n 1 nvidia-smi"
alias tb="tensorboard --logdir runs/segment"
```

---

**최종 수정**: 2025-10-22
**관리자**: AI Development Team
