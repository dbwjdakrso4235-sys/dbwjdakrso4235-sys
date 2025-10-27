# YOLOv11 Segmentation - Silage Bale Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-red.svg)](https://pytorch.org/)
[![Ultralytics 8.3+](https://img.shields.io/badge/ultralytics-8.3+-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

곤포사일리지(Silage Bale) 자동 검출 및 세그멘테이션 시스템

> 🚀 **처음 시작하시나요?** [`START_HERE.md`](START_HERE.md) 파일을 먼저 확인하세요!

[English](#english) | [한국어](#한국어)

---

## 한국어

### 📋 프로젝트 개요

드론/위성 촬영 이미지에서 곤포사일리지를 자동으로 검출하고 세그멘테이션하여 재고 관리 및 물량 파악을 자동화하는 AI 시스템입니다.

### ✨ 주요 기능

- **고정밀 검출**: mAP50 92.2% (목표 75-85% 초과 달성)
- **실시간 추론**: 11.3ms/이미지 (~36 FPS)
- **자동 카운팅**: 객체 개수 자동 집계
- **4-band 이미지 지원**: TIF (R,G,B,NIR) → RGB 자동 변환
- **완전 자동화**: 전처리 → 학습 → 추론 파이프라인

### 🎯 성능 지표

| 메트릭 | 값 | 평가 |
|--------|-----|------|
| **mAP50 (Mask)** | 92.2% | ⭐⭐⭐ 우수 |
| **mAP50-95 (Mask)** | 85.3% | ⭐⭐⭐ 우수 |
| **Precision** | 96.5% | ⭐⭐⭐ 매우 높음 |
| **Recall** | 86.3% | ⭐⭐ 양호 |
| **추론 속도** | 11.3ms | ⭐⭐⭐ 실시간 |
| **검출률** | 97% | ⭐⭐⭐ 우수 |

### 🚀 빠른 시작

#### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/dbwjdakrso4235-sys.git
cd dbwjdakrso4235-sys

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python rasterio numpy matplotlib tensorboard tqdm pyyaml
```

#### 2. 데이터 전처리

```bash
# 4-band TIF → RGB PNG 변환
python scripts/preprocess_dataset.py \
    --input E:/namwon_ai/dataset_silage_bale \
    --output E:/namwon_ai/dataset_silage_bale_rgb \
    --format png
```

#### 3. 모델 학습

```bash
# Optimized 설정으로 학습
python scripts/train.py \
    --data E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml \
    --model n \
    --epochs 150 \
    --batch 8 \
    --imgsz 1024 \
    --lr0 0.001 \
    --optimizer AdamW \
    --patience 100 \
    --name silage_optimized

# 또는 YAML 설정 파일 사용
yolo segment train data=configs/train_optimized.yaml
```

#### 4. 추론

```bash
# 단일 이미지 추론
python scripts/inference.py \
    --model runs/segment/silage_optimized/weights/best.pt \
    --source path/to/image.jpg \
    --imgsz 1024 \
    --save

# 배치 추론 (폴더)
python scripts/inference.py \
    --model runs/segment/silage_optimized/weights/best.pt \
    --source path/to/images/ \
    --imgsz 1024 \
    --save --save-txt --analyze
```

### 📁 프로젝트 구조

```
dbwjdakrso4235-sys/
├── configs/
│   ├── train_optimized.yaml      # 최적화된 학습 설정
│   └── train_advanced.yaml       # Advanced 모델 설정
├── scripts/
│   ├── train.py                  # 학습 스크립트
│   ├── inference.py              # 추론 스크립트
│   └── preprocess_dataset.py    # 데이터 전처리
├── utils/
│   └── preprocess.py             # 전처리 유틸리티
├── inference_system/             # ⭐ 곤포사일리지 추론 시스템
│   ├── src/
│   │   ├── crop_processor.py    # SHP 기반 TIF 크롭
│   │   ├── inference_engine.py  # YOLO 추론 엔진
│   │   └── pipeline.py          # 통합 파이프라인
│   ├── examples/
│   │   ├── test_crop.py
│   │   ├── test_inference.py
│   │   ├── test_valid_polygons.py
│   │   └── test_full_pipeline.py
│   ├── output/                   # 추론 결과
│   └── README.md                 # 시스템 가이드
├── Dev_md/
│   ├── 01_규칙_Rules.md         # 개발 규칙
│   ├── 02_가이드_Guide.md       # 사용 가이드
│   ├── 03_개발일지_DevLog_20251022.md
│   ├── 06_하이퍼파라미터_최적화.md
│   ├── 07_최종보고서_Final_Report.md
│   ├── 08_향후계획_Future_Roadmap.md
│   ├── 09_곤포사일리지_추론시스템_개발계획.md  # ⭐ 추론 시스템 설계
│   └── claude.md                 # Claude 작업 분담
├── runs/                         # 학습/추론 결과
├── README.md                     # 본 문서
└── .gitignore
```

### 📊 데이터셋

- **총 이미지**: 324개
  - Train: 259 (79.9%)
  - Val: 32 (9.9%)
  - Test: 33 (10.2%)
- **총 객체**: 970개
- **클래스**: 1 (곤포사일리지)
- **해상도**: 1024 x 1024 pixels
- **포맷**: 4-band TIF (R,G,B,NIR) → RGB PNG

### 🔧 하이퍼파라미터

#### 최적화된 설정 (Optimized)
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

자세한 설정은 [`configs/train_optimized.yaml`](configs/train_optimized.yaml) 참조

### 📈 학습 결과

#### 학습 곡선
```
Epoch   1:   0.5% mAP50
Epoch   2:  31.2% mAP50 (65배 증가!)
Epoch  50:  80.3% mAP50 (목표 달성)
Epoch 100:  89.4% mAP50
Epoch 150:  92.2% mAP50 (최종)
```

#### Loss 수렴
| Loss | 초기 | 최종 | 감소율 |
|------|------|------|--------|
| box_loss | 1.188 | 0.372 | 68.7% |
| seg_loss | 2.273 | 0.509 | 77.6% |
| cls_loss | 3.411 | 0.347 | 89.8% |

학습 곡선 및 상세 분석은 [`runs/segment/silage_optimized/results.png`](runs/segment/silage_optimized/results.png) 참조

### 🎯 추론 결과

#### Test 세트 평가
- **총 이미지**: 33개
- **검출 성공**: 32개 (97%)
- **검출 실패**: 1개 (3%)
- **총 검출 객체**: 120개
- **평균 Confidence**: 84.4%
- **추론 속도**: 11.3ms/이미지

#### 개수별 분포
| 검출 개수 | 이미지 수 | 비율 |
|----------|----------|------|
| 1개 | 9장 | 28.1% |
| 2개 | 10장 | 31.3% |
| 3-5개 | 10장 | 31.3% |
| 7개 이상 | 3장 | 9.4% |
| 최대 | 31개 | - |

### 🗺️ 곤포사일리지 추론 시스템 (NEW!)

대용량 TIF 이미지와 Shapefile을 활용한 지리공간 기반 곤포사일리지 자동 검출 시스템

#### 주요 기능
- **SHP 기반 선택적 처리**: Shapefile로 정의된 영역만 크롭하여 처리 효율 극대화
- **대용량 TIF 지원**: 25.8GB TIF 파일을 메모리에 올리지 않고 부분 읽기
- **GeoPackage 출력**: 검출 결과를 지리 좌표계와 함께 저장
- **자동 통계 생성**: JSON, CSV, TXT 형식의 상세 보고서 자동 생성

#### 빠른 시작

```bash
# 전체 파이프라인 실행
python inference_system/examples/test_full_pipeline.py

# 또는 명령줄 인터페이스 사용
python inference_system/src/pipeline.py \
    --tif F:/namwon_ai/input_tif/금지면_1차.tif \
    --shp F:/namwon_ai/saryo_jeongbo/saryo_parcel.shp \
    --model runs/segment/silage_optimized/weights/best.pt \
    --output inference_system/output
```

#### 시스템 성능
- **크롭 속도**: 100개/분 (목표 50개/분의 2배)
- **추론 속도**: 0.6초/폴리곤
- **처리 성공률**: 100% (15/15 폴리곤)
- **메모리 효율**: 대용량 TIF를 메모리에 올리지 않고 처리

#### 출력 형식
```
output/
├── silage_bale_detections.gpkg    # GeoPackage (좌표계 보존)
├── visualizations/                 # 검출 결과 시각화
└── reports/
    ├── statistics.json             # 전체 통계
    ├── polygon_details.csv         # 폴리곤별 상세
    └── summary.txt                 # 요약 보고서
```

상세 가이드: [`inference_system/README.md`](inference_system/README.md)

### 💻 시스템 요구사항

#### 최소 사양
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10+
- **GPU**: NVIDIA GPU (4GB+ VRAM)
- **CUDA**: 11.8+
- **RAM**: 8GB+
- **저장공간**: 10GB+

#### 권장 사양
- **GPU**: NVIDIA RTX 3060 이상 (8GB+ VRAM)
- **RAM**: 16GB+
- **저장공간**: 20GB+

### 📚 문서

- **사용 가이드**: [`Dev_md/02_가이드_Guide.md`](Dev_md/02_가이드_Guide.md)
- **개발 규칙**: [`Dev_md/01_규칙_Rules.md`](Dev_md/01_규칙_Rules.md)
- **개발 일지**: [`Dev_md/03_개발일지_DevLog_20251022.md`](Dev_md/03_개발일지_DevLog_20251022.md)
- **하이퍼파라미터 최적화**: [`Dev_md/06_하이퍼파라미터_최적화.md`](Dev_md/06_하이퍼파라미터_최적화.md)
- **최종 보고서**: [`Dev_md/07_최종보고서_Final_Report.md`](Dev_md/07_최종보고서_Final_Report.md)
- **향후 계획**: [`Dev_md/08_향후계획_Future_Roadmap.md`](Dev_md/08_향후계획_Future_Roadmap.md)

### 🛠️ 주요 명령어

```bash
# 학습
python scripts/train.py --data <dataset.yaml> --model n --epochs 150

# 추론
python scripts/inference.py --model <model.pt> --source <images/>

# TensorBoard 모니터링
tensorboard --logdir runs/segment

# 모델 검증
yolo segment val model=<model.pt> data=<dataset.yaml>
```

### 🔬 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **Deep Learning** | PyTorch 2.7.1, Ultralytics YOLOv11 |
| **Computer Vision** | OpenCV 4.12.0, rasterio 1.4.3 |
| **Scientific Computing** | NumPy 2.1.2, Matplotlib 3.10.7 |
| **Monitoring** | TensorBoard 2.20.0 |
| **GPU** | CUDA 11.8, cuDNN |

### 🚀 향후 계획

#### 단기 (1-2주)
- [ ] Advanced 모델 학습 (yolo11s-seg)
- [ ] API 서버 구축 (FastAPI)
- [ ] Docker 패키징

#### 중기 (1-2개월)
- [ ] NIR 밴드 활용 연구
- [ ] 웹 인터페이스 구축
- [ ] 모델 최적화 (ONNX/TensorRT)

#### 장기 (3개월+)
- [ ] 다중 작물 지원
- [ ] 시계열 분석
- [ ] 모바일 앱 개발

자세한 로드맵은 [`Dev_md/08_향후계획_Future_Roadmap.md`](Dev_md/08_향후계획_Future_Roadmap.md) 참조

### 🤝 기여

기여를 환영합니다! Pull Request를 보내주세요.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

### 👥 개발팀

- **AI Development**: Claude Sonnet 4.5
- **Project Management**: LX
- **Documentation**: AI Team

### 🙏 감사의 말

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11 프레임워크
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [rasterio](https://rasterio.readthedocs.io/) - 지리공간 이미지 처리

### 📞 연락처

프로젝트 관련 문의: [이슈 등록](https://github.com/YOUR_USERNAME/dbwjdakrso4235-sys/issues)

---

## English

### 📋 Project Overview

An AI system for automatic detection and segmentation of silage bales from drone/satellite imagery for automated inventory management and quantity assessment.

### ✨ Key Features

- **High-Precision Detection**: 92.2% mAP50 (exceeding 75-85% target)
- **Real-time Inference**: 11.3ms/image (~36 FPS)
- **Automatic Counting**: Automated object counting
- **4-band Image Support**: TIF (R,G,B,NIR) → RGB automatic conversion
- **Fully Automated**: End-to-end preprocessing → training → inference pipeline

### 🎯 Performance Metrics

| Metric | Value | Rating |
|--------|-------|--------|
| **mAP50 (Mask)** | 92.2% | ⭐⭐⭐ Excellent |
| **mAP50-95 (Mask)** | 85.3% | ⭐⭐⭐ Excellent |
| **Precision** | 96.5% | ⭐⭐⭐ Very High |
| **Recall** | 86.3% | ⭐⭐ Good |
| **Inference Speed** | 11.3ms | ⭐⭐⭐ Real-time |
| **Detection Rate** | 97% | ⭐⭐⭐ Excellent |

### 🚀 Quick Start

#### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/dbwjdakrso4235-sys.git
cd dbwjdakrso4235-sys

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python rasterio numpy matplotlib tensorboard tqdm pyyaml
```

#### 2. Data Preprocessing

```bash
# Convert 4-band TIF → RGB PNG
python scripts/preprocess_dataset.py \
    --input E:/namwon_ai/dataset_silage_bale \
    --output E:/namwon_ai/dataset_silage_bale_rgb \
    --format png
```

#### 3. Train Model

```bash
# Train with optimized settings
python scripts/train.py \
    --data E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml \
    --model n \
    --epochs 150 \
    --batch 8 \
    --imgsz 1024 \
    --lr0 0.001 \
    --optimizer AdamW \
    --patience 100 \
    --name silage_optimized

# Or use YAML config
yolo segment train data=configs/train_optimized.yaml
```

#### 4. Run Inference

```bash
# Single image inference
python scripts/inference.py \
    --model runs/segment/silage_optimized/weights/best.pt \
    --source path/to/image.jpg \
    --imgsz 1024 \
    --save

# Batch inference (folder)
python scripts/inference.py \
    --model runs/segment/silage_optimized/weights/best.pt \
    --source path/to/images/ \
    --imgsz 1024 \
    --save --save-txt --analyze
```

### 📚 Documentation

For detailed documentation, see [`Dev_md/`](Dev_md/) directory:
- Usage Guide (Korean)
- Development Log (Korean)
- Hyperparameter Optimization (Korean)
- Final Report (Korean)
- Future Roadmap (Korean)

### 💻 System Requirements

#### Minimum
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10+
- **GPU**: NVIDIA GPU (4GB+ VRAM)
- **CUDA**: 11.8+
- **RAM**: 8GB+

#### Recommended
- **GPU**: NVIDIA RTX 3060+ (8GB+ VRAM)
- **RAM**: 16GB+
- **Storage**: 20GB+

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 📞 Contact

For inquiries: [Create an issue](https://github.com/YOUR_USERNAME/dbwjdakrso4235-sys/issues)

---

**Made with ❤️ by AI Development Team**
