# VSCode 작업 재개 가이드

**작성일**: 2025-10-23
**목적**: Visual Studio Code에서 작업을 빠르게 재개하는 방법

---

## 🎯 빠른 시작 (5분)

### 1. VSCode 열기
```
폴더 열기: C:\Users\LX\dbwjdakrso4235-sys
```

### 2. 터미널 열기 (단축키: Ctrl+`)

### 3. 현재 상태 확인

#### 옵션 A: 학습이 실행 중인지 확인
```bash
# 프로세스 확인
ps aux | grep train.py

# 또는
pgrep -f train.py
```

**결과 해석**:
- **숫자가 나옴** (예: 12345) → 학습 실행 중 ✅
- **아무것도 안 나옴** → 학습 중단됨 ❌

#### 옵션 B: 학습 결과 확인
```bash
# 최신 학습 결과 확인
ls -lt runs/segment/
```

가장 위에 있는 폴더가 최신 실험입니다.

---

## 📂 주요 파일 위치

### 작업 재개 시 확인할 파일들

#### 1️⃣ **개발 일지** (무엇을 했는지 확인)
```
📁 Dev_md/
├── 03_개발일지_1일차_20251022.md    ← 1일차 작업 내용
├── 04_개발일지_2일차_20251023.md    ← 2일차 작업 내용 (오늘)
└── claude.md                         ← Sonnet/Opus 작업 분담
```

**확인 방법**:
1. VSCode 왼쪽 Explorer에서 `Dev_md/` 폴더 찾기
2. `04_개발일지_2일차_20251023.md` 열기
3. 마지막 "향후 작업" 섹션 확인

#### 2️⃣ **학습 결과** (어디까지 학습했는지 확인)
```
📁 runs/segment/silage_optimized/
├── weights/
│   ├── best.pt       ← 최고 성능 모델
│   └── last.pt       ← 마지막 epoch 모델
├── results.csv       ← epoch별 메트릭 (Excel로 열기)
├── results.png       ← 학습 곡선 그래프
└── args.yaml         ← 학습 설정
```

**확인 방법**:
1. VSCode에서 `runs/segment/silage_optimized/results.csv` 열기
2. 마지막 줄 확인 → 몇 epoch까지 학습했는지 확인

#### 3️⃣ **추론 시스템** (추론 작업 확인)
```
📁 inference_system/
├── src/
│   ├── crop_processor.py      ← SHP 기반 크롭
│   ├── inference_engine.py    ← YOLO 추론
│   └── pipeline.py            ← 통합 파이프라인
├── examples/
│   └── test_full_pipeline.py  ← 테스트 스크립트
└── output/
    ├── test_valid/             ← 크롭 테스트 결과
    └── full_pipeline_test/     ← 파이프라인 테스트 결과
```

**확인 방법**:
1. `inference_system/output/full_pipeline_test/reports/summary.txt` 열기
2. 테스트 결과 확인

#### 4️⃣ **로그 파일** (실시간 진행 상황)
```
📁 logs/
└── train_YYYYMMDD_HHMMSS.log  ← 학습 로그
```

**확인 방법** (터미널에서):
```bash
# 로그 파일 찾기
ls -lt logs/

# 실시간 확인
tail -f logs/train_*.log
```

---

## 🔄 작업 재개 시나리오

### 시나리오 1: 학습이 완료되었을 때 ✅

#### 확인
```bash
# 학습 프로세스 확인
ps aux | grep train.py
# → 아무것도 안 나옴 (완료됨)

# 결과 확인
cat runs/segment/silage_optimized/results.csv | tail -5
```

#### 다음 작업
1. **최종 성능 확인**
   ```bash
   python scripts/inference.py \
       --model runs/segment/silage_optimized/weights/best.pt \
       --source E:/namwon_ai/dataset_silage_bale_rgb/images/test \
       --save --analyze
   ```

2. **보고서 업데이트**
   - `Dev_md/07_최종보고서_Final_Report.md` 열기
   - 최종 결과 작성

3. **GitHub 커밋**
   ```bash
   git add runs/segment/silage_optimized/
   git commit -m "chore: Add training results (150 epochs)"
   git push origin main
   ```

### 시나리오 2: 학습이 진행 중일 때 🔄

#### 확인
```bash
# 학습 프로세스 확인
ps aux | grep train.py
# → 12345 python scripts/train.py ... (실행 중)

# 실시간 로그 확인
tail -f logs/train_*.log
```

#### 다음 작업
1. **TensorBoard로 모니터링**
   ```bash
   # 새 터미널 열기 (Ctrl+Shift+`)
   tensorboard --logdir runs/segment

   # 브라우저에서 열기: http://localhost:6006
   ```

2. **GPU 사용량 확인**
   ```bash
   # 1초마다 업데이트
   watch -n 1 nvidia-smi
   ```

3. **다른 작업 진행**
   - 문서 작성
   - 추론 시스템 테스트
   - 다음 실험 계획

### 시나리오 3: 학습이 중단되었을 때 ❌

#### 확인
```bash
# 학습 프로세스 확인
ps aux | grep train.py
# → 아무것도 안 나옴

# 마지막 체크포인트 확인
ls -lh runs/segment/silage_optimized/weights/
```

#### 다음 작업
1. **재개할지 결정**
   ```bash
   # results.csv에서 마지막 epoch 확인
   tail -5 runs/segment/silage_optimized/results.csv

   # 예: 50/150 epoch까지 완료 → 재개 고려
   # 예: 149/150 epoch까지 완료 → 재개 불필요
   ```

2. **학습 재개** (원한다면)
   ```bash
   python scripts/train.py \
       --data "E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml" \
       --model runs/segment/silage_optimized/weights/last.pt \
       --epochs 150 \
       --resume
   ```

3. **또는 새로 시작**
   ```bash
   bash scripts/run_training_persistent.sh
   ```

### 시나리오 4: 추론 시스템 테스트 🧪

#### 확인
```bash
# 추론 결과 확인
ls -lh inference_system/output/full_pipeline_test/
```

#### 다음 작업
1. **대규모 테스트** (275개 폴리곤)
   ```bash
   cd inference_system

   # 전체 교차 폴리곤 처리
   python src/pipeline.py \
       --tif F:/namwon_ai/input_tif/금지면_1차.tif \
       --shp F:/namwon_ai/saryo_jeongbo/saryo_parcel.shp \
       --model ../runs/segment/silage_optimized/weights/best.pt \
       --output output/large_scale_test
   ```

2. **결과 분석**
   ```bash
   # 통계 확인
   cat inference_system/output/large_scale_test/reports/summary.txt

   # 시각화 확인
   explorer.exe inference_system/output/large_scale_test/visualizations/
   ```

3. **GeoPackage 열기** (QGIS 등)
   ```
   파일: inference_system/output/large_scale_test/silage_bale_detections.gpkg
   ```

---

## 🛠️ VSCode 유용한 단축키

### 터미널
| 단축키 | 기능 |
|--------|------|
| `Ctrl+` ` | 터미널 열기/닫기 |
| `Ctrl+Shift+` ` | 새 터미널 |
| `Ctrl+Shift+5` | 터미널 분할 |
| `Alt+↑/↓` | 터미널 전환 |

### 파일 탐색
| 단축키 | 기능 |
|--------|------|
| `Ctrl+P` | 파일 빠른 열기 |
| `Ctrl+Shift+E` | Explorer 포커스 |
| `Ctrl+B` | 사이드바 토글 |
| `Ctrl+\` | 에디터 분할 |

### 검색
| 단축키 | 기능 |
|--------|------|
| `Ctrl+F` | 파일 내 검색 |
| `Ctrl+Shift+F` | 프로젝트 전체 검색 |
| `Ctrl+H` | 찾기 및 바꾸기 |
| `F3` / `Shift+F3` | 다음/이전 결과 |

### Git
| 단축키 | 기능 |
|--------|------|
| `Ctrl+Shift+G` | Git 패널 열기 |
| `Ctrl+Enter` | 커밋 |

---

## 📌 빠른 체크리스트

VSCode 열면 이 순서대로 확인하세요:

### ✅ 5분 체크리스트

- [ ] **1. 터미널 열기** (`Ctrl+` `)
- [ ] **2. 학습 상태 확인**
  ```bash
  ps aux | grep train.py
  ```
- [ ] **3. 마지막 개발일지 확인**
  ```
  Dev_md/04_개발일지_2일차_20251023.md
  ```
- [ ] **4. 향후 작업 확인**
  - 개발일지 → "향후 작업 계획 (Opus)" 섹션
- [ ] **5. 상황별 다음 작업 진행**
  - 학습 완료 → 추론 테스트
  - 학습 진행 중 → TensorBoard 모니터링
  - 학습 중단 → 재개 또는 분석

---

## 🎯 추천 워크플로우

### 아침에 출근했을 때

```bash
# 1. 프로젝트 폴더로 이동
cd C:\Users\LX\dbwjdakrso4235-sys

# 2. VSCode 열기
code .

# 3. 터미널에서 상태 확인
ps aux | grep train.py           # 학습 진행 여부
git status                        # 변경 사항 확인
git pull origin main              # 최신 코드 받기

# 4. 로그 확인 (학습 중이라면)
tail -f logs/train_*.log

# 5. TensorBoard 실행 (선택)
tensorboard --logdir runs/segment
```

### 작업 종료 전

```bash
# 1. 변경사항 저장
git status
git add .
git commit -m "chore: Update work in progress"

# 2. 학습 상태 확인
ps aux | grep train.py

# 3. 로그 마지막 확인
tail -20 logs/train_*.log

# 4. 백업 (선택)
git push origin main
```

---

## 📁 VSCode 추천 확장 프로그램

### 필수
- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **GitLens** (Git 히스토리)

### 추천
- **Markdown All in One** (문서 작성)
- **YAML** (설정 파일)
- **Remote - SSH** (원격 작업)

### 유용
- **Material Icon Theme** (파일 아이콘)
- **Todo Tree** (TODO 관리)
- **Error Lens** (에러 표시)

---

## 🔍 빠른 참조: 파일별 역할

```
dbwjdakrso4235-sys/
│
├── 📝 개발 문서
│   ├── Dev_md/03_개발일지_1일차_20251022.md  ← 1일차 작업
│   ├── Dev_md/04_개발일지_2일차_20251023.md  ← 2일차 작업 (오늘)
│   ├── Dev_md/07_최종보고서_Final_Report.md  ← 최종 보고서
│   └── Dev_md/claude.md                       ← 작업 분담
│
├── 🎯 학습 관련
│   ├── scripts/train.py                       ← 학습 스크립트
│   ├── scripts/run_training_persistent.sh    ← 지속 학습 스크립트
│   ├── configs/train_optimized.yaml          ← 학습 설정
│   └── runs/segment/silage_optimized/        ← 학습 결과
│
├── 🔬 추론 관련
│   ├── scripts/inference.py                   ← 추론 스크립트
│   ├── inference_system/src/pipeline.py      ← 추론 파이프라인
│   └── inference_system/output/              ← 추론 결과
│
└── 📊 로그
    └── logs/train_*.log                       ← 학습 로그
```

---

## 💡 프로 팁

### 1. 빠른 파일 열기
```
Ctrl+P → 파일명 입력
예: "dev" → 04_개발일지_2일차_20251023.md
```

### 2. 터미널 명령어 히스토리
```
↑/↓ 화살표 키로 이전 명령어 재실행
```

### 3. 여러 터미널 동시 사용
```
Ctrl+Shift+` → 새 터미널
터미널 1: 학습 실행
터미널 2: 로그 모니터링 (tail -f)
터미널 3: Git 작업
```

### 4. 분할 화면
```
Ctrl+\ → 에디터 분할
왼쪽: 코드
오른쪽: 로그 또는 문서
```

### 5. Git 시각화
```
Ctrl+Shift+G → Git 패널
변경 사항 한눈에 확인
```

---

## 🆘 문제 해결

### Q: 터미널이 Python을 못 찾아요
**A**: VSCode 설정 확인
```
Ctrl+, → "python.defaultInterpreterPath" 검색
→ Python 경로 설정 (예: C:\Users\LX\AppData\Local\Programs\Python\Python310\python.exe)
```

### Q: Git이 안 돼요
**A**: Git 설치 확인
```bash
git --version
# 없다면: https://git-scm.com/download/win
```

### Q: 한글이 깨져요
**A**: 파일 인코딩 확인
```
VSCode 하단 바 → "UTF-8" 확인
깨지면 → "Save with Encoding" → UTF-8 선택
```

### Q: 파일이 너무 많아서 느려요
**A**: .gitignore 확인
```bash
# 제외할 폴더
echo "runs/" >> .gitignore
echo "logs/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

---

**작성자**: Claude Sonnet 4.5
**최종 수정**: 2025-10-23

**다음에 VSCode를 열면 이 파일을 먼저 확인하세요!** 📖
