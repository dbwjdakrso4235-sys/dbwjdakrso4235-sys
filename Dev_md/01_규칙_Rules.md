# 프로젝트 개발 규칙 (Development Rules)

**작성일**: 2025-10-22
**프로젝트**: YOLOv11 Segmentation - Silage Bale Detection

---

## 1. 코드 작성 규칙

### 1.1 Python 코딩 스타일
- PEP 8 스타일 가이드 준수
- 함수 및 변수명: snake_case
- 클래스명: PascalCase
- 상수명: UPPER_CASE
- 들여쓰기: 4 spaces

### 1.2 주석 규칙
```python
# 한 줄 주석은 명확하고 간결하게

def function_name(param1, param2):
    """
    함수 설명

    Args:
        param1 (type): 설명
        param2 (type): 설명

    Returns:
        type: 설명
    """
    pass
```

### 1.3 파일 구조
```
project/
├── data/           # 데이터 관련
├── models/         # 모델 정의
├── utils/          # 유틸리티 함수
├── configs/        # 설정 파일
├── scripts/        # 학습/추론 스크립트
└── Dev_md/         # 문서화
```

---

## 2. 데이터 처리 규칙

### 2.1 이미지 전처리
- **4-band to 3-band RGB 변환 필수**
  - NIR 밴드 제거
  - RGB 3채널만 사용
  - 정규화: [0, 255] → [0, 1]

### 2.2 데이터셋 구조
- Train/Val/Test 분할 유지
- 원본 데이터 변경 금지
- 전처리된 데이터는 별도 저장

### 2.3 라벨 포맷
- YOLO segmentation format
- Normalized coordinates [0, 1]
- Class ID는 0-based indexing

---

## 3. 모델 학습 규칙

### 3.1 체크포인트 관리
- 매 epoch마다 자동 저장
- Best model 별도 보관
- 모델 파일명: `yolov11_seg_{date}_{metric}.pt`

### 3.2 로깅
- TensorBoard 사용
- 학습 메트릭 기록
  - Loss (box, seg, cls)
  - mAP50, mAP50-95
  - Precision, Recall

### 3.3 재현성
- Random seed 고정
- 하이퍼파라미터 저장
- 환경 설정 기록 (requirements.txt)

---

## 4. 버전 관리 규칙

### 4.1 Git Commit 메시지
```
[Type] 제목

- 상세 내용 1
- 상세 내용 2

Type:
- feat: 새로운 기능
- fix: 버그 수정
- docs: 문서 수정
- refactor: 코드 리팩토링
- test: 테스트 코드
- chore: 빌드/설정 변경
```

### 4.2 브랜치 전략
- `main`: 안정 버전
- `dev`: 개발 버전
- `feature/*`: 기능 개발
- `fix/*`: 버그 수정

---

## 5. 문서화 규칙

### 5.1 Dev_md 폴더 구조
- `01_규칙_Rules.md`: 개발 규칙
- `02_가이드_Guide.md`: 사용 가이드
- `03_개발일지_DevLog_YYYYMMDD.md`: 일자별 개발 로그
- `04_보고서_Report_YYYYMMDD.md`: 결과 보고서
- `05_명령어_Commands.md`: 자주 사용하는 명령어
- `claude.md`: AI 모델 역할 분담

### 5.2 문서 작성 원칙
- 날짜 표기: YYYY-MM-DD 또는 YYYYMMDD
- 명확하고 구체적인 작성
- 코드 예시 포함
- 결과 이미지/표 첨부

---

## 6. 코드 리뷰 체크리스트

- [ ] 코드 스타일 준수
- [ ] 주석 및 docstring 작성
- [ ] 에러 핸들링 구현
- [ ] 테스트 코드 작성
- [ ] 문서 업데이트
- [ ] 성능 최적화 고려

---

## 7. 보안 및 데이터 관리

### 7.1 민감 정보 관리
- API 키는 `.env` 파일 사용
- `.gitignore`에 민감 파일 추가
- 절대 경로 사용 지양

### 7.2 데이터 백업
- 주요 결과물 정기 백업
- 모델 체크포인트 버전 관리
- 실험 결과 기록 보관

---

**최종 수정**: 2025-10-22
**작성자**: AI Development Team
