#!/bin/bash
# 지속적인 학습 실행 스크립트 (터미널 종료 후에도 계속 실행)
#
# 사용법:
#   bash scripts/run_training_persistent.sh

echo "================================================================"
echo "YOLOv11 지속적 학습 스크립트"
echo "터미널을 종료해도 학습이 계속 진행됩니다"
echo "================================================================"
echo

# 로그 파일 경로
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "로그 파일: $LOG_FILE"
echo "진행 상황 확인: tail -f $LOG_FILE"
echo

# nohup으로 백그라운드 실행 (터미널 종료와 무관)
nohup python scripts/train.py \
    --data "E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml" \
    --model n \
    --epochs 150 \
    --batch 8 \
    --imgsz 1024 \
    --lr0 0.001 \
    --optimizer AdamW \
    --patience 100 \
    --name silage_optimized \
    > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
PID=$!
echo $PID > "$LOG_DIR/train.pid"

echo "================================================================"
echo "✅ 학습이 백그라운드에서 시작되었습니다!"
echo "================================================================"
echo
echo "프로세스 ID: $PID"
echo "로그 파일: $LOG_FILE"
echo
echo "명령어:"
echo "  - 진행 상황 확인: tail -f $LOG_FILE"
echo "  - 프로세스 확인: ps aux | grep $PID"
echo "  - 중단: kill $PID"
echo "  - 또는: pkill -f train.py"
echo
echo "================================================================"
