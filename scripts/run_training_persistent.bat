@echo off
REM Windows 배치 파일 - 지속적인 학습 실행
REM 사용법: run_training_persistent.bat

echo ================================================================
echo YOLOv11 지속적 학습 스크립트 (Windows)
echo 터미널을 종료해도 학습이 계속 진행됩니다
echo ================================================================
echo.

REM 로그 디렉토리 생성
if not exist logs mkdir logs

REM 타임스탬프
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"

set "LOG_FILE=logs\train_%TIMESTAMP%.log"

echo 로그 파일: %LOG_FILE%
echo.

REM 백그라운드 실행 (start /B)
start /B python scripts\train.py ^
    --data "E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml" ^
    --model n ^
    --epochs 150 ^
    --batch 8 ^
    --imgsz 1024 ^
    --lr0 0.001 ^
    --optimizer AdamW ^
    --patience 100 ^
    --name silage_optimized ^
    > "%LOG_FILE%" 2>&1

echo ================================================================
echo ✅ 학습이 백그라운드에서 시작되었습니다!
echo ================================================================
echo.
echo 로그 파일: %LOG_FILE%
echo.
echo 명령어:
echo   - 진행 상황 확인: type "%LOG_FILE%"
echo   - 실시간 확인 (PowerShell): Get-Content "%LOG_FILE%" -Wait -Tail 10
echo   - 프로세스 확인: tasklist ^| findstr python
echo   - 중단: taskkill /F /IM python.exe
echo.
echo ================================================================
pause
