#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Detection 스케줄 학습 큐
#   1) 현재 돌고 있는 2,000장 학습(_schdule.yml)이 끝날 때까지 대기
#   2) 아래 CONFIGS 를 순서대로 실행 (앞 단계 실패해도 다음 단계 계속 진행)
#
# 실행 (터미널 끊겨도 유지):
#   cd /workspace/ocr_LLM/PaddleOCR
#   nohup bash train_queue.sh > ./output/queue_master.log 2>&1 &
#   tail -f ./output/queue_master.log      # 진행 확인
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd /workspace/ocr_LLM/PaddleOCR
mkdir -p output

CONDA_ENV=paddle_train

# 큐에 넣을 config들 (실행 순서대로). 8000/10000 만들면 주석 해제.
CONFIGS=(
  configs/det/PP-OCRv5/PP-OCRv5_mobile_det_schedule_4000.yml
  configs/det/PP-OCRv5/PP-OCRv5_mobile_det_schedule_6000.yml
  # configs/det/PP-OCRv5/PP-OCRv5_mobile_det_schedule_8000.yml
  # configs/det/PP-OCRv5/PP-OCRv5_mobile_det_schedule_10000.yml
)

# ── 1) 현재 2,000장 학습 종료 대기 ───────────────────────────────────────────
# 현재 실행 중인 학습은 _schdule.yml (철자 그대로) 을 사용 중.
echo "[$(date '+%F %T')] 현재 학습(_schdule.yml) 종료 대기 중..."
while pgrep -f "tools/train.py.*_schdule\.yml" >/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] 현재 학습 종료 확인 → 큐 시작"

# ── 2) 큐 순차 실행 ──────────────────────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "[$(date '+%F %T')] [건너뜀] config 없음: $cfg"
        continue
    fi
    tag=$(basename "$cfg" .yml)
    log="./output/${tag}_train.log"
    echo "[$(date '+%F %T')] >>> START  $tag   (log: $log)"
    conda run -n "$CONDA_ENV" --no-capture-output python3 tools/train.py -c "$cfg" \
        > "$log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] <<< DONE   $tag   (exit code: $rc)"
done

echo "[$(date '+%F %T')] ✅ 큐 전체 완료"
