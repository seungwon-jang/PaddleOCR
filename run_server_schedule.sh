#!/usr/bin/env bash
#
# PP-OCRv5 Server Detection — 데이터 스케줄 순차 학습 (2000 → 10000)
#
# 각 stage는 공식 server pretrained에서 독립적으로 시작한다.
# (총 step 60,000 고정 + 데이터 양만 증가 → "고정 compute, 데이터 양 변화" 실험)
# stage를 이전 체크포인트에서 이어 학습하려면 아래 CONTINUATION 주석 참고.
#
# 사용법:
#   cd /home/kist/watt/PaddleOCR
#   bash run_server_schedule.sh
#   nohup bash run_server_schedule.sh > schedule_all.log 2>&1 &   # 백그라운드(권장: tmux/ setsid)

set -uo pipefail

# ── 스크립트 위치 기준으로 PaddleOCR 디렉토리로 이동 ──────────────────────────
cd "$(dirname "$0")"

# ── 사용자 환경 설정 ─────────────────────────────────────────────────────────
GPU=4                                                          # CUDA_VISIBLE_DEVICES
CPU_AFFINITY="17-63:2"                                         # taskset -c
DATA_DIR="/home/kist/watt/watt_data_tmp/det_medi_schedule"    # train_N.txt / val.txt 가 있는 곳
VAL_TXT="${DATA_DIR}/val.txt"                                  # eval 라벨 (전체 val)
PRETRAINED="/home/kist/watt/watt_data_tmp/paddle_pretrained_data/PP-OCRv5_server_det_pretrained"  # 확장자(.pdparams) 제외
OUT_BASE="./checkpoints"                                       # 체크포인트 + 로그 저장 베이스
SCHEDULE=(2000 4000 6000 8000 10000)                          # 실행 순서

CONFIG_DIR="configs/det/PP-OCRv5/det_server"

# ── 사전 점검 ────────────────────────────────────────────────────────────────
if [[ ! -f "${PRETRAINED}.pdparams" ]]; then
    echo "[오류] pretrained 파일 없음: ${PRETRAINED}.pdparams"
    echo "       다운로드: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams"
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[오류] 데이터 디렉토리 없음: ${DATA_DIR}"
    exit 1
fi

# ── 스케줄 순차 실행 ─────────────────────────────────────────────────────────
for N in "${SCHEDULE[@]}"; do
    CONFIG="${CONFIG_DIR}/PP-OCRv5_server_det_schedule_${N}.yml"
    TRAIN_TXT="${DATA_DIR}/train_${N}.txt"
    SAVE_DIR="${OUT_BASE}/medi_server_${N}"
    LOG="${SAVE_DIR}/train.log"

    echo "============================================================"
    echo " STAGE ${N}  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   config : ${CONFIG}"
    echo "   train  : ${TRAIN_TXT}"
    echo "   save   : ${SAVE_DIR}"
    echo "============================================================"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "[오류] config 없음: ${CONFIG} — 중단"; exit 1
    fi
    if [[ ! -f "${TRAIN_TXT}" ]]; then
        echo "[오류] train 라벨 없음: ${TRAIN_TXT} — 중단"; exit 1
    fi

    mkdir -p "${SAVE_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    FLAGS_allocator_strategy=naive_best_fit FLAGS_fraction_of_gpu_memory_to_use=0.85 \
    taskset -c "${CPU_AFFINITY}" python3 tools/train.py \
        -c "${CONFIG}" \
        -o \
        Global.save_model_dir="${SAVE_DIR}" \
        Global.pretrained_model="${PRETRAINED}" \
        Train.dataset.data_dir="${DATA_DIR}" \
        Train.dataset.label_file_list="[${TRAIN_TXT}]" \
        Eval.dataset.data_dir="${DATA_DIR}" \
        Eval.dataset.label_file_list="[${VAL_TXT}]" \
        2>&1 | tee -a "${LOG}"

    # tee 뒤 파이프라인 종료코드 = python3 의 코드 (pipefail)
    STATUS=${PIPESTATUS[0]}
    if [[ "${STATUS}" -ne 0 ]]; then
        echo "[중단] STAGE ${N} 실패 (exit ${STATUS}) — 이후 스케줄 실행 안 함"
        exit "${STATUS}"
    fi
    echo "[완료] STAGE ${N}  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

# ── CONTINUATION 모드로 바꾸려면 ──────────────────────────────────────────────
#   각 stage를 이전 stage 결과에서 이어 학습하고 싶으면:
#   위 -o 에서 Global.pretrained_model 줄을 빼고, 대신 (2000 stage 제외) 직전
#   stage의 best_accuracy 를 넘긴다:
#     Global.checkpoints="${OUT_BASE}/medi_server_<직전N>/best_accuracy"
done

echo "============================================================"
echo " 전체 스케줄 완료: ${SCHEDULE[*]}"
echo "============================================================"
