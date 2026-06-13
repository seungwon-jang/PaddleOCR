# 최소 재현 스크립트: eval dataloader 워커 spawn(fork) segfault 확인용
# train.py와 동일하게 forkserver 설정 → paddle GPU 초기화 → 실제 Eval config로
# dataloader 만들어 워커 spawn. 실제 학습은 안 함. eval 데이터 1배치만 당겨봄.
#
# 사용:
#   python3 repro_eval_fork.py <config.yml>            # cv2 스레드 그대로(기본 32)
#   CV2_SINGLE=1 python3 repro_eval_fork.py <config.yml>  # cv2.setNumThreads(0) 적용
from __future__ import absolute_import, division, print_function

import multiprocessing
multiprocessing.set_start_method("forkserver", force=True)  # train.py:21 과 동일

import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import cv2
print(f"[repro] cv2 {cv2.__version__}, 시작 시 getNumThreads()={cv2.getNumThreads()}", flush=True)
if os.environ.get("CV2_SINGLE") == "1":
    cv2.setNumThreads(0)
    print(f"[repro] cv2.setNumThreads(0) 적용 → getNumThreads()={cv2.getNumThreads()}", flush=True)

import logging
logger = logging.getLogger("repro")
logging.basicConfig(level=logging.INFO)

import paddle
from ppocr.data import build_dataloader
from tools.program import load_config


def main():
    cfg_path = sys.argv[1]
    config = load_config(cfg_path)

    # 학습 데이터 디렉토리/라벨은 실제 학습과 동일하게 (스케줄 데이터)
    data_dir = "/data/det_medi_schedule"
    val_txt = f"{data_dir}/val.txt"
    config["Eval"]["dataset"]["data_dir"] = data_dir
    config["Eval"]["dataset"]["label_file_list"] = [val_txt]

    eval_loader_cfg = config["Eval"]["loader"]
    if os.environ.get("SHM_OFF") == "1":
        eval_loader_cfg["use_shared_memory"] = False
        print("[repro] SHM_OFF=1 → Eval use_shared_memory=False 강제 적용", flush=True)
    print(f"[repro] Eval num_workers={eval_loader_cfg['num_workers']}, "
          f"use_shared_memory={eval_loader_cfg.get('use_shared_memory', True)}", flush=True)

    device = paddle.set_device("gpu")
    # 실제 학습처럼 GPU에 텐서 올려 CUDA 컨텍스트 초기화 (fork 전 멀티스레드 상태 재현)
    x = paddle.randn([1, 3, 640, 640])
    y = (x * 2).sum()
    paddle.device.cuda.synchronize()
    print(f"[repro] CUDA 초기화 완료 (dummy sum={float(y):.3f}), fork 직전 cv2 threads={cv2.getNumThreads()}", flush=True)

    import copy, gc, glob
    rounds = int(os.environ.get("ROUNDS", "60"))
    print(f"[repro] 반복 eval dataloader 생성/spawn/파괴 {rounds}회 (실제 학습의 매 eval 흉내)...", flush=True)
    pid = os.getpid()
    for r in range(rounds):
        paddle.device.cuda.empty_cache()
        eval_cfg = copy.deepcopy(config)
        loader = build_dataloader(eval_cfg, "Eval", device, logger)   # <-- 매 eval 워커 spawn
        n = 0
        for idx, batch in enumerate(loader):
            n += 1
            if n >= 3:
                break
        del loader
        gc.collect()
        fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        shm_count = len(glob.glob("/dev/shm/paddle_*"))
        if r % 5 == 0 or r == rounds - 1:
            print(f"[repro] round {r:3d}: OK ({n} batch) | open_fds={fd_count} shm_files={shm_count}", flush=True)
    print(f"[repro] ✅ 전체 {rounds}회 성공: segfault 안 남.", flush=True)


if __name__ == "__main__":
    main()
