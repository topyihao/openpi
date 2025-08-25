#!/bin/bash
# 安全运行训练脚本，避免 systemd-oomd 杀进程
# 用法: ./run_train_safe.sh

systemd-run --user --scope --same-dir \
  -p MemoryMax=infinity \
  -p MemoryHigh=infinity \
  -p MemorySwapMax=infinity \
  -p ManagedOOMPreference=avoid \
  bash -lc '
mkdir -p ~/Disk2/Yihao/tmp_ckpt ~/Disk2/Yihao/logs

# 设置环境变量
export TMPDIR=~/Disk2/Yihao/tmp_ckpt
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# 文件句柄数上限
ulimit -n 65535

# 执行训练命令
stdbuf -oL -eL uv run scripts/train.py pi0_fast_aloha_clean_dish \
  --exp-name=yihao_pi0_fast_aloha_clean_dish --overwrite \
  2>&1 | tee -a ~/Disk2/Yihao/logs/train_$(date +%F_%H-%M).log
'