#! /bin/bash

# make log file directory if not exists
mkdir -p logs

# log file name with timestamp
LOG_FILE="logs/sglang_pro-base_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=/_alias:${PYTHONPATH:-} \
  PORT=30010 \
  MODEL=/hf/DeepSeek-V4-Pro \
  SGLANG_FORCE_MXFP4_SERIALIZED=1 \
  bash /sgl-pr/launch_dsv4_pro_mxfp4.sh 2>&1 | tee $LOG_FILE