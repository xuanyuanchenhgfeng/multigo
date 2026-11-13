#!/usr/bin/env bash
# 阶段1：训练验证（Swiss-Prot时间切割）
# Linux/WSL/Git-Bash 通用；依赖: Python3.9+ 及 config.yaml 配置的外部工具
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_PREP=""
if [ "${1:-}" = "--skip-prep" ]; then SKIP_PREP="--skip-prep"; fi

python scripts/run_train_val.py ${SKIP_PREP}
