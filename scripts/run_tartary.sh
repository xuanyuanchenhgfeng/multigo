#!/usr/bin/env bash
# 阶段2：苦荞预测（完整pipeline）
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_tartary.py
