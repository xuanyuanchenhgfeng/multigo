#!/bin/bash
# 本机 InterProScan local 模式全量重跑 Struct-Channel（阶段1）
# 用 nohup 启动，SSH 断开不影响。java 必须在 PATH（interproscan.sh 用 `type -p java`）。
export JAVA_HOME=/home/zsl/tools/jdk-11.0.2
export PATH=$JAVA_HOME/bin:/home/zsl/tools/mmseqs/bin:$PATH
export PYTHONUNBUFFERED=1

cd /home/zsl/project
LOG=logs/local_ips_run.log
echo "=== Local InterProScan 全量重跑 start at $(date) ===" >> "$LOG"
python3 scripts/run_train_val.py --skip-prep >> "$LOG" 2>&1
RC=$?
echo "=== Local InterProScan 全量重跑 end at $(date), exit code=$RC ===" >> "$LOG"
