# -*- coding: utf-8 -*-
"""阶段2：苦荞蛋白组功能预测
用法: python scripts/run_tartary.py [--config config.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="阶段2：苦荞GO预测")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    log.info("阶段2：苦荞预测启动")

    pipe = Pipeline(cfg)
    pipe.run_stage2()
    log.info("阶段2完成。输出: %s", cfg["paths"]["results_dir"])


if __name__ == "__main__":
    main()
