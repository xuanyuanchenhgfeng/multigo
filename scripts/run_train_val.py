# -*- coding: utf-8 -*-
"""阶段1：PSB26 时间基训练验证（Swiss-Prot）
用法: python scripts/run_train_val.py [--skip-prep] [--demo] [--only split3_test]
目标集: split2_val(验证) / split3_test(测试) / arabidopsis(近缘跨物种)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="阶段1：PSB26时间基训练验证")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-prep", action="store_true",
                    help="跳过数据划分/建库（已生成时使用）")
    ap.add_argument("--demo", action="store_true", help="强制演示模式（小样本）")
    ap.add_argument("--only", default="",
                    help="仅运行指定目标集（逗号分隔: split2_val,split3_test,arabidopsis）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.demo:
        cfg.setdefault("runtime", {})["demo_sample_size"] = 200
    log = setup_logging(cfg)
    log.info("阶段1：训练验证启动 (config=%s)", args.config)

    if not args.skip_prep:
        from src.data_prep import run_data_prep
        stats = run_data_prep(cfg)
        log.info("数据准备完成: %s", stats)

    only = [t.strip() for t in args.only.split(",") if t.strip()] or None
    pipe = Pipeline(cfg)
    pipe.run_stage1(only=only)
    log.info("阶段1完成。结果见 %s", cfg["paths"]["results_dir"])


if __name__ == "__main__":
    main()
