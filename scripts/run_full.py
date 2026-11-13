# -*- coding: utf-8 -*-
"""无人值守全流程：阶段1（全量训练验证） -> 阶段2（苦荞预测）
用法: python scripts/run_full.py [--config config.yaml]
两阶段独立try/except：任一阶段失败不阻断另一阶段，最终输出汇总。
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="全流程：阶段1+阶段2（无人值守）")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])

    summary = {"stage1": None, "stage2": None}
    t0 = time.time()

    # ---------------- 阶段1：全量训练验证 ----------------
    log.info("=" * 70)
    log.info("阶段1（全量）：split2_val + split3_test + arabidopsis")
    log.info("=" * 70)
    pipe = None
    try:
        from src.pipeline import Pipeline
        pipe = Pipeline(cfg)
        pipe.run_stage1()
        summary["stage1"] = "OK"
    except Exception as e:  # noqa: BLE001
        summary["stage1"] = f"FAILED: {e}"
        log.error("阶段1异常（继续阶段2）:\n%s", traceback.format_exc())

    # ---------------- 阶段2：苦荞预测 ----------------
    log.info("=" * 70)
    log.info("阶段2（全量）：苦荞蛋白组 GO 预测")
    log.info("=" * 70)
    try:
        if pipe is None:
            from src.pipeline import Pipeline
            pipe = Pipeline(cfg)
        pipe.run_stage2()
        summary["stage2"] = "OK"
    except Exception as e:  # noqa: BLE001
        summary["stage2"] = f"FAILED: {e}"
        log.error("阶段2异常:\n%s", traceback.format_exc())

    # ---------------- 汇总 ----------------
    summary["elapsed_hours"] = round((time.time() - t0) / 3600, 2)
    out = results_dir / "full_run_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("全流程结束: %s -> %s", summary, out)


if __name__ == "__main__":
    main()
