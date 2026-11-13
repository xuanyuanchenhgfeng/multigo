# -*- coding: utf-8 -*-
"""仲裁参数敏感性：min_score × max_terms_per_protein 网格（测试集，12 库通道产物）。

正文表 7 与补充表 S11 的仲裁参数部分由本脚本产出。通道产物固定，只改仲裁参数，
评估口径与 revision_evaluate.py 完全一致（修订本体 + train-only IC + 101 点网格）。

用法:
    env -u LD_LIBRARY_PATH python3 -X utf8 scripts/revision_arbiter_sensitivity.py \
        [--out results/revision/arbiter_sensitivity.json]
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.arbiter import Arbiter  # noqa: E402
from src.config import load_config, setup_logging  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.go_dag import GODag  # noqa: E402

WORK = PROJECT / "work"
PRED_DIR = WORK / "revision" / "sensitivity"
RES_DIR = PROJECT / "results" / "revision"

MIN_SCORES = (0.10, 0.15, 0.20)
MAX_TERMS = (5, 10, 15, 30, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES_DIR / "arbiter_sensitivity.json"))
    args = ap.parse_args()

    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    log = setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only", "本脚本只用于修订版口径（train-only IC）"

    seq_tsv = str(WORK / "seq_channel_split3_test.tsv")
    struct_tsv = str(WORK / "struct_channel_split3_test.tsv")
    lit_tsv = str(WORK / "lit_channel_split3_test_dcut.tsv")
    truth = str(PROJECT / "data/annotations/split3_go.tsv")
    train = str(PROJECT / "data/annotations/split1_go.tsv")

    grid = {}
    for ms in MIN_SCORES:
        for mt in MAX_TERMS:
            c = json.loads(json.dumps(cfg))
            c["arbiter"]["min_score"] = ms
            c["arbiter"]["max_terms_per_protein"] = mt
            arb = Arbiter(c, dag)
            pred = str(PRED_DIR / f"pred_{ms}_{mt}.tsv")
            audit = str(PRED_DIR / f"audit_{ms}_{mt}.jsonl")
            arb.arbitrate(seq_tsv, struct_tsv, lit_tsv, pred, audit)
            rep = ev.evaluate(pred, truth, f"sens-{ms}-{mt}", train_tsv=train)
            grid[f"{ms}|{mt}"] = [round(rep["fmax"], 4), round(rep["smin"], 3),
                                  round(rep["aupr"], 4)]
            log.info("仲裁敏感性 [%s|%s] Fmax=%.4f Smin=%.3f AUPR=%.4f",
                     ms, mt, rep["fmax"], rep["smin"], rep["aupr"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(grid, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    log.info("仲裁参数敏感性 -> %s", args.out)


if __name__ == "__main__":
    main()
