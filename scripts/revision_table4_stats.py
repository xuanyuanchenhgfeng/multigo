# -*- coding: utf-8 -*-
"""表 4（通道命中率/输出覆盖/仲裁统计）口径的确定性重算。

输入全部来自在档通道产物与仲裁审计，不依赖运行日志：
  work/{seq,struct}_channel_split3_test.tsv、work/lit_channel_split3_test_dcut.tsv、
  work/meta_split3_test.tsv、work/revision/preds/{predictions,audit}_split3_test_*.
文献通道检索式回退级别另有 scripts/lit_query_fallback_stats.py（表 4 末三行）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_truth  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
PRED = WORK / "revision" / "preds"
RES = PROJECT / "results" / "revision"


def prots(tsv):
    return {r["protein_id"] for r in read_tsv(str(tsv))}


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"], relations=["is_a", "part_of"])
    ev = Evaluator(cfg, dag)
    truth = load_truth(str(PROJECT / "data/annotations/split3_go.tsv"),
                       exclude_iea=ev.exclude_iea, dag=dag)
    tp = {p: ev._prop(g) for p, g in truth.items()}
    targets = [p for p in tp if tp[p]]

    seq = prots(WORK / "seq_channel_split3_test.tsv")
    struct_rows = list(read_tsv(str(WORK / "struct_channel_split3_test.tsv")))
    struct = {r["protein_id"] for r in struct_rows}
    lit = prots(WORK / "lit_channel_split3_test_dcut.tsv")
    lit_pre = prots(WORK / "lit_channel_split3_test.tsv")
    queries = prots(WORK / "meta_split3_test.tsv")
    nq = len(queries) or 1159

    any_ch = seq | struct | lit
    out = {
        "n_queries": nq, "n_targets": len(targets),
        "seq_hit_prots": len(seq), "struct_hit_prots": len(struct),
        "struct_rows": len(struct_rows),
        "lit_hit_prots": len(lit), "lit_hit_prots_pretrunc": len(lit_pre),
        "lit_rows": len(list(read_tsv(
            str(WORK / "lit_channel_split3_test_dcut.tsv")))),
        "at_least_one": len(any_ch), "no_evidence": nq - len(any_ch),
        "seq_no_hit_among_targets": len([p for p in targets if p not in seq]),
    }
    for combo in ("multi", "seq-only", "seq+struct", "seq+lit"):
        pd = PRED / f"predictions_split3_test_{combo}.tsv"
        got = prots(pd)
        out[f"pred_prots_{combo}"] = len(got)
        out[f"pred_prots_{combo}_in_targets"] = len(got & set(targets))
    audit = {}
    for ln in (PRED / "audit_split3_test_multi.jsonl").read_text(
            encoding="utf-8").splitlines():
        d = json.loads(ln)
        audit["n"] = audit.get("n", 0) + 1
        for k in ("unknown", "same_merges", "parent_child_collapses",
                  "dag_validation_removed"):
            v = d.get(k)
            if k == "unknown":
                audit["unknown"] = audit.get("unknown", 0) + bool(v)
            elif isinstance(v, dict):
                audit[k] = audit.get(k, 0) + len(v)
            elif isinstance(v, list):
                audit[k] = audit.get(k, 0) + len(v)
        audit["branch_conflicts"] = audit.get(
            "branch_conflicts", 0) + len(d.get("branch_conflicts") or [])
    out["audit"] = audit
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "table4_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
