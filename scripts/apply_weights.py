# -*- coding: utf-8 -*-
"""分本体最优权重固定应用于测试集（协议：搜索仅用val，test一次性应用）。"""
import json
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import src.pipeline  # noqa
from src.arbiter import Arbiter
from src.evaluator import Evaluator
from src.go_dag import GODag
import yaml

ASPECTS = {"F": (0.15, 0.55, 0.30), "P": (0.15, 0.70, 0.15),
           "C": (0.20, 0.30, 0.50)}

cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
dag = GODag(cfg["paths"]["go_obo"],
            relations=cfg.get("go_dag", {}).get("relations"))
ev = Evaluator(cfg, dag)
arb = Arbiter(cfg, dag)
tag = "split3_test"

merged = {}  # protein -> {go: score}
for code, ws in ASPECTS.items():
    tmpd = tempfile.mkdtemp(prefix=f"wapply_{code}_")
    for src, dst in [("seq", "seq"), ("struct", "struct"), ("lit", "lit")]:
        src_p = BASE / "work" / f"{src}_channel_{tag}.tsv"
        (Path(tmpd) / f"{dst}.tsv").write_text(
            src_p.read_text(encoding="utf-8"), encoding="utf-8")
    arb.weights = {"seq": ws[0], "struct": ws[1], "lit": ws[2]}
    arb.arbitrate(f"{tmpd}/seq.tsv", f"{tmpd}/struct.tsv",
                  f"{tmpd}/lit.tsv", f"{tmpd}/pred.tsv",
                  f"{tmpd}/audit.jsonl")
    for ln in (Path(tmpd) / "pred.tsv").read_text(
            encoding="utf-8").splitlines()[1:]:
        parts = ln.split("\t")
        pid, g, s = parts[0], parts[1], float(parts[3])
        if parts[2] == code:
            merged.setdefault(pid, {})
            merged[pid][g] = max(s, merged[pid].get(g, 0.0))

out = BASE / "work" / "predictions_split3_test_peront_weights.tsv"
with open(out, "w", encoding="utf-8") as f:
    f.write("protein_id\tgo_id\tscore\n")
    for pid, d in merged.items():
        for g, s in d.items():
            f.write(f"{pid}\t{g}\t{s:.4f}\n")

r = ev.evaluate(str(out), str(BASE / "data/annotations/split3_go.tsv"),
                "peront-weights", train_tsv=str(
                    BASE / "data/annotations/split1_go.tsv"),
                out_json=str(BASE / "results/eval_peront_weights.json"))
print("分本体最优权重(test整体): Fmax=%.4f Smin=%.4f AUPR=%.4f AUC=%.4f"
      % (r["fmax"], r["smin"], r["aupr"], r["auc"]))
for code in ("F", "P", "C"):
    a = r["by_aspect"][code]
    print(a["ontology"], "Fmax=%.4f AUPR=%.4f" % (a["fmax"], a["aupr"]))
