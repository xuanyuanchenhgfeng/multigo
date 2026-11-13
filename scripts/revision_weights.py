# -*- coding: utf-8 -*-
"""修订版：分本体权重 val 网格搜索 + test 一次性应用。

修订口径：2023-10 本体、IC 仅训练库、文献=日期截断版。
搜索只在 split2_val 上进行（步长 0.05 单纯形网格 231 点），
各本体最优权重固定后在 split3_test 上一次性应用。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator, load_truth  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
RES = PROJECT / "results" / "revision"
STEP = 0.05
ASPECTS = {"MFO": "F", "BPO": "P", "CCO": "C"}


def grid():
    ws, n = [], round(1.0 / STEP)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            ws.append((round(i * STEP, 2), round(j * STEP, 2),
                       round((n - i - j) * STEP, 2)))
    return ws


def arbite(arb, tag, weights, suffix_lit="_dcut"):
    files = {}
    for ch, suf in (("seq", ""), ("struct", ""), ("lit", suffix_lit)):
        files[ch] = (WORK / f"{ch}_channel_{tag}{suf}.tsv").read_text(
            encoding="utf-8")
    arb.weights = {"seq": weights[0], "struct": weights[1], "lit": weights[2]}
    tmp = Path(tempfile.mkdtemp(prefix="rw_"))
    for ch, txt in files.items():
        (tmp / f"{ch}.tsv").write_text(txt, encoding="utf-8")
    out = tmp / "pred.tsv"
    arb.arbitrate(str(tmp / "seq.tsv"), str(tmp / "struct.tsv"),
                  str(tmp / "lit.tsv"), str(out), str(tmp / "audit.jsonl"))
    preds = {}
    for ln in out.read_text(encoding="utf-8").splitlines()[1:]:
        p0 = ln.split("\t")
        preds.setdefault(p0[0], {})[p0[1]] = max(
            float(p0[3]), preds.get(p0[0], {}).get(p0[1], 0))
    return preds


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"], relations=["is_a", "part_of"])
    ev = Evaluator(cfg, dag)
    truth_val = load_truth(str(PROJECT / "data/annotations/split2_go.tsv"),
                           dag=dag)
    train_truth = load_truth(str(PROJECT / "data/annotations/split1_go.tsv"),
                             dag=dag)
    truth_p = {p: ev._prop(g) for p, g in truth_val.items()}
    ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])

    arb = Arbiter(cfg, dag)
    results = {}
    best = {n: (-1, None) for n in ASPECTS}
    for ws in grid():
        raw = arbite(arb, "split2_val", ws)
        preds = {p: ev._propagate_preds(g) for p, g in raw.items()}
        row = {}
        for name, code in ASPECTS.items():
            pa = {p: {g: s for g, s in d.items() if dag.aspect(g) == code}
                  for p, d in preds.items()}
            ta = {p: {g for g in gs if dag.aspect(g) == code}
                  for p, gs in truth_p.items()}
            pa = {p: d for p, d in pa.items() if d}
            ta = {p: g for p, g in ta.items() if g}
            f, _s, _a, _d = ev.fmax_smin_aupr(pa, ta, ic)
            row[name] = round(f, 4)
            if f > best[name][0]:
                best[name] = (f, ws)
        results[str(ws)] = row
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "weight_search_val.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best_ws = {n: v[1] for n, v in best.items()}
    print("val 最优权重:", best_ws,
          {n: round(v[0], 4) for n, v in best.items()})

    # ---- test 一次性应用（逐本体合并）----
    merged = {}
    arb2 = Arbiter(cfg, dag)
    for name, code in ASPECTS.items():
        raw = arbite(arb2, "split3_test", best_ws[name])
        for pid, d in raw.items():
            for g, s in d.items():
                if dag.aspect(g) == code:
                    merged.setdefault(pid, {})
                    merged[pid][g] = max(s, merged[pid].get(g, 0))
    out = WORK / "revision" / "preds" / "predictions_split3_test_peront.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("protein_id\tgo_id\tscore\n")
        for pid, d in merged.items():
            for g, s in d.items():
                f.write(f"{pid}\t{g}\t{s:.4f}\n")
    rep = ev.evaluate(str(out), str(PROJECT / "data/annotations/split3_go.tsv"),
                      "peront-revised",
                      train_tsv=str(PROJECT / "data/annotations/split1_go.tsv"))
    summary = {"best_weights": best_ws,
               "overall": {k: rep[k] for k in
                           ("fmax", "smin", "aupr", "auc")},
               "by_aspect": {a["ontology"]: {k: a[k] for k in
                            ("fmax", "smin", "aupr", "auc")}
                             for a in rep["by_aspect"].values()}}
    (RES / "eval_peront_weights.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
