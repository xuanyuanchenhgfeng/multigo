# -*- coding: utf-8 -*-
"""
分本体权重搜索（投稿补强：权重稳健性）
======================================
在验证集（split2_val）上按本体网格搜索三通道权重（步长0.05，单纯形），
得到各本体最优权重后固定应用于测试集，与全局经验权重(0.40/0.35/0.25)对比。
协议纪律：搜索仅用 val，test 只做一次性应用。
"""
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import src.pipeline  # noqa: F401
from src.arbiter import Arbiter
from src.evaluator import Evaluator
from src.go_dag import GODag
import yaml

STEP = 0.05
TAG_VAL, TAG_TEST = "split2_val", "split3_test"
ASPECTS = {"MFO": "F", "BPO": "P", "CCO": "C"}


def grid():
    ws = []
    n = round(1.0 / STEP)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            ws.append((round(i * STEP, 2), round(j * STEP, 2),
                       round(k * STEP, 2)))
    return ws


def arbite(arb, cfg, dag, tag, weights):
    """按给定权重跑一次仲裁，返回 {protein: {go: score}}（读库文件重放）。"""
    seq_t = (BASE / "work" / f"seq_channel_{tag}.tsv").read_text(
        encoding="utf-8").splitlines()
    st_t = (BASE / "work" / f"struct_channel_{tag}.tsv").read_text(
        encoding="utf-8").splitlines()
    lt_t = (BASE / "work" / f"lit_channel_{tag}.tsv").read_text(
        encoding="utf-8").splitlines()
    arb.weights = {"seq": weights[0], "struct": weights[1], "lit": weights[2]}
    # Arbiter 主入口是 arbitrate(文件路径...)——写临时文件
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix=f"wsearch_{tag}_"))
    (tmp / "seq.tsv").write_text("\n".join(seq_t) + "\n", encoding="utf-8")
    (tmp / "struct.tsv").write_text("\n".join(st_t) + "\n", encoding="utf-8")
    (tmp / "lit.tsv").write_text("\n".join(lt_t) + "\n", encoding="utf-8")
    out = tmp / "pred.tsv"
    arb.arbitrate(str(tmp / "seq.tsv"), str(tmp / "struct.tsv"),
                  str(tmp / "lit.tsv"), str(out), str(tmp / "audit.jsonl"))
    preds = {}
    for ln in out.read_text(encoding="utf-8").splitlines()[1:]:
        parts = ln.split("\t")
        p, g, s = parts[0], parts[1], parts[3]
        preds.setdefault(p, {})[g] = max(float(s), preds.get(p, {}).get(g, 0))
    return preds


def main():
    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)

    # 预加载 val 真值与 IC（一次）
    from src.evaluator import load_truth, load_predictions
    truth_val = load_truth(str(BASE / "data/annotations/split2_go.tsv"),
                           exclude_iea=True, dag=dag)
    train_truth = load_truth(str(BASE / "data/annotations/split1_go.tsv"),
                             exclude_iea=True, dag=dag)
    truth_p = {p: ev._prop(gos) for p, gos in truth_val.items()}
    ic = ev.compute_ic([ev._prop(gos) for gos in train_truth.values()]
                       + list(truth_p.values()))

    arb = Arbiter(cfg, dag)
    results = {}
    for ws in grid():
        preds_raw = arbite(arb, cfg, dag, TAG_VAL, ws)
        preds = {p: ev._propagate_preds(gos) for p, gos in preds_raw.items()}
        per = {}
        for name, code in ASPECTS.items():
            pa = {p: {g: s for g, s in d.items()
                      if dag.aspect(g) == code}
                  for p, d in preds.items()}
            ta = {p: {g for g in gos if dag.aspect(g) == code}
                  for p, gos in truth_p.items()}
            pa = {p: d for p, d in pa.items() if d}
            ta = {p: g for p, g in ta.items() if g}
            f, _s, _a, _d = ev.fmax_smin_aupr(pa, ta, ic)
            per[name] = round(f, 4)
        results[str(ws)] = per
        print(ws, per, flush=True)

    out = BASE / "results" / "weight_search_val.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    best = {n: max(results, key=lambda w: results[w][n]) for n in ASPECTS}
    print("\n最优分本体权重:", best)


if __name__ == "__main__":
    main()
