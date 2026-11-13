# -*- coding: utf-8 -*-
"""修订版 bootstrap：8 个方法（四配置 + 四基线）配对差值 95% CI。

蛋白级重采样 B=2000，同批重采样索引用于全部方法（配对比较）。
IC 仅训练库（与修订评估一致）。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
PRED_DIR = WORK / "revision" / "preds"
RES_DIR = PROJECT / "results" / "revision"
T = 101
B = 2000
SEED = 20261104

CONFIGS = {
    "fusion": str(PRED_DIR / "predictions_split3_test_multi.tsv"),
    "seq-only": str(PRED_DIR / "predictions_split3_test_seq-only.tsv"),
    "seq+struct": str(PRED_DIR / "predictions_split3_test_seq+struct.tsv"),
    "seq+lit": str(PRED_DIR / "predictions_split3_test_seq+lit.tsv"),
    "interpro": str(WORK / "baseline_interpro_split3_test.tsv"),
    "union": str(WORK / "baseline_union_split3_test.tsv"),
    "nn_top1": str(WORK / "baseline_nn_top1_split3_test.tsv"),
    "nn_top3": str(WORK / "baseline_nn_top3_split3_test.tsv"),
}
PAIRS = [("fusion", k) for k in
         ("seq-only", "seq+struct", "seq+lit", "interpro", "union")]


def precompute(ev, preds, truth_p, targets):
    n = len(targets)
    prec = np.zeros((n, T)); mask = np.zeros((n, T)); rec = np.zeros((n, T))
    ru = np.zeros((n, T)); mi = np.zeros((n, T))
    for i, p in enumerate(targets):
        tt = truth_p[p]
        pg = preds.get(p, {})
        by_t = [[] for _ in range(T)]
        for g, s in pg.items():
            by_t[min(int(s * 100 + 1e-9), T - 1)].append(g)
        sets, acc = [], set()
        for ti in range(T - 1, -1, -1):
            acc |= set(by_t[ti]); sets.append(set(acc))
        sets.reverse()
        for ti, pt in enumerate(sets):
            inter = len(pt & tt)
            if pt:
                prec[i, ti] = inter / len(pt); mask[i, ti] = 1.0
            rec[i, ti] = inter / len(tt)
            ru[i, ti] = sum(ev.ic.get(g, 0.0) for g in tt - pt)
            mi[i, ti] = sum(ev.ic.get(g, 0.0) for g in pt - tt)
    return {"prec": prec, "mask": mask, "rec": rec, "ru": ru, "mi": mi}


def metrics_on(idx, a):
    p, m, r = a["prec"][idx], a["mask"][idx], a["rec"][idx]
    ru, mi = a["ru"][idx], a["mi"][idx]
    denom = m.sum(axis=0, keepdims=True); denom[denom == 0] = 1
    P = (p * m).sum(axis=0) / denom.sum(axis=0)
    R = r.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = 2 * P * R / (P + R)
    F = np.nan_to_num(F)
    fmax = F.max()
    s = np.sqrt(ru.mean(axis=0) ** 2 + mi.mean(axis=0) ** 2)
    # AUPR：101 阈值点按召回排序后梯形积分（与 evaluator 同口径）
    order = np.argsort(R)
    rs, ps = R[order], P[order]
    aupr = float(np.sum(np.diff(rs) * (ps[1:] + ps[:-1]) / 2.0))
    return fmax, float(s.min()), aupr


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"], relations=["is_a", "part_of"])
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    truth = load_truth(str(PROJECT / "data/annotations/split3_go.tsv"),
                       dag=dag)
    truth_p = {p: ev._prop(g) for p, g in truth.items()}
    train_truth = load_truth(str(PROJECT / "data/annotations/split1_go.tsv"),
                             dag=dag)
    ic_sets = [ev._prop(g) for g in train_truth.values()]
    if ev.ic_source != "train_only":
        ic_sets += list(truth_p.values())
    ev.ic = ev.compute_ic(ic_sets)
    targets = [p for p in truth_p if truth_p[p]]
    print("targets:", len(targets))

    arrays, point, pred_dicts = {}, {}, {}
    for name, path in CONFIGS.items():
        preds = {p: ev._propagate_preds(g)
                 for p, g in load_predictions(path).items()}
        pred_dicts[name] = preds
        arrays[name] = precompute(ev, preds, truth_p, targets)
        fmax, smin, aupr = metrics_on(np.arange(len(targets)), arrays[name])
        auc_v, _ = ev.auc(preds, truth_p, targets)
        point[name] = (fmax, smin, aupr, auc_v)

    rng = np.random.default_rng(SEED)
    n = len(targets)
    boots = {k: [] for k in CONFIGS}
    deltas = {f"{a}-{b}": [] for a, b in PAIRS}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_targets = [targets[i] for i in idx]
        vals = {}
        for name in CONFIGS:
            fmax, smin, aupr = metrics_on(idx, arrays[name])
            auc_v, _ = ev.auc(pred_dicts[name], truth_p, boot_targets)
            v = (fmax, smin, aupr, auc_v)
            vals[name] = v
            boots[name].append(v)
        for a, c in PAIRS:
            deltas[f"{a}-{c}"].append(
                (vals[a][0] - vals[c][0], vals[a][1] - vals[c][1],
                 vals[a][2] - vals[c][2], vals[a][3] - vals[c][3]))

    out = {"B": B, "seed": SEED, "n_targets": n,
           "point": {k: {"fmax": round(v[0], 4), "smin": round(v[1], 4),
                         "aupr": round(v[2], 4), "auc": round(v[3], 4)}
                     for k, v in point.items()}, "ci": {}, "delta_ci": {}}
    for k, lst in boots.items():
        arr = np.array(lst)
        out["ci"][k] = {
            "fmax_ci": [round(float(np.percentile(arr[:, 0], 2.5)), 4),
                        round(float(np.percentile(arr[:, 0], 97.5)), 4)],
            "smin_ci": [round(float(np.percentile(arr[:, 1], 2.5)), 4),
                        round(float(np.percentile(arr[:, 1], 97.5)), 4)],
            "aupr_ci": [round(float(np.percentile(arr[:, 2], 2.5)), 4),
                        round(float(np.percentile(arr[:, 2], 97.5)), 4)],
            "auc_ci": [round(float(np.percentile(arr[:, 3], 2.5)), 4),
                       round(float(np.percentile(arr[:, 3], 97.5)), 4)]}
    for k, lst in deltas.items():
        arr = np.array(lst)
        out["delta_ci"][k] = {
            "dFmax": round(float(arr[:, 0].mean()), 4),
            "dFmax_ci": [round(float(np.percentile(arr[:, 0], 2.5)), 4),
                         round(float(np.percentile(arr[:, 0], 97.5)), 4)],
            "dSmin": round(float(arr[:, 1].mean()), 4),
            "dSmin_ci": [round(float(np.percentile(arr[:, 1], 2.5)), 4),
                         round(float(np.percentile(arr[:, 1], 97.5)), 4)],
            "dAupr": round(float(arr[:, 2].mean()), 4),
            "dAupr_ci": [round(float(np.percentile(arr[:, 2], 2.5)), 4),
                         round(float(np.percentile(arr[:, 2], 97.5)), 4)],
            "dAuc": round(float(arr[:, 3].mean()), 4),
            "dAuc_ci": [round(float(np.percentile(arr[:, 3], 2.5)), 4),
                        round(float(np.percentile(arr[:, 3], 97.5)), 4)]}
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "bootstrap_ci.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["point"], ensure_ascii=False, indent=1))
    print(json.dumps(out["delta_ci"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
