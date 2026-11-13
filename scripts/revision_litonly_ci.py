# -*- coding: utf-8 -*-
"""文献单源行消融臂的冻结 F1 区间，及其与最强对照的配对差（表 5 下半所需）。

revision_strict.py 已给出消融臂与学习仲裁本身在同一 t* 下的配对差区间，但表 2 主结果
口径下最近邻 Top1 是冻结 F1 上唯一与学习仲裁无法区分的路线，因此"删掉文献单源行"这一
交付策略是否改变与最强对照的相对位置，必须单独检验而不是靠点估计读数。

约定与正文一致：每条路线都用它自己在严格验证窗按 1-SE 规则选出的 t*，同一次蛋白级
重采样用于所有路线（配对自助），B 与种子同主表。

产物：results/revision/litonly_ci.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402
import revision_bootstrap as rb  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
RES_DIR = PROJECT / "results" / "revision"
STRICT = WORK / "revision" / "preds" / "_strict"
SEED = rl.SEED
B = int(os.environ.get("MULTIGO_BOOTSTRAP", 2000))
ABL = {"learn_nolitonly": STRICT / "litonly" /
                     "predictions_split3_test_learn_nolitonly_strict.tsv",
       "learn": STRICT / "predictions_split3_test_learn.tsv",
       "nn_top1": STRICT / "baseline_nn_top1_split3_test.tsv",
       "fusion": STRICT / "predictions_split3_test_multi.tsv"}


def load_preds(ev, path):
    return {p: ev._propagate_preds(g) for p, g in load_predictions(path).items()}


def pctl(a):
    arr = np.asarray(a, dtype=float)
    return [round(float(np.percentile(arr, 2.5)), 4),
            round(float(np.percentile(arr, 97.5)), 4)]


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    train_truth = load_truth(str(ANN / "split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=dag)
    ev.ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])
    truth3 = load_truth(str(ANN / "split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    tg3 = [p for p in tp3 if tp3[p]]

    se = json.loads((RES_DIR / "strict_eval.json").read_text(encoding="utf-8"))
    ti = {k: se["routes"][k]["t_index_one_se"] for k in ABL if k in se["routes"]}
    ti["learn_nolitonly"] = se["lit_only_ablation"]["strict"]["same_t_star"]["t_star"]
    ti["learn_nolitonly"] = int(round(ti["learn_nolitonly"] * 100))

    arrs = {k: rb.precompute(ev, load_preds(ev, str(v)), tp3, tg3)
            for k, v in ABL.items()}
    n = len(tg3)
    grid = {}
    for k, a in arrs.items():
        fmax, smin, aupr = rb.metrics_on(np.arange(n), a)
        grid[k] = {"fmax": round(float(fmax), 4), "aupr": round(float(aupr), 4),
                   "smin": round(float(smin), 4),
                   "auc": round(ev.auc(load_preds(ev, str(ABL[k])), tp3, tg3)[0], 4)}
        print(k, grid[k], flush=True)
    rng = np.random.default_rng(SEED)
    preds_abl = load_preds(ev, str(ABL["learn_nolitonly"]))
    boot = {k: [] for k in ABL}
    abl_grid = []
    pairs = {"learn_nolitonly-nn_top1": [], "learn_nolitonly-fusion": [],
             "learn_nolitonly-learn": []}
    for _ in range(B):
        ix = rng.integers(0, n, size=n)
        val = {}
        for k, a in arrs.items():
            p, r, f, _c = rev.prf1({x: y[ix] for x, y in a.items()}, ti[k])
            boot[k].append(f)
            val[k] = f
        fm, sm, ap = rb.metrics_on(ix, arrs["learn_nolitonly"])
        abl_grid.append((fm, sm, ap,
                         ev.auc(preds_abl, tp3, [tg3[i] for i in ix])[0]))
        for key in pairs:
            a, b = key.split("-")
            pairs[key].append(val[a] - val[b])

    g = np.array(abl_grid)
    out = {"B": B, "seed": SEED, "n_targets": n,
           "config": "datecoded_pre_T0",
           "grid": grid,
           "grid_ci_learn_nolitonly": {
               k: pctl(g[:, j]) for j, k in
               enumerate(("fmax", "smin", "aupr", "auc"))},
           "t_star": {k: ti[k] / 100 for k in ABL},
           "frozen_F1": {k: {"point": round(rev.prf1(arrs[k], ti[k])[2], 4),
                             "ci95": pctl(boot[k])} for k in ABL},
           "paired_delta": {k: {"delta": round(float(np.mean(v)), 4),
                               "ci95": pctl(v)} for k, v in pairs.items()}}
    (RES_DIR / "litonly_ci.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
