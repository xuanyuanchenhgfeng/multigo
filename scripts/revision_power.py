# -*- coding: utf-8 -*-
"""统计功效分析（审稿意见 3：测试窗太小、BPO 更小，差异能不能被分辨）。

审稿人给的三个规模数字（1 130 蛋白、BPO 184 蛋白、深层 BPO 96 蛋白/288 条注释）
里，前两个与本项目的真值统计一致，第三个来自旧协议脚本的术语分层，需要在修订口径
下重算并核对。本脚本一次做三件事：

  1. 规模核对：按"直接注释 / 传播后注释"两种口径给出每本体的蛋白数、注释条数、
     每蛋白注释数分布，纠正正文里引用过的错误规模；
  2. 分层：MFO/BPO/CCO 与 BPO 深度三分位（浅/中/深），深度定义与 bpo_analysis.py
     一致（到 GO:0008150 的 is_a/part_of 最短步数），分层只描述规模，不参与调参；
  3. 可分辨差异：在每一层用同一批蛋白做配对重采样（B=2000），给出 learn 相对
     fusion / union / vote2 的冻结阈 F1 差值 95% CI、CI 半宽、标准误，并按
     MDD_80 = (z_0.025 + z_0.20)·SE ≈ 2.80·SE 报告 80% 功效下的最小可分辨差值。
     阈值 t* 一律用"全验证窗 + 1-SE 规则"选定后冻结，分层只作用于测试窗评估，
     因此各层用的是同一个可部署工作点，层间可比。

结论口径：整体层能分辨 ~0.02 量级的 F1 差；深层 BPO 那一层的 MDD 比整体大一倍以上，
所以那一层里"谁比谁好"的不显著不能读作"没有差别"。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402
import bpo_analysis as bp  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
PRED_DIR = WORK / "revision" / "preds"
RES_DIR = PROJECT / "results" / "revision"
# MULTIGO_CONFIG=strict 把全部输入换成 scripts/revision_strict.py 重跑出来的严格
# 时间完整配置产物（结构通道只留截止前可判定的签名库），结果写独立文件，
# 不覆盖已发表的全库配置 power_analysis.json。
CONFIG = os.environ.get("MULTIGO_CONFIG", "full")
BASELINE_DIR = WORK
OUT_JSON = "power_analysis.json"
if CONFIG == "strict":
    PRED_DIR = WORK / "revision" / "preds" / "_strict"
    BASELINE_DIR = PRED_DIR
    OUT_JSON = "power_analysis_strict.json"
SEED = rl.SEED
B = int(os.environ.get("MULTIGO_BOOTSTRAP", 2000))
Z80 = 2.80                          # z_0.025 + z_0.20，双侧 5%、80% 功效
ARMS = {
    "fusion": "predictions_split3_test_multi.tsv",
    "learn": "predictions_split3_test_learn.tsv",
    "union": None,                   # 基线走 work/ 下的同协议产物
    "vote2": "predictions_split3_test_vote2.tsv",
}
VAL_ARMS = {
    "fusion": "predictions_split2_val_multi.tsv",
    "learn": "predictions_split2_val_learn.tsv",
    "union": None,
    "vote2": "predictions_split2_val_vote2.tsv",
}
PAIRS = [("learn", "fusion"), ("learn", "union"), ("learn", "vote2"),
         ("fusion", "union")]


def val_paths():
    """验证窗四个路线的预测文件（union/vote2/fusion 用交付路径，learn 用标定路径）。"""
    out = {}
    for name, fn in VAL_ARMS.items():
        if name == "union":
            out[name] = str(BASELINE_DIR / "baseline_union_split2_val.tsv")
        elif fn is None:
            out[name] = None
        else:
            out[name] = str(PRED_DIR / fn)
    return out


def test_paths():
    out = {}
    for name, fn in ARMS.items():
        if name == "union":
            out[name] = str(BASELINE_DIR / "baseline_union_split3_test.tsv")
        else:
            out[name] = str(PRED_DIR / fn)
    return out


def load_preds(ev, path):
    return {p: ev._propagate_preds(g) for p, g in load_predictions(path).items()}


def scale_counts(truth_raw, tp, targets, dag):
    """直接注释与传播后注释两套口径的规模核对。"""
    out = {}
    keep = set(targets)
    for asp, label in (("F", "MFO"), ("P", "BPO"), ("C", "CCO")):
        direct = sum(1 for p, gos in truth_raw.items() if p in keep
                     for g in gos if dag.aspect(g) == asp)
        prots = [p for p in targets if any(dag.aspect(g) == asp for g in tp[p])]
        prop = sum(len([g for g in tp[p] if dag.aspect(g) == asp])
                   for p in prots)
        per = [sum(1 for g in truth_raw[p] if dag.aspect(g) == asp)
               for p in prots]
        out[label] = {
            "n_proteins_direct": len(prots),
            "n_direct_annotations": direct,
            "n_propagated_annotations": prop,
            "mean_direct_per_protein": round(float(np.mean(per)), 2) if per else 0,
            "max_direct_per_protein": int(max(per)) if per else 0,
            "proteins_ge_64_direct": int(sum(1 for x in per if x >= 64)),
        }
    out["overall"] = {
        "n_proteins": len(targets),
        "n_direct_annotations": sum(len(g) for p, g in truth_raw.items()
                                    if p in keep),
        "n_propagated_annotations": sum(len(tp[p]) for p in targets),
        "mean_propagated_per_protein": round(
            float(np.mean([len(tp[p]) for p in targets])), 2),
    }
    return out


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    train_truth = load_truth(str(ANN / "split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=dag)
    ev.ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])

    truth_raw = load_truth(str(ANN / "split3_go.tsv"),
                           exclude_iea=ev.exclude_iea, dag=dag)
    truth3 = load_truth(str(ANN / "split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    truth2 = load_truth(str(ANN / "split2_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    tg3 = [p for p in tp3 if tp3[p]]
    tg2 = [p for p in tp2 if tp2[p]]

    out = {"scale": scale_counts(truth_raw, tp3, tg3, dag),
           "bootstrap": {"B": B, "seed": SEED, "z80": Z80,
                         "mdd_formula": "MDD_80 = 2.80 × SE(Δ配对)"}}

    # ---- 分层定义（只描述规模，不参与任何调参）----
    pool = {g for s in tp3.values() for g in s if ev.dag.aspect(g) == "P"}
    dc = {}
    deps = sorted(d for d in (bp.depth_of(ev.dag, g, dc) for g in pool) if d >= 0)
    d33, d66 = deps[len(deps) // 3], deps[2 * len(deps) // 3]
    deep = {g for g in pool if dc.get(g, -1) > d66}
    mid = {g for g in pool if d33 < dc.get(g, -1) <= d66}
    shallow = {g for g in pool if 0 <= dc.get(g, -1) <= d33}
    strata = {"overall": (None, None)}
    for asp, label in (("F", "MFO"), ("P", "BPO"), ("C", "CCO")):
        strata[label] = (asp, None)
    strata["BPO_shallow"] = ("P", shallow)
    strata["BPO_mid"] = ("P", mid)
    strata["BPO_deep"] = ("P", deep)

    vp = val_paths()
    preds2 = {k: load_preds(ev, p) for k, p in vp.items() if p}
    preds3 = {k: load_preds(ev, p) for k, p in test_paths().items()}

    # ---- 各路线全验证窗冻结阈（1-SE 规则），分层只切测试窗 ----
    tstar = {}
    rng_se = np.random.default_rng(SEED)
    for name in preds3:
        pv2 = preds2[name]
        a2 = rev.precompute(ev, pv2, tp2, tg2)
        curve = rl.f1_curve(np.arange(len(tg2)), a2)
        _, ti_1se, se, tol = rl.pick_threshold(curve, rng_se, a2, len(tg2))
        tstar[name] = {"t_index": ti_1se, "t_star": round(ti_1se / 100, 2),
                       "val_F1_at_tstar": round(float(curve[ti_1se]), 4),
                       "val_F1_se": se}
    out["frozen_thresholds"] = tstar
    print(json.dumps(tstar, ensure_ascii=False), flush=True)

    rng = np.random.default_rng(SEED)
    layers = {}
    for label, (asp, allowed) in strata.items():
        # 行集与 precompute 内部的过滤必须一致，否则重采样索引会越界
        tg = [p for p in tg3
              if (allowed is None or (tp3[p] & allowed))
              and (asp is None
                   or any(ev.dag.aspect(g) == asp for g in tp3[p]))]
        arrs = {}
        for name, p3 in preds3.items():
            if allowed is not None:
                a = bp.precompute(ev, {pp: {g: s for g, s in d.items()
                                            if g in allowed}
                                       for pp, d in p3.items()},
                                 {pp: {g for g in s if g in allowed}
                                  for pp, s in tp3.items()}, tg, allowed=allowed)
            else:
                a = rev.precompute(ev, p3, tp3, tg, asp)
            arrs[name] = a
        point = {n: round(bp.pr_f1(a, tstar[n]["t_index"])[2], 4)
                 for n, a in arrs.items()}
        n = len(tg)
        terms = (allowed if allowed is not None else
                 {g for p in tg for g in tp3[p]
                  if asp is None or ev.dag.aspect(g) == asp})
        res = {"n_proteins": n,
               "n_terms": len(terms),
               "n_annotations": int(sum(len(tp3[p] & terms) for p in tg)),
               "frozen_F1": point}
        # 配对重采样：同一批重采样索引用于所有路线
        draws = {n_: [] for n_ in arrs}
        deltas = {f"{a}-{b}": [] for a, b in PAIRS}
        idx_all = np.arange(n)
        for _ in range(B):
            idx = rng.integers(0, n, n) if n > 1 else idx_all
            vals = {}
            for name, a in arrs.items():
                vals[name] = bp.pr_f1({k: v[idx] for k, v in a.items()},
                                      tstar[name]["t_index"])[2]
                draws[name].append(vals[name])
            for a, b in PAIRS:
                deltas[f"{a}-{b}"].append(vals[a] - vals[b])
        for k, v in deltas.items():
            arr = np.array(v)
            se = float(arr.std(ddof=1))
            lo, hi = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
            res.setdefault("paired_delta_F1", {})[k] = {
                "delta": round(float(arr.mean()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "ci_half_width": round((hi - lo) / 2, 4),
                "se": round(se, 4),
                "MDD_80": round(Z80 * se, 4),
                "significant": bool(lo > 0 or hi < 0)}
        for name, v in draws.items():
            arr = np.array(v)
            res.setdefault("F1_ci95", {})[name] = [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4)]
        layers[label] = res
        print(label, json.dumps(res["frozen_F1"]),
              "n=", n, flush=True)
    out["strata"] = layers
    out["depth_terciles"] = {"d33": d33, "d66": d66,
                             "n_deep_terms": len(deep),
                             "n_terms_pool": len(pool)}

    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / OUT_JSON).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出", RES_DIR / OUT_JSON)
    print(json.dumps(out["scale"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
