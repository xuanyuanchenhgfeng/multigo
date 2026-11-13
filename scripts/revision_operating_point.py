# -*- coding: utf-8 -*-
"""分命名空间工作点（部署口径）——回答审稿意见 5 的阈值问题。

一根全局分数下限在验证窗看着最优，冻结到测试窗却把 BPO 打死：学习型仲裁的 BPO
概率天然偏低（验证窗 BPO 候选正例率 1.8%，λ=10 又进一步收缩），全局下限 0.05 一卡，
测试窗 BPO 只剩 6 行输出、F1≈0；同一模型在 BPO 上的验证窗 F1 峰值出现在下限 0.005
处（0.3533）。这不是模型的缺陷，是部署规则把一个三峰分布用一根线切。

规则（与全文其余部分同一条纪律，不用测试窗信息选参数）：
  1. 固定权重与学习式仲裁两条路线，各自在验证窗按命名空间扫同一组分数下限网格；
  2. 选点用 1-SE 规则：取"与峰值相差不足 1 SE 的下限中最高的一档"（最保守的一端），
     避免把 BPO 这种扁曲线上的噪声峰当信号；
  3. 下限冻结后应用测试窗：按命名空间下限筛选原始仲裁输出，先传播再算
     蛋白中心微平均 P / 宏平均 R / F1，整体与分命名空间都给。

口径声明：本脚本一律"筛选后传播"（gate 规则），与主表的"传播后阈值"（t* 规则）
是两个不同的部署约定，数值不可直接互比；主表的可部署 F1 与其配对区间由
scripts/revision_learning.py 产出，本脚本只比较同一 gate 约定下的四种工作点。

MULTIGO_CONFIG=strict 时改跑严格时间完整配置（表 2 主结果口径）：结构通道换成
work/revision/preds/_strict/struct_strict_{窗口}.tsv，学习模型的拟合样本同步换成受限
后的验证窗候选，产物写 results/revision/operating_point_strict.json。两条路线的单一
全局下限在两种配置下同值（0.15 与 0.05，前者是设计参数、后者由各自的验证窗标定）。
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
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
from src.learning import ArbitrationModel, CandidateFeatures, term_depth  # noqa: E402
import revision_learning as rl  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
PRED_DIR = WORK / "revision" / "preds"
CONFIG = os.environ.get("MULTIGO_CONFIG", "full")
STRICT_DIR = PRED_DIR / "_strict"
OUT = PRED_DIR / ("_oppoint_strict" if CONFIG == "strict" else "_oppoint")
RES_DIR = PROJECT / "results" / "revision"
OUT_JSON = RES_DIR / ("operating_point_strict.json" if CONFIG == "strict"
                      else "operating_point.json")
SEED = rl.SEED
B = int(os.environ.get("MULTIGO_BOOTSTRAP", 2000))
GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35)
ASPECTS = ("F", "P", "C")
GLOBAL_FLOOR = {"fusion": 0.15, "learn": 0.05}     # 各自现行交付下限
N_SE_BOOT = 200


def prop(ev, path):
    return {p: ev._propagate_preds(g) for p, g in load_predictions(path).items()}


def prop_sets(ev, emitted):
    return {p: ev._propagate_preds(d) for p, d in emitted.items()}


def run(cfg, dag, scorer, gate, tag, name):
    arb = Arbiter(cfg, dag)
    if scorer is not None:
        arb.scorer = scorer
    arb.min_score = gate
    p = str(OUT / f"{name}_{tag}.tsv")
    ch = list(rl.channels(tag))
    if CONFIG == "strict":
        ch[1] = str(STRICT_DIR / f"struct_strict_{tag}.tsv")
    arb.arbitrate(ch[0], ch[1], ch[2], p, str(OUT / f"audit_{name}_{tag}.jsonl"))
    return p


def counts(preds, truth, targets, aspect_of, aspect=None):
    """每蛋白 (命中数, 输出数, 真值数)；配对重采样只需要这三个可加量。"""
    n = len(targets)
    inter, npred, ntruth = (np.zeros(n), np.zeros(n), np.zeros(n))
    for i, p in enumerate(targets):
        tt, pr = truth.get(p, set()), set(preds.get(p, {}))
        if aspect:
            tt = {g for g in tt if aspect_of(g) == aspect}
            pr = {g for g in pr if aspect_of(g) == aspect}
        ntruth[i], npred[i], inter[i] = len(tt), len(pr), len(pr & tt)
    return inter, npred, ntruth


def f1_from(inter, npred, ntruth):
    """蛋白中心：精度按有输出的蛋白微平均，召回按目标蛋白宏平均（同 evaluator）。"""
    cov = npred > 0
    p = float(inter[cov].sum() / npred[cov].sum()) if cov.any() else 0.0
    ok = ntruth > 0
    r = float((inter[ok] / ntruth[ok]).mean()) if ok.any() else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), int(cov.sum())


def pick_gate(curve, ses, grid):
    ti = int(np.argmax(curve))
    tol = max(ses[ti], 1e-6)
    cand = [i for i in range(len(grid)) if curve[i] >= curve[ti] - tol]
    return grid[int(max(cand))], grid[ti], ses[ti]


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    OUT.mkdir(parents=True, exist_ok=True)
    train_truth = load_truth(str(ANN / "split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=dag)
    ev.ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])
    prev = Counter(g for gos in train_truth.values() for g in gos)

    tp2 = {p: ev._prop(g) for p, g in load_truth(
        str(ANN / "split2_go.tsv"), exclude_iea=ev.exclude_iea, dag=dag).items()}
    tp3 = {p: ev._prop(g) for p, g in load_truth(
        str(ANN / "split3_go.tsv"), exclude_iea=ev.exclude_iea, dag=dag).items()}
    tg2 = [p for p in tp2 if tp2[p]]
    tg3 = [p for p in tp3 if tp3[p]]
    aspect_of = dag.aspect      # 方法，按 (g) 调用

    arb0 = Arbiter(cfg, dag)
    feats = CandidateFeatures(arb0, dag, ev.ic, dict(prev), term_depth(dag))
    v2 = rl.channels("split2_val")
    if CONFIG == "strict":
        import struct_library_audit as sa
        keep = set(json.loads((RES_DIR / "struct_library_audit.json")
                              .read_text(encoding="utf-8"))
                   ["pre_T0_date_coded_libraries"])
        struct_val = STRICT_DIR / "struct_strict_split2_val.tsv"
        if not struct_val.exists():
            sa.restrict_struct(Path(v2[1]), struct_val, keep)
        samples, _ = sa.build_samples_restricted(
            arb0, feats, str(struct_val), v2[2], tp2, tg2, dag)
    else:
        samples, _ = rl.build_samples(arb0, feats, rl.collect(arb0, *v2),
                                      tp2, tg2)
    model = ArbitrationModel(feats, seed=SEED).fit(samples)

    rng = np.random.default_rng(SEED)
    arms = {"fusion": None, "learn": model}
    result = {"note": "gate 规则（筛选后传播）；主表 t* 规则（传播后阈值）见 "
                      + ("strict_eval.json" if CONFIG == "strict"
                         else "bootstrap_learn.json")
                      + "，两套约定不可互比", "config": CONFIG,
              "grid": list(GRID), "B": B, "seed": SEED, "n_se_boot": N_SE_BOOT,
              "per_aspect_gate": {}, "val_curve": {}}

    # ---- 1) 验证窗：每条路线 × 每个命名空间扫下限，1-SE 选点 ----
    for name, scorer in arms.items():
        pre = {g: prop(ev, run(cfg, dag, scorer, g, "split2_val",
                               f"{name}_g{g:g}")) for g in GRID}
        for asp in ASPECTS:
            tg = [p for p in tg2 if any(aspect_of(g) == asp for g in tp2[p])]
            cur, ses = [], []
            for g in GRID:
                inter, npred, ntruth = counts(pre[g], tp2, tg, aspect_of, asp)
                f = f1_from(inter, npred, ntruth)[2]
                draws = [f1_from(inter[idx], npred[idx], ntruth[idx])[2]
                         for idx in (rng.integers(0, len(tg), len(tg))
                                     for _ in range(N_SE_BOOT))]
                cur.append(f)
                ses.append(float(np.std(draws, ddof=1)))
            g_star, g_argmax, se = pick_gate(np.array(cur), ses, GRID)
            result["per_aspect_gate"].setdefault(name, {})[asp] = {
                "n_val_proteins": len(tg), "gate_one_se": g_star,
                "gate_argmax": g_argmax, "se_at_peak": round(se, 5),
                "val_F1_at_one_se": round(cur[GRID.index(g_star)], 4),
                "val_F1_at_argmax": round(cur[GRID.index(g_argmax)], 4)}
            result["val_curve"].setdefault(name, {})[asp] = {
                f"{g:g}": [round(c, 4), round(s, 5)]
                for g, c, s in zip(GRID, cur, ses)}
            print(name, asp, json.dumps(result["per_aspect_gate"][name][asp]),
                  flush=True)

    # ---- 2) 测试窗：gate 规则评估（分命名空间 vs 单一全局下限）----
    cand = {}
    for name, scorer in arms.items():
        gates = {a: result["per_aspect_gate"][name][a]["gate_one_se"]
                 for a in ASPECTS}
        low = min(gates.values())
        raw = load_predictions(run(cfg, dag, scorer, low, "split3_test",
                                  f"{name}_final"))
        emitted = {p: {g: s for g, s in d.items() if s >= gates[aspect_of(g)]}
                   for p, d in raw.items()}
        cand[f"{name}_peraspect"] = prop_sets(ev, emitted)
        cand[f"{name}_globalfloor"] = prop(ev, run(cfg, dag, scorer,
                                                   GLOBAL_FLOOR[name],
                                                   "split3_test",
                                                   f"{name}_globalfloor"))

    summary, arrays = {}, {}
    NAME = {"F": "MFO", "P": "BPO", "C": "CCO"}
    for label, preds in cand.items():
        inter, npred, ntruth = counts(preds, tp3, tg3, aspect_of)
        pr, rc, f, cov = f1_from(inter, npred, ntruth)
        row = {"overall": {"P": round(pr, 4), "R": round(rc, 4),
                           "F1": round(f, 4), "covered": cov}}
        for asp in ASPECTS:
            tg = [p for p in tg3 if any(aspect_of(g) == asp for g in tp3[p])]
            i2, n2, t2 = counts(preds, tp3, tg, aspect_of, asp)
            pr2, rc2, f2, cov2 = f1_from(i2, n2, t2)
            row[NAME[asp]] = {"P": round(pr2, 4), "R": round(rc2, 4),
                              "F1": round(f2, 4), "covered": cov2,
                              "n_proteins": len(tg)}
        summary[label] = row
        arrays[label] = (inter, npred, ntruth)
        print(label, json.dumps(row, ensure_ascii=False), flush=True)

    # ---- 3) 同一 gate 约定内的配对比较 ----
    pairs = [("learn_peraspect", "learn_globalfloor"),
             ("fusion_peraspect", "fusion_globalfloor"),
             ("learn_peraspect", "fusion_peraspect"),
             ("learn_globalfloor", "fusion_globalfloor")]
    boot = {}
    n = len(tg3)
    for a, b in pairs:
        d = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            d.append(f1_from(*(x[idx] for x in arrays[a]))[2]
                     - f1_from(*(x[idx] for x in arrays[b]))[2])
        arr = np.array(d)
        lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        boot[f"{a}-{b}"] = {"delta": round(float(arr.mean()), 4),
                            "ci95": [round(lo, 4), round(hi, 4)],
                            "se": round(float(arr.std(ddof=1)), 4),
                            "significant": bool(lo > 0 or hi < 0)}
        print(f"{a}-{b}: ΔF1 {boot[f'{a}-{b}']['delta']:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] (B={B})", flush=True)

    result["test_gate_rule"] = summary
    result["paired_delta_F1"] = boot
    RES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出", OUT_JSON)


if __name__ == "__main__":
    main()
