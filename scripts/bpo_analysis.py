# -*- coding: utf-8 -*-
"""BPO 短板分析：IC/深度分层报告 + 阈值校准 + 误判案例（split3_test）。

防泄漏纪律：
  - 分层边界基于 IC（train+split3 真值传播统计，与 PSB26 评估协议一致，非标签泄漏）
  - 阈值校准只用 split2_val（val 上扫网格找 t*），在 split3_test 上固定应用
  - oracle（test 上逐档扫 101 网格）仅作上界参照，不作为可报告数字

口径：蛋白中心（与 evaluator/bootstrap_ci 完全一致——点估计已逐位验证一致）。
用法: python3 -X utf8 scripts/bpo_analysis.py
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
T = 101
ROOT_BPO = "GO:0008150"


def precompute(ev, preds, truth_p, targets, allowed=None):
    """per-protein×101 数组（allowed: GO 子集过滤，None=不过滤）。"""
    n = len(targets)
    arrs = {k: np.zeros((n, T)) for k in ("prec", "mask", "rec", "ru", "mi")}
    for i, p in enumerate(targets):
        tt = truth_p.get(p, set())
        pg = preds.get(p, {})
        if allowed is not None:
            tt = tt & allowed
            pg = {g: s for g, s in pg.items() if g in allowed}
        if not tt:
            continue
        by_t = [[] for _ in range(T)]
        for g, s in pg.items():
            ti = min(int(s * 100 + 1e-9), T - 1)
            by_t[ti].append(g)
        pt_sets, acc = [], set()
        for ti in range(T - 1, -1, -1):
            acc = acc | set(by_t[ti])
            pt_sets.append(set(acc))
        pt_sets.reverse()
        for ti, pt in enumerate(pt_sets):
            inter = len(pt & tt)
            if pt:
                arrs["prec"][i, ti] = inter / len(pt)
                arrs["mask"][i, ti] = 1.0
            arrs["rec"][i, ti] = inter / len(tt)
    return arrs


def pr_f1(arrs, ti):
    """固定阈值列 ti 的 (P, R, F1)（蛋白中心）。"""
    prec = arrs["prec"][:, ti]
    mask = arrs["mask"][:, ti]
    rec = arrs["rec"][:, ti]
    p = float((prec * mask).sum() / mask.sum()) if mask.sum() else 0.0
    r = float(rec.mean())
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fmax_grid(arrs):
    """oracle：101 网格最优 F1，返回 (fmax, t*)。"""
    best = (0.0, 0)
    for ti in range(T):
        _, _, f = pr_f1(arrs, ti)
        if f > best[0]:
            best = (f, ti)
    return best


def depth_of(dag, go_id, cache):
    """BFS 到 BPO 根的最短步数（is_a+part_of 合并 parents）。"""
    if go_id in cache:
        return cache[go_id]
    if go_id == ROOT_BPO:
        cache[go_id] = 0
        return 0
    seen, q = {go_id}, deque([(go_id, 0)])
    while q:
        cur, d = q.popleft()
        for par in (dag.terms[cur].parents if cur in dag.terms else ()):
            if par == ROOT_BPO:
                cache[go_id] = d + 1
                return d + 1
            if par not in seen:
                seen.add(par)
                q.append((par, d + 1))
    cache[go_id] = -1  # 不连通
    return -1


def main():
    cfg_path = (sys.argv[1] if len(sys.argv) > 1
                else str(PROJECT / "config.yaml"))
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        PROJECT / "results" / "bpo_analysis.json")
    revised = "revision" in Path(cfg_path).name
    cfg = load_config(cfg_path)
    log = setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    pred_dir = (PROJECT / "work/revision/preds"
                if revised else PROJECT / "work")
    multi3 = ("predictions_split3_test_multi.tsv" if revised
              else "predictions_split3_test.tsv")
    multi2 = ("predictions_split2_val_multi.tsv" if revised
              else "predictions_split2_val.tsv")

    # ---- 加载（split3_test 评估 + split2_val 校准 + train IC）----
    truth3 = load_truth(str(PROJECT / "data/annotations/split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=ev.dag)
    truth2 = load_truth(str(PROJECT / "data/annotations/split2_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=ev.dag)
    train_truth = load_truth(str(PROJECT / "data/annotations/split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=ev.dag)
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    ic_sets = [ev._prop(g) for g in train_truth.values()]
    if ev.ic_source != "train_only":
        ic_sets += list(tp3.values())
    ic = ev.compute_ic(ic_sets)
    preds3 = {p: ev._propagate_preds(g) for p, g in
              load_predictions(str(pred_dir / multi3)).items()}
    preds2 = {p: ev._propagate_preds(g) for p, g in
              load_predictions(str(pred_dir / multi2)).items()}
    pred3_seq = {p: ev._propagate_preds(g) for p, g in
                 load_predictions(str(pred_dir /
                                      ("predictions_split3_test_seq-only.tsv"
                                       if revised else
                                       "predictions_split3_test_seq-only.tsv"))).items()}

    out = {"分层报告": {}, "阈值校准": {}, "误判案例": []}

    # ---- BPO 术语池与分层（IC / 深度各三档，33%/66% 分位）----
    pool = set()
    for g in tp3.values():
        pool |= {x for x in g if ev.dag.aspect(x) == "P"}
    ic_vals = sorted(ic.get(g, 0.0) for g in pool)
    q33 = ic_vals[len(ic_vals) // 3]
    q66 = ic_vals[2 * len(ic_vals) // 3]
    dc, layers_ic, layers_dep = {}, {}, {}
    dep_vals = []
    for g in pool:
        d = depth_of(ev.dag, g, dc)
        if d >= 0:
            dep_vals.append(d)
    dep_vals.sort()
    d33 = dep_vals[len(dep_vals) // 3]
    d66 = dep_vals[2 * len(dep_vals) // 3]

    def layer_of_ic(g):
        v = ic.get(g, 0.0)
        return "低IC" if v <= q33 else ("中IC" if v <= q66 else "高IC")

    def layer_of_dep(g):
        v = dc.get(g, -1)
        if v < 0:
            return None
        return "浅层" if v <= d33 else ("中层" if v <= d66 else "深层")

    for g in pool:
        layers_ic.setdefault(layer_of_ic(g), set()).add(g)
        ld = layer_of_dep(g)
        if ld:
            layers_dep.setdefault(ld, set()).add(g)

    targets_all = [p for p in tp3 if tp3[p]]

    def strat(layer_set, label, pred_dict):
        tg = [p for p in targets_all
              if tp3[p] & layer_set]
        arrs = precompute(ev, pred_dict, tp3, tg, allowed=layer_set)
        fm, t_or = fmax_grid(arrs)
        n_term = len(layer_set)
        n_anno = sum(len(tp3[p] & layer_set) for p in tg)
        rec = {"terms": n_term, "proteins": len(tg), "annotations": n_anno,
               "oracle_fmax": round(fm, 4), "oracle_t": round(t_or / 100, 2)}
        return rec, arrs, tg

    for lname, lset in layers_ic.items():
        rec, _, _ = strat(lset, lname, preds3)
        out["分层报告"].setdefault("IC分层_3Channel", {})[lname] = rec
        rec_s, _, _ = strat(lset, lname + "_seq", pred3_seq)
        out["分层报告"].setdefault("IC分层_seq-only", {})[lname] = rec_s
    for lname, lset in layers_dep.items():
        rec, _, _ = strat(lset, lname, preds3)
        out["分层报告"].setdefault("深度分层_3Channel", {})[lname] = rec

    # ---- 阈值校准：val 上找 t*，test 上固定应用 ----
    for aspect, name in (("P", "BPO"), ("F", "MFO"), ("C", "CCO")):
        # open 池口径：真值 ∪ 预测侧的本体术语全集（与主表 evaluator 一致，
        # 预测了池外术语同样计入 FP，避免 closed 池高估 precision）
        pool2 = {g for s in tp2.values() for g in s
                 if ev.dag.aspect(g) == aspect}
        pool2 |= {g for d in preds2.values() for g in d
                  if ev.dag.aspect(g) == aspect}
        tg2 = [p for p in tp2 if {g for g in tp2[p]
                                  if ev.dag.aspect(g) == aspect}]
        arrs2 = precompute(ev, preds2, tp2, tg2, allowed=pool2)
        f_val, t_star = fmax_grid(arrs2)
        # test 上固定应用（open 池）
        pool3 = {g for s in tp3.values() for g in s
                 if ev.dag.aspect(g) == aspect}
        pool3 |= {g for d in preds3.values() for g in d
                  if ev.dag.aspect(g) == aspect}
        tg3 = [p for p in targets_all if tp3[p] & pool3]
        arrs3 = precompute(ev, preds3, tp3, tg3, allowed=pool3)
        fm_or, t_or3 = fmax_grid(arrs3)
        p_c, r_c, f_c = pr_f1(arrs3, t_star)
        out["阈值校准"][name] = {
            "val校准阈值": round(t_star / 100, 2),
            "val上的Fmax": round(f_val, 4),
            "test_校准阈值_F1": round(f_c, 4),
            "test_校准阈值_P": round(p_c, 4), "test_校准阈值_R": round(r_c, 4),
            "test_oracle_Fmax(上界)": round(fm_or, 4),
            "test_oracle_t": round(t_or3 / 100, 2),
        }

    # ---- BPO 分层阈值校准：每 IC 档在 val 上独立找 t* ----
    bpo_cal = {}
    tg3_bpo = [p for p in targets_all if tp3[p] & layers_ic.get("高IC", set())
               | layers_ic.get("中IC", set()) | layers_ic.get("低IC", set())]
    for lname in ("低IC", "中IC", "高IC"):
        lset = layers_ic.get(lname, set())
        tg2 = [p for p in tp2 if tp2[p] & lset]
        arrs2 = precompute(ev, preds2, tp2, tg2, allowed=lset)
        f_val, t_star = fmax_grid(arrs2)
        tg3 = [p for p in targets_all if tp3[p] & lset]
        arrs3 = precompute(ev, preds3, tp3, tg3, allowed=lset)
        fm_or, _ = fmax_grid(arrs3)
        p_c, r_c, f_c = pr_f1(arrs3, t_star)
        bpo_cal[lname] = {"val校准阈值": round(t_star / 100, 2),
                          "test_校准F1": round(f_c, 4),
                          "test_oracle_Fmax": round(fm_or, 4),
                          "test_P": round(p_c, 4), "test_R": round(r_c, 4)}
    out["阈值校准"]["BPO_IC分层阈值"] = bpo_cal

    # ---- 误判案例：高IC档中 ru（漏报IC和）与 mi（误报IC和）最大的蛋白 ----
    hi = layers_ic.get("高IC", set())
    tg3 = [p for p in targets_all if tp3[p] & hi]
    arrs = precompute(ev, preds3, tp3, tg3, allowed=hi)
    fm, t_or = fmax_grid(arrs)
    ti = t_or
    rows = []
    for i, p in enumerate(tg3):
        tt = tp3[p] & hi
        pg = {g: s for g, s in preds3.get(p, {}).items() if g in hi}
        pt = {g for g, s in pg.items() if s >= ti / 100}
        miss = tt - pt
        fp = pt - tt
        ru_v = sum(ic.get(g, 0) for g in miss)
        mi_v = sum(ic.get(g, 0) for g in fp)
        rows.append((ru_v + mi_v, p, sorted(miss)[:3], sorted(fp)[:3]))
    rows.sort(reverse=True)
    out["误判案例"] = [
        {"protein": p, "IC误差和": round(v, 2), "高IC漏报": m, "高IC误报": fp}
        for v, p, m, fp in rows[:5]]

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("BPO 分析完成 -> %s", out_path)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
