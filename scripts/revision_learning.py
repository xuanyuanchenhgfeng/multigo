# -*- coding: utf-8 -*-
"""学习式仲裁 + 同协议决策规则基线（Vote-≥2）+ 分证据来源精度。

新增三条路线，全部沿用固定权重路线的决策语义（min_score、每蛋白 15 术语上限、
DAG 校验、未知判定、传播后阈值），因此可与正文表 3/表 4 的方法直接配对比较：

  learn   —— 候选级分本体 logistic 仲裁，仅在验证窗候选上拟合（按蛋白分组 5 折
             选 L2 强度），系数冻结后应用于测试窗；
  vote2   —— 规则基线：只有 >=2 个通道同时给出该候选时才保留，保留者的排序仍用
             融合分，其余流程不变；
  fusion  —— 正文的固定权重 0.40/0.35/0.25（对照）。

产物：
  results/revision/arbiter_model.json     模型系数、折间CV、出折ECE
  results/revision/learn_eval.json        三路线指标、校准F1、共同覆盖子集、ECE
  results/revision/bootstrap_learn.json   蛋白级配对 bootstrap（B=2000）
  results/revision/source_precision.json  按证据来源组合的经验精度
"""
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
from src.learning import (ArbitrationModel, CandidateFeatures, ece,  # noqa: E402
                          term_depth, FEATURES)
import revision_evaluate as rev  # noqa: E402
import revision_bootstrap as rb  # noqa: E402

WORK = PROJECT / "work"
PRED_DIR = WORK / "revision" / "preds"
RES_DIR = PROJECT / "results" / "revision"
ANN = PROJECT / "data" / "annotations"
SEED = 20261104
B = int(os.environ.get("MULTIGO_BOOTSTRAP", rb.B))   # 冒烟测试用 MULTIGO_BOOTSTRAP=20
GATES = (0.05, 0.10, 0.15, 0.25, 0.35)              # 学习式路线的分数下限网格
BOOT_SE = 200                                       # 1-SE 工作点规则用的验证窗重采样
T = rev.T


def channels(tag):
    return (str(WORK / f"seq_channel_{tag}.tsv"),
            str(WORK / f"struct_channel_{tag}.tsv"),
            str(WORK / f"lit_channel_{tag}_dcut.tsv"))


def collect(arb, seq_p, struct_p, lit_p):
    return (arb._load_seq(seq_p), arb._load_struct(struct_p), arb._load_lit(lit_p))


def build_samples(arb, feats, loaded, truth_prop, targets):
    seq, struct, lit = loaded
    rows, fixed_scores = [], []
    for pid in targets:
        cands, _, _ = arb.collect(seq.get(pid, {}), struct.get(pid, {}),
                                  lit.get(pid, {}))
        for go, ev in cands.items():
            aspect = arb.dag.aspect(go)
            if aspect not in ("F", "P", "C"):
                continue
            rows.append((pid, aspect, feats.vector(ev, go),
                         1 if go in truth_prop.get(pid, set()) else 0))
            fixed_scores.append((aspect, arb.score(ev, go),
                                 1 if go in truth_prop.get(pid, set()) else 0))
    return rows, fixed_scores


def run_arbiter(cfg, dag, scorer, tag, out_name, gate=None):
    arb = Arbiter(cfg, dag)
    if scorer is not None:
        arb.scorer = scorer
    if gate is not None:
        arb.min_score = gate
    s, t, l = channels(tag)
    out = PRED_DIR / out_name
    arb.arbitrate(s, t, l, str(out), str(PRED_DIR / out_name.replace(
        "predictions_", "audit_").replace(".tsv", ".jsonl")))
    return str(out)


def run_combo(cfg, dag, combo, tag, subdir):
    """验证窗的通道消融预测：正文只交付测试窗消融，验证窗这些文件仅用于定阈值，
    落在 preds/_val_ablation/ 下，不进交付包。"""
    chs = rev.COMBOS[combo]
    arb = Arbiter(cfg, dag)
    path = {c: rev.channel_path(tag, c, True) if c in chs else ""
            for c in ("seq", "struct", "lit")}
    out = PRED_DIR / subdir / f"predictions_{tag}_{combo}.tsv"
    arb.arbitrate(path["seq"], path["struct"], path["lit"], str(out),
                  str(out.parent / f"audit_{tag}_{combo}.jsonl"))
    return str(out)


def tune_gate(ev, cfg, dag, model, tp2, tg2, v2):
    """学习式路线的分数下限在验证窗标定（与固定权重同属仲裁层参数，测试窗不参与）。

    logistic 输出的是概率，与 0.40/0.35/0.25 加权分数量纲不同，直接沿用 0.15 会
    把候选砍得过稀（拟南芥窗每蛋白只剩 1.7 个术语）；下限网格取验证窗 Fmax 最优。
    """
    tmp = PRED_DIR / "_gate_tune"
    tmp.mkdir(parents=True, exist_ok=True)
    rows, best = {}, None
    for g in GATES:
        arb = Arbiter(cfg, dag)
        arb.scorer, arb.min_score = model, g
        p = str(tmp / f"val_gate_{g:g}.tsv")
        arb.arbitrate(v2[0], v2[1], v2[2], p, str(tmp / f"audit_gate_{g:g}.jsonl"))
        m2, _, _, _ = metrics_of(ev, tp2, tg2, p)
        rows[f"{g:g}"] = {"val_fmax": m2["fmax"], "val_aupr": m2["aupr"],
                          "val_mean_terms_raw": m2["mean_terms_raw"],
                          "val_covered": m2["covered_propagated"]}
        if best is None or m2["fmax"] > best[1]:
            best = (g, m2["fmax"])
    shutil.rmtree(tmp)
    return best[0], rows


def metrics_of(ev, truth_prop, targets, pred_path, idx=None):
    """三指标 + AUC + 该路线自身 Fmax 工作点上的输出规模（覆盖/术语数）。

    输出规模单列是为了回答"AUPR 优势是否只来自覆盖蛋白数与每蛋白术语数"：
    同一份 arr 还用于共同覆盖子集重算，两种口径并排落盘。
    """
    raw = load_predictions(pred_path)
    preds = {p: ev._propagate_preds(g) for p, g in raw.items()}
    arr = rb.precompute(ev, preds, truth_prop, targets)
    sel = np.arange(len(targets)) if idx is None else np.asarray(idx)
    fmax, smin, aupr = rb.metrics_on(sel, arr)
    sub = [targets[i] for i in sel]
    auc, n_classes = ev.auc(preds, truth_prop, sub)
    covered = sum(1 for p in sub if preds.get(p))
    ti = int(np.argmax([rev.prf1(arr, k)[2] for k in range(T)]))
    t_fmax = ti / 100.0
    nterms = [len([g for g, s in raw.get(p, {}).items() if s >= t_fmax])
              for p in sub]
    emitted = {g for p in sub for g, s in preds.get(p, {}).items()
               if s >= t_fmax}
    return ({"fmax": round(fmax, 4), "smin": round(smin, 4),
             "aupr": round(aupr, 4), "auc": round(auc, 4),
             "auc_n_classes": n_classes,
             "covered_propagated": covered,
             "covered_at_fmax": int(arr["mask"][:, ti].sum()),
             "fmax_threshold": t_fmax,
             "mean_terms_raw": round(float(np.mean(
                 [len(raw.get(p, {})) for p in sub])), 2),
             "mean_terms_at_fmax": round(float(np.mean(nterms)), 2),
             "distinct_terms_at_fmax": len(emitted)}, preds, arr, emitted)


def f1_curve(idx, a):
    """101 个网格点上的蛋白中心微平均 F1（P 按有输出的蛋白微平均、R 宏平均）。"""
    p, m, r = a["prec"][idx], a["mask"][idx], a["rec"][idx]
    denom = m.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1
    P = (p * m).sum(axis=0) / denom.sum(axis=0)
    R = r.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = 2 * P * R / (P + R)
    return np.nan_to_num(F)


def pick_threshold(curve, rng, a, n, boot=BOOT_SE):
    """验证窗 F1 曲线的 1-SE 工作点：在"与最大值相差不足一个标准误"的阈值里
    取最高者（即最保守、精度最高的那一端）。

    直接取 argmax 会踩到审稿意见指出的退化：BPO/CCO 的验证窗 F1 随阈值下降到
    网格下界 0.00 仍单调上升，把 0.00 冻结到测试窗会让精度崩塌。1-SE 规则只用
    验证窗信息，对全部路线同一条规则，不区分本体。
    """
    ti_opt = int(np.argmax(curve))
    draws = np.array([f1_curve(rng.integers(0, n, n), a) for _ in range(boot)])
    se = float(draws[:, ti_opt].std(ddof=1))      # 最优阈值处的重采样标准误
    tol = max(se, 1e-6)
    cand = [ti for ti in range(T) if curve[ti] >= curve[ti_opt] - tol]
    return ti_opt, int(max(cand)), round(se, 5), round(tol, 5)


def fixed_f1(ev, truth2_prop, targets2, preds2, truth3_prop, targets3, preds3,
             rng=None):
    """验证窗选 t*、冻结应用到测试窗；同时报 argmax 与 1-SE 两个工作点。"""
    a2 = rev.precompute(ev, preds2, truth2_prop, targets2)
    a3 = rev.precompute(ev, preds3, truth3_prop, targets3)
    curve = f1_curve(np.arange(len(targets2)), a2)
    ti_opt, ti_1se, se, tol = pick_threshold(curve, rng, a2, len(targets2))
    out = {}
    for tag, ti in (("argmax", ti_opt), ("one_se", ti_1se)):
        pc, rc, fc, cov = rev.prf1(a3, ti)
        out[f"t_star_{tag}"] = round(ti / 100, 2)
        out[f"t_index_{tag}"] = ti
        out[f"val_F1_{tag}"] = round(float(curve[ti]), 4)
        out[f"test_P_{tag}"] = round(pc, 4)
        out[f"test_R_{tag}"] = round(rc, 4)
        out[f"test_F1_{tag}"] = round(fc, 4)
        out[f"test_covered_{tag}"] = int(cov)
    out["val_F1_se"] = se
    out["val_F1_curve_max"] = round(float(curve.max()), 4)
    out["val_F1_curve"] = [round(float(x), 4) for x in curve]
    return out


def per_aspect_calibration(ev, preds2, tp2, tg2, preds3, tp3, tg3, dag, rng):
    """分命名空间阈值：各自在验证窗本体内用同一条 1-SE 规则选点，冻结并回测试窗。

    与全局阈值的差别只在"每个本体各一个工作点"，规则本身一致；报告两者是为了
    说明分本体阈值迁移在本协议下并不稳定（BPO 验证窗精度与测试窗精度差数倍）。
    """
    out, t_star = {}, {}
    for asp, code in (("MFO", "F"), ("BPO", "P"), ("CCO", "C")):
        a2 = rev.precompute(ev, preds2, tp2, tg2, code)
        # precompute 按"该本体有真值的蛋白"过滤过行，折样本数须与过滤后一致
        n2 = sum(1 for p in tg2
                 if any(dag.aspect(g) == code for g in tp2[p]))
        curve = f1_curve(np.arange(n2), a2)
        ti_opt, ti_1se, se, tol = pick_threshold(curve, rng, a2, n2)
        t_star[asp] = ti_1se / 100.0
        out[f"val_F1_{asp}"] = round(float(curve[ti_1se]), 4)
        out[f"val_F1_{asp}_argmax"] = round(float(curve[ti_opt]), 4)
        out[f"t_star_{asp}_argmax"] = ti_opt / 100.0
    tot_pred = tot_hit = tot_truth = 0
    covered = 0
    per_aspect = {}
    for asp, code in (("MFO", "F"), ("BPO", "P"), ("CCO", "C")):
        p_a = h_a = t_a = 0
        for p in tg3:
            tt = {g for g in tp3[p] if dag.aspect(g) == code}
            pp = {g for g, s in preds3.get(p, {}).items()
                  if dag.aspect(g) == code and s >= t_star[asp] + 1e-12}
            h = len(pp & tt)
            p_a += len(pp); h_a += h; t_a += len(tt)
            tot_pred += len(pp); tot_hit += h; tot_truth += len(tt)
        per_aspect[asp] = {"P": round(h_a / p_a, 4) if p_a else 0.0,
                           "R": round(h_a / t_a, 4) if t_a else 0.0,
                           "F1": round(2 * h_a / (p_a + t_a), 4)
                                 if p_a + t_a else 0.0,
                           "emitted": p_a}
    for p in tg3:
        if any(s >= t_star[{"F": "MFO", "P": "BPO", "C": "CCO"}[dag.aspect(g)]]
               + 1e-12 for g, s in preds3.get(p, {}).items()
               if dag.aspect(g) in "FPC"):
            covered += 1
    prec = tot_hit / tot_pred if tot_pred else 0.0
    rec = tot_hit / tot_truth if tot_truth else 0.0
    return {"t_star_per_aspect": {k: round(v, 2) for k, v in t_star.items()},
            "test_overall_P": round(prec, 4), "test_overall_R": round(rec, 4),
            "test_overall_F1": round(2 * prec * rec / (prec + rec), 4)
                               if prec + rec else 0.0,
            "test_overall_covered": covered,
            "test_emitted_terms": tot_pred,
            "per_aspect": per_aspect, **out}


def per_term_auc(ev, preds, truth, targets):
    """类中心 AUC 的逐项版本（与 Evaluator.auc 同语义），返回 {term: auc}。

    用于把"发射了哪些术语"与"排序质量"分开：未被任何预测覆盖的术语在该口径下
    恒为 0.5（全部并列零分），因此术语发射规模会直接进入宏平均 AUC。
    """
    import bisect
    pos_by_term = defaultdict(list)
    for p in targets:
        for g in truth.get(p, ()):
            pos_by_term[g].append(p)
    n_all = len(targets)
    tset = set(targets)
    index = defaultdict(dict)
    for p, gos in preds.items():
        if p not in tset:
            continue
        for g, s in gos.items():
            index[g][p] = s
    res = {}
    for g, pos_ps in pos_by_term.items():
        n_pos = len(pos_ps)
        if n_pos >= n_all:
            continue
        pos_set = set(pos_ps)
        idx = index.get(g, {})
        pos_scores = [idx.get(p, 0.0) for p in pos_ps]
        neg_scores = [s for p, s in idx.items() if p not in pos_set]
        n_zero_neg = n_all - n_pos - len(neg_scores)
        srt = sorted(neg_scores)
        auc = 0.0
        for s in pos_scores:
            lo, hi = bisect.bisect_left(srt, s), bisect.bisect_right(srt, s)
            less = lo + (n_zero_neg if s > 0 else 0)
            eq = (hi - lo) + (n_zero_neg if s == 0 else 0)
            auc += less + 0.5 * eq
        res[g] = (auc / (n_pos * (n_all - n_pos)), bool(idx))
    return res


def auc_breakdown(per_term, targets, truth, dag):
    """把类中心宏平均 AUC 拆成"术语是否被发射"与"排序质量"两部分。

    未被任何预测覆盖的 GO 类在该口径下恒为 0.5（全体样本并列零分），因此发射
    术语更多的路线会机械地抬高宏平均 AUC；同时给出只在已发射类上的均值。
    """
    n_all = len(targets)
    pos_by_term = defaultdict(list)
    for p in targets:
        for g in truth.get(p, ()):
            pos_by_term[g].append(p)
    rows = {}
    for g, pos_ps in pos_by_term.items():
        if len(pos_ps) >= n_all:
            continue
        a, emitted = per_term.get(g, (0.5, False))
        rows[g] = (a, emitted, dag.aspect(g))
    if not rows:
        return {"n_classes": 0}
    em = [v[0] for v in rows.values() if v[1]]
    out = {"n_classes": len(rows),
           "n_classes_emitted": len(em),
           "macro_auc": round(float(np.mean([v[0] for v in rows.values()])), 4),
           "macro_auc_emitted_only": round(float(np.mean(em)), 4) if em else None}
    for name, code in (("MFO", "F"), ("BPO", "P"), ("CCO", "C")):
        v = [a for a, _e, c in rows.values() if c == code]
        out[f"n_classes_{name}"] = len(v)
        out[f"auc_{name}"] = round(float(np.mean(v)), 4) if v else None
    return out


def source_precision(pred_path, truth_prop, aspect_of):
    src_rows = defaultdict(lambda: [0, 0])
    by_aspect = defaultdict(lambda: [0, 0])
    n_terms = 0
    for r in read_tsv(pred_path):
        pid, go, src = r["protein_id"], r["go_id"], r["sources"]
        hit = 1 if go in truth_prop.get(pid, set()) else 0
        src_rows[src][0] += 1
        src_rows[src][1] += hit
        by_aspect[aspect_of(go)][0] += 1
        by_aspect[aspect_of(go)][1] += hit
        n_terms += 1
    return {"rows": n_terms,
            "by_source": {k: {"rows": v[0], "in_truth": v[1],
                              "precision": round(v[1] / v[0], 4)}
                          for k, v in sorted(src_rows.items(),
                                             key=lambda x: -x[1][0])},
            "by_aspect": {k: {"rows": v[0], "precision": round(v[1] / v[0], 4)}
                          for k, v in sorted(by_aspect.items())}}


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"

    train_truth = load_truth(str(ANN / "split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=dag)
    ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])
    ev.ic = ic
    prevalence = Counter(g for gos in train_truth.values() for g in gos)
    depth = term_depth(dag)

    truth2 = load_truth(str(ANN / "split2_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    truth3 = load_truth(str(ANN / "split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    tg2 = [p for p in tp2 if tp2[p]]
    tg3 = [p for p in tp3 if tp3[p]]
    print(f"val targets={len(tg2)} test targets={len(tg3)}")

    arb0 = Arbiter(cfg, dag)
    feats = CandidateFeatures(arb0, dag, ic, dict(prevalence), depth)
    v2 = channels("split2_val")
    val_loaded = collect(arb0, *v2)
    samples, fixed_val = build_samples(arb0, feats, val_loaded, tp2, tg2)
    print("val 候选行数 =", len(samples),
          "正例 =", sum(s[3] for s in samples))

    model = ArbitrationModel(feats, seed=SEED).fit(samples)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(RES_DIR / "arbiter_model.json"))
    print(json.dumps(model.selection, ensure_ascii=False))

    # 固定权重分数当概率用的校准误差（对照，出折不适用故用全部验证候选）
    fx = defaultdict(lambda: ([], []))
    for aspect, sc, lab in fixed_val:
        fx[aspect][0].append(sc)
        fx[aspect][1].append(lab)
    fixed_ece = {a: ece(*fx[a])["ece"] for a in ("F", "P", "C")}
    fixed_ece_all = ece([r[1] for r in fixed_val], [r[2] for r in fixed_val])["ece"]

    # ---- 三条路线 × 两个窗口 ----
    gate_learn, gate_rows = tune_gate(ev, cfg, dag, model, tp2, tg2, v2)
    print("learn 分数下限（验证窗标定）=", gate_learn,
          json.dumps(gate_rows, ensure_ascii=False))
    vote = lambda ev_, go: (arb0.score(ev_, go)   # noqa: E731
                            if len(ev_.sources) >= 2 else 0.0)
    arms = {}
    for name, scorer, gate in (("fusion", None, None),
                               ("learn", model, gate_learn),
                               ("vote2", vote, None)):
        p2 = run_arbiter(cfg, dag, scorer, "split2_val",
                         f"predictions_split2_val_{name}.tsv", gate)
        p3 = run_arbiter(cfg, dag, scorer, "split3_test",
                         f"predictions_split3_test_{name}.tsv", gate)
        m3, preds3, arr3, emitted3 = metrics_of(ev, tp3, tg3, p3)
        m2 = metrics_of(ev, tp2, tg2, p2)[0]
        arms[name] = {"test": m3, "val": m2,
                      "_preds3": preds3, "_arr3": arr3, "_path3": p3,
                      "_path2": p2, "_emitted3": emitted3}
    for name in arms:
        prob, lab = [], []
        for p in tg3:
            tt = tp3[p]
            for g, s in arms[name]["_preds3"].get(p, {}).items():
                prob.append(s)
                lab.append(1 if g in tt else 0)
        arms[name]["test_ece_on_propagated"] = ece(prob, lab)
        per = per_term_auc(ev, arms[name]["_preds3"], tp3, tg3)
        arms[name]["auc_breakdown"] = auc_breakdown(per, tg3, tp3, dag)
    # 只在双方都发射的类上比较 AUC，去掉"发射规模"这一混杂来源
    shared = (arms["learn"]["_emitted3"] & arms["fusion"]["_emitted3"])
    pl = per_term_auc(ev, arms["learn"]["_preds3"], tp3, tg3)
    pf = per_term_auc(ev, arms["fusion"]["_preds3"], tp3, tg3)
    shared_rows = {g: (pl[g][0], pf[g][0]) for g in shared if g in pl and g in pf}
    matched_auc = {
        "n_classes": len(shared_rows),
        "learn": round(float(np.mean([v[0] for v in shared_rows.values()])), 4)
                 if shared_rows else None,
        "fusion": round(float(np.mean([v[1] for v in shared_rows.values()])), 4)
                  if shared_rows else None}

    # ---- 共同覆盖子集：排除输出规模差异带来的 AUPR 优势 ----
    covered = {name: {p for p in tg3 if arms[name]["_preds3"].get(p)}
               for name in arms}
    common = sorted(set.intersection(*covered.values()))
    idx_common = [tg3.index(p) for p in common]
    subset = {name: dict(zip(("fmax", "smin", "aupr"),
                             rb.metrics_on(np.asarray(idx_common),
                                           arms[name]["_arr3"])),
                            auc=round(ev.auc(arms[name]["_preds3"], tp3,
                                             common)[0], 4),
                            n=len(common))
               for name in arms}

    # ---- 同协议横向对比：三条仲裁路线 + 通道消融 + 转移基线 ----
    abl = "_val_ablation"
    configs = {
        "fusion": (str(PRED_DIR / "predictions_split2_val_multi.tsv"),
                   str(PRED_DIR / "predictions_split3_test_multi.tsv")),
        "seq-only": (run_combo(cfg, dag, "seq-only", "split2_val", abl),
                     str(PRED_DIR / "predictions_split3_test_seq-only.tsv")),
        "seq+struct": (run_combo(cfg, dag, "seq+struct", "split2_val", abl),
                       str(PRED_DIR / "predictions_split3_test_seq+struct.tsv")),
        "seq+lit": (run_combo(cfg, dag, "seq+lit", "split2_val", abl),
                    str(PRED_DIR / "predictions_split3_test_seq+lit.tsv")),
        "learn": (arms["learn"]["_path2"], arms["learn"]["_path3"]),
        "vote2": (arms["vote2"]["_path2"], arms["vote2"]["_path3"]),
        "interpro": (str(WORK / "baseline_interpro_split2_val.tsv"),
                     str(WORK / "baseline_interpro_split3_test.tsv")),
        "union": (str(WORK / "baseline_union_split2_val.tsv"),
                  str(WORK / "baseline_union_split3_test.tsv")),
        "nn_top3": (str(WORK / "baseline_nn_top3_split2_val.tsv"),
                    str(WORK / "baseline_nn_top3_split3_test.tsv")),
        # nn_top1 置于末尾：它不影响任何随机数流，但放在中间会挪动后续 arm 的
        # rng_se 消耗顺序，使已发表的阈值选择发生无意义漂移。
        "nn_top1": (str(WORK / "baseline_nn_top1_split2_val.tsv"),
                    str(WORK / "baseline_nn_top1_split3_test.tsv")),
    }
    arrays, points, pred_dicts, non_oracle = {}, {}, {}, {}
    rng_se = np.random.default_rng(SEED)        # 工作点选择用的验证窗重采样
    for name, (p2, p3) in configs.items():
        preds2 = {p: ev._propagate_preds(g)
                  for p, g in load_predictions(p2).items()}
        preds3 = {p: ev._propagate_preds(g)
                  for p, g in load_predictions(p3).items()}
        pred_dicts[name] = preds3
        arrays[name] = rb.precompute(ev, preds3, tp3, tg3)
        fmax, smin, aupr = rb.metrics_on(np.arange(len(tg3)), arrays[name])
        points[name] = (fmax, smin, aupr, ev.auc(preds3, tp3, tg3)[0])
        non_oracle[name] = {
            **fixed_f1(ev, tp2, tg2, preds2, tp3, tg3, preds3, rng=rng_se),
            **per_aspect_calibration(ev, preds2, tp2, tg2, preds3, tp3, tg3,
                                     dag, rng=rng_se)}
    # 部署工作点统一取 1-SE 规则（见 pick_threshold），argmax 口径同表并列
    ti_fixed = {k: v["t_index_one_se"] for k, v in non_oracle.items()}

    # ---- 配对 bootstrap（含冻结阈值 F1，回答"Fmax 是 oracle 口径"） ----
    pairs = [("learn", o) for o in
             ("fusion", "union", "interpro", "seq-only", "seq+struct", "vote2",
              "nn_top1", "nn_top3")]
    rng = np.random.default_rng(SEED)
    n = len(tg3)
    boots = {k: [] for k in configs}
    f1boots = {k: [] for k in configs}
    deltas = {f"{a}-{b}": [] for a, b in pairs}
    f1deltas = {f"{a}-{b}": [] for a, b in pairs}
    f1_point = {name: rev.prf1(arrays[name], ti_fixed[name])[2]
                for name in configs}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = [tg3[i] for i in idx]
        vals, f1vals = {}, {}
        for name in configs:
            fmax, smin, aupr = rb.metrics_on(idx, arrays[name])
            vals[name] = (fmax, smin, aupr,
                          ev.auc(pred_dicts[name], tp3, sub)[0])
            f1vals[name] = rev.prf1({k: a[idx] for k, a in
                                     arrays[name].items()}, ti_fixed[name])[2]
            boots[name].append(vals[name])
            f1boots[name].append(f1vals[name])
        for a, b in pairs:
            deltas[f"{a}-{b}"].append(tuple(vals[a][i] - vals[b][i]
                                            for i in range(4)))
            f1deltas[f"{a}-{b}"].append(f1vals[a] - f1vals[b])

    def ci(lst, names=("fmax", "smin", "aupr", "auc")):
        arr = np.array(lst)
        return {f"{k}_ci": [round(float(np.percentile(arr[:, j], 2.5)), 4),
                             round(float(np.percentile(arr[:, j], 97.5)), 4)]
                for j, k in enumerate(names)}

    boot_out = {"B": B, "seed": SEED, "n_targets": n,
                "note": "Fmax/Smin/AUPR/AUC 为测试窗网格口径（Fmax 取网格最大值，"
                        "是 oracle 上界）；F1_1se 为验证窗 F1 曲线 1-SE 规则选阈值、"
                        "冻结应用测试窗的可部署精度；F1_argmax 为同一流程改用验证窗 "
                        "F1 最大点的情形，两者都不使用测试窗信息选阈值",
                "point": {k: {"fmax": round(v[0], 4), "smin": round(v[1], 4),
                              "aupr": round(v[2], 4), "auc": round(v[3], 4),
                              "fixed_F1": round(f1_point[k], 4),
                              "t_star": round(ti_fixed[k] / 100, 2),
                              "F1_argmax": round(non_oracle[k]["test_F1_argmax"], 4),
                              "t_star_argmax": non_oracle[k]["t_star_argmax"]}
                          for k, v in points.items()},
                "ci": {k: ci(v) for k, v in boots.items()},
                "fixed_F1_ci": {k: [round(float(np.percentile(np.array(v), 2.5)), 4),
                                     round(float(np.percentile(np.array(v), 97.5)), 4)]
                                for k, v in f1boots.items()},
                "delta_ci": {k: {"dFmax": round(float(np.mean([d[0] for d in v])), 4),
                                 "dSmin": round(float(np.mean([d[1] for d in v])), 4),
                                 "dAupr": round(float(np.mean([d[2] for d in v])), 4),
                                 "dAuc": round(float(np.mean([d[3] for d in v])), 4),
                                 "dFixed_F1": round(float(np.mean(v_f1)), 4),
                                 "dFixed_F1_ci": [
                                     round(float(np.percentile(np.array(v_f1), 2.5)), 4),
                                     round(float(np.percentile(np.array(v_f1), 97.5)), 4)],
                                 **ci(v, ("dFmax", "dSmin", "dAupr", "dAuc"))}
                             for (k, v), v_f1 in zip(deltas.items(), f1deltas.values())}}
    (RES_DIR / "bootstrap_learn.json").write_text(
        json.dumps(boot_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 分证据来源经验精度（融合与学习两路线） ----
    sp = {"fusion": source_precision(arms["fusion"]["_path3"], tp3, dag.aspect),
          "learn": source_precision(arms["learn"]["_path3"], tp3, dag.aspect),
          "vote2": source_precision(arms["vote2"]["_path3"], tp3, dag.aspect)}
    (RES_DIR / "source_precision.json").write_text(
        json.dumps(sp, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 外部物种核查：拟南芥 T1 严格窗，模型系数冻结、不再拟合 ----
    ext_truth = load_truth(str(ANN / "arabidopsis_go_t1.tsv"),
                           exclude_iea=ev.exclude_iea, dag=dag)
    extp = {p: ev._prop(g) for p, g in ext_truth.items()}
    extg = [p for p in extp if extp[p]]
    external = {}
    for name, scorer, gate in (("fusion", None, None),
                               ("learn", model, gate_learn)):
        pp = run_arbiter(cfg, dag, scorer, "arabidopsis",
                         f"predictions_arabidopsis_{name}.tsv", gate)
        m, preds, _, _ = metrics_of(ev, extp, extg, pp)
        external[name] = m
    print(f"arabidopsis targets={len(extg)} external:",
          json.dumps(external, ensure_ascii=False))

    out = {
        "protocol_note": "learn 仅在验证窗候选上拟合，按蛋白分组5折选L2强度，"
                         "系数冻结后应用测试窗；三条路线共用 min_score/上限/DAG校验/传播口径",
        "n_val_candidates": len(samples),
        "n_val_positives": int(sum(s[3] for s in samples)),
        "features": list(FEATURES),
        "model_selection": model.selection,
        "oof_ece": {a: model.to_dict()["oof"][a]["ece"] for a in ("F", "P", "C")},
        "fixed_score_ece_val": {"by_aspect": {k: round(v, 5) for k, v in fixed_ece.items()},
                                "overall": round(fixed_ece_all, 5)},
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in arms.items()},
        "learn_gate": {"selected": gate_learn, "grid": gate_rows},
        "per_aspect_note": "分本体工作点只作敏感性检查：λ=10 把 BPO 概率整体压到"
                           "分数下限以下，验证窗 BPO 的 F1 曲线恒为 0，1-SE 规则在"
                           "该本体退回网格上界；且分本体合并用的是全局微平均，与"
                           "蛋白中心的 Fmax/F1 口径不同，数值系统性偏低。部署口径"
                           "是三本体共用一个全局阈值",
        "deploy_table": non_oracle,
        "common_coverage_subset": subset,
        "matched_emission_auc": matched_auc,
        "external_arabidopsis_T1": external,
        "bootstrap": "results/revision/bootstrap_learn.json",
        "source_precision": "results/revision/source_precision.json",
    }
    (RES_DIR / "learn_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["arms"], ensure_ascii=False, indent=1))
    print(json.dumps(out["common_coverage_subset"], ensure_ascii=False, indent=1))
    print(json.dumps(boot_out["delta_ci"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
