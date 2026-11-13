# -*- coding: utf-8 -*-
"""严格时间完整配置下 2.3 节机理解释所需的量（表 2 主结果口径）。

背景：2.3 节回答"学习仲裁的增益买自哪里"，用的是共同覆盖子集、发射类别数、
发射概率校准误差与模型系数四组数字。此前这些数字只在 InterProScan 5.72-103.0
全库配置下算过（results/revision/learn_eval.json），而表 2 主结果已改为严格配置
（结构通道只保留版本串自带日期码且不晚于 T0 的签名库，见 revision_strict.py）。
配置换了，机理解释不能仍引用旧配置的数：本脚本在同一严格配置下重算这四组量。

图 4 的三张诊断面板也取本脚本：val_curve 给两条路线在严格验证窗的 101 点 F1-阈值
曲线与 argmax / 1-SE 两个选点（同一 1-SE 规则、同一随机种子，与 revision_strict.py
选出的 t* 一致），arms[*].test_reliability 给测试窗传播后口径的分箱可靠性表。

复用而不重复劳动：预测文件直接取 revision_strict.py 的落盘产物
（work/revision/preds/_strict/），阈值取该脚本在严格验证窗用 1-SE 规则选出的
t*；模型重新拟合一次以取系数与出折校准，拟合样本、λ 网格与随机种子与
revision_strict.py 完全相同，因此系数就是部署的那一套。

产物：results/revision/strict_mech.json
"""
from __future__ import annotations

import json
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
from src.evaluator import (Evaluator, load_predictions,  # noqa: E402
                           load_truth)
from src.learning import (ArbitrationModel, CandidateFeatures, ece,  # noqa: E402
                          term_depth, FEATURES)
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402
import revision_bootstrap as rb  # noqa: E402
import struct_library_audit as sa  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
RES_DIR = PROJECT / "results" / "revision"
OUT_DIR = WORK / "revision" / "preds" / "_strict"
SEED = rl.SEED
ARMS = {"fusion": "multi", "learn": "learn"}


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

    se = json.loads((RES_DIR / "strict_eval.json").read_text(encoding="utf-8"))
    t_star = {a: se["routes"][a]["t_star_one_se"] for a in ("fusion", "learn")}
    ti = {a: se["routes"][a]["t_index_one_se"] for a in ("fusion", "learn")}

    # ---- 严格配置的模型（系数与出折校准即部署那一套）----
    keep = set(json.loads((RES_DIR / "struct_library_audit.json")
                          .read_text(encoding="utf-8"))
               ["pre_T0_date_coded_libraries"])
    v2 = rl.channels("split2_val")
    struct_val = OUT_DIR / "struct_strict_split2_val.tsv"
    if not struct_val.exists():
        sa.restrict_struct(Path(v2[1]), struct_val, keep)
    arb0 = Arbiter(cfg, dag)
    feats = CandidateFeatures(arb0, dag, ev.ic, dict(prevalence), depth)
    samples, _ = rl.build_samples(
        arb0, feats, rl.collect(arb0, v2[0], str(struct_val), v2[2]), tp2, tg2)
    model = ArbitrationModel(feats, seed=SEED).fit(samples)
    oof_ece = {}
    for a in ("F", "P", "C"):
        o = model.oof[a]
        oof_ece[a] = {"ece": round(ece(o["oof_prob"], o["oof_label"])["ece"], 4),
                      "n_row": o["n_row"], "n_pos": o["n_pos"],
                      "pos_rate": round(o["n_pos"] / o["n_row"], 4)}
    coefs = {a: {"beta": [round(float(b), 3) for b in model.models[a].beta],
                 "lambda": model.selection[a]["lambda"]}
             for a in ("F", "P", "C")}
    for a in coefs:
        coefs[a]["b0"] = coefs[a]["beta"][0]
        coefs[a]["w"] = dict(zip(FEATURES, coefs[a]["beta"][1:]))

    # ---- 两臂在严格配置下的测试窗产物 ----
    out = {"config": "datecoded_pre_T0", "T0": sa.T0,
           "t_star": t_star, "feature_names": list(FEATURES),
           "n_val_candidates": len(samples),
           "n_val_positives": int(sum(s[3] for s in samples)),
           "lambda": {a: model.selection[a]["lambda"] for a in ("F", "P", "C")},
           "cv_logloss": {a: model.selection[a]["cv_logloss"]
                          for a in ("F", "P", "C")},
           "oof_ece": oof_ece, "coefficients": coefs,
           "arms": {}, "deploy": {}, "val_curve": {}}
    preds, arrs, emitted = {}, {}, {}
    for arm, fname in ARMS.items():
        p3 = str(OUT_DIR / f"predictions_split3_test_{fname}.tsv")
        m3, preds[arm], arr3, emitted[arm] = rl.metrics_of(ev, tp3, tg3, p3)
        arrs[arm] = arr3
        per = rl.per_term_auc(ev, preds[arm], tp3, tg3)
        pc, rc, fc, cov = rev.prf1(arr3, ti[arm])
        prob, lab = [], []
        for p in tg3:
            tt = tp3[p]
            for g, s in preds[arm].get(p, {}).items():
                prob.append(s)
                lab.append(1 if g in tt else 0)
        e = ece(prob, lab)
        out["arms"][arm] = {"pred_path": p3, "fmax": m3["fmax"],
                            "aupr": m3["aupr"], "smin": m3["smin"],
                            "auc": m3["auc"],
                            "fmax_threshold": m3["fmax_threshold"],
                            "distinct_terms_at_fmax": len(emitted[arm]),
                            "test_ece_on_propagated": round(e["ece"], 4),
                            "test_ece_n_rows": e["n"],
                            "test_reliability": e["reliability"],
                            "auc_breakdown": rl.auc_breakdown(per, tg3, tp3, dag)}
        out["deploy"][arm] = {"t_star": t_star[arm], "P": round(pc, 4),
                              "R": round(rc, 4), "F1": round(fc, 4),
                              "covered": int(cov)}
        p2 = str(OUT_DIR / f"predictions_split2_val_{fname}.tsv")
        out["val_curve"][arm] = rl.fixed_f1(
            ev, tp2, tg2,
            {p: ev._propagate_preds(g) for p, g in load_predictions(p2).items()},
            tp3, tg3, preds[arm], rng=np.random.default_rng(SEED))
        print(arm, json.dumps(out["deploy"][arm], ensure_ascii=False),
              flush=True)

    # ---- 共同覆盖子集：排除输出规模差异带来的优势 ----
    cov_set = {a: {p for p in tg3 if preds[a].get(p)} for a in ARMS}
    common = sorted(set.intersection(*cov_set.values()))
    idx = np.asarray([tg3.index(p) for p in common])
    out["common_coverage_subset"] = {"n": len(common)}
    for a in ARMS:
        fmax, smin, aupr = rb.metrics_on(idx, arrs[a])
        out["common_coverage_subset"][a] = {
            "fmax": round(float(fmax), 4), "aupr": round(float(aupr), 4),
            "smin": round(float(smin), 4),
            "auc": round(ev.auc(preds[a], tp3, common)[0], 4)}

    # ---- 限定在两法都发射的类上比 AUC ----
    shared = emitted["learn"] & emitted["fusion"]
    pl = rl.per_term_auc(ev, preds["learn"], tp3, tg3)
    pf = rl.per_term_auc(ev, preds["fusion"], tp3, tg3)
    rows = {g: (pl[g][0], pf[g][0]) for g in shared if g in pl and g in pf}
    out["matched_emission_auc"] = {
        "n_classes": len(rows),
        "learn": round(float(np.mean([v[0] for v in rows.values()])), 4),
        "fusion": round(float(np.mean([v[1] for v in rows.values()])), 4)}

    # ---- 每蛋白平均输出术语数（Fmax 工作点与冻结工作点）----
    for a in ARMS:
        raw = {p: len({g for g, s in preds[a].get(p, {}).items()
                       if s >= t_star[a]})
               for p in tg3}
        out["arms"][a]["mean_terms_at_frozen_tstar"] = round(
            float(np.mean([v for v in raw.values() if v])), 3)
        out["arms"][a]["n_covered_frozen"] = sum(1 for v in raw.values() if v)

    # ---- 分来源的原始候选行精度（仲裁保留、尚未传播）----
    abl_p = str(OUT_DIR / "litonly" /
               "predictions_split3_test_learn_nolitonly_strict.tsv")
    out["source_precision"] = {
        k: rl.source_precision(v, tp3, dag.aspect)
        for k, v in (("fusion", out["arms"]["fusion"]["pred_path"]),
                     ("learn", out["arms"]["learn"]["pred_path"]),
                     ("learn_nolitonly", abl_p))}
    print("source_precision", json.dumps(
        {k: v["by_source"] for k, v in out["source_precision"].items()},
        ensure_ascii=False), flush=True)
    lit_only = defaultdict(lambda: [0, 0])
    for r in read_tsv(out["arms"]["learn"]["pred_path"]):
        if r["sources"] != "lit":
            continue
        v = lit_only[dag.aspect(r["go_id"])]
        v[0] += 1
        v[1] += 1 if r["go_id"] in tp3.get(r["protein_id"], set()) else 0
    out["lit_only_rows_by_aspect"] = {
        k: {"rows": v[0], "precision": round(v[1] / v[0], 4) if v[0] else None}
        for k, v in sorted(lit_only.items())}
    print("lit_only_rows_by_aspect", json.dumps(
        out["lit_only_rows_by_aspect"], ensure_ascii=False), flush=True)

    (RES_DIR / "strict_mech.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("deploy", "common_coverage_subset", "matched_emission_auc",
                       "oof_ece", "lambda")}, ensure_ascii=False, indent=1))
    print("->", RES_DIR / "strict_mech.json")


if __name__ == "__main__":
    main()
