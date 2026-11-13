# -*- coding: utf-8 -*-
"""真跨物种压力测试下的学习仲裁（补审稿意见 A3：跨物种只算了规则臂）。

跨物种设置与 revision_cross_species.py 完全一致：检索库剔除全部拟南芥条目
（Swiss-Prot taxid 3702 与 GOA 库侧 8 505 条），DIAMOND 重比对得到的序列通道产物
直接复用，结构通道用签名库批次的原生产物，文献通道用出版日期截断版。

学习臂的纪律：模型系数、分数下限与部署阈值 t* 全部取主协议验证窗
（split2_val）标定后的冻结值，在拟南芥查询上不做任何重新拟合、重新标定或
重新选点——这正是"标定窗跨物种是否失效"要问的东西。规则臂同批重跑一遍作为参照，
两组查询（真跨物种 noath 与同物种留出）× 两种配置各算一次：
  full   —— 结构通道为 5.72-103.0 全量签名库（含截止后信息，只作上界参照）；
  strict —— 结构通道只保留版本串自带日期码且不晚于 T0 的签名库（表 2 主结果口径）。

产物：results/revision/eval_cross_species_learn.json（表 6 的学习仲裁行）
"""
from __future__ import annotations

import json
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
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator, load_truth  # noqa: E402
from src.learning import ArbitrationModel, CandidateFeatures, term_depth  # noqa: E402
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402
import revision_bootstrap as rb  # noqa: E402
import struct_library_audit as sa  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
CROSS = WORK / "revision" / "cross_species"
OUT_DIR = CROSS / "learn"
RES_DIR = PROJECT / "results" / "revision"
SEED = rl.SEED
T0 = sa.T0


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
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    tg2 = [p for p in tp2 if tp2[p]]

    ext_truth = load_truth(str(ANN / "arabidopsis_go_t1.tsv"),
                           exclude_iea=ev.exclude_iea, dag=dag)
    tpX = {p: ev._prop(g) for p, g in ext_truth.items()}
    tgX = [p for p in tpX if tpX[p]]
    print(f"跨物种可评估蛋白 = {len(tgX)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = set(json.loads((RES_DIR / "struct_library_audit.json")
                          .read_text(encoding="utf-8"))
               ["pre_T0_date_coded_libraries"])
    struct_full = str(WORK / "struct_channel_arabidopsis.tsv")
    struct_strict = str(OUT_DIR / "struct_strict_arabidopsis.tsv")
    restriction = sa.restrict_struct(Path(struct_full), Path(struct_strict), keep)
    print("跨物种结构行:", json.dumps(restriction, ensure_ascii=False), flush=True)

    seq_noath = str(CROSS / "seq_channel_arabidopsis_noath.tsv")
    lit = str(WORK / "lit_channel_arabidopsis_dcut.tsv")
    v2 = rl.channels("split2_val")
    arb0 = Arbiter(cfg, dag)
    feats = CandidateFeatures(arb0, dag, ev.ic, dict(prevalence), depth)

    # 冻结参数：全库配置取已发表产物，严格配置取 revision_strict.py 的产物
    le = json.loads((RES_DIR / "learn_eval.json").read_text(encoding="utf-8"))
    bl = json.loads((RES_DIR / "bootstrap_learn.json").read_text(encoding="utf-8"))
    se = json.loads((RES_DIR / "strict_eval.json").read_text(encoding="utf-8"))
    plans = {
        "full": {"struct_val": v2[1], "struct_ext": struct_full,
                 "gate": le["learn_gate"]["selected"],
                 "t_star": bl["point"]["learn"]["t_star"]},
        "strict": {"struct_val": se_restricted_struct(se, keep, v2),
                   "struct_ext": struct_strict,
                   "gate": se["learn_gate"]["selected"],
                   "t_star": se["routes"]["learn"]["t_star_one_se"]},
    }

    out = {"T0": T0, "n_targets": len(tgX), "restriction": restriction,
           "protocol": "学习臂的系数、分数下限与部署 t* 全部冻结自各自配置的主协议"
                       "验证窗，拟南芥两组查询上都不做任何再标定",
           "settings": {"cross": "检索库剔除全部拟南芥（16 948 条）",
                        "same": "检索库含 70% 拟南芥蛋白（25 453 条）的同物种留出"},
           "arms": {}}
    seqs = {"cross": seq_noath, "same": str(WORK / "seq_channel_arabidopsis.tsv")}
    for conf, pl in plans.items():
        samples, _ = sa.build_samples_restricted(
            arb0, feats, pl["struct_val"], v2[2], tp2, tg2, dag)
        model = ArbitrationModel(feats, seed=SEED).fit(samples)
        ti = int(round(pl["t_star"] * 100))
        ts_route = {"learn": pl["t_star"],
                    "fusion": (bl["point"]["fusion"]["t_star"] if conf == "full"
                               else se["routes"]["fusion"]["t_star_one_se"])}
        for setting, seqp in seqs.items():
          for arm, scorer, gate in (("learn", model, pl["gate"]),
                                    ("fusion", None, None)):
            arb = Arbiter(cfg, dag)
            if scorer is not None:
                arb.scorer = scorer
            if gate is not None:
                arb.min_score = gate
            p = str(OUT_DIR / f"pred_arabidopsis_{setting}_{arm}_{conf}.tsv")
            arb.arbitrate(seqp, pl["struct_ext"], lit, p,
                          str(OUT_DIR / f"audit_{setting}_{arm}_{conf}.jsonl"))
            m, preds, _, _ = rl.metrics_of(ev, tpX, tgX, p)
            arr = rb.precompute(ev, preds, tpX, tgX)
            pc, rc, fc, cov = rev.prf1(arr, ti)
            pr_, rr_, fr_, covr = rev.prf1(arr, int(round(ts_route[arm] * 100)))
            aspect = {}
            for code, label in (("F", "MFO"), ("P", "BPO"), ("C", "CCO")):
                a = rev.precompute(ev, preds, tpX, tgX, code)
                tg = [q for q in tgX
                      if any(dag.aspect(g) == code for g in tpX[q])]
                fmax_a, aupr_a = grid_curve(a)
                pa, ra, fa, _ = rev.prf1(a, ti)
                aspect[label] = {"fmax": round(fmax_a, 4), "aupr": round(aupr_a, 4),
                                 "precision": round(pa, 4),
                                 "recall": round(ra, 4),
                                 "frozen_F1": round(fa, 4), "n": len(tg)}
            out["arms"][f"{setting}_{arm}_{conf}"] = {
                "fmax": m["fmax"], "aupr": m["aupr"], "smin": m["smin"],
                "auc": m["auc"], "coverage": m["covered_propagated"] / len(tgX),
                "frozen_at_main_tstar": {
                    "t_star": round(ti / 100, 2), "P": round(pc, 4),
                    "R": round(rc, 4), "F1": round(fc, 4),
                    "covered": int(cov)},
                "frozen_at_route_tstar": {
                    "t_star": round(ts_route[arm], 2), "P": round(pr_, 4),
                    "R": round(rr_, 4), "F1": round(fr_, 4),
                    "covered": int(covr)},
                "by_aspect": aspect,
                "mean_terms_raw": m["mean_terms_raw"],
            }
            print(setting, conf, arm,
                  json.dumps(out["arms"][f"{setting}_{arm}_{conf}"],
                             ensure_ascii=False), flush=True)
    (RES_DIR / "eval_cross_species_learn.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", RES_DIR / "eval_cross_species_learn.json")


def grid_curve(a):
    """测试窗网格上的 Fmax 与 AUPR，口径同 revision_bootstrap.metrics_on。

    precompute 只给出 prec/mask/rec 三张数组，Smin 需要的 ru/mi 不在其中，
    而跨物种表只引用 Fmax 与 AUPR，故这里按同一算法重算这两项、跳过 Smin。
    """
    p, m, r = a["prec"], a["mask"], a["rec"]
    denom = m.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1
    P = (p * m).sum(axis=0) / denom.sum(axis=0)
    R = r.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.nan_to_num(2 * P * R / (P + R))
    order = np.argsort(R)
    aupr = float(np.sum(np.diff(R[order]) * (P[order][1:] + P[order][:-1]) / 2.0))
    return float(F.max()), aupr


def se_restricted_struct(se, keep, v2):
    """严格配置的验证窗结构通道文件（由 revision_strict.py 落盘，缺失则重建）。"""
    dst = WORK / "revision" / "preds" / "_strict" / "struct_strict_split2_val.tsv"
    if not dst.exists():
        sa.restrict_struct(Path(v2[1]), dst, keep)
    return str(dst)


if __name__ == "__main__":
    main()
