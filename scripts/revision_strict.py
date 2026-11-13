# -*- coding: utf-8 -*-
"""严格时间完整配置下的全路线重算（表 2 主结果口径）+ 文献单源行消融。

背景：表 2 的八条路线里除仅序列与两个最近邻基线外，都用到结构通道，而结构通道全部
批次由 InterProScan 5.72-103.0 产出（数据发布 2024-11-28，晚于知识截止
T0=2023-11-08 共 386 天）。签名快照整体在截止之后，那几行数字因此只能当上界参照，
不能写成"可部署性能"。struct_library_audit.py 量化了这个代价，但只重跑了规则与学习
两条路线；本脚本把严格配置扩到表 2 的全部路线，并在同一份配对自助里给出区间。

严格配置的定义（与 struct_library_audit.py 的 datecoded_pre_T0 完全一致）：
  结构通道只保留版本串自带日期码且不晚于 T0 的签名库，清单取自
  results/revision/struct_library_audit.json 的 pre_T0_date_coded_libraries，
  该清单由批次 XML 的 <libraryManagement> version 字段实测得到，不靠人工指定。
  其余通道不动：序列通道比对的是训练库序列本身，文献通道用出版日期 <=T0 的截断重跑。

参数一律在严格配置自己的验证窗重新标定（学习路线重拟合 λ 与分数下限，各路线的 t*
用同一验证窗的 1-SE 规则重选；规则路线的 0.40/0.35/0.25 与 min_score 是设计参数，
两种配置同值），因此严格配置与全库配置的差只反映证据增减，不混杂参数漂移。

第二件事：文献单源候选行消融。把"只有文献通道给出"的候选从学习仲裁里整类删掉
（分数置 0，即 min_score 之下），与保留它们的工作点配对比较，回答"这类独立精度
不到 4% 的证据该不该进交付"。两种配置各算一次。

产物：results/revision/strict_eval.json
      work/revision/preds/_strict/            严格配置全部预测与审计文件
      work/revision/preds/_strict/litonly/    文献单源消融的预测与审计文件
分层与 MDD_80 由 scripts/revision_power.py 在 MULTIGO_CONFIG=strict 下重算。
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
from src.arbiter import Arbiter  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
from src.learning import ArbitrationModel, CandidateFeatures, term_depth  # noqa: E402
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402
import revision_bootstrap as rb  # noqa: E402
import struct_library_audit as sa  # noqa: E402
import baseline_transfer as bt  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
RES_DIR = PROJECT / "results" / "revision"
PRED_DIR = WORK / "revision" / "preds"
OUT_DIR = PRED_DIR / "_strict"
LIT_DIR = OUT_DIR / "litonly"
SEED = rl.SEED
B = int(os.environ.get("MULTIGO_BOOTSTRAP", 2000))
T0 = sa.T0
WINDOWS = ("split2_val", "split3_test")
# 与交付包同名，便于 revision_power.py / make_supplementary_tables.py 直接切换目录
ROUTES = ("fusion", "seq-only", "seq+struct", "seq+lit", "learn", "vote2",
          "interpro", "union", "nn_top3", "nn_top1")
PAIRS = [("learn", o) for o in
         ("fusion", "union", "interpro", "seq-only", "seq+struct", "seq+lit",
          "vote2", "nn_top1", "nn_top3")]


def write_pred_tsv(preds: dict, path: Path) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write("protein_id\tgo_id\tscore\n")
        for p, d in preds.items():
            for g, s in d.items():
                f.write(f"{p}\t{g}\t{s:.6f}\n")
    return str(path)


def load_preds(ev, path):
    return {p: ev._propagate_preds(g) for p, g in load_predictions(path).items()}


def pctl(lst, q=(2.5, 97.5)):
    arr = np.asarray(lst, dtype=float)
    return [round(float(np.percentile(arr, q[0])), 4),
            round(float(np.percentile(arr, q[1])), 4)]


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LIT_DIR.mkdir(parents=True, exist_ok=True)
    keep = set(json.loads((RES_DIR / "struct_library_audit.json")
                          .read_text(encoding="utf-8"))
               ["pre_T0_date_coded_libraries"])
    ch = {w: rl.channels(w) for w in WINDOWS}
    struct_strict = {}
    restriction = {"keep_libraries": sorted(keep),
                   "source": "results/revision/struct_library_audit.json::"
                             "pre_T0_date_coded_libraries"}
    for w in WINDOWS:
        dst = OUT_DIR / f"struct_strict_{w}.tsv"
        struct_strict[w] = str(dst)
        restriction[w] = sa.restrict_struct(Path(ch[w][1]), dst, keep)
    print("严格配置结构行数:", json.dumps(restriction, ensure_ascii=False),
          flush=True)

    arb0 = Arbiter(cfg, dag)
    vote = lambda ev_, go: (arb0.score(ev_, go)                        # noqa: E731
                            if len(ev_.sources) >= 2 else 0.0)

    def arbitrate(scorer, gate, win, out_dir, name, struct="strict",
                  seq=True, lit=True):
        """跑一次仲裁。struct 取 "strict"/"full"/"off"，决定结构通道用哪份文件。"""
        s, t, l = ch[win]
        sp = {"strict": struct_strict[win], "full": t, "off": ""}[struct]
        arb = Arbiter(cfg, dag)
        if scorer is not None:
            arb.scorer = scorer
        if gate is not None:
            arb.min_score = gate
        p = str(out_dir / f"predictions_{win}_{name}.tsv")
        arb.arbitrate(s if seq else "", sp, l if lit else "", p,
                      str(out_dir / f"audit_{win}_{name}.jsonl"))
        return p

    # ---- 1) 学习路线在严格配置下重新拟合（λ 与分数下限都重新标定）----
    feats = CandidateFeatures(arb0, dag, ev.ic, dict(prevalence), depth)
    samples, _ = sa.build_samples_restricted(
        arb0, feats, struct_strict["split2_val"], ch["split2_val"][2], tp2, tg2, dag)
    model = ArbitrationModel(feats, seed=SEED).fit(samples)
    gate_learn, gate_rows = sa.gate_tune_restricted(
        ev, cfg, dag, model, struct_strict["split2_val"], ch["split2_val"][2],
        tp2, tg2, "strict_learn")
    print("learn 分数下限（严格验证窗标定）=", gate_learn, flush=True)

    # ---- 2) 仲裁类路线 ----
    configs = {}
    for name, scorer, gate, struct, lit in (
            ("fusion", None, None, "strict", True),
            ("seq-only", None, None, "off", False),
            ("seq+struct", None, None, "strict", False),
            ("seq+lit", None, None, "off", True),
            ("learn", model, gate_learn, "strict", True),
            ("vote2", vote, None, "strict", True)):
        fname = "multi" if name == "fusion" else name
        configs[name] = tuple(arbitrate(scorer, gate, w, OUT_DIR, fname,
                                        struct=struct, lit=lit)
                              for w in WINDOWS)

    # ---- 3) 转移基线：InterPro/Union 的签名侧改用严格配置 ----
    for win in WINDOWS:
        base = bt.build(read_tsv(ch[win][0]), read_tsv(struct_strict[win]))
        for name in ("nn_top1", "nn_top3", "interpro", "union"):
            p = write_pred_tsv(base[name],
                               OUT_DIR / f"baseline_{name}_{win}.tsv")
            configs.setdefault(name, [None, None])[WINDOWS.index(win)] = p
    configs = {k: tuple(v) for k, v in configs.items()}

    # ---- 4) 全部路线：网格指标 + 验证窗冻结阈（1-SE）----
    arrays, points, pred_dicts, non_oracle, emitted = {}, {}, {}, {}, {}
    rng_se = np.random.default_rng(SEED)
    for name in ROUTES:
        p2, p3 = configs[name]
        preds2, preds3 = load_preds(ev, p2), load_preds(ev, p3)
        pred_dicts[name] = preds3
        arrays[name] = rb.precompute(ev, preds3, tp3, tg3)
        fmax, smin, aupr = rb.metrics_on(np.arange(len(tg3)), arrays[name])
        m3, _, _, em3 = rl.metrics_of(ev, tp3, tg3, p3)
        points[name] = (fmax, smin, aupr, ev.auc(preds3, tp3, tg3)[0])
        emitted[name] = em3
        non_oracle[name] = {
            **rl.fixed_f1(ev, tp2, tg2, preds2, tp3, tg3, preds3, rng=rng_se),
            **rl.per_aspect_calibration(ev, preds2, tp2, tg2, preds3, tp3, tg3,
                                        dag, rng=rng_se),
            "grid": {k: m3[k] for k in ("mean_terms_raw", "covered_propagated",
                                        "distinct_terms_at_fmax")}}
        print(f"{name:11s} Fmax={fmax:.4f} "
              f"F1_1se={non_oracle[name]['test_F1_one_se']:.4f} "
              f"AUPR={aupr:.4f}", flush=True)
    ti_fixed = {k: v["t_index_one_se"] for k, v in non_oracle.items()}

    # ---- 5) 同一份配对自助 ----
    rng = np.random.default_rng(SEED)
    n = len(tg3)
    boots = {k: [] for k in ROUTES}
    f1boots = {k: [] for k in ROUTES}
    deltas = {f"{a}-{b}": [] for a, b in PAIRS}
    f1deltas = {f"{a}-{b}": [] for a, b in PAIRS}
    f1_point = {name: rev.prf1(arrays[name], ti_fixed[name])[2] for name in ROUTES}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = [tg3[i] for i in idx]
        vals, f1vals = {}, {}
        for name in ROUTES:
            fmax, smin, aupr = rb.metrics_on(idx, arrays[name])
            vals[name] = (fmax, smin, aupr, ev.auc(pred_dicts[name], tp3, sub)[0])
            f1vals[name] = rev.prf1({k: a[idx] for k, a in
                                     arrays[name].items()}, ti_fixed[name])[2]
            boots[name].append(vals[name])
            f1boots[name].append(f1vals[name])
        for a, b in PAIRS:
            deltas[f"{a}-{b}"].append(tuple(vals[a][i] - vals[b][i] for i in range(4)))
            f1deltas[f"{a}-{b}"].append(f1vals[a] - f1vals[b])
        print(".", end="", flush=True)
    print(flush=True)

    def ci(lst, names=("fmax", "smin", "aupr", "auc")):
        arr = np.array(lst)
        return {f"{k}_ci": pctl(arr[:, j]) for j, k in enumerate(names)}

    routes_out = {name: {
        "fmax": round(points[name][0], 4), "fmax_ci": ci(boots[name])["fmax_ci"],
        "aupr": round(points[name][2], 4), "aupr_ci": ci(boots[name])["aupr_ci"],
        "smin": round(points[name][1], 4), "smin_ci": ci(boots[name])["smin_ci"],
        "auc": round(points[name][3], 4), "auc_ci": ci(boots[name])["auc_ci"],
        "frozen_F1": round(f1_point[name], 4),
        "frozen_F1_ci": pctl(f1boots[name]),
        **{k: v for k, v in non_oracle[name].items() if k != "val_F1_curve"}}
        for name in ROUTES}

    # ---- 6) 文献单源候选行消融（严格配置 + 全库配置各一次）----
    # 分数置 0 = 落在 min_score 之下，即整类候选不进仲裁，与 vote2 同一手法。
    full_pt = json.loads((RES_DIR / "bootstrap_learn.json")
                         .read_text(encoding="utf-8"))["point"]["learn"]
    gm = json.loads((RES_DIR / "learn_eval.json").read_text(encoding="utf-8"))
    model_full = ArbitrationModel(feats, seed=SEED).fit(
        rl.build_samples(arb0, feats, rl.collect(arb0, *ch["split2_val"]),
                         tp2, tg2)[0])

    def nolit_scorer(base):
        def f(ev_, go):
            return 0.0 if ev_.sources == {"lit"} else base(ev_, go)
        return f

    ablation = {}
    ref_path = {"strict": configs["learn"][1],
                "full": str(PRED_DIR / "predictions_split3_test_learn.tsv")}
    ref_ti = {"strict": ti_fixed["learn"],
              "full": int(round(full_pt["t_star"] * 100))}
    ref_arr = {"strict": arrays["learn"],
               "full": rb.precompute(ev, load_preds(ev, ref_path["full"]),
                                     tp3, tg3)}
    base_row = {
        "strict": {"frozen_F1": routes_out["learn"]["frozen_F1"],
                   "aupr": routes_out["learn"]["aupr"],
                   "fmax": routes_out["learn"]["fmax"],
                   "covered": routes_out["learn"]["test_covered_one_se"],
                   "t_star": routes_out["learn"]["t_star_one_se"]},
        "full": {"frozen_F1": full_pt["fixed_F1"], "aupr": full_pt["aupr"],
                 "fmax": full_pt["fmax"], "covered": None,
                 "t_star": round(ref_ti["full"] / 100, 2)}}
    for conf, m, g, struct in (("strict", model, gate_learn, "strict"),
                               ("full", model_full,
                                gm["learn_gate"]["selected"], "full")):
        paths = tuple(arbitrate(nolit_scorer(m.score), g, w, LIT_DIR,
                                f"learn_nolitonly_{conf}", struct=struct)
                      for w in WINDOWS)
        p2, p3 = paths
        preds2, preds3 = load_preds(ev, p2), load_preds(ev, p3)
        arr = rb.precompute(ev, preds3, tp3, tg3)
        fixed = rl.fixed_f1(ev, tp2, tg2, preds2, tp3, tg3, preds3,
                            rng=np.random.default_rng(SEED))
        ti_own = fixed["t_index_one_se"]
        pc, rc, fc, cov = rev.prf1(arr, ti_own)
        gmax, gsmin, gaupr = rb.metrics_on(np.arange(len(tg3)), arr)
        ti_ref = ref_ti[conf]
        base = base_row[conf]
        # 同一工作点上的对照：只在参考路线冻结的 t* 上比，排除选点漂移的混杂
        at_ref = rev.prf1(arr, ti_ref)
        ref_f1 = rev.prf1(ref_arr[conf], ti_ref)
        # 参考路线在部署 t* 上交付的"只有文献一路给出"的行数与命中率
        rows = [r for r in read_tsv(ref_path[conf]) if r["sources"] == "lit"
                and float(r["score"]) >= ti_ref / 100.0]
        hit = sum(1 for r in rows if r["go_id"] in tp3.get(r["protein_id"], set()))
        r2 = np.random.default_rng(SEED)
        dr_same = []
        for _ in range(B):
            ix = r2.integers(0, n, size=n)
            sa_ = {k: a[ix] for k, a in arr.items()}
            sb_ = {k: a[ix] for k, a in ref_arr[conf].items()}
            dr_same.append(rev.prf1(sa_, ti_ref)[2] - rev.prf1(sb_, ti_ref)[2])
        ablation[conf] = {
            "predictions": {"val": p2, "test": p3},
            "own_threshold": {"t_star": fixed["t_star_one_se"],
                              "frozen_F1": round(fc, 4),
                              "precision": round(pc, 4), "recall": round(rc, 4),
                              "covered": int(cov)},
            "aupr": round(gaupr, 4), "fmax": round(gmax, 4),
            "reference": base,
            "delta_own_vs_reference_frozen_F1": round(fc - base["frozen_F1"], 4),
            "same_t_star": {"t_star": round(ti_ref / 100, 2),
                            "ablated_F1": round(at_ref[2], 4),
                            "reference_F1": round(ref_f1[2], 4),
                            "delta": round(at_ref[2] - ref_f1[2], 4),
                            "delta_ci": pctl(dr_same),
                            "ablated_covered": int(at_ref[3]),
                            "reference_covered": int(ref_f1[3])},
            "delta_reference_at_reference_arm_own_t": round(
                rev.prf1(ref_arr[conf], ti_own)[2] - ref_f1[2], 4),
            "delta_aupr": round(gaupr - base["aupr"], 4),
            "delta_fmax": round(gmax - base["fmax"], 4),
            "lit_only_rows_in_reference_delivery": len(rows),
            "lit_only_row_precision": round(hit / len(rows), 4) if rows else None,
        }
        print("消融", conf, json.dumps(
            {k: ablation[conf][k] for k in ("own_threshold", "reference",
                                            "same_t_star",
                                            "lit_only_rows_in_reference_delivery",
                                            "lit_only_row_precision")},
            ensure_ascii=False), flush=True)

    out = {
        "T0": T0,
        "config": "datecoded_pre_T0",
        "note": "结构通道只保留版本串自带日期码且不晚于 T0 的签名库；其余通道与全库"
                "配置相同。学习路线的 λ 与分数下限、各路线的 t* 都在严格配置自己的验证"
                "窗重新标定。Fmax 为测试窗网格最大值（oracle 上界），冻结 F1 为验证窗"
                "1-SE 规则选阈值原样应用测试窗的可部署精度。",
        "restriction": restriction,
        "learn_gate": {"selected": gate_learn, "grid": gate_rows},
        "model_selection": model.selection,
        "n_val_candidates": len(samples),
        "n_val_positives": int(sum(s[3] for s in samples)),
        "routes": routes_out,
        "bootstrap": {"B": B, "seed": SEED, "n_targets": n,
                      "delta_ci": {
                          k: {"dFmax": round(float(np.mean([d[0] for d in v])), 4),
                              "dSmin": round(float(np.mean([d[1] for d in v])), 4),
                              "dAupr": round(float(np.mean([d[2] for d in v])), 4),
                              "dAuc": round(float(np.mean([d[3] for d in v])), 4),
                              "dFixed_F1": round(float(np.mean(v_f1)), 4),
                              "dFixed_F1_ci": pctl(v_f1),
                              **ci(v, ("dFmax", "dSmin", "dAupr", "dAuc"))}
                          for (k, v), v_f1 in zip(deltas.items(),
                                                  f1deltas.values())}},
        "lit_only_ablation": ablation,
    }
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "strict_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("fmax", "aupr", "smin", "auc", "frozen_F1",
                                    "frozen_F1_ci", "t_star_one_se", "test_P_one_se",
                                    "test_R_one_se", "test_covered_one_se")}
                      for k, v in out["routes"].items()},
                     ensure_ascii=False, indent=1), flush=True)
    print(json.dumps(out["bootstrap"]["delta_ci"], ensure_ascii=False, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
