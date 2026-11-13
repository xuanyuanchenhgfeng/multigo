# -*- coding: utf-8 -*-
"""修订版总评估（大修 P0/P1 重算）。

口径变化（相对初版）：
  1. GO 本体 = 2023-10-09 快照（不晚于知识截止 2023-11-08）；
  2. Smin 的 IC 仅由训练库真值统计（评估真值不参与指标构建）；
  3. 文献通道候选来自出版日期 <=2023-11-08 的截断重跑（_dcut.tsv）；
  4. 新增：验证集校准阈值 t* 在测试集固定应用的 P/R/F1；
     覆盖率/功能未知率/每蛋白预测术语数；EXP 六码真值白名单敏感性；
     4 个基线同口径重评。
序列通道产物沿用（DIAMOND 比对的是训练库序列本身，不含截止后信息）。结构域通道
不能这么说：全部批次由 InterProScan 5.72-103.0（数据发布 2024-11-28，晚于截止
386 天）产出，签名快照整体在截止之后；其影响范围与扣除后的指标由
scripts/struct_library_audit.py 量化，正文按该脚本的口径表述。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
PRED_DIR = WORK / "revision" / "preds"
RES_DIR = PROJECT / "results" / "revision"
T = 101
EXP6 = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

TAGS = {
    "split2_val": ("split2_go.tsv", ["multi"]),
    "split3_test": ("split3_go.tsv",
                    ["multi", "seq-only", "seq+struct", "seq+lit"]),
    "arabidopsis": ("arabidopsis_go_t1.tsv", ["multi"]),
}
COMBOS = {
    "multi": ("seq", "struct", "lit"),
    "seq-only": ("seq",),
    "seq+struct": ("seq", "struct"),
    "seq+lit": ("seq", "lit"),
}
BASELINES = ["nn_top1", "nn_top3", "interpro", "union"]


def channel_path(tag, ch, revised):
    if ch == "lit":
        return str(WORK / f"lit_channel_{tag}_dcut.tsv")
    return str(WORK / f"{ch}_channel_{tag}.tsv")


def build_predictions(cfg, dag):
    """用修订本体 + 截断文献重建全部仲裁预测。"""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for tag, (_truth, combos) in TAGS.items():
        for combo in combos:
            out = PRED_DIR / f"predictions_{tag}_{combo}.tsv"
            audit = PRED_DIR / f"audit_{tag}_{combo}.jsonl"
            if out.exists():
                continue
            chs = COMBOS[combo]
            arb = Arbiter(cfg, dag)
            arb.arbitrate(
                channel_path(tag, "seq", True) if "seq" in chs else "",
                channel_path(tag, "struct", True) if "struct" in chs else "",
                channel_path(tag, "lit", True) if "lit" in chs else "",
                str(out), str(audit))


def precompute(ev, preds, truth_p, targets, aspect=None):
    """per-protein×101 数组（口径同 bootstrap_ci）。"""
    if aspect:
        targets = [p for p in targets
                   if any(ev.dag.aspect(g) == aspect for g in truth_p[p])]
    n = len(targets)
    arrs = {k: np.zeros((n, T)) for k in ("prec", "mask", "rec")}
    for i, p in enumerate(targets):
        tt = truth_p[p]
        pg = preds.get(p, {})
        if aspect:
            tt = {g for g in tt if ev.dag.aspect(g) == aspect}
            pg = {g: s for g, s in pg.items() if ev.dag.aspect(g) == aspect}
        if not tt:
            continue
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
                arrs["prec"][i, ti] = inter / len(pt)
                arrs["mask"][i, ti] = 1.0
            arrs["rec"][i, ti] = inter / len(tt)
    return arrs


def prf1(arrs, ti):
    m = arrs["mask"][:, ti]
    p = float((arrs["prec"][:, ti] * m).sum() / m.sum()) if m.sum() else 0.0
    r = float(arrs["rec"][:, ti].mean())
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, float(m.sum())


def calibration(ev, cfg):
    """验证集 t*（overall/MFO/BPO/CCO）固定应用于测试集的 P/R/F1。"""
    truth2 = load_truth(str(PROJECT / "data/annotations/split2_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=ev.dag)
    truth3 = load_truth(str(PROJECT / "data/annotations/split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=ev.dag)
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    pv = {p: ev._propagate_preds(g) for p, g in
          load_predictions(str(PRED_DIR / "predictions_split2_val_multi.tsv")).items()}
    pt = {p: ev._propagate_preds(g) for p, g in
          load_predictions(str(PRED_DIR / "predictions_split3_test_multi.tsv")).items()}
    tg2 = [p for p in tp2 if tp2[p]]
    tg3 = [p for p in tp3 if tp3[p]]
    out = {}
    for scope, asp in (("overall", None), ("MFO", "F"), ("BPO", "P"), ("CCO", "C")):
        a2 = precompute(ev, pv, tp2, tg2, asp)
        a3 = precompute(ev, pt, tp3, tg3, asp)
        fvals = [prf1(a2, ti)[2] for ti in range(T)]
        t_star = int(np.argmax(fvals))
        pv_, rv_, fv_ = prf1(a2, t_star)[:3]
        pc, rc, fc, ncover = prf1(a3, t_star)
        fo, to = prf1(a3, int(np.argmax([prf1(a3, ti)[2] for ti in range(T)])))[:3:2]
        # oracle 三元组
        oracle = max(((prf1(a3, ti)[2], ti) for ti in range(T)))
        out[scope] = {"t_star": round(t_star / 100, 2),
                      "val_F1_at_tstar": round(fv_, 4),
                      "test_fixed_P": round(pc, 4),
                      "test_fixed_R": round(rc, 4),
                      "test_fixed_F1": round(fc, 4),
                      "test_fixed_covered": ncover,
                      "test_oracle_Fmax": round(oracle[0], 4),
                      "test_oracle_t": round(oracle[1] / 100, 2)}
    return out


def coverage_stats(ev):
    """测试集四配置：原始覆盖、Fmax 阈值下每蛋白预测数、seq 无命中率。"""
    truth = load_truth(str(PROJECT / "data/annotations/split3_go.tsv"),
                       exclude_iea=ev.exclude_iea, dag=ev.dag)
    tp = {p: ev._prop(g) for p, g in truth.items()}
    targets = [p for p in tp if tp[p]]
    seq_hits = {r["protein_id"] for r in
                read_tsv(str(WORK / "seq_channel_split3_test.tsv"))}
    out = {"n_truth_targets": len(targets),
           "n_query_proteins": sum(1 for _ in
                                   read_tsv(str(WORK / "meta_split3_test.tsv"))),
           "seq_no_hit_targets": len([p for p in targets if p not in seq_hits])}
    for combo in ("multi", "seq-only", "seq+struct", "seq+lit"):
        raw = load_predictions(str(PRED_DIR /
                                   f"predictions_split3_test_{combo}.tsv"))
        preds = {p: ev._propagate_preds(g) for p, g in raw.items()}
        asp = None
        arrs = precompute(ev, preds, tp, targets, asp)
        ti_fmax = int(np.argmax([prf1(arrs, ti)[2] for ti in range(T)]))
        m = arrs["mask"][:, ti_fmax]
        n_any = sum(1 for p in targets if preds.get(p))
        # 每蛋白输出术语数（Fmax 阈值，未传播口径=仲裁直接输出）
        nterms = [len([g for g, s in raw.get(p, {}).items()
                       if s >= ti_fmax / 100]) for p in targets]
        out[combo] = {
            "raw_coverage": round(n_any / len(targets), 4),
            "fmax_threshold": round(ti_fmax / 100, 2),
            "covered_at_fmax": int(m.sum()),
            "mean_terms_per_protein_at_fmax": round(float(np.mean(nterms)), 2),
            "mean_terms_among_covered": round(
                float(np.mean([x for x in nterms if x > 0])) if any(nterms) else 0, 2),
        }
    return out


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    build_predictions(cfg, dag)

    train = str(PROJECT / "data/annotations/split1_go.tsv")
    summary = {}

    # ---- 三目标集 × 配置 ----
    for tag, (truth_name, combos) in TAGS.items():
        truth = str(PROJECT / "data/annotations" / truth_name)
        for combo in combos:
            pred = str(PRED_DIR / f"predictions_{tag}_{combo}.tsv")
            rep = ev.evaluate(pred, truth, f"{tag}-{combo}", train_tsv=train)
            summary[f"{tag}|{combo}"] = rep

    # ---- 基线（12 库结构域产物，新本体+train-only IC 重评）----
    for b in BASELINES:
        pred = str(WORK / f"baseline_{b}_split3_test.tsv")
        rep = ev.evaluate(pred, str(PROJECT / "data/annotations/split3_go.tsv"),
                          f"baseline-{b}", train_tsv=train)
        summary[f"split3_test|baseline_{b}"] = rep

    # ---- EXP 六码敏感性（测试集四配置）----
    exp_sens = {}
    for combo in ("multi", "seq-only", "seq+struct", "seq+lit"):
        pred = str(PRED_DIR / f"predictions_split3_test_{combo}.tsv")
        rep = ev.evaluate(pred, str(PROJECT / "data/annotations/split3_go.tsv"),
                          f"exp6-{combo}", train_tsv=train,
                          include_codes=EXP6)
        exp_sens[combo] = {k: rep[k] for k in
                           ("fmax", "smin", "aupr", "auc", "coverage", "n_targets")}
    (RES_DIR / "exp6_sensitivity.json").write_text(
        json.dumps(exp_sens, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 固定阈值校准 & 覆盖统计 ----
    calib = calibration(ev, cfg)
    cover = coverage_stats(ev)
    (RES_DIR / "calibration.json").write_text(
        json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    (RES_DIR / "coverage_stats.json").write_text(
        json.dumps(cover, ensure_ascii=False, indent=2), encoding="utf-8")

    (RES_DIR / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 关键结果 ===")
    for k, v in summary.items():
        print(f"{k:42s} Fmax={v['fmax']:.4f} Smin={v['smin']:.4f} "
              f"AUPR={v['aupr']:.4f} AUC={v['auc']:.4f} cov={v['coverage']:.3f}")
    print("\n校准:", json.dumps(calib, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
