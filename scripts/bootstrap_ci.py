# -*- coding: utf-8 -*-
"""按蛋白 paired bootstrap 置信区间（投稿统计证据）。

对 split3_test 四组消融（multi / seq-only / seq+struct / seq+lit）的
Fmax / Smin / AUPR 计算按蛋白有放回重采样的 95% CI（percentile 法），
并对配对差值（3-Channel vs 单/双Channel）给出差值 CI 与 P(delta>0)。

口径与 evaluator.fmax_smin_aupr 完全一致（蛋白中心）：
  precision = 有预测蛋白的per-protein精度均值；recall = 全部真值蛋白均值
  预测沿 is_a/part_of 传播、去三根；IC 固定用全量(train+eval)真值计算
  （知识库属性，不随重采样变化，与 PSB26 协议一致）
  AUPR = 101阈值 (recall,precision) 按recall排序梯形积分
实现：预计算每蛋白×101阈值 (prec, mask, rec, ru, mi) 数组，重采样仅 numpy 聚合。
用法: nohup python3 -X utf8 scripts/bootstrap_ci.py > logs/bootstrap_ci.log 2>&1 &
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402  打破循环导入
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
B = 2000
SEED = 20260917

GROUPS = {
    "multi": "work/predictions_split3_test.tsv",
    "seq-only": "work/predictions_split3_test_seq-only.tsv",
    "seq+struct": "work/predictions_split3_test_seq+struct.tsv",
    "seq+lit": "work/predictions_split3_test_seq+lit.tsv",
}
PAIRS = [("multi", "seq-only"), ("multi", "seq+struct"), ("multi", "seq+lit")]
SCOPES = [("overall", None), ("MFO", "F"), ("BPO", "P"), ("CCO", "C")]


def precompute(ev, pred_tsv, truth_tsv, train_tsv, aspect=None):
    """返回 per-protein×101 数组字典（蛋白顺序由真值文件决定，跨组一致）。"""
    preds_raw = load_predictions(pred_tsv)
    truth = load_truth(truth_tsv, exclude_iea=ev.exclude_iea, dag=ev.dag)
    preds = {p: ev._propagate_preds(g) for p, g in preds_raw.items()}
    truth_p = {p: ev._prop(g) for p, g in truth.items()}

    train_truth = load_truth(train_tsv, exclude_iea=ev.exclude_iea, dag=ev.dag)
    ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()] +
                       list(truth_p.values()))

    targets = [p for p in truth_p if truth_p[p]]
    if aspect:
        targets = [p for p in targets
                   if any(ev.dag.aspect(g) == aspect for g in truth_p[p])]
    T = 101
    n = len(targets)
    arrs = {k: np.zeros((n, T)) for k in ("prec", "mask", "rec", "ru", "mi")}
    for i, p in enumerate(targets):
        tt = truth_p[p]
        pg = preds.get(p, {})
        if aspect:
            tt = {g for g in tt if ev.dag.aspect(g) == aspect}
            pg = {g: s for g, s in pg.items() if ev.dag.aspect(g) == aspect}
        if not tt:
            continue
        # 阈值分桶：ti = floor(s*100+eps)，pt(t=ti/100) = U_{tj>=ti} by_t[tj]
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
                for g in pt - tt:
                    arrs["mi"][i, ti] += ic.get(g, 0.0)
            for g in tt - pt:
                arrs["ru"][i, ti] += ic.get(g, 0.0)
            arrs["rec"][i, ti] = inter / len(tt)
    return arrs


def aggregate(arrs, idx):
    """蛋白索引 idx 聚合 (fmax, smin, aupr)。idx: (n,) 或 (B,n)。"""
    axis = idx.ndim - 1
    prec, mask = arrs["prec"][idx], arrs["mask"][idx]
    rec, ru, mi = arrs["rec"][idx], arrs["ru"][idx], arrs["mi"][idx]
    p_n = mask.sum(axis=axis)
    precision = np.where(p_n > 0, (prec * mask).sum(axis=axis)
                        / np.maximum(p_n, 1), 0.0)
    recall = rec.mean(axis=axis)
    f = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    s = np.sqrt(ru.mean(axis=axis) ** 2 + mi.mean(axis=axis) ** 2)
    order = np.argsort(recall, axis=axis, kind='stable')
    r_s = np.take_along_axis(recall, order, axis=axis)
    p_s = np.take_along_axis(precision, order, axis=axis)
    dx = np.diff(r_s, axis=axis)
    aupr = np.sum(np.where(dx > 0, dx * (p_s[..., :-1] + p_s[..., 1:]) / 2, 0),
                  axis=axis)
    return f.max(axis=axis), s.min(axis=axis), aupr


def main():
    cfg = load_config(str(PROJECT / "config.yaml"))
    log = setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    truth_tsv = str(PROJECT / "data/annotations/split3_go.tsv")
    train_tsv = str(PROJECT / "data/annotations/split1_go.tsv")

    cache = {}
    for gname, ppath in GROUPS.items():
        for sname, aspect in SCOPES:
            arrs = precompute(ev, str(PROJECT / ppath), truth_tsv,
                              train_tsv, aspect)
            n = arrs["prec"].shape[0]
            cache[(gname, sname)] = arrs
            log.info("预计算完成: %s/%s (n=%d)", gname, sname, n)

    rng = np.random.default_rng(SEED)
    out = {"protocol": f"paired bootstrap by protein, B={B}, seed={SEED}, "
                       f"percentile 95% CI, IC固定(全量train+eval真值)",
           "group": {}, "paired_delta": {}}
    for (gname, sname), arrs in cache.items():
        n = arrs["prec"].shape[0]
        idx0 = np.arange(n)
        f0, s0, a0 = aggregate(arrs, idx0)
        idx = rng.integers(0, n, size=(B, n))
        fb, sb, ab = aggregate(arrs, idx)
        out["group"].setdefault(gname, {})[sname] = {
            "n_proteins": n,
            "fmax": round(float(f0), 4),
            "fmax_ci95": [round(float(np.percentile(fb, 2.5)), 4),
                          round(float(np.percentile(fb, 97.5)), 4)],
            "smin": round(float(s0), 4),
            "smin_ci95": [round(float(np.percentile(sb, 2.5)), 4),
                          round(float(np.percentile(sb, 97.5)), 4)],
            "aupr": round(float(a0), 4),
            "aupr_ci95": [round(float(np.percentile(ab, 2.5)), 4),
                         round(float(np.percentile(ab, 97.5)), 4)],
        }
        log.info("[%s/%s] Fmax=%.4f [%.4f,%.4f] AUPR=%.4f [%.4f,%.4f]",
                 gname, sname, f0,
                 *out["group"][gname][sname]["fmax_ci95"], a0,
                 *out["group"][gname][sname]["aupr_ci95"])

    # 配对差值（overall 口径；同一批重采样索引）
    rng = np.random.default_rng(SEED)
    for a_name, b_name in PAIRS:
        arrs_a, arrs_b = cache[(a_name, "overall")], cache[(b_name, "overall")]
        n = arrs_a["prec"].shape[0]
        idx = rng.integers(0, n, size=(B, n))
        fa, _, aa = aggregate(arrs_a, idx)
        fb, _, ab = aggregate(arrs_b, idx)
        df, da = fa - fb, aa - ab
        out["paired_delta"][f"{a_name} - {b_name}"] = {
            "delta_fmax_mean": round(float(df.mean()), 4),
            "delta_fmax_ci95": [round(float(np.percentile(df, 2.5)), 4),
                                round(float(np.percentile(df, 97.5)), 4)],
            "delta_aupr_mean": round(float(da.mean()), 4),
            "delta_aupr_ci95": [round(float(np.percentile(da, 2.5)), 4),
                                round(float(np.percentile(da, 97.5)), 4)],
            "p_delta_fmax_gt0": round(float((df > 0).mean()), 4),
            "p_delta_aupr_gt0": round(float((da > 0).mean()), 4),
        }
        log.info("[paired %s - %s] dFmax=%.4f dAUPR=%.4f",
                 a_name, b_name, df.mean(), da.mean())

    out_path = PROJECT / "results" / "bootstrap_ci.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("bootstrap 完成 -> %s", out_path)


if __name__ == "__main__":
    main()
