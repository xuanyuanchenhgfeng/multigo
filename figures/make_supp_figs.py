# -*- coding: utf-8 -*-
"""生成论文4张补充图（300dpi 灰度 PNG，期刊非彩色要求）。

数据全部来自本仓库 `results/revision/*.json` 和 `work/revision/preds/*.tsv`：
  supp_fig1  PR 曲线            —— 重算 revision predictions_split3_test_multi.tsv
  supp_fig2  仲裁权重敏感性热图  —— results/revision/weight_search_val.json
  supp_fig3  拟南芥跨物种分本体  —— results/revision/eval_summary.json (arabidopsis|multi)
  supp_fig4  GO-DAG 冲突/冗余    —— results/revision/split3_test_arbiter_conflict_stats.json

四张补充图都要读 `work/revision/preds/`、`work/revision/*_multi.tsv` 与 GO 的 OBO 文件，
这三样是重建产物、不入版本库：跑之前先按 §8 步骤 1–2 生成，或取 Release 附件
multigo_data_v1.0.tar.gz 解开后把工程根指过来。输出目录同 make_figures.py。
"""
import ast
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

CJK_FONT = os.environ.get("MULTIGO_CJK_FONT",
                          str(Path.home() / ".fonts/NotoSansSC.ttf"))
if Path(CJK_FONT).is_file():        # 图面标签全为西文，中文字体只作兜底
    fm.fontManager.addfont(CJK_FONT)
from journal_fonts import use_times
FONT_FAMILY = use_times(plt)

OUT = Path(os.environ.get("MULTIGO_FIG_OUT") or Path(__file__).resolve().parent)
DPI = 300
PROJ = Path(__file__).resolve().parents[1]
RES_REV = PROJ / "results" / "revision"
PRED_REV = PROJ / "work" / "revision" / "preds"

sys.path.insert(0, str(PROJ))
import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402


def _eval_context():
    cfg = load_config(str(PROJ / "config_revision.yaml"))
    dag = GODag(str(PROJ / cfg["paths"]["go_obo"]),
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    return cfg, dag, ev


CFG = load_config(str(PROJ / "config_revision.yaml"))


def _sweep_pr(ev, preds, truth):
    """101 阈值网格扫描，返回 (recalls, precisions)。"""
    targets = [p for p in truth if truth[p]]
    n = len(targets)
    recalls, precisions = [], []
    for t in ev.thresholds:
        p_sum, p_n = 0.0, 0
        r_sum = 0.0
        for p in targets:
            tt = truth[p]
            pt = {g for g, s in preds.get(p, {}).items() if s >= t}
            inter = len(pt & tt)
            if pt:
                p_sum += inter / len(pt)
                p_n += 1
            r_sum += inter / len(tt)
        precisions.append(p_sum / p_n if p_n else 0.0)
        recalls.append(r_sum / n)
    return recalls, precisions


def _aupr(recalls, precisions):
    order = sorted(range(len(recalls)), key=lambda i: recalls[i])
    v = 0.0
    for a, b in zip(order, order[1:]):
        dx = recalls[b] - recalls[a]
        if dx > 0:
            v += dx * (precisions[a] + precisions[b]) / 2.0
    return v


def fmt_num(v, nd):
    """与正文一致的十舍十入（ROUND_HALF_UP），避免图与表末位差 0.001。"""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd),
                                                rounding=ROUND_HALF_UP))

# ---------------- 补充图1 PR 曲线（split3_test revision 预测） ----------------
def fig1_pr():
    _, _, ev = _eval_context()
    preds_raw = load_predictions(str(PRED_REV / "predictions_split3_test_multi.tsv"))
    truth = load_truth(str(PROJ / "data/annotations/split3_go.tsv"),
                       exclude_iea=ev.exclude_iea, dag=ev.dag)
    preds = {p: ev._propagate_preds(g) for p, g in preds_raw.items()}
    truth_p = {p: ev._prop(g) for p, g in truth.items()}

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    styles = [("Overall", None, "0.05", "-"),
              ("MFO", "F", "0.28", "-"),
              ("BPO", "P", "0.48", "--"),
              ("CCO", "C", "0.68", ":")]
    for label, aspect, g, ls in styles:
        if aspect is None:
            pr, tr = preds, truth_p
        else:
            pr = {p: {g2: s for g2, s in gos.items() if ev.dag.aspect(g2) == aspect}
                  for p, gos in preds.items()}
            pr = {p: g for p, g in pr.items() if g}
            tr = {p: {g2 for g2 in gos if ev.dag.aspect(g2) == aspect}
                  for p, gos in truth_p.items()}
            tr = {p: g for p, g in tr.items() if g}
        r, p = _sweep_pr(ev, pr, tr)
        a = _aupr(r, p)
        ax.plot(r, p, color=g, lw=1.8, ls=ls,
                label=f"{label}  (AUPR={a:.3f})")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Protein-centric PR curves (test set)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "supp_fig1_pr.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("supp_fig1_pr.png 生成完成")


# ---------------- 补充图2 仲裁权重敏感性热图 ----------------
def fig2_weights():
    ws = json.load(open(RES_REV / "weight_search_val.json", encoding="utf-8"))
    aspects = [("MFO", "molecular function"), ("BPO", "biological process"),
               ("CCO", "cellular component")]
    W = CFG["arbiter"]["weights"]
    step = 0.05
    n = int(round(1 / step)) + 1
    chosen = (W["seq"], W["struct"])
    fig = plt.figure(figsize=(9.2, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.07],
                          left=0.07, right=0.93, top=0.78, bottom=0.17,
                          wspace=0.32)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for ax in axes[1:]:
        ax.sharex(axes[0]); ax.sharey(axes[0])
    im = None
    grids = {}
    for key, _ in aspects:
        g = np.full((n, n), np.nan)
        for k, v in ws.items():
            seq, struct, _ = ast.literal_eval(k)
            if seq + struct > 1.0 + 1e-9:
                continue
            si = int(round(seq / step)); sti = int(round(struct / step))
            g[sti, si] = v[key]
        grids[key] = g
    finite = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    vmin, vmax = float(finite.min()), float(finite.max())
    for ax, (key, name) in zip(axes, aspects):
        grid = grids[key]
        im = ax.imshow(grid, origin="lower", cmap="Greys",
                       extent=(-0.025, 1.025, -0.025, 1.025),
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.plot(chosen[0], chosen[1], marker="*", color="white",
                markersize=17, zorder=5, markeredgewidth=0)
        ax.plot(chosen[0], chosen[1], marker="*", color="black",
                markersize=13, zorder=6)
        ax.set_title(f"{name} ({key})\ngrid {np.nanmin(grid):.2f}-{np.nanmax(grid):.2f}",
                     fontsize=9)
        ax.set_xlabel("Sequence weight w_seq", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Structure weight w_struct", fontsize=8)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Validation Fmax", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(f"Arbiter weight grid search (validation Fmax; * = empirical "
                 f"{W['seq']:.2f}/{W['struct']:.2f}/{W['lit']:.2f}; "
                 f"w_lit = 1 - w_seq - w_struct)",
                 fontsize=9.5, y=0.93)
    fig.savefig(OUT / "supp_fig2_weight_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("supp_fig2_weight_heatmap.png 生成完成")


# ---------------- 补充图3 拟南芥同物种留出 vs 真跨物种 ----------------
def fig3_arabidopsis():
    ev = json.load(open(RES_REV / "eval_summary.json", encoding="utf-8"))
    cs = json.load(open(RES_REV / "eval_cross_species.json", encoding="utf-8"))
    hold = ev["arabidopsis|multi"]
    cross = cs["multi_noath"]
    regimes = [("Same-species hold-out", hold),
               ("Genuine cross-species", cross)]
    metrics = [("fmax", "Fmax"), ("aupr", "AUPR"), ("smin", "Smin")]
    onts = ["Overall", "MFO", "BPO", "CCO"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, (fld, title) in zip(axes, metrics):
        x = np.arange(len(onts)); w = 0.36
        vals = []
        for i, (label, res) in enumerate(regimes):
            v = ([res[fld]] + [res["by_aspect"][a][fld] for a in "FPC"])
            vals.append(v)
            ax.bar(x + (i - 0.5) * w, v, w, color=["0.18", "0.72"][i],
                   edgecolor="0.05", hatch=["", "//"][i],
                   label=label)
        for xi, pair in enumerate(zip(*vals)):
            for i, v in enumerate(pair):
                ax.annotate(fmt_num(v, 2 if fld == "smin" else 3),
                            xy=(xi + (i - 0.5) * w, v), xytext=(0, 2),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.2, color="0.05",
                            rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(onts, fontsize=7.2)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, max(max(v) for v in vals) * 1.32)
        ax.tick_params(labelsize=7.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Metric value", fontsize=8)
    fig.legend(handles=axes[0].containers,
               labels=[r[0] for r in regimes], fontsize=7.2, frameon=False,
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               handlelength=1.4, columnspacing=1.4)
    fig.suptitle("Ontology-specific performance of the Arabidopsis external\n"
                 "validations (fusion; library keeps 70% of Arabidopsis vs none;\n"
                 "lower Smin is better)",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(OUT / "supp_fig3_arabidopsis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("supp_fig3_arabidopsis.png 生成完成")


# ---------------- 补充图4 GO-DAG 冲突/冗余统计 ----------------
def fig4_dag():
    # 审计计数改读 audit_stats.json：单文件冲突统计只到旧检索库批次为止
    st = json.load(open(RES_REV / "audit_stats.json",
                        encoding="utf-8"))["测试集·三通道融合"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    cats = ["Cross-namespace\nconflict", "Branch conflict",
            "Parent-child\nfolding", "Same-term merge"]
    vals = [st["namespace_conflicts"], st["branch_conflicts"],
            st["parent_child_collapses"], st["same_merges"]]
    ypos = np.arange(len(cats))[::-1]
    bars = ax.barh(ypos, vals, 0.6, color="0.28", edgecolor="0.05")
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.02, y, f"{v:,}", va="center", fontsize=7)
    ax.set_yticks(ypos); ax.set_yticklabels(cats, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)", fontsize=8)
    ax.set_title("(a) Topological arbitration statistics (test set)", fontsize=9)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    bf = st["branch_conflict_by_aspect"]
    onts = ["MFO (F)", "BPO (P)", "CCO (C)"]
    bvals = [bf["F"], bf["P"], bf["C"]]
    x = np.arange(3)
    ax.bar(x, bvals, 0.55, color="0.5", edgecolor="0.05")
    for xi, v in enumerate(bvals):
        ax.text(xi, v + max(bvals) * 0.02, f"{v:,}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(onts, fontsize=8)
    ax.set_ylabel("Number of branch conflicts", fontsize=8)
    ax.set_title("(b) Branch conflicts by ontology", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "supp_fig4_dag.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("supp_fig4_dag.png 生成完成")


if __name__ == "__main__":
    # supp_fig1 要现算 PR 曲线，依赖三样重建产物（不入版本库）；其余三张只读入库 JSON
    need = [PROJ / CFG["paths"]["go_obo"],
            PRED_REV / "predictions_split3_test_multi.tsv",
            PROJ / "data" / "annotations" / "split3_go.tsv"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        print("跳过 supp_fig1，缺少：", *missing, sep="\n  ")
    else:
        fig1_pr()
    fig2_weights()
    fig3_arabidopsis()
    fig4_dag()
    print("ALL 4 SUPP FIGURES DONE ->", OUT)
