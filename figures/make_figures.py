# -*- coding: utf-8 -*-
"""生成论文灰度主图（fig1–fig3）与补充图 S5（fig5），300dpi PNG，期刊非彩色要求。

所有评估数字从本仓库 `results/revision/*.json` 动态读取，确保与
revision_evaluate / revision_bootstrap 结果完全一致；脚本内不写死任何指标值，
也不向仓库工作区写文件（真值统计直接读 results/revision/truth_stats.json）。

用法：python figures/make_figures.py [1 2 3 5]     缺省为全部。
输出目录默认为脚本所在目录，可用 MULTIGO_FIG_OUT 指到随稿提交的 figures/。
图 2 需要 results/revision/truth_stats.json（入库），缺该文件才会现读
data/partitions 与 data/annotations，那两处是重建产物、不在版本库内。
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

CJK_FONT = os.environ.get("MULTIGO_CJK_FONT",
                          str(Path.home() / ".fonts/NotoSansSC.ttf"))
if Path(CJK_FONT).is_file():        # 图面标签全为西文，中文字体只作兜底
    fm.fontManager.addfont(CJK_FONT)
from journal_fonts import use_times
FONT_FAMILY = use_times(plt)

OUT = Path(os.environ.get("MULTIGO_FIG_OUT") or Path(__file__).resolve().parent)
PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "revision"
PRED = PROJ / "work" / "revision" / "preds"
DPI = 300

sys.path.insert(0, str(PROJ))
from src.config import load_config  # noqa: F401,E402

def strict_view():
    """把严格时间完整配置的三份结果拼成 bootstrap_learn.json 的形状（图 3 用）。

    strict_eval.json 的点估计与区间在 routes / bootstrap 两块，字段名不同；
    线索交付行（表 2 第九行）由 scripts/revision_litonly_ci.py 落在
    litonly_ci.json。只改字段名，不重算指标。
    """
    se = json.load(open(RES / "strict_eval.json", encoding="utf-8"))
    li = json.load(open(RES / "litonly_ci.json", encoding="utf-8"))
    point, ci, f1ci = {}, {}, {}
    for k, v in se["routes"].items():
        point[k] = {"fmax": v["fmax"], "smin": v["smin"], "aupr": v["aupr"],
                    "auc": v["auc"], "fixed_F1": v["frozen_F1"],
                    "t_star": v["t_star_one_se"],
                    "F1_argmax": v["test_F1_argmax"],
                    "t_star_argmax": v["t_star_argmax"]}
        ci[k] = {f"{m}_ci": v[f"{m}_ci"] for m in ("fmax", "smin", "aupr", "auc")}
        f1ci[k] = v["frozen_F1_ci"]
    g = li["grid"]["learn_nolitonly"]
    key = "learn_nolitonly"
    point[key] = {"fmax": g["fmax"], "smin": g["smin"], "aupr": g["aupr"],
                  "auc": g["auc"], "fixed_F1": li["frozen_F1"][key]["point"],
                  "t_star": li["t_star"][key]}
    ci[key] = {f"{m}_ci": v for m, v in li["grid_ci_learn_nolitonly"].items()}
    f1ci[key] = li["frozen_F1"][key]["ci95"]
    delta = dict(se["bootstrap"]["delta_ci"])
    delta["learn_nolitonly-nn_top1"] = {
        "dFixed_F1": li["paired_delta"]["learn_nolitonly-nn_top1"]["delta"],
        "dFixed_F1_ci": li["paired_delta"]["learn_nolitonly-nn_top1"]["ci95"]}
    bb = se["bootstrap"]
    return {"B": bb["B"], "seed": bb["seed"], "n_targets": bb["n_targets"],
            "note": se["note"], "point": point, "ci": ci,
            "fixed_F1_ci": f1ci, "delta_ci": delta}


# ---------------- 读取 revision 结果 ----------------
# FIG_CONFIG=strict 把图 1 与图 3 换成表 2 主结果口径（严格时间完整配置）。
# 严格口径的数字由 scripts/revision_strict.py / revision_power.py /
# revision_litonly_ci.py 写出，本脚本只做字段改名与拼接，不重算任何指标。
FIG_CONFIG = os.environ.get("FIG_CONFIG", "full")
SUFFIX = "" if FIG_CONFIG == "full" else "_strict"

EVAL = json.load(open(RES / "eval_summary.json", encoding="utf-8"))
BOOT = json.load(open(RES / "bootstrap_ci.json", encoding="utf-8"))
BPO = json.load(open(RES / "bpo_analysis.json", encoding="utf-8"))
BOOTL = json.load(open(RES / "bootstrap_learn.json", encoding="utf-8"))
POWER = json.load(open(RES / "power_analysis.json", encoding="utf-8"))
LEARNEV = json.load(open(RES / "learn_eval.json", encoding="utf-8"))
if FIG_CONFIG == "strict":
    MECH = json.load(open(RES / "strict_mech.json", encoding="utf-8"))
    LEARNEV = {**LEARNEV, "features": MECH["feature_names"],
               "n_val_candidates": MECH["n_val_candidates"],
               "n_val_positives": MECH["n_val_positives"]}
    POWER = json.load(open(RES / "power_analysis_strict.json", encoding="utf-8"))
    BOOTL = strict_view()

CFG = load_config(str(PROJ / "config_revision.yaml"))
ASPECTS = ("F", "P", "C")
SPLIT_FILES = {"train": "split1", "val": "split2", "test": "split3"}
# 图5 曲线标签：_curves() 生成、fig5() 按字符串匹配取用，必须完全一致
LAB_MFO = "molecular function (MFO)"
LAB_BPO = "biological process (BPO)"
LAB_CCO = "cellular component (CCO)"


def _fasta_count(name):
    with open(PROJ / "data/partitions" / f"{name}.fasta") as fh:
        return sum(1 for ln in fh if ln.startswith(">"))


def truth_stats():
    """表 1 口径的真值统计：2023-10-09 快照内、白名单证据码的直接注释。

    返回并集术语数、各划分各本体的真值蛋白数与注释条目数，并落盘为
    results/revision/truth_stats.json 供正文与补充材料引用。
    该文件已由 revision 流程产出，作图时一律直接读取，不再重算、不再回写。
    """
    from src.go_dag import GODag
    from src.evaluator import load_truth
    dag = GODag(str(PROJ / CFG["paths"]["go_obo"]),
                relations=CFG.get("go_dag", {}).get("relations"))
    wl = set(CFG["split"]["evidence_codes"])
    union = {a: set() for a in ASPECTS}
    prot, ent = {}, {}
    for split, base in SPLIT_FILES.items():
        t = load_truth(str(PROJ / f"data/annotations/{base}_go.tsv"),
                       exclude_iea=True, dag=dag, include_codes=wl)
        for p, gos in t.items():
            for g in gos:
                a = dag.aspect(g)
                union[a].add(g)
                prot.setdefault((split, a), set()).add(p)
                ent[(split, a)] = ent.get((split, a), 0) + 1
    out = {"protocol": ("直接注释、13 种白名单证据码、"
                        f"{CFG['paths']['go_obo']}"),
           "go_terms": len(dag.terms),
           "union_terms": {a: len(union[a]) for a in ASPECTS},
           "truth_proteins": {f"{s}|{a}": len(prot[(s, a)])
                              for s in SPLIT_FILES for a in ASPECTS},
           "truth_entries": {f"{s}|{a}": ent[(s, a)]
                             for s in SPLIT_FILES for a in ASPECTS},
           "dedup_proteins": {s: len(set().union(
               *[prot[(s, a)] for a in ASPECTS])) for s in SPLIT_FILES},
           "fasta_counts": {k: _fasta_count(v) for k, v in
                            (("train", "split1"), ("val", "split2"),
                             ("test", "split3"), ("ath_lib", "arabidopsis_lib"),
                             ("ath_val", "arabidopsis_val"))}}
    return out


TS = (json.load(open(RES / "truth_stats.json", encoding="utf-8"))
      if (RES / "truth_stats.json").exists() else truth_stats())


def fmt(s):
    """20231108 -> 2023-11-08"""
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def iso(s):
    """2023-11-08 -> 2023-11-08"""
    return s.replace("/", "-")

def fmt_num(v, nd):
    """与正文一致的十舍十入（ROUND_HALF_UP），避免图与表末位差 0.001。"""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd),
                                                rounding=ROUND_HALF_UP))


# 表 2 的路线（seq+struct 与 seq+lit 两条消融配置只列于补充表 S4/S23，不入上排）；
# 严格口径的表 2 多一行"文献单源候选行改线索交付"，图 3 与表同源，因此也多一根条。
ROUTES = ("seq-only", "interpro", "nn_top1", "nn_top3", "union", "vote2",
          "fusion", "learn")
if FIG_CONFIG == "strict":
    ROUTES = ROUTES + ("learn_nolitonly",)
ROUTE_LABEL = {"seq-only": "Sequence only", "nn_top3": "NN Top3",
               "union": "Naive union", "nn_top1": "NN Top1",
               "interpro": "InterPro direct", "vote2": "Vote ≥2",
               "fusion": "Rule-based (this study)", "learn": "Learned (this study)",
               "learn_nolitonly": "Learned, lit-only rows as leads",
               "seq+struct": "Seq+struct", "seq+lit": "Seq+lit"}
DELTA_LABEL = {"fusion": "learned − rule-based", "union": "learned − naive union",
               "interpro": "learned − InterPro direct",
               "seq-only": "learned − sequence only",
               "seq+struct": "learned − sequence + structure",
               "vote2": "learned − vote ≥2", "nn_top1": "learned − NN Top1",
               "nn_top3": "learned − NN Top3"}
GRAY = {"learn": "0.12", "fusion": "0.38", "learn_nolitonly": "0.24",
        "rest": "0.62"}
# 森林图里除"学习仲裁 − 各路线"之外单列的对照（严格口径才有）
EXTRA_DELTA = ({"learn_nolitonly-nn_top1":
                "delivery arm \u2212 NN Top1"} if FIG_CONFIG == "strict" else {})

# ============================================================
# 图1 系统架构（纯示意图，无数据依赖）
# ============================================================
def fig1():
    n_tr = TS["fasta_counts"]["train"]
    n_va = TS["fasta_counts"]["val"]
    n_te = TS["fasta_counts"]["test"]
    t0 = fmt(str(CFG["split"]["train_end"]))
    t1 = fmt(str(CFG["split"]["val_end"]))
    t2 = fmt(str(CFG["split"]["test_end"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, label, sub, g="0.92", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                     fc=g, ec="0.1", lw=1.0, ls=ls))
        ax.text(x+w/2, y+h*0.74, label, ha="center", va="center", fontsize=9.5)
        ax.text(x+w/2, y+h*0.36, sub, ha="center", va="center",
                fontsize=6.8, color="0.25", linespacing=1.45)

    def arr(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=11, color="0.1", lw=1.1))

    def mini(x, y, w, h, lines, g="0.98", fs=6.0):
        """仲裁层内的可切换打分函数小方框（多行正文，无副标题）。"""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035",
                     fc=g, ec="0.1", lw=0.8))
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=fs, color="0.1", linespacing=1.5)

    ax.text(5, 6.78, "Query protein sequences (Swiss-Prot train / val / test; "
                     "Arabidopsis external set)",
            ha="center", fontsize=8.5)
    seq = CFG["seq_channel"]
    ev_exp = -int(round(np.log10(seq["evalue"])))
    box(0.2, 4.75, 2.9, 1.25, "Sequence channel",
        "DIAMOND --sensitive\n"
        f"E≤$10^{{-{ev_exp}}}$, identity ≥{seq['min_identity']:.2f}\n"
        f"qcov ≥{seq['min_qcov']:.2f}; Top-{seq['top_hits']} GO transfer", "0.88")
    n_dbs = len(CFG["struct_channel"]["applications"].split(","))
    strict_note = ("" if FIG_CONFIG == "full" else
                   "\nmain caliber: date-coded pre-T0 libraries only")
    box(3.55, 4.75, 2.9, 1.25, "Structure channel",
        f"InterProScan, {n_dbs} signature DBs\n"
        "InterPro2GO mapping\n"
        f"coverage ≥{CFG['struct_channel']['min_coverage']:.2f}; share→conf"
        f"{strict_note}", "0.88")
    box(6.9, 4.75, 2.9, 1.25, "Literature channel",
        f"PubMed search (≤{iso(CFG['lit_channel']['pub_date_cutoff'])})\n"
        "gene+species → gene → protein\n"
        "term co-occurrence matching", "0.88")
    arr(3.0, 6.4, 1.65, 6.05); arr(5.0, 6.5, 5.0, 6.05); arr(7.0, 6.4, 8.35, 6.05)

    # ---- 仲裁层：一条决策链 + 可切换打分函数（规则 / 学习双模式） ----
    W = CFG["arbiter"]["weights"]
    A = CFG["arbiter"]
    n_feat = len(LEARNEV["features"])
    ax.add_patch(FancyBboxPatch((0.35, 2.62), 8.2, 1.73, boxstyle="round,pad=0.05",
                 fc="0.93", ec="0.1", lw=1.1))
    ax.text(4.45, 4.16, "Arbiter layer (dual mode): one decision chain, "
                        "switchable score",
            ha="center", va="center", fontsize=7.9)
    box(0.55, 2.74, 4.05, 1.28, "Shared decision chain",
        "same-term merge; ancestor-parent fold\n"
        f"branch-conflict record; score < {A['min_score']:.2f} dropped\n"
        f"max {A['max_terms_per_protein']} terms per protein; DAG validation\n"
        "explicit \"function unknown\" call", "0.98")
    ax.text(6.60, 3.90, "Score function (switchable)", ha="center", va="center",
            fontsize=7.2)
    mini(4.85, 3.44, 3.50, 0.34,
         [f"Rule mode: fixed weights "
          f"{W['seq']:.2f} / {W['struct']:.2f} / {W['lit']:.2f}"], "0.86", fs=6.0)
    mini(4.85, 2.78, 3.50, 0.60,
         [f"Learned mode: per-namespace L2 logistic regression",
          f"{n_feat} candidate-level features, calibrated on the",
          f"validation window; coefficients frozen for the test window"],
         "0.86", fs=5.7)
    ax.text(6.60, 3.38, "or", ha="center", va="center", fontsize=6.0, color="0.25")
    for x, x_top in ((1.65, 1.65), (5.0, 5.0), (8.35, 8.0)):
        arr(x, 4.75, x_top, 4.42)
    ax.text(9.42, 3.40, "GO ontology\n(is_a/part_of)", ha="center", va="center",
            fontsize=7.0, color="0.2")
    arr(8.55, 3.40, 8.92, 3.40, style="<|-|>")

    box(2.35, 1.35, 2.4, 1.0, "GAF 2.2 annotation",
        f"{CFG['output']['gaf_evidence_code']} evidence code", "0.92")
    box(5.25, 1.35, 2.4, 1.0, "Evaluation module",
        "frozen F1 (primary)\nFmax\u2020 / Smin / AUPR / AUC", "0.92")
    arr(3.6, 2.62, 3.6, 2.40); arr(6.0, 2.62, 6.0, 2.40)
    box(0.6, 0.1, 8.8, 0.9, "", "", "0.98", ls="--")
    go_ver = CFG["paths"]["go_obo"].split("go-basic-")[-1].replace(".obo", "")
    ax.text(5.0, 0.55, "Time-based three-way split (date of first whitelisted-evidence annotation)\n"
                       f"T0 = {t0} knowledge cutoff (train ≤ T0: {n_tr:,})   "
                       f"T1 = {t1} validation end (val: {n_va:,})   "
                       f"T2 = {t2} test end (test: {n_te:,})\n"
                       f"Ontology snapshot {go_ver} ({TS['go_terms']:,} terms)   "
                       f"Truth evidence codes: {len(CFG['split']['evidence_codes'])}   "
                       f"Validation-window candidates: {LEARNEV['n_val_candidates']:,}",
            ha="center", va="center", fontsize=6.4, color="0.15",
            linespacing=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_framework.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 图2 数据划分与本体统计（注释数来自 data_prep，评估口径数字在 fig3）
# ============================================================
def fig2():
    fc = TS["fasta_counts"]
    tp = TS["truth_proteins"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    dd = TS["dedup_proteins"]
    segs = [(f"Train library\n≤{fmt(str(CFG['split']['train_end']))}", 0, 26, "0.35",
             f"{fc['train']:,} seq-searchable\n{dd['train']:,} with truth"),
            (f"Valid.\n≤{fmt(str(CFG['split']['val_end']))}", 26, 14, "0.60",
             f"{fc['val']:,} queries\n{dd['val']:,} evaluable"),
            (f"Test\n≤{fmt(str(CFG['split']['test_end']))}", 40, 12, "0.82",
             f"{fc['test']:,} queries\n{dd['test']:,} evaluable")]
    for lab, s, w, g, n in segs:
        ax.barh(1.6, w, left=s, height=0.55, color=g, edgecolor="0.1", lw=0.8)
        ax.text(s+w/2, 2.02, lab, ha="center", va="bottom", fontsize=7.2)
        ax.text(s+w/2, 1.6, n, ha="center", va="center", fontsize=6.4,
                color="white" if g == "0.35" else "0.05")
    ax.barh(0.5, 14, left=26, height=0.55, color="0.48", edgecolor="0.1",
            lw=0.8, hatch="///")
    ax.text(33, 0.5, f"Arabidopsis {fc['ath_val']:,}", ha="center", va="center",
            fontsize=7.4, bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                    ec="none", alpha=0.85))
    ax.text(41.2, 0.5, f"Lib-side {fc['ath_lib']:,}\ninto training DB", ha="left",
            va="center", fontsize=6.4, color="0.3")
    ax.text(0, 1.06, "Swiss-Prot (date of first experimental annotation)",
            fontsize=7.0, color="0.25")
    ax.set_xlim(0, 56); ax.set_ylim(0, 2.6)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(a) Time-based data split", fontsize=9)

    ax = axes[1]
    onts = ["MFO", "BPO", "CCO"]
    x = np.arange(3); w = 0.25
    tr = [tp[f"train|{a}"] for a in ASPECTS]
    va = [tp[f"val|{a}"] for a in ASPECTS]
    te = [tp[f"test|{a}"] for a in ASPECTS]
    ax.bar(x-w, tr, w, color="0.25", edgecolor="0.1", label="Training library")
    ax.bar(x, va, w, color="0.55", edgecolor="0.1", label="Validation set")
    ax.bar(x+w, te, w, color="0.85", edgecolor="0.1", label="Test set")
    ax.set_xticks(x); ax.set_xticklabels(onts, fontsize=8)
    ax.set_ylabel("Number of truth proteins", fontsize=8)
    for xi, (a, b, c) in enumerate(zip(tr, va, te)):
        for bx, v in ((xi-w, a), (xi, b), (xi+w, c)):
            ax.annotate(f"{v:,}", xy=(bx, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.6, color="0.05")
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(0, max(tr) * 1.18)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.00), handlelength=1.2, columnspacing=1.0)
    ax.set_title("(b) Truth proteins per ontology", fontsize=9, pad=22)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dataset.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 图3 上排：主结果四指标（eval_summary.json）
#     下排：四配置消融 + bootstrap CI（bootstrap_ci.json）
# ============================================================
def fig3():
    """图 3：上＝表 2 八条路线的可部署指标（叠加 oracle Fmax），中＝ΔF1 配对差
    森林图，下＝分层冻结 F1 与 80% 功效下的可检测下限阴影。"""
    pt = BOOTL["point"]
    f1ci, mci = BOOTL["fixed_F1_ci"], BOOTL["ci"]
    order = sorted(ROUTES, key=lambda k: pt[k]["fixed_F1"])   # 最差在下，最好在上
    y = np.arange(len(order))
    fig = plt.figure(figsize=(7.2, 6.9 if FIG_CONFIG == "full" else 7.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.05]
                          if FIG_CONFIG == "full" else [1.0, 1.5, 1.0],
                          hspace=0.66,
                          wspace=0.07, left=0.175, right=0.985,
                          top=0.925, bottom=0.045)

    # ---------------- 上排：冻结 F1 与 AUPR（横向条，共用路线名） ----------------
    for j, (key, label, cikey) in enumerate(
            (("fixed_F1", "Frozen F1 (deployable)", "fixed"),
             ("aupr", "AUPR", "aupr"))):
        ax = fig.add_subplot(gs[0, j])
        vals = np.array([pt[k][key] for k in order])
        lo = np.array([f1ci[k][0] if cikey == "fixed"
                       else mci[k]["aupr_ci"][0] for k in order])
        hi = np.array([f1ci[k][1] if cikey == "fixed"
                       else mci[k]["aupr_ci"][1] for k in order])
        ax.barh(y, vals, 0.6, color=[GRAY.get(k, GRAY["rest"]) for k in order],
                edgecolor="0.05", lw=0.6)
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="0.05", capsize=2.4, lw=0.9)
        ax.plot([pt[k]["fmax"] for k in order], y, ls="none", marker="s",
                ms=4.2, mfc="none", mec="0.05", mew=0.9,
                label="grid Fmax (oracle)")
        ax.set_xlim(0, max(hi.max(), max(pt[k]["fmax"] for k in order)) * 1.30)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([ROUTE_LABEL[k] for k in order] if j == 0 else [""] * len(y),
                           fontsize=6.6)
        ax.tick_params(axis="y", length=0 if j else 3)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel(label, fontsize=7.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for yi, v, h, k in zip(y, vals, hi, order):
            ax.annotate(fmt_num(v, 3), (max(h, pt[k]["fmax"]) + 0.012, yi),
                        xytext=(0, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=5.6, color="0.05")
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.672),
                       fontsize=6.4, frameon=False)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ---------------- 中排：配对差森林图 ----------------
    axf = fig.add_subplot(gs[1, :])
    rows = [(DELTA_LABEL[k.split("-", 1)[1]],
             d["dFixed_F1"], d["dFixed_F1_ci"], "0.25")
            for k, d in BOOTL["delta_ci"].items()
            if k.startswith("learn-") and k.split("-", 1)[1] in ROUTES]
    rows += [(EXTRA_DELTA[k], d["dFixed_F1"], d["dFixed_F1_ci"], "0.45")
             for k, d in BOOTL["delta_ci"].items() if k in EXTRA_DELTA]
    fu = POWER["strata"]["overall"]["paired_delta_F1"]["fusion-union"]
    rows.append(("rule-based \u2212 naive union \u2021", fu["delta"], fu["ci95"], "0.5"))
    rows.sort(key=lambda r: r[1])
    yy = np.arange(len(rows))
    for yi, (lab, dv, ci, col) in zip(yy, rows):
        crosses = ci[0] <= 0 <= ci[1]
        axf.plot([ci[0], ci[1]], [yi, yi], color=col, lw=1.5,
                 ls="-" if not crosses else (0, (3, 2)))
        axf.plot([dv], [yi], marker="o", ms=5.0,
                 mfc="white" if crosses else col, mec=col, mew=1.2)
        axf.annotate(f"{dv:+.4f} ({ci[0]:.4f}\u2013{ci[1]:.4f})",
                     (ci[1], yi), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=6.0, color="0.1")
    axf.axvline(0, color="0.1", lw=0.9)
    axf.set_yticks(yy)
    axf.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    axf.set_ylim(-0.7, len(rows) - 0.3)
    axf.set_xlim(min([c[0] for _, _, c, _ in rows] + [0]) - 0.02,
                 max([c[1] for _, _, c, _ in rows] + [0]) + 0.115)
    axf.set_xlabel("Paired \u0394 frozen F1, B = 2 000 protein-level bootstrap",
                   fontsize=7.6)
    axf.set_title("Paired differences (open marker = interval crosses zero)",
                  fontsize=8.2)
    axf.grid(axis="x", color="0.8", lw=0.5)
    axf.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axf.spines[sp].set_visible(False)
    axf.tick_params(axis="x", labelsize=7)

    # ---------------- 下排：分层冻结 F1 与 MDD80 ----------------
    axb = fig.add_subplot(gs[2, :])
    strata = ["overall", "MFO", "BPO", "CCO", "BPO_shallow", "BPO_mid", "BPO_deep"]
    names = ["Overall", "MFO", "BPO", "CCO", "BPO shallow", "BPO mid", "BPO deep"]
    x = np.arange(len(strata))
    bw = 0.33
    band_top = np.array([POWER["strata"][s]["frozen_F1"]["fusion"] +
                         POWER["strata"][s]["paired_delta_F1"]["learn-fusion"]["MDD_80"]
                         for s in strata])
    for off, col, tag in ((-bw / 2 - 0.01, "0.38", "rule-based"),
                          (bw / 2 + 0.01, "0.12", "learned")):
        vals = np.array([POWER["strata"][s]["frozen_F1"][
                        "fusion" if tag == "rule-based" else "learn"] for s in strata])
        cis = [POWER["strata"][s]["F1_ci95"][
               "fusion" if tag == "rule-based" else "learn"] for s in strata]
        lo = np.array([c[0] for c in cis]); hi = np.array([c[1] for c in cis])
        axb.bar(x + off, vals, bw, color=col, edgecolor="0.05", lw=0.6, label=tag)
        axb.errorbar(x + off, vals, yerr=[vals - lo, hi - vals], fmt="none",
                     ecolor="0.05", capsize=2.0, lw=0.8)
        for xi, t, val in zip(x + off, np.maximum(hi, band_top), vals):
            axb.annotate(fmt_num(val, 3), (xi, t), xytext=(0, 2),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=5.4, color="0.05")
    for xi, s in zip(x, strata):
        mdd = POWER["strata"][s]["paired_delta_F1"]["learn-fusion"]["MDD_80"]
        base = POWER["strata"][s]["frozen_F1"]["fusion"]
        axb.add_patch(Rectangle((xi - 0.46, base), 0.92, mdd, fc="0.78",
                                 ec="0.45", lw=0.5, ls=":", alpha=0.5))
    axb.set_xticks(x)
    axb.set_xticklabels(names, fontsize=6.8)
    axb.set_ylim(0, 0.86)
    axb.set_ylabel("Frozen F1", fontsize=8)
    axb.set_title("Stratified frozen F1; shaded band = minimum detectable "
                  "paired difference at 80% power (values in Table 3)", fontsize=8.0)
    axb.legend(handles=axb.get_legend_handles_labels()[0] +
                     [Rectangle((0, 0), 1, 1, fc="0.78", ec="0.45", lw=0.5,
                                ls=":", alpha=0.5)],
               labels=axb.get_legend_handles_labels()[1] + ["MDD$_{80}$"],
               fontsize=6.4, frameon=False, loc="upper left", ncol=3)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.tick_params(axis="y", labelsize=7)
    fig.savefig(OUT / "fig3_main_ablation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 补充图 S5 BPO 归因分析（原正文图 5，移入补充材料）
# (a) 阈值-性能曲线  —— _curves() 从 revision 预测重算
# (b) 生物过程分层   —— 从 bpo_analysis.json 读取 oracle_fmax
# ============================================================
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    curves, t = _curves()
    for (lab, y, g, h) in curves:
        ax.plot(t, y, color=g, lw=1.4, label=lab, ls="-" if h == "" else h)
    # 校准阈值与各本体 oracle 最优点均取自 revision JSON / 同口径重算
    calib = BPO["阈值校准"]["BPO"]
    t_star, f1_at_tstar = calib["val校准阈值"], calib["test_校准阈值_F1"]
    bppo = next(y for (lab, y, g, h) in curves if lab == LAB_BPO)
    idx = int(round(t_star / t[1]))
    y_at = bppo[idx]
    for lab, key in ((LAB_MFO, "MFO"), (LAB_BPO, "BPO"), (LAB_CCO, "CCO")):
        c = BPO["阈值校准"][key]
        yi = bppo if key == "BPO" else next(
            y for (lb, y, g, h) in curves if lb == lab)
        oi = int(round(c["test_oracle_t"] / t[1]))
        ax.plot([c["test_oracle_t"]], [yi[oi]], marker="o", ms=4.5,
                color="0.05", zorder=6)
    ax.annotate(f"BPO t* (val) = {t_star:.2f}\n"
                f"test F1 = {f1_at_tstar:.4f}\n"
                f"(= oracle, see ○)",
                xy=(t_star, y_at), xytext=(0.04, 0.03),
                fontsize=7.0, color="0.05", ha="left", va="bottom",
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="0.6", lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9,
                                shrinkA=4, shrinkB=3))
    ymax = max(max(y) for (_, y, _, _) in curves)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Decision threshold t", fontsize=8)
    ax.set_ylabel("Protein-centric F1 (test)", fontsize=8)
    ax.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="none", handlelength=1.6, borderaxespad=0.2)
    ax.set_title("(a) Threshold curves, oracle (○), calibration", fontsize=9, pad=26)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    # 从 bpo_analysis.json 读 oracle_fmax
    ic3_bin  = BPO["分层报告"]["IC分层_3Channel"]
    icseq_bin = BPO["分层报告"]["IC分层_seq-only"]
    deep3_bin = BPO["分层报告"]["深度分层_3Channel"]
    # IC 分层：低 / 中 / 高
    ic3   = [ic3_bin["低IC"]["oracle_fmax"],
             ic3_bin["中IC"]["oracle_fmax"],
             ic3_bin["高IC"]["oracle_fmax"]]
    icseq = [icseq_bin["低IC"]["oracle_fmax"],
             icseq_bin["中IC"]["oracle_fmax"],
             icseq_bin["高IC"]["oracle_fmax"]]
    # 深度分层：浅 / 中 / 深
    deep3 = [deep3_bin["浅层"]["oracle_fmax"],
             deep3_bin["中层"]["oracle_fmax"],
             deep3_bin["深层"]["oracle_fmax"]]
    x = np.arange(3); w = 0.24
    order_ic = ["low IC", "mid IC", "high IC"]
    order_dep = ["shallow", "medium", "deep"]
    ax.bar(x-w, ic3, w, color="0.15", edgecolor="0.05", label="Fusion (IC)")
    ax.bar(x, icseq, w, color="0.55", edgecolor="0.05", hatch="//",
           label="Sequence only (IC)")
    ax.bar(x+w, deep3, w, color="0.82", edgecolor="0.05", hatch="..",
           label="Fusion (depth)")
    for xi, (a, b, c) in enumerate(zip(ic3, icseq, deep3)):
        for off, v in ((-w, a), (0, b), (w, c)):
            ax.annotate(f"{v:.3f}", xy=(xi+off, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.4, color="0.05", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb} | {dp}" for lb, dp in zip(order_ic, order_dep)],
                       fontsize=7.0)
    ax.set_ylabel("Within-stratum Fmax", fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=0.8,
              handlelength=1.2)
    ax.set_title("(b) BPO stratified performance", fontsize=9, pad=26)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(0, max(ic3 + icseq + deep3) * 1.26)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "supp_fig5_bpo.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def _curves():
    """重算三个命名空间各自的蛋白中心 F1-阈值曲线（101 点网格）。

    真值与预测都限制到同一命名空间，口径与 evaluator 的分本体 Fmax 一致，
    因此各曲线峰值应复现表 2 的分本体 Fmax。
    """
    from src.evaluator import Evaluator, load_predictions, load_truth
    ev = _eval_context()
    t3 = load_truth(str(PROJ / "data/annotations/split3_go.tsv"),
                    exclude_iea=ev.exclude_iea, dag=ev.dag)
    tp3_full = {p: ev._prop(g) for p, g in t3.items()}
    pr_full = {p: ev._propagate_preds(g) for p, g in
               load_predictions(str(PRED / "predictions_split3_test_multi.tsv")).items()}
    ts = np.asarray(ev.thresholds)
    out = []
    for aspect, lab, g, h in (("F", LAB_MFO, "0.1", ""),
                              ("P", LAB_BPO, "0.4", "--"),
                              ("C", LAB_CCO, "0.62", ":")):
        tp3 = {p: {x for x in s if ev.dag.aspect(x) == aspect}
               for p, s in tp3_full.items()}
        tp3 = {p: s for p, s in tp3.items() if s}
        pr = {p: {g2: sc for g2, sc in d.items() if ev.dag.aspect(g2) == aspect}
              for p, d in pr_full.items()}
        tg = sorted(tp3)
        f1 = []
        for t_val in ts:
            p_sum, p_n, r_sum = 0.0, 0, 0.0
            for p in tg:
                tt = tp3[p]
                pt = {g2 for g2, sc in pr.get(p, {}).items() if sc >= t_val}
                inter = len(pt & tt)
                if pt:
                    p_sum += inter / len(pt); p_n += 1
                r_sum += inter / len(tt)
            prec = p_sum / p_n if p_n else 0.0
            rec = r_sum / len(tg)
            f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out.append((lab, f1, g, h))
    return out, ts


def _eval_context():
    from src.evaluator import Evaluator
    from src.go_dag import GODag
    dag = GODag(str(PROJ / CFG["paths"]["go_obo"]),
                relations=CFG.get("go_dag", {}).get("relations"))
    return Evaluator(CFG, dag)


# 图 1–3 由本脚本驱动，图 4 见 make_fig4_operating.py；fig5 出补充图 S5。
# 命令行可只画其中几幅：python make_figures.py 1 3
FIGNO = {"1": fig1, "2": fig2, "3": fig3, "5": fig5}
_sel = [a for a in sys.argv[1:] if a in FIGNO]
for k in (_sel or ["1", "2", "3", "5"]):
    FIGNO[k]()
print("完成 ->", OUT)
