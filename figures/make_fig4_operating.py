# -*- coding: utf-8 -*-
"""图 4：两套仲裁模式工作点的三张诊断图，灰度+纹理，300 dpi。

左＝验证窗 F1-阈值曲线与两个选点（argmax、1-SE）各自冻结到测试窗的 F1；
中＝测试窗传播后口径下的可靠性图与期望校准误差；
右＝术语发射宽度与逐 GO 宏 AUC 的关系，含两法共同发射类上的反超。

全部数字读自本仓库 `results/revision/`，脚本内不重算任何指标：
缺省（FIG_CONFIG=full）读 learn_eval.json（InterProScan 全库上界口径），
FIG_CONFIG=strict 读 strict_mech.json（表 2 主结果口径，结构通道只留版本串自带
日期码且不晚于 T0 的签名库）。两份文件的字段名不同，靠下面的适配器对齐。
输出目录默认为脚本所在目录，可用 MULTIGO_FIG_OUT 指到随稿提交的 figures/。
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUT = Path(os.environ.get("MULTIGO_FIG_OUT") or Path(__file__).resolve().parent)
RES = Path(__file__).resolve().parents[1] / "results" / "revision"
CJK_FONT = os.environ.get("MULTIGO_CJK_FONT",
                          str(Path.home() / ".fonts/NotoSansSC.ttf"))
if Path(CJK_FONT).is_file():        # 图面标签全为西文，中文字体只作兜底
    fm.fontManager.addfont(CJK_FONT)
from journal_fonts import use_times
use_times(plt)

FIG_CONFIG = os.environ.get("FIG_CONFIG", "full")
if FIG_CONFIG == "strict":
    EV = json.load(open(RES / "strict_mech.json", encoding="utf-8"))
    DEP = EV["val_curve"]
    ARMS = {a: {"test": {"distinct_terms_at_fmax": v["distinct_terms_at_fmax"],
                         "auc": v["auc"]},
                "test_ece_on_propagated": {"ece": v["test_ece_on_propagated"],
                                           "reliability": v["test_reliability"]}}
            for a, v in EV["arms"].items()}
else:
    EV = json.load(open(RES / "learn_eval.json", encoding="utf-8"))
    DEP, ARMS = EV["deploy_table"], EV["arms"]
ARMS_ORDER = ("fusion", "learn")
STYLE = {"fusion": ("0.35", "-", "Rule-based arbiter"),
         "learn": ("0.08", "--", "Learned arbiter")}

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
fig.subplots_adjust(left=0.065, right=0.985, top=0.845, bottom=0.185, wspace=0.30)

# ---------------- 左：验证窗 F1-阈值曲线 ----------------
ax = axes[0]
grid = np.arange(101) / 100.0
for arm in ARMS_ORDER:
    col, ls, lab = STYLE[arm]
    curve = np.asarray(DEP[arm]["val_F1_curve"], dtype=float)
    ax.plot(grid, curve, color=col, ls=ls, lw=1.3, label=lab)
    for tag, mk, off in (("argmax", "o", (-3, -11)), ("one_se", "s", (4, 4))):
        t = DEP[arm][f"t_star_{tag}"]
        ax.plot([t], [DEP[arm][f"val_F1_{tag}"]], marker=mk, ms=4.4,
                mfc="white", mec=col, mew=1.2, ls="none")
        ax.annotate(f"{DEP[arm][f'test_F1_{tag}']:.4f}",
                    (t, DEP[arm][f"val_F1_{tag}"]), xytext=off,
                    textcoords="offset points", ha="left" if off[0] > 0 else "right",
                    fontsize=5.4, color=col,
                    bbox=dict(fc="white", ec="none", pad=0.4))
ax.set_xlabel("Threshold", fontsize=7.4)
ax.set_ylabel("Validation-window F1", fontsize=7.4)
ax.set_title("Operating points (labels = test F1)", fontsize=8.0)
ax.tick_params(labelsize=6.6)
ax.legend(fontsize=6.0, frameon=False, loc="upper right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---------------- 中：测试窗可靠性图 ----------------
ax = axes[1]
ax.plot([0, 1], [0, 1], color="0.55", lw=0.8, ls=":", label="perfect")
for arm in ARMS_ORDER:
    col, _, lab = STYLE[arm]
    rel = ARMS[arm]["test_ece_on_propagated"]["reliability"]
    conf = np.array([b["mean_prob"] for b in rel])
    acc = np.array([b["empirical"] for b in rel])
    ax.bar(conf, acc, width=0.072, color=col, alpha=0.20, edgecolor=col, lw=0.6)
    ax.plot(conf, acc, marker="o", ms=3.6, color=col, ls=":", lw=0.8,
            label=f"{lab} (ECE {ARMS[arm]['test_ece_on_propagated']['ece']:.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Predicted score", fontsize=7.4)
ax.set_ylabel("Observed positive rate", fontsize=7.4)
ax.set_title("Reliability on test window", fontsize=8.0)
ax.tick_params(labelsize=6.6)
ax.legend(fontsize=5.4, frameon=False, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---------------- 右：发射宽度与宏 AUC ----------------
ax = axes[2]
x = np.arange(len(ARMS_ORDER))
emitted = [ARMS[a]["test"]["distinct_terms_at_fmax"] for a in ARMS_ORDER]
macro = [ARMS[a]["test"]["auc"] for a in ARMS_ORDER]
matched = EV["matched_emission_auc"]
for xi, arm in zip(x, ARMS_ORDER):
    col, _, _ = STYLE[arm]
    ax.plot([xi - 0.07, xi - 0.07], [0.5, macro[xi]], color=col, lw=0.8)
    ax.plot([xi - 0.07], [macro[xi]], marker="o", ms=6.0, color=col, ls="none")
    ax.annotate(f"{macro[xi]:.3f}", (xi - 0.07, macro[xi]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=6.0)
    ax.annotate(f"{emitted[xi]:,} terms\nat Fmax", (xi, 0.458), ha="center",
                va="bottom", fontsize=5.2, color="0.3",
                bbox=dict(fc="white", ec="none", pad=0.6))
    shared = matched[arm]
    ax.plot([xi + 0.13, xi + 0.13], [0.5, shared], color=col, lw=0.7,
            ls=(0, (1, 1.6)))
    ax.plot([xi + 0.13], [shared], marker="D", ms=4.6, mfc="white", mec=col,
            mew=1.1, ls="none", zorder=5)
    ax.annotate(f"{shared:.3f}", (xi + 0.13, shared), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=5.6, color=col)
ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(["Rule-based", "Learned"], fontsize=6.8)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0.45, 0.92)
ax.set_ylabel("Macro AUC", fontsize=7.4)
ax.set_title(f"Emitted breadth vs AUC\n"
             f"(open diamond = {matched['n_classes']} jointly emitted classes)",
             fontsize=7.6)
ax.tick_params(labelsize=6.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(OUT / "fig4_operating.png", dpi=300, bbox_inches="tight")
print("FIG4 SAVED ->", OUT / "fig4_operating.png")
