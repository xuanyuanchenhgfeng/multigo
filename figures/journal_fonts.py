# -*- coding: utf-8 -*-
"""期刊图件字体：优先真正的 Times New Roman，缺失时退回度量兼容的 Liberation Serif / TeX Gyre Termes。

投稿模板要求图中西文一律 Times New Roman。本机没有该字体时，用字宽与之逐字相同的克隆体，
换成装有 TrueType 版 Times 的机器重跑会自动选到真字体，图面字距不发生变化。
"""
from pathlib import Path

import matplotlib.font_manager as fm

SERIF_PRIORITY = ("Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif")
SEARCH_DIRS = (
    Path.home() / ".local/share/fonts/times-substitute",
    Path.home() / ".fonts",
    Path("/usr/share/fonts/truetype/liberation"),
)


def use_times(plt):
    for d in SEARCH_DIRS:
        if d.is_dir():
            for f in sorted(d.glob("*.otf")) + sorted(d.glob("*.ttf")):
                try:
                    fm.fontManager.addfont(str(f))
                except Exception:
                    pass
    have = {f.name for f in fm.fontManager.ttflist}
    stack = [c for c in SERIF_PRIORITY if c in have]
    if not stack:
        raise SystemExit("未找到 Times 族或其克隆体，请先安装 Liberation Serif / TeX Gyre Termes")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = stack + ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    return stack[0]
