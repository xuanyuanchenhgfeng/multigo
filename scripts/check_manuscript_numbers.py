#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 manuscript.md 里的每个已发表数字是否等于产物中的值。

用法：python3 scripts/check_manuscript_numbers.py
口径：主结果一律为严格时间完整配置（MULTIGO_CONFIG=strict），产物在
results/revision/ 下由 revision_strict.py、revision_strict_mech.py、
revision_litonly_ci.py、revision_power.py、revision_operating_point.py、
lit_source_spotcheck.py、revision_learning.py 与 struct_library_audit.py 生成。
四舍五入与正文一致（ROUND_HALF_UP），因此表格单元格可以逐字比对。
改完 manuscript.md 必须紧接着跑 论文/build_docx.py 重编译 manuscript.docx：
本脚本最后一段会核对 docx 是否落后于 md，并抽查表 2 的数字是否已进 docx。
"""
from __future__ import annotations

import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _results() -> Path:
    env = os.environ.get("MULTIGO_RESULTS")
    if env:
        return Path(env)
    for cand in (ROOT / "results" / "revision",
                 ROOT.parent / "数据" / "修订版结果"):
        if cand.is_dir():
            return cand
    sys.exit("找不到 results/revision 产物目录，请用 MULTIGO_RESULTS 指定")


RES = _results()


def _manuscript() -> Path:
    env = os.environ.get("MULTIGO_MANUSCRIPT")
    if env:
        return Path(env)
    for cand in (ROOT.parent / "Materials" / "论文" / "manuscript.md",
                 ROOT.parent / "论文" / "manuscript.md"):
        if cand.exists():
            return cand
    sys.exit("找不到 manuscript.md，请用 MULTIGO_MANUSCRIPT 指定稿件路径")


MAN = _manuscript()

FAILS: list[str] = []
TEXT = MAN.read_text(encoding="utf8")


def num(v, nd):
    """按正文口径取 nd 位小数（四舍五入），负号用正文的 U+2212。"""
    s = str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd),
                                             rounding=ROUND_HALF_UP))
    return s.replace("-", "−")


def ci(pair, nd):
    """区间写法：下界为负而上界非负时，上界补显式正号（正文约定）。"""
    lo, hi = num(pair[0], nd), num(pair[1], nd)
    if float(pair[0]) < 0 <= float(pair[1]):
        hi = "+" + hi
    return f"{lo}–{hi}"


def cell(metric, ci_pair, nd):
    return f"{num(metric, nd)}（{ci(ci_pair, nd)}）"


def signed(v, nd):
    """正文中配对差的写法：+0.0575 / −0.0112。"""
    s = num(abs(v), nd)
    return ("+" if v >= 0 else "−") + s


def need(what, s, ctx=""):
    if s not in TEXT:
        FAILS.append(f"MISSING {what}: 「{s}」{ctx}")


def load(name):
    return json.loads((RES / name).read_text(encoding="utf8"))


def table_rows(header_key: str, ncols: int):
    """取出正文中表头以 header_key 开头的那张表的数据行。"""
    lines = TEXT.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith(f"| {header_key}"))
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    return [r for r in rows if len(r) >= ncols]


# ---------------------------------------------------------------- 表 2
se = load("strict_eval.json")
st = se["routes"]
li = load("litonly_ci.json")
ND3, ND4 = 3, 4
ROW2 = {
    "仅序列": "seq-only", "InterPro 直接映射": "interpro", "最近邻 Top1": "nn_top1",
    "最近邻 Top3": "nn_top3", "朴素并集": "union", "投票≥2": "vote2",
    "规则仲裁（本文）": "fusion", "学习仲裁（本文）": "learn",
}
rows2 = {r[0]: r for r in table_rows("路线", 8)}
need("表 2", "| 学习仲裁·文献单源行按线索交付（本文） |", "行缺失")
for label, key in ROW2.items():
    r = rows2.get(label)
    if r is None:
        FAILS.append(f"表 2 缺行：{label}")
        continue
    d = st[key]
    for col, m, c, nd in ((1, "fmax", "fmax_ci", ND3), (2, "aupr", "aupr_ci", ND3),
                          (3, "smin", "smin_ci", 2), (4, "auc", "auc_ci", ND3),
                          (5, "frozen_F1", "frozen_F1_ci", ND4)):
        need(f"表 2 {label} {m}", cell(d[m], d[c], nd), f"（该行第 {col + 1} 列）")
    need(f"表 2 {label} t*", f"| {num(d['t_star_one_se'], 2)} | {d['test_covered_one_se']} |")
nl = li["grid_ci_learn_nolitonly"]
deliv = [r for r in TEXT.split("\n") if r.startswith("| 学习仲裁·文献单源行按线索交付")]
if deliv:
    for s in (cell(li["grid"]["learn_nolitonly"]["fmax"], nl["fmax"], ND3),
              cell(li["grid"]["learn_nolitonly"]["aupr"], nl["aupr"], ND3),
              cell(li["grid"]["learn_nolitonly"]["auc"], nl["auc"], ND3),
              "**0.4441（0.4226–0.4662）**", "| 0.15 | 723 |"):
        need("表 2 交付行", s)
else:
    FAILS.append("表 2 缺交付行")

# --------------------------------------------------------- 配对差（2.2 节）
for k, v in se["bootstrap"]["delta_ci"].items():
    if k in ("learn-seq+struct", "learn-seq+lit"):
        continue          # 这两条消融配置只列于补充表 S4/S23，正文不引用
    need(f"2.2 节配对差 {k}",
         f"{signed(v['dFixed_F1'], ND4)}（{ci(v['dFixed_F1_ci'], ND4)}）")

# ---------------------------------------------------------------- 表 3
pw = load("power_analysis_strict.json")["strata"]
rows3 = {r[0]: r for r in table_rows("分层", 8)}
NAME3 = {"整体": "overall", "MFO": "MFO", "BPO": "BPO", "CCO": "CCO",
         "BPO 浅层": "BPO_shallow", "BPO 中层": "BPO_mid", "BPO 深层": "BPO_deep"}
for label, key in NAME3.items():
    d = pw[key]
    if label not in rows3:
        FAILS.append(f"表 3 缺行：{label}")
        continue
    f = d["frozen_F1"]
    pd_ = d["paired_delta_F1"]["learn-fusion"]
    need(f"表 3 {label}",
         f"| {num(f['fusion'], ND4)} | {num(f['learn'], ND4)} | "
         f"{signed(pd_['delta'], ND4)}（{ci(pd_['ci95'], ND4)}） | {num(pd_['MDD_80'], ND4)} |")
    need(f"表 3 {label} 规模",
         f"| {d['n_proteins']:,}".replace(",", " ") +
         f" | {d['n_terms']:,}".replace(",", " ") +
         f" | {d['n_annotations']:,}".replace(",", " ") + " |")

# ---------------------------------------------------------------- 表 4
au = load("struct_library_audit.json")
sens = au["sensitivity"]
for arm, key in (("规则仲裁", "fusion"), ("学习仲裁", "learn")):
    a, f, d = (sens[f"{key}_datecoded_pre_T0"], sens[f"{key}_full"],
               sens[f"{key}_dropped"])
    need(f"表 4 {arm}",
         f"{num(a['test_grid']['fmax'], ND4)}（{num(a['test_F1_one_se'], ND4)}）→ "
         f"{num(f['test_grid']['fmax'], ND4)}（{num(f['test_F1_one_se'], ND4)}）→ "
         f"{num(d['test_grid']['fmax'], ND4)}（{num(d['test_F1_one_se'], ND4)}）")
need("表 4 间隔", f"晚 {au['gap_days_after_T0']} d；其后共 {au['gap_n_releases_after_T0']} 次发布")
need("表 4 保留行", "验证窗 11/5 627（0.20%，8 个蛋白）；测试窗 17/1 712（0.99%，17 个蛋白）")

# ---------------------------------------------------------------- 表 5
sp = load("lit_spotcheck_strict.json")
PCT = lambda v, nd: f"{num(v * 100, nd)}%"
for label, d in (("全部文献候选", sp["all_lit_rows"]),
                 ("共现文献数 ≥ 2", sp["by_citations"][">=2"]),
                 ("共现文献数 ≥ 3", sp["by_citations"][">=3"]),
                 ("其中 MFO", sp["by_aspect"]["F"]),
                 ("其中 BPO", sp["by_aspect"]["P"]),
                 ("其中 CCO", sp["by_aspect"]["C"]),
                 ("学习仲裁保留的文献单源行", sp["lit_only_emitted_by_learn"])):
    rows5 = {r[0]: r for r in table_rows("分组", 5)}
    r = rows5.get(label)
    if r is None:
        FAILS.append(f"表 5 缺行：{label}")
        continue
    need(f"表 5 {label} 行数", f"| {d['n']:,}".replace(",", " ") + " |", f"（{label}）")
    for col, m, nd in ((2, "title_hit", 1), (3, "gene_in_sentence", 1),
                       (4, "truth_hit", 2)):
        if r[col] != "—":
            need(f"表 5 {label} {m}", f"| {PCT(d[m], nd)} |", f"（{label} 第 {col} 列）")
d5 = li["paired_delta"]["learn_nolitonly-learn"]
need("表 5 交付消融", f"{signed(d5['delta'], ND4)}（{ci(d5['ci95'], ND4)}）")
for k in ("learn_nolitonly-nn_top1", "learn_nolitonly-fusion"):
    v = li["paired_delta"][k]
    need(f"表 5/正文 {k}", f"{signed(v['delta'], ND4)}（{ci(v['ci95'], ND4)}）")

# ---------------------------------------------------------------- 表 6
xs = load("eval_cross_species_learn.json")["arms"]
for arm, key in (("规则仲裁", "fusion"), ("学习仲裁", "learn")):
    for setting, pre in (("同物种留出", "same"), ("真跨物种", "cross")):
        d = xs[f"{pre}_{key}_strict"]
        need(f"表 6 {setting} {arm}",
             f"{num(d['fmax'], ND3)} | {num(d['smin'], 2)} | {num(d['aupr'], ND3)} | "
             f"{num(d['auc'], ND3)} | {num(d['coverage'], ND3)} | "
             f"{num(d['frozen_at_route_tstar']['F1'], ND4)}")
need("表 6 注（全量口径 Fmax）",
     "同物种留出规则 0.446 / 学习 0.343，真跨物种规则 0.282 / 学习 0.239")

# ------------------------------------------------------- 严格机制量（图 4）
mech = load("strict_mech.json")
lc = mech["val_curve"]
for arm, prose in (("fusion", "0.3406"), ("learn", "0.4064")):
    got = num(lc[arm]["test_F1_argmax"], ND4)
    if got != prose:
        FAILS.append(f"2.7 节 argmax 落点漂移：{arm} strict_mech={got}，正文={prose}")
need("2.7 节工作点", "规则仲裁的验证窗 argmax 阈值 t\\* = 0.23 冻结应用测试窗得 F1 = 0.3406，"
                    "1-SE 规则给出 t\\* = 0.39、F1 = 0.3475")
need("2.7 节工作点（学习）", "argmax 0.13 得 0.4064，1-SE 0.15 得 0.4048，差 0.0016")
need("2.7 节峰值标准误", f"峰值标准误分别为 {num(st['fusion']['val_F1_se'], 4)} 与 "
                        f"{num(st['learn']['val_F1_se'], 4)}")
need("图 4 ECE", "学习仲裁 ECE = 0.081，把规则模式的加权和当概率算则为 0.247")
mm = mech["matched_emission_auc"]
need("2.3 节共同类 AUC", f"{num(mm['learn'], ND3)} 高于规则仲裁 {num(mm['fusion'], ND3)}")

# ------------------------------------------------------------------ 摘要
for s in ("0.4048", "0.3475", "+0.0575", "0.428", "0.205", "0.081", "0.247",
          "0.4441", "0.4161", "0.2877", "386 d", "1 130", "16 948", "5.14%"):
    need("摘要/正文共用", s, "（主口径关键值）")

# -------------------------------------------------------------- 引用完整性
cited = set()
for m in re.finditer(r"\[(\d+(?:[-–,]\d+)*)\]", TEXT.split("## 参考文献")[0]):
    for part in re.split(r"[,–-]", m.group(1)):
        if part.strip().isdigit():
            cited.add(int(part))
    span = m.group(1)
    if re.fullmatch(r"\d+[-–]\d+", span):
        a, b = map(int, re.split(r"[-–]", span))
        cited |= set(range(a, b + 1))
listed = {int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", TEXT, re.M)}
for n in sorted(listed - cited):
    FAILS.append(f"参考文献 [{n}] 列于文末但正文未引用")
for n in sorted(cited - listed):
    FAILS.append(f"正文引用 [{n}] 在文末无对应条目")

# ------------------------------------------------- manuscript.docx 同步性
DOCX = MAN.with_suffix(".docx")
if not DOCX.exists():
    FAILS.append(f"{DOCX.name} 不存在：manuscript.md 的改动没有编译进 docx")
elif DOCX.stat().st_mtime < MAN.stat().st_mtime:
    FAILS.append(f"{DOCX.name} 早于 manuscript.md：请跑 论文/build_docx.py 重编译")
else:
    try:
        from docx import Document
    except ImportError:
        Document = None
    if Document is not None:
        doc = Document(str(DOCX))
        dtxt = "\n".join(p.text for p in doc.paragraphs)
        dtxt += "\n" + "\n".join(c.text for tb in doc.tables
                                 for r in tb.rows for c in r.cells)
        # docx 表内是英文书写，括号与空格被改写，比对时只留数字与正负号
        keep = lambda s: re.sub(r"[^0-9.+−–]", "", s)
        dkeep = keep(dtxt)
        if len(doc.tables) != 6:
            FAILS.append(f"docx 表格数 {len(doc.tables)}，应为 6")
        if "基于时间划分与可追溯证据仲裁" not in dtxt:
            FAILS.append("docx 首题与 manuscript.md 不一致")
        for label, key in ROW2.items():
            d = st[key]
            s = keep(cell(d["frozen_F1"], d["frozen_F1_ci"], ND4))
            if s not in dkeep:
                FAILS.append(f"docx 缺表 2 的 {label} 冻结 F1 单元格（{s}）")
        if keep("**0.4441（0.4226–0.4662）**") not in dkeep:
            FAILS.append("docx 缺表 2 交付行的 0.4441（0.4226–0.4662）")

print(f"共检查 {len(TEXT.splitlines())} 行正文、{len(listed)} 条参考文献"
      f"（正文引用 {len(cited)} 条），发现 {len(FAILS)} 处问题")
for f in FAILS:
    print("  ✗", f)
sys.exit(1 if FAILS else 0)
