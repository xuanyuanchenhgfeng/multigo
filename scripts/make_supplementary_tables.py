"""生成补充材料中数字密集型的表格片段（S3、S4、S17-S24）。

所有数值只从 results/revision/ 下的产物 JSON 读取，不在本脚本中重算任何指标。
主口径为严格时间完整配置：结构通道只保留版本串自带日期码且不晚于 T0 = 2023-11-08 的
签名库（HAMAP/PIRSR/PROSITEPATTERNS/PROSITEPROFILES），全部路线在同一受限验证窗重新标定。
全量签名库的数字只保留在明确标注“只作上界”的分块内，两套口径同表分块并列。
用法：
    python3 -X utf8 scripts/make_supplementary_tables.py [--out 文件]
输出的 Markdown 片段按补充表编号分节，逐节粘贴进 Materials/补充材料/supplementary.md。
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "revision"

ARMS = ["learn", "fusion", "vote2", "union", "interpro",
        "nn_top1", "nn_top3", "seq-only", "seq+struct", "seq+lit"]
CN = {"learn": "学习仲裁（本文）", "fusion": "规则仲裁（三通道融合）",
      "vote2": "投票≥2", "union": "朴素并集 Union",
      "interpro": "InterPro 直接映射", "nn_top1": "同源 Top1 转移",
      "nn_top3": "同源 Top3 转移", "seq-only": "仅序列通道",
      "seq+struct": "序列+结构", "seq+lit": "序列+文献",
      "learn_nolitonly": "学习仲裁·文献单源行置零"}
PAIR_CN = {"fusion": "规则仲裁", "union": "朴素并集", "interpro": "InterPro 直接映射",
           "seq-only": "仅序列通道", "seq+struct": "序列+结构", "seq+lit": "序列+文献",
           "vote2": "投票≥2",
           "nn_top1": "同源 Top1", "nn_top3": "同源 Top3"}
FEAT_CN = {"seq_conf": "序列置信度", "struct_conf": "结构置信度",
           "lit_conf": "文献置信度", "n_src": "支持来源数",
           "seq_x_struct": "序列×结构", "seq_x_lit": "序列×文献",
           "struct_x_lit": "结构×文献", "ic_norm": "术语信息内容（归一）",
           "log_prevalence": "术语流行度（log）", "depth_norm": "DAG 深度（归一）",
           "collapsed": "祖-父折叠标记"}
EXP6_CN = {"multi": "三通道融合（规则仲裁）", "seq-only": "仅序列通道",
           "seq+struct": "序列+结构", "seq+lit": "序列+文献"}
ASP = {"F": "MFO", "P": "BPO", "C": "CCO"}
STRICT = "严格时间完整配置"
UPPER = "口径：全量签名库，含截止后信息，只作上界"
# 标题与句中不能用 UPPER 的“口径：”前缀，这里给一个可嵌入的同义串
UPPER_IN = "全量签名库口径（含截止后信息，只作上界）"
UPPER_TBL = "全量签名库配置"
# 表头与括号内使用无括号版，避免出现嵌套括号
UPPER_T = "全量签名库口径，只作上界"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def num(x):
    return f"{x:,}".replace(",", " ")


def q(x, nd=4):
    """与正文同一套四舍五入（ROUND_HALF_UP）与负号写法（U+2212）。"""
    s = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd),
                                             rounding=ROUND_HALF_UP))
    return s.replace("-", "−")


def ci(v, nd=4):
    """区间：（下界–上界）；下界为负而上界非负时上界补显式 +，正文同约定。"""
    lo, hi = q(v[0], nd), q(v[1], nd)
    if float(v[0]) < 0 <= float(v[1]):
        hi = "+" + hi
    return f"（{lo}–{hi}）"


def normalize(text):
    """统一正文排版：ASCII 连字符负号改 U+2212，弯引号改与正文一致的正引号。"""
    text = re.sub(r"(?<=[ \t|（,])-(?=\d)", "−", text)
    return text.replace("“", '"').replace("”", '"')


def _s4_rows(src):
    """src 给出统一的 (点估计, 区间) 取值方式，两套口径共用一张表头。"""
    L = ["| 路线 | 冻结工作点 t\\* | Fmax（95% CI） | 冻结 F1（95% CI） | "
         "Smin（95% CI） | AUPR（95% CI） | AUC（95% CI） |",
         "|---|---:|---|---|---|---|---|"]
    for a in ARMS:
        t, p, c, f = src(a)
        L.append(f"| {CN[a]} | {t:.2f} | {p['fmax']:.4f}{ci(c['fmax'])} | "
                 f"{p['f1']:.4f}{ci(f)} | {p['smin']:.4f}{ci(c['smin'])} | "
                 f"{p['aupr']:.4f}{ci(c['aupr'])} | {p['auc']:.4f}{ci(c['auc'])} |")
    L.append("")
    return L


def s3(se, bl):
    bs, d = se["bootstrap"], se["bootstrap"]["delta_ci"]
    L = []
    L.append(f"## 补充表 S3 学习仲裁对各对照路线的配对 bootstrap 差值"
             f"（{STRICT}，测试集整体，n = {num(bs['n_targets'])}，"
             f"B = {num(bs['B'])}，种子 {bs['seed']}）\n")
    L.append("差值定义为“学习仲裁 − 对照”，在同一蛋白级重采样中配对计算，"
             "区间为重采样分布的 2.5% 与 97.5% 分位。Smin 越低越好、AUC 为类中心宏平均。\n")
    L.append("| 对照 | ΔFmax | ΔSmin | ΔAUPR | ΔAUC | Δ冻结 F1 |")
    L.append("|---|---|---|---|---|---|")
    L += _delta_rows(d)
    L.append("")
    L += _s3_note(d)
    L.append(f"**{UPPER}** 全量签名库配置下的同一批比较，共 {len(bl['delta_ci'])} 条对照路线"
             "（序列+文献 一对在该批产物中未落盘），只作上界参照，"
             "不参与正文任何比较：\n")
    L.append("| 对照 | ΔFmax | ΔSmin | ΔAUPR | ΔAUC | Δ冻结 F1 |")
    L.append("|---|---|---|---|---|---|")
    L += _delta_rows(bl["delta_ci"])
    L.append("")
    return L


def _delta_rows(d):
    out = []
    for key, v in d.items():
        out.append(f"| {PAIR_CN[key.split('-', 1)[1]]} | "
                   f"{v['dFmax']:+.4f}{ci(v['dFmax_ci'])} | "
                   f"{v['dSmin']:+.4f}{ci(v['dSmin_ci'])} | "
                   f"{v['dAupr']:+.4f}{ci(v['dAupr_ci'])} | "
                   f"{v['dAuc']:+.4f}{ci(v['dAuc_ci'])} | "
                   f"{v['dFixed_F1']:+.4f}{ci(v['dFixed_F1_ci'])} |")
    return out


def _s3_note(d):
    nm = lambda k: PAIR_CN[k.split("-", 1)[1]]
    cross = [nm(k) for k, v in d.items() if v["dFixed_F1_ci"][0] <= 0 <= v["dFixed_F1_ci"][1]]
    auc_pos = [nm(k) for k, v in d.items() if v["dAuc_ci"][0] > 0]
    f1 = ("都显著为正" if not cross
          else f"除{'、'.join(cross)}（区间跨零）外都显著为正")
    return ["区间不含零即视为显著。全部 " + str(len(d)) + " 条对照路线的 ΔFmax 区间下界都在零以上，"
            f"Δ冻结 F1 {f1}；ΔAUC 显著为正的只有 {'、'.join(auc_pos)}，对其余路线为负或跨零，"
            "正文 2.3 节给出该反向的解释。\n"]


def s4(se, bl, exp6):
    r, bs = se["routes"], se["bootstrap"]
    L = []
    L.append(f"## 补充表 S4 十条路线的整体口径点估计与 95% 置信区间（{STRICT}，"
             f"测试集 n = {num(bs['n_targets'])}，B = {num(bs['B'])}）\n")

    def strict(a):
        v = r[a]
        return (v["t_star_one_se"],
                {k: v[k] for k in ("fmax", "smin", "aupr", "auc")}
                | {"f1": v["frozen_F1"]},
                {k: v[f"{k}_ci"] for k in ("fmax", "smin", "aupr", "auc")},
                v["frozen_F1_ci"])

    L += _s4_rows(strict)
    L.append("Fmax 取 101 点网格上的最大值，是依赖测试窗的 oracle 上界；"
             "冻结 F1 的阈值由验证窗 1-SE 规则选出后冻结应用测试窗，不使用测试窗信息，"
             "为可部署口径。两条口径同列报告，正文主结论只引用冻结 F1 与 AUPR/Smin。"
             "seq+struct 与 seq+lit 是通道消融配置，正文 2.5 节的两因素读法取自本表："
             "文献通道的增益为 规则仲裁 − 序列+结构（结构在场）与 序列+文献 − 仅序列（结构不在场），"
             "结构通道的增益为 序列+结构 − 仅序列 与 规则仲裁 − 序列+文献；"
             "以规则仲裁为参照的配对差值只在旧的全量口径产物中落盘，见补充表 S23。\n")
    p, c, f1ci = bl["point"], bl["ci"], bl["fixed_F1_ci"]

    def full(a):
        v = p[a]
        return (v["t_star"], {**v, "f1": v["fixed_F1"]},
                {k: c[a][f"{k}_ci"] for k in ("fmax", "smin", "aupr", "auc")}, f1ci[a])

    L.append(f"**{UPPER}** 十条路线在全量签名库配置下的同口径点估计"
             "（仅序列通道 与 序列+文献 两行不含结构通道，签名库限制对它们没有作用，"
             "故与严格配置块同值）：\n")
    L += _s4_rows(full)
    n13, n6 = num(bl["n_targets"]), num(exp6["multi"]["n_targets"])
    L.append(f"**{UPPER}** 的另一组参照收窄的是真值证据码：把真值白名单由 13 码收窄到 EXP 六码"
             f"（EXP/IDA/IPI/IMP/IGI/IEP）后，可评估蛋白由 {n13} 减至 {n6}，"
             "四配置（仍是全量签名库口径）重算如下；与上一块 13 码口径逐项对照，"
             "Fmax/AUPR/AUC 的差异都小于 0.01、Smin 的差异小于 0.1，排序与结论不变。"
             "严格配置下的 EXP 六码重算未落盘，因此本块只作上界参照。\n")
    L.append("| 配置 | 可评估蛋白数 | Fmax | Smin | AUPR | AUC | 覆盖率 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k in ("multi", "seq-only", "seq+struct", "seq+lit"):
        v = exp6[k]
        L.append(f"| {EXP6_CN[k]} | {num(v['n_targets'])} | {v['fmax']:.4f} | "
                 f"{v['smin']:.4f} | {v['aupr']:.4f} | {v['auc']:.4f} | {v['coverage']:.4f} |")
    L.append("")
    return L


def s17(pw, se):
    st = pw["strata"]
    order = ["overall", "MFO", "BPO", "CCO", "BPO_shallow", "BPO_mid", "BPO_deep"]
    name = {"overall": "全部可评估蛋白", "MFO": "MFO 限定术语集",
            "BPO": "BPO 限定术语集", "CCO": "CCO 限定术语集",
            "BPO_shallow": "BPO 浅层（深度 ≤4）", "BPO_mid": "BPO 中层（=5）",
            "BPO_deep": "BPO 深层（≥6）"}
    ft = pw["frozen_thresholds"]
    L = []
    L.append(f"## 补充表 S17 分层冻结 F1 全表与配对差值（{STRICT}，阈值冻结后只限定术语集，"
             f"B = {num(pw['bootstrap']['B'])}）\n")
    L.append("分层不重新选阈值：四条路线的工作点取各队在严格配置验证窗 1-SE 规则下冻结的结果"
             f"（学习仲裁 {ft['learn']['t_star']}、规则仲裁 {ft['fusion']['t_star']}、"
             f"朴素并集 {ft['union']['t_star']}、投票≥2 {ft['vote2']['t_star']}），"
             "与正文表 2、表 3 同源。学习仲裁在验证窗 F1 曲线上的标准误在本脚本为 "
             f"{ft['learn']['val_F1_se']:.5f}，主评估脚本给出的为 "
             f"{se['routes']['learn']['val_F1_se']:.4f}，两套脚本各自重采样验证窗、"
             "使用不同的随机流，1-SE 边界的判定与选出的阈值不受影响。"
             f"MDD₈₀ = {pw['bootstrap']['z80']:.2f} × SE(Δ配对)，"
             "为该层在 80% 功效下可检出的最小配对差；区间跨零的层不能读出方向。\n")
    L.append("| 层 | 蛋白数 | 术语数 | 真值注解数 | 学习仲裁 | 规则仲裁 | 朴素并集 | 投票≥2 | "
             "Δ(学−规) 95%CI | MDD₈₀ | Δ(学−并) 95%CI | Δ(规−并) 95%CI |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|")
    for k in order:
        s = st[k]
        f1, dl, fc = s["frozen_F1"], s["paired_delta_F1"], s["F1_ci95"]

        def cell(pair):
            v = dl[pair]
            mark = "" if v["significant"] else "†"
            return (f"{v['delta']:+.4f}{ci(v['ci95'])}{mark}")
        L.append(f"| {name[k]} | {num(s['n_proteins'])} | {num(s['n_terms'])} | "
                 f"{num(s['n_annotations'])} | "
                 f"{f1['learn']:.4f}{ci(fc['learn'])} | "
                 f"{f1['fusion']:.4f}{ci(fc['fusion'])} | "
                 f"{f1['union']:.4f}{ci(fc['union'])} | "
                 f"{f1['vote2']:.4f}{ci(fc['vote2'])} | "
                 f"{cell('learn-fusion')} | {dl['learn-fusion']['MDD_80']:.4f} | "
                 f"{cell('learn-union')} | {cell('fusion-union')} |")
    L.append("")
    L.append("† 表示 95% 区间包含零。学习仲裁在 BPO 及其三个深度层的冻结 F1 都为 0，"
             f"因为严格配置下它在 BPO 层（{num(pw['strata']['BPO']['n_proteins'])} 个可评估蛋白）"
             "不发射任何 BPO 术语、命中数为零（保留的行数与来源拆分见补充表 S20），"
             "正文 2.4 节按部署回归报告；"
             "朴素并集不做来源加权、把低置信结构候选一并发射，因而在 BPO 浅层仍是最高的对照，"
             "正文 2.7 节以分命名空间工作点分析该现象。\n")
    L.append("| 真值尺度核对 | MFO | BPO | CCO | 合计 |")
    L.append("|---|---:|---:|---:|---:|")
    sc = pw["scale"]
    for lab, key in [("有直接注解的蛋白数", "n_proteins_direct"),
                     ("直接注解条数", "n_direct_annotations"),
                     ("传播后注解条数", "n_propagated_annotations"),
                     ("每蛋白直接注解均值", "mean_direct_per_protein"),
                     ("每蛋白直接注解最大值", "max_direct_per_protein")]:
        vals = [sc[a][key] for a in ("MFO", "BPO", "CCO")]
        tot = sum(vals) if key != "mean_direct_per_protein" else sc["overall"].get(
            "mean_propagated_per_protein")
        if key == "mean_direct_per_protein":
            vals = [f"{v:.2f}" for v in vals]
            tot = "—"
        elif key == "max_direct_per_protein":
            tot = "—"
        else:
            vals = [num(v) for v in vals]
            tot = num(tot)
        L.append(f"| {lab} | {vals[0]} | {vals[1]} | {vals[2]} | {tot} |")
    L.append("")
    L.append(f"按 {num(sc['overall']['n_direct_annotations'])} 条直接注解与 "
             f"{num(sc['overall']['n_propagated_annotations'])} 条传播注解的实测分布，"
             f"每蛋白直接注解最多 "
             f"{max(sc[a]['max_direct_per_protein'] for a in ('MFO', 'BPO', 'CCO'))} 条，"
             f"各本体“注解不少于 64 条”的蛋白数都是 "
             f"{sc['MFO']['proteins_ge_64_direct']}，"
             "不存在“注解不少于 64 条”的富 annotator 亚层；"
             "因此不能按该口径切分高注解密度蛋白，正文 2.4 节改用本体深度三分位分层。"
             f"深层术语池 {pw['depth_terciles']['n_terms_pool']} 个，切点 "
             f"{pw['depth_terciles']['d33']} 与 {pw['depth_terciles']['d66']}，"
             f"深度 ≥6 的术语 {pw['depth_terciles']['n_deep_terms']} 个。\n")
    return L


def s19(sm, se, am):
    co, lam, oof = (sm["coefficients"], sm["lambda"], sm["oof_ece"])
    L = []
    L.append(f"## 补充表 S19 学习仲裁的系数、特征标准化参数与逐 λ 交叉验证对数损失"
             f"（{STRICT}，验证窗候选 {num(sm['n_val_candidates'])} 行、"
             f"{num(sm['n_val_positives'])} 行命中传播真值）\n")
    L.append("逻辑回归 `p = σ(w·z + b)`，z 为按命名空间各特征标准化后的候选行特征，"
             "L2 强度 λ 由分组交叉验证的对数损失选出，系数与分数下限都在受限验证窗上重新标定、"
             "测试窗触碰之前冻结。特征已标准化，系数可按绝对值直接比较。\n")
    L.append("| 特征 | MFO | BPO | CCO |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| 截距 | {co['F']['b0']:+.3f} | {co['P']['b0']:+.3f} | {co['C']['b0']:+.3f} |")
    for f in sm["feature_names"]:
        L.append(f"| {f}（{FEAT_CN[f]}） | {co['F']['w'][f]:+.3f} | "
                 f"{co['P']['w'][f]:+.3f} | {co['C']['w'][f]:+.3f} |")
    L.append(f"| 选定的 λ | {lam['F']:g} | {lam['P']:g} | {lam['C']:g} |")
    L.append("| 验证窗候选行数 | " + " | ".join(
        f"{num(oof[a]['n_row'])}" for a in ("F", "P", "C")) + " |")
    L.append("| 折外期望校准误差 ECE | " + " | ".join(
        f"{oof[a]['ece']:.5f}（正例率 {oof[a]['pos_rate']:.4f}）"
        for a in ("F", "P", "C")) + " |")
    L.append("")
    L.append("| λ | MFO 对数损失 | BPO 对数损失 | CCO 对数损失 |")
    L.append("|---|---:|---:|---:|")
    grid = se["model_selection"]["F"]["cv_logloss"]
    for x in grid:
        L.append(f"| {float(x):g} | " + " | ".join(
            f"{se['model_selection'][a]['cv_logloss'][x]:.5f}" for a in ("F", "P", "C")) + " |")
    L.append("")
    g = se["learn_gate"]
    L.append(f"学习路线的分数下限在同一条受限验证窗上重新搜索，选定 {g['selected']}：\n")
    L.append("| 分数下限 | 验证窗 Fmax | 验证窗 AUPR | 每蛋白平均原始术语数 | 验证窗覆盖蛋白数 |")
    L.append("|---|---:|---:|---:|---:|")
    for x, v in g["grid"].items():
        L.append(f"| {float(x):g} | {v['val_fmax']:.4f} | {v['val_aupr']:.4f} | "
                 f"{v['val_mean_terms_raw']:.2f} | {num(v['val_covered'])} |")
    L.append("")
    dep, ar = sm["deploy"], sm["arms"]
    L.append("| 部署口径（严格配置，测试窗） | 学习仲裁 | 规则仲裁 |")
    L.append("|---|---:|---:|")
    L.append(f"| 冻结 t\\* | {dep['learn']['t_star']:.2f} | {dep['fusion']['t_star']:.2f} |")
    L.append(f"| 微平均 P | {dep['learn']['P']:.4f} | {dep['fusion']['P']:.4f} |")
    L.append(f"| 宏平均 R | {dep['learn']['R']:.4f} | {dep['fusion']['R']:.4f} |")
    L.append(f"| 覆盖蛋白数 | {num(dep['learn']['covered'])} | {num(dep['fusion']['covered'])} |")
    L.append(f"| 每蛋白平均发射术语数 | {ar['learn']['mean_terms_at_frozen_tstar']:.1f} | "
             f"{ar['fusion']['mean_terms_at_frozen_tstar']:.1f} |")
    L.append(f"| 测试窗传播后口径 ECE | {ar['learn']['test_ece_on_propagated']:.4f} | "
             f"{ar['fusion']['test_ece_on_propagated']:.4f} |")
    L.append(f"| 两法共同覆盖的蛋白数 | {num(sm['common_coverage_subset']['n'])} | 同左 |")
    cc = sm["common_coverage_subset"]
    L.append(f"| 共同覆盖子集上的 Fmax | {cc['learn']['fmax']:.4f} | {cc['fusion']['fmax']:.4f} |")
    L.append(f"| 共同覆盖子集上的 AUPR | {cc['learn']['aupr']:.4f} | {cc['fusion']['aupr']:.4f} |")
    me = sm["matched_emission_auc"]
    L.append(f"| 同发射类（{num(me['n_classes'])} 类）宏 AUC | {me['learn']:.4f} | "
             f"{me['fusion']:.4f} |")
    L.append(f"| 各自已发射类的宏 AUC | {ar['learn']['auc_breakdown']['macro_auc_emitted_only']:.4f} | "
             f"{ar['fusion']['auc_breakdown']['macro_auc_emitted_only']:.4f} |")
    L.append("")
    # 严格配置的标准化参数未单独落盘，均值/标准差只有全库口径的产物，故按上界参照列出
    L.append(f"**{UPPER}** 各特征在验证窗候选上的均值与标准差（严格配置未落盘这批参数，"
             "只作量级参照，不可与上表的严格配置系数直接相乘还原）：\n")
    L.append("| 特征 | 命名空间 | 均值 | 标准差 |")
    L.append("|---|---|---:|---:|")
    for a in ("F", "P", "C"):
        for f in am["features"]:
            L.append(f"| {f} | {ASP[a]} | {am['models'][a]['mean'][f]:.4f} | "
                     f"{am['models'][a]['std'][f]:.4f} |")
    L.append("")
    L.append("三个命名空间的 λ 都选到网格上界，逐 λ 的对数损失差异出现在第 4 至第 5 位小数"
             "（见上表 MFO 与 CCO 两列），说明 11 个特征上没有过拟合迹象；"
             "λ 偏大与验证窗候选正例率低（正文 2.3 节）、尤其 BPO 的袋外正例率极低一致，"
             "其后果是该本体的整个概率分布落到全局工作点之下（正文 2.4 节）。\n")
    return L


def s20(sm, sp):
    keys = ["seq", "struct", "lit", "seq|struct", "lit|seq", "lit|struct", "lit|seq|struct"]
    head = {"seq": "仅序列", "struct": "仅结构", "lit": "仅文献",
            "seq|struct": "序列+结构", "lit|seq": "文献+序列",
            "lit|struct": "文献+结构", "lit|seq|struct": "三路同意"}
    L = []
    L.append(f"## 补充表 S20 保留候选行按证据来源组合拆分的精度"
             f"（{STRICT}，测试窗，先筛选后传播口径）\n")
    L.append("“保留候选行”为仲裁层在各自最低分处保留、尚未做 DAG 传播的原始行，"
             "与正文 2.6 节口径相同；精度为该集合内传播前候选命中测试窗真值的比例，"
             "与仲裁权重无关。“文献单源行置零”是把仅文献行分数置 0 后的消融臂，"
             "与正文表 2 的置零消融行、补充表 S24 同一手法；规则仲裁在该口径下不保留任何仅文献行，"
             "故其仅文献列为空。\n")
    L += _s20_rows(sm["source_precision"],
                   ("learn", "fusion", "learn_nolitonly"), head, keys)
    la = sm["lit_only_rows_by_aspect"]
    L.append("学习仲裁保留的仅文献单源行按命名空间拆分（这些行不含 BPO 术语，"
             "故 BPO 无对应行，正文表 3 的注释同此）：\n")
    L.append("| 命名空间 | 仅文献单源保留行数 | 精度 |")
    L.append("|---|---:|---:|")
    for a in ("F", "P", "C"):
        if a not in la:
            continue
        v = la[a]
        L.append(f"| {ASP[a]} | {num(v['rows'])} | {q(v['precision'] * 100, 2)}% |")
    L.append("")
    L.append(f"**{UPPER}** 同一张表在全量签名库配置下的取值（对照路线为投票≥2 而非置零消融）：\n")
    L += _s20_rows(sp, ("learn", "fusion", "vote2"), head, keys)
    return L


def _s20_rows(sp, arms, head, keys):
    L = ["| 路线 | 来源组合 | 保留行数 | 命中真值 | 精度 |",
         "|---|---|---:|---:|---:|"]
    for arm in arms:
        for k in keys:
            bs = sp[arm]["by_source"]
            if k not in bs:
                continue
            v = bs[k]
            L.append(f"| {CN[arm]} | {head[k]} | {num(v['rows'])} | "
                     f"{num(v['in_truth'])} | {q(v['precision'] * 100, 2)}% |")
    L.append("")
    L.append("| 路线 | 命名空间 | 保留行数 | 精度 |")
    L.append("|---|---|---:|---:|")
    for arm in arms:
        for a in ("C", "F", "P"):
            v = sp[arm]["by_aspect"][a]
            L.append(f"| {CN[arm]} | {ASP[a]} | {num(v['rows'])} | "
                     f"{q(v['precision'] * 100, 2)}% |")
    L.append("")
    return L


def s21(sa):
    L = []
    lt = sa["library_temporal_table"]
    L.append("## 补充表 S21 结构通道签名库版本 census 与受限重跑（测试集）\n")
    vers = sorted({v for s in sa["per_target_set_xml"].values()
                   for v in s["interproscan_versions"]})
    L.append(f"全部 {len(sa['per_target_set_xml'])} 批、共 "
             f"{num(sum(s['n_xml_files'] for s in sa['per_target_set_xml'].values()))} 个 "
             "InterProScan 输出 XML（含主协议三批 "
             f"{num(sum(sa['per_target_set_xml'][k]['n_xml_files'] for k in ('split2_train', 'split3_val', 'split3_test')))} 个）"
             f"的 `<libraryManagement>` 段只报告分发版本 {'、'.join(vers)}"
             "，签名库知识截止无法从版本字段逐项还原。"
             f"不晚于 T0（{sa['T0']}）的最后一个官方版本为 "
             f"{sa['last_release_not_later_than_T0']}，"
             f"所用版本在其后 {sa['gap_n_releases_after_T0']} 个版本、间隔 "
             f"{sa['gap_days_after_T0']} d。"
             "下表按“版本号是否自带日期”把签名库分为两类：只有自带日期的库能与 T0 对齐。\n")
    test_libs = sa["per_target_set_xml"]["split3_test"]["libraries"]
    L.append("| 签名库 | XML 中的拼写 | 版本 | 版本号自带日期 | 与 T0 的关系 | 测试窗签名命中数 |")
    L.append("|---|---|---|---|---|---:|")
    for lib, v in sorted(lt.items(),
                         key=lambda kv: -test_libs.get(kv[0], {}).get("n_matches", 0)):
        n = test_libs.get(lib, {}).get("n_matches", 0)
        rel = ("早于 T0" if v["temporal_class"] == "pre_T0"
               else f"不可判定，随 {vers[0]}（{v['bundled_release_date']}）分发")
        L.append(f"| {lib} | {v['xml_spelling'][0]} | {v['version']} | "
                 f"{'是' if v['date_coded'] else '否'} | {rel} | {num(n)} |")
    L.append("")
    r, sh = sa["restriction"], sa["share_struct_rows_from_pre_T0_datecoded"]
    L.append(f"只保留自带日期且不晚于 T0 的 {len(sa['keep_pre_T0_libraries'])} 个库"
             f"（{'、'.join(sa['keep_pre_T0_libraries'])}）时，"
             f"验证窗结构候选由 {num(r['datecoded_pre_T0']['val']['n_rows_in'])} 行降至 "
             f"{r['datecoded_pre_T0']['val']['n_rows_kept']} 行、"
             f"测试窗由 {num(r['datecoded_pre_T0']['test']['n_rows_in'])} 行降至 "
             f"{r['datecoded_pre_T0']['test']['n_rows_kept']} 行，"
             f"即主配置中该批签名库占结构候选的比例为验证窗 {q(sh['split2_val'] * 100, 2)}%、"
             f"测试窗 {q(sh['split3_test'] * 100, 2)}%；把无法与 T0 对齐的那批签名库从结构通道删去，"
             "即得正文表 4 与补充表 S4 所用的严格配置。\n")
    sens = sa["sensitivity"]
    L.append("| 结构通道配置 | 路线 | 测试窗 Fmax | 测试窗 AUPR | 冻结 F1（1-SE） | 验证窗 Fmax |")
    L.append("|---|---|---:|---:|---:|---:|")
    for cfg, lab in (("full", f"全部 {len(lt)} 库（{UPPER_T}）"),
                     ("datecoded_pre_T0", f"仅日期编码且不晚于 T0 的 "
                      f"{len(sa['keep_pre_T0_libraries'])} 库（{STRICT}）"),
                     ("dropped", "完全去掉结构通道")):
        for arm in ("fusion", "learn"):
            s = sens[f"{arm}_{cfg}"]
            L.append(f"| {lab} | {CN[arm]} | {s['test_grid']['fmax']:.4f} | "
                     f"{s['test_grid']['aupr']:.4f} | {s['test_F1_one_se']:.4f} | "
                     f"{s['val']['fmax']:.4f} |")
    L.append("")
    dep = sa["test_window_struct_dependence"]
    L.append(f"测试窗 {num(dep['n_targets'])} 个可评估蛋白中，{num(dep['targets_with_struct_evidence'])} "
             f"个有结构证据、{dep['targets_with_struct_as_only_channel']} 个只有结构证据，"
             "后者是签名库版本泄漏直接影响指标的蛋白规模上限；"
             f"严格配置下只剩 {num(sa['restriction']['datecoded_pre_T0']['test']['n_proteins_kept'])} "
             "个蛋白仍有结构证据。\n")
    return L


def s22(op, op_full):
    L = []
    L.append(f"## 补充表 S22 分命名空间工作点的验证窗曲线与选点（{STRICT}，"
             f"先筛选后传播口径，网格 {'、'.join(str(g) for g in op['grid'])}，"
             f"标准误由 {num(op['n_se_boot'])} 次验证窗蛋白级自助给出，B = {num(op['B'])}）\n")
    L.append("正文表 5 下半给出该口径下的测试窗结果，本表给出选点过程。")
    L += _op_block(op)
    L.append(f"**{UPPER}** 同一张表在全量签名库配置下的选点过程：")
    L += _op_block(op_full)
    return L


def _op_block(op):
    L = ["两条路线的 MFO/BPO/CCO 下限都来自各自验证窗的 1-SE 规则。"
         "学习仲裁在 BPO 的 F1 随下限升高单调下降、到 0.05 已归零，"
         f"1-SE 规则因此给出网格下界 {op['grid'][0]}，是一个边界解而非内点，"
         "说明这一命名空间的可用工作点不在网格内部。\n"]
    L.append("| 路线 | 命名空间 | 验证窗蛋白数 | 1-SE 下限 | argmax 下限 | 峰值处 SE | "
             "1-SE 处验证 F1 | argmax 处验证 F1 |")
    L.append("|---|---|---:|---|---|---:|---:|---:|")
    for arm in ("learn", "fusion"):
        for a in ("F", "P", "C"):
            g = op["per_aspect_gate"][arm][a]
            L.append(f"| {CN[arm]} | {ASP[a]} | {num(g['n_val_proteins'])} | "
                     f"{g['gate_one_se']} | {g['gate_argmax']} | {g['se_at_peak']:.5f} | "
                     f"{g['val_F1_at_one_se']:.4f} | {g['val_F1_at_argmax']:.4f} |")
    L.append("")
    L.append("| 路线 | 命名空间 | " + " | ".join(str(g) for g in op["grid"]) + " |")
    L.append("|---|---|" + "---:|" * len(op["grid"]))
    for arm in ("learn", "fusion"):
        for a in ("F", "P", "C"):
            row = op["val_curve"][arm][a]
            L.append(f"| {CN[arm]} | {ASP[a]} | " + " | ".join(
                f"{row[str(g)][0]:.4f}" for g in op["grid"]) + " |")
    L.append("")
    d = op["paired_delta_F1"]
    L.append("| 比较（均为先筛选后传播口径，指标为全本体微平均 F1） | Δ | 95% CI | 是否显著 |")
    L.append("|---|---:|---|---|")
    lab = {"learn_peraspect-learn_globalfloor": "学习仲裁：分命名空间下限 − 单一全局下限",
           "fusion_peraspect-fusion_globalfloor": "规则仲裁：分命名空间下限 − 单一全局下限",
           "learn_peraspect-fusion_peraspect": "同用分命名空间下限时：学习仲裁 − 规则仲裁",
           "learn_globalfloor-fusion_globalfloor": "同用单一全局下限时：学习仲裁 − 规则仲裁"}
    for k, v in d.items():
        L.append(f"| {lab[k]} | {v['delta']:+.4f} | {ci(v['ci95'])} | "
                 f"{'是' if v['significant'] else '否'} |")
    L.append("")
    gates = sorted({str(op["per_aspect_gate"]["fusion"][a]["gate_one_se"])
                    for a in ("F", "P", "C")})
    L.append("学习仲裁放开 BPO 闸门可以把 BPO 内部的 F1 从关闭状态救回，"
             "但全本体微平均 F1 显著下降，因此部署口径选择单一全局下限，"
             "正文 2.7 节按这一取舍陈述。规则仲裁的两种工作点差别很小，"
             f"其分命名空间下限为 {'、'.join(gates)}。\n")
    return L


def s18(ls):
    L = []
    L.append(f"## 补充表 S18 文献通道抽检样本（{STRICT}的预测文件，"
             f"{ls['sample_n']} 行分层抽样，正文 2.6 节与表 5）\n")
    L.append(f"分层按 命名空间 × 共现篇数 × 是否输出 抽取，判定对象为测试窗 "
             f"{num(ls['all_lit_rows']['n'])} 行文献候选，其 PMID 由本地检索缓存复原"
             f"（{num(ls['recovery']['n_rows_resolved'])}/{num(ls['recovery']['n_rows_all'])}），"
             "不依赖在线接口。"
             "标题命中 = 候选术语短语出现在某篇命中文献的标题中；"
             "基因名命中 = 查询蛋白的基因名/别名出现在摘要句子中；"
             "命中真值 = 该候选（含传播父 term）出现在测试窗真值集中，即独立精度判定。"
             "证据链只列 PMID 与首个命中文本片段。"
             "最后一列为该候选在严格配置的学习仲裁输出中的来源标记，"
             "组合记法（lit|seq 等）与补充表 S20 相同，未输出者记为空。\n")
    L.append("| 蛋白 | GO 术语 | 本体 | 共现篇数 | 标题命中 | 基因名命中 | 命中真值 | "
             "学习仲裁输出 | 证据链（截断） |")
    L.append("|---|---|---|---:|---|---|---|---|---|")
    yes = lambda v: "是" if v == "1" else "否"
    esc = lambda s: s.replace("|", "\\|")  # 来源组合里的竖线会打断 Markdown 表格
    with open(ROOT / ls["sample_file"], encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            prot, go, asp, name, cites, th, gs, hit, learn, _query, ev = f[:11]
            frag = " ".join(ev.split("||")[0].split())
            if len(frag) > 150:
                frag = frag[:150] + "…"
            L.append(f"| {prot} | {go}（{name}） | {ASP[asp]} | {cites} | {yes(th)} | "
                     f"{yes(gs)} | {yes(hit)} | {esc(learn)} | {esc(frag)} |")
    L.append("")
    return L


def s23(bc):
    d = bc["delta_ci"]
    L = []
    L.append(f"## 补充表 S23 通道消融与朴素基线的配对 bootstrap 差值"
             f"（以规则仲裁为参照、{UPPER_T}，测试集整体，"
             f"n = {num(bc['n_targets'])}，B = {num(bc['B'])}）\n")
    L.append("差值定义为“规则仲裁 − 对照”，用于分离通道贡献；"
             "学习仲裁相对同一批对照路线的差值见补充表 S3，两者不可混用。"
             f"本表是通道消融配对的唯一落盘产物，只在{UPPER_IN}下计算，"
             f"{STRICT}下未重算以规则仲裁为参照的配对差值；"
             f"{STRICT}的通道消融点估计见补充表 S4 的 序列+结构 与 序列+文献 两行，"
             "学习仲裁为参照的严格配置配对差见补充表 S3。\n")
    L.append("| 对照 | ΔFmax | ΔSmin | ΔAUPR | ΔAUC |")
    L.append("|---|---|---|---|---|")
    for k, v in d.items():
        L.append(f"| {PAIR_CN[k.split('-', 1)[1]]} | "
                 f"{v['dFmax']:+.4f}{ci(v['dFmax_ci'])} | "
                 f"{v['dSmin']:+.4f}{ci(v['dSmin_ci'])} | "
                 f"{v['dAupr']:+.4f}{ci(v['dAupr_ci'])} | "
                 f"{v['dAuc']:+.4f}{ci(v['dAuc_ci'])} |")
    L.append("")
    return L


def s24(se, lc, bl):
    ab = se["lit_only_ablation"]
    st, fu = ab["strict"], ab["full"]
    r, f1 = se["routes"], lc["frozen_F1"]
    z, y = st["own_threshold"], fu["own_threshold"]
    ref = fu["reference"]

    def keep(name):
        """未置零的参照臂在路线名后加标记，避免嵌套括号。"""
        return f"{name[:-1]}，未置零）" if name.endswith("）") else f"{name}（未置零）"

    L = []
    L.append(f"## 补充表 S24 文献单源行的交付消融（把仅文献行的仲裁分数置 0，"
             f"{STRICT}，测试集整体，n = {num(lc['n_targets'])}，B = {num(lc['B'])}，"
             f"种子 {lc['seed']}）\n")
    L.append("手法与投票≥2 相同：在仲裁层输入里把只有文献通道支持"
             "（同一术语上没有序列或结构签名）的候选行分数置 0，其余候选行、标准化参数、"
             "DAG 传播与工作点选择规则都不变，两臂因此严格可配对。"
             "阈值仍在验证窗按 1-SE 规则选出后冻结应用测试窗，"
             "P、R、覆盖蛋白数都取该臂自身冻结工作点处的值，口径与正文表 2 相同（先传播后阈值）；"
             "Fmax 与 AUPR 由整条分数排序给出、与所选阈值无关。\n")
    head = ["| 交付口径 | 路线 | 冻结工作点 t\\* | P | R | 冻结 F1（95% CI） | Fmax | AUPR | "
            "覆盖蛋白数 |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    row = lambda c, n, t, p, rr, f, fx, au, cv: (
        f"| {c} | {n} | {t} | {p} | {rr} | {f} | {fx} | {au} | {cv} |")
    L += head
    L.append(row(STRICT, CN["learn_nolitonly"], f"{z['t_star']:.2f}",
                 f"{z['precision']:.4f}", f"{z['recall']:.4f}",
                 f"{z['frozen_F1']:.4f}{ci(f1['learn_nolitonly']['ci95'])}",
                 f"{st['fmax']:.4f}", f"{st['aupr']:.4f}", num(z['covered'])))
    for a in ("learn", "fusion", "nn_top1"):
        v = r[a]
        L.append(row(STRICT, keep(CN[a]), f"{v['t_star_one_se']:.2f}",
                     f"{v['test_P_one_se']:.4f}", f"{v['test_R_one_se']:.4f}",
                     f"{v['frozen_F1']:.4f}{ci(v['frozen_F1_ci'])}",
                     f"{v['fmax']:.4f}", f"{v['aupr']:.4f}",
                     num(v["test_covered_one_se"])))
    L.append("")
    L.append("配对差值在同一批蛋白级重采样中计算，定义为“置零臂 − 对照臂”，"
             "各臂都取自己的冻结工作点；区间下界大于零即视为置零带来增益。\n")
    L.append("| 比较 | Δ冻结 F1（95% CI） | 判定 |")
    L.append("|---|---|---|")
    for base in ("learn", "fusion", "nn_top1"):
        v = lc["paired_delta"][f"learn_nolitonly-{base}"]
        if v["ci95"][0] > 0:
            jd = "置零带来增益"
        elif v["ci95"][0] <= 0 <= v["ci95"][1]:
            jd = "跨零，读不出方向"
        else:
            jd = "置零带来损失"
        L.append(f"| 置零臂 − {CN[base]} | {v['delta']:+.4f}{ci(v['ci95'])} | {jd} |")
    L.append("")
    same = st["same_t_star"]
    L.append(f"严格配置下学习仲裁在自己的工作点 {z['t_star']} 上真正发射了 "
             f"{num(st['lit_only_rows_in_reference_delivery'])} 行仅文献单源行，精度为 "
             f"{q(st['lit_only_row_precision'] * 100, 2)}%。把这些行置零后覆盖蛋白由 "
             f"{num(st['reference']['covered'])} 减到 {num(z['covered'])}，"
             f"冻结 F1 从 {st['reference']['frozen_F1']:.4f} 升到 {z['frozen_F1']:.4f}，"
             f"配对差 {same['delta']:+.4f}{ci(same['delta_ci'])}即这一动作的增益"
             f"（两臂都在 {same['t_star']} 上取值）；"
             f"网格口径的 Fmax 与 AUPR 基本不变"
             f"（ΔFmax {st['delta_fmax']:+.4f}、ΔAUPR {st['delta_aupr']:+.4f}），"
             "说明增益全部来自发射集合中低精度行的剔除，不来自分数排序的改动。"
             "该行级精度的来源拆分见补充表 S20；正文 2.6 节据此把文献通道读作佐证通道，"
             "而不是可独立交付的来源。\n")
    L.append(f"**{UPPER}** 同一消融在全量签名库配置下的取值"
             "（这一批产物里置零臂没有落盘冻结 F1 的置信区间、参照臂没有落盘 P、R、"
             "覆盖蛋白数，因此同列留空）：\n")
    L += head
    L.append(row(UPPER_TBL, CN["learn_nolitonly"], f"{y['t_star']:.2f}",
                 f"{y['precision']:.4f}", f"{y['recall']:.4f}", f"{y['frozen_F1']:.4f}",
                 f"{fu['fmax']:.4f}", f"{fu['aupr']:.4f}", num(y['covered'])))
    L.append(row(UPPER_TBL, "学习仲裁，未置零", f"{ref['t_star']:.2f}", "—", "—",
                 f"{ref['frozen_F1']:.4f}{ci(bl['fixed_F1_ci']['learn'])}",
                 f"{ref['fmax']:.4f}", f"{ref['aupr']:.4f}", "—"))
    L.append("")
    s2 = fu["same_t_star"]
    L.append(f"全量签名库配置下学习仲裁在自己的工作点 {ref['t_star']} 上只发射 "
             f"{num(fu['lit_only_rows_in_reference_delivery'])} 行仅文献单源行、精度 "
             f"{q(fu['lit_only_row_precision'] * 100, 2)}%，"
             f"因此同一阈值 {s2['t_star']} 处的配对差只有 {s2['delta']:+.4f}"
             f"{ci(s2['delta_ci'])}（覆盖蛋白 {num(s2['reference_covered'])} 对 "
             f"{num(s2['ablated_covered'])}）。置零臂自己重选工作点后冻结 F1 为 "
             f"{y['frozen_F1']:.4f}，比未置零的 {ref['frozen_F1']:.4f} 低 "
             f"{abs(fu['delta_own_vs_reference_frozen_F1']):.4f}："
             "被删的行虽然几乎没有真值，却改变了验证窗 F1 曲线的形状，"
             "1-SE 规则因此把阈值挪到测试窗上的劣质位置，这是选点副作用而非通道结论，"
             "与严格配置的方向相反。两套口径的差别只在被发射的仅文献行数（"
             f"{num(st['lit_only_rows_in_reference_delivery'])} 对 "
             f"{num(fu['lit_only_rows_in_reference_delivery'])}），"
             "正文 2.6 节的交付结论以严格配置为准。\n")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "supp_tables.md"))
    args = ap.parse_args()
    se = load("strict_eval.json")
    bl = load("bootstrap_learn.json")
    sm = load("strict_mech.json")
    L = []
    L += s3(se, bl)
    L += s4(se, bl, load("exp6_sensitivity.json"))
    L += s17(load("power_analysis_strict.json"), se)
    L += s18(load("lit_spotcheck_strict.json"))
    L += s19(sm, se, load("arbiter_model.json"))
    L += s20(sm, load("source_precision.json"))
    L += s21(load("struct_library_audit.json"))
    L += s22(load("operating_point_strict.json"), load("operating_point.json"))
    L += s23(load("bootstrap_ci.json"))
    L += s24(se, load("litonly_ci.json"), bl)
    Path(args.out).write_text(normalize("\n".join(L).rstrip() + "\n"),
                              encoding="utf-8")
    print(f"已写出 {args.out}（{len(L)} 行）")


if __name__ == "__main__":
    main()
