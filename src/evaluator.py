# -*- coding: utf-8 -*-
"""
评估模块：PSB26 / CAFA 四指标（Fmax / Smin / AUPR / AUC）
=========================================================
对齐 PSB26 (Pacific Symposium on Biocomputing 2026) 评估协议：

- Fmax: 蛋白质中心最大F1。阈值网格 0.00-1.00 步长0.01（101点）；
        精度=有预测蛋白均值，召回=全部有真值蛋白均值
- Smin: 语义距离 sqrt(ru² + mi²) 在阈值网格上的最小值；
        IC(t) = log2(min(直接父节点计数) / 项计数)，由 训练库真值+评估真值
        的传播注释统计（跨全部命名空间，与 PSB26 协议一致）；
        ru=漏报项IC之和均值，mi=误报项IC之和均值
- AUPR: 101阈值点的 (召回,精度) 曲线按召回排序后的梯形积分
- AUC:  类中心指标——逐GO类 ROC AUC（Mann-Whitney 秩法，并列取中秩），
        对">=1正例且>=1负例"的类取平均
- 评估前预测与真值沿 is_a/part_of 传播（预测分数取蕴含该术语的最大分），
  剔除三个子本体根术语（PSB26 的 Fmax/Smin/AUPR 同样剔除根术语）
- 分子本体 MFO(F) / 生物过程 BPO(P) / 细胞组分 CCO(C) 分别报告
- 本流水线为确定性工具链：单次运行（对照方法才需多次运行取均值±标准差，
  确定性方法的方差为 0，报告中注明）

Coverage（有≥1条预测的目标蛋白比例）为项目自有补充指标，非 PSB26 协议。
"""
from __future__ import annotations

import bisect
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .channels.base import read_tsv
from .go_dag import GODag

log = logging.getLogger("evaluator")

IEA_CODES = {"IEA"}
ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}   # MFO/BPO/CCO 根
ASPECT_ONTOLOGY = {"F": "MFO", "P": "BPO", "C": "CCO"}


def load_predictions(tsv_path: str) -> Dict[str, Dict[str, float]]:
    """预测TSV -> {protein: {go: score}}"""
    preds: Dict[str, Dict[str, float]] = {}
    for r in read_tsv(tsv_path):
        try:
            s = float(r.get("score") or 0)
        except ValueError:
            s = 0.0
        preds.setdefault(r["protein_id"], {})[r["go_id"]] = max(
            s, preds.get(r["protein_id"], {}).get(r["go_id"], 0.0))
    return preds


def load_truth(tsv_path: str, exclude_iea: bool = True,
               dag: Optional[GODag] = None,
               include_codes: Optional[set] = None) -> Dict[str, Set[str]]:
    """真值TSV -> {protein: {go}}（可排除IEA；可限定证据码白名单）"""
    truth: Dict[str, Set[str]] = {}
    for r in read_tsv(tsv_path):
        ev = r.get("evidence", "")
        if include_codes is not None and ev not in include_codes:
            continue
        if exclude_iea and ev in IEA_CODES:
            continue
        go = r["go_id"]
        if dag is not None:
            go = dag.normalize(go)
            if not dag.is_valid(go):
                continue
        truth.setdefault(r["protein_id"], set()).add(go)
    return truth


class Evaluator:
    def __init__(self, cfg: dict, dag: GODag):
        self.cfg = cfg
        self.dag = dag
        ec = cfg.get("evaluation", {})
        self.propagate = bool(ec.get("propagate_terms", True))
        self.exclude_iea = bool(ec.get("exclude_iea_truth", True))
        # IC 统计口径：train_only=仅训练库真值（不接触评估真值，防指标泄漏）；
        # train_eval=训练库真值+评估真值（旧版/PSB26 参考实现口径，仅用于对照）
        self.ic_source = ec.get("ic_source", "train_only")
        # PSB26 阈值网格：0.00-1.00 步长 0.01（101点）
        self.thresholds = [i / 100.0 for i in range(101)]

    # ------------------------------------------------------------ 传播
    def _prop(self, gos: Set[str]) -> Set[str]:
        if not self.propagate:
            return {g for g in gos if g not in ROOT_TERMS}
        out = self.dag.propagate(gos)
        return out - ROOT_TERMS

    def _propagate_preds(self, gos: Dict[str, float]) -> Dict[str, float]:
        """预测传播至全部祖先：祖先分数 = 蕴含它的最大预测分（PSB26 语义），
        并剔除根术语。"""
        out: Dict[str, float] = {}
        for g, s in gos.items():
            g = self.dag.normalize(g)
            if not self.dag.is_valid(g) or g in ROOT_TERMS:
                continue
            out[g] = max(out.get(g, 0.0), s)
            if self.propagate:
                for a in self.dag.ancestors(g):
                    if a in ROOT_TERMS:
                        continue
                    out[a] = max(out.get(a, 0.0), s)
        return out

    # ------------------------------------------------------------ 信息内容 IC
    def compute_ic(self, annot_sets: List[Set[str]]) -> Dict[str, float]:
        """PSB26 IC：ic(t) = log2(min(直接父节点计数) / 计数(t))，
        基于传播注释集统计（跨全部命名空间）。根/无父术语 IC=0；
        未出现的术语查询时按 0 处理（PSB26 get_ic 语义）。"""
        cnt: Counter = Counter()
        for gos in annot_sets:
            cnt.update(gos)
        ic: Dict[str, float] = {}
        for go_id, n in cnt.items():
            t = self.dag.terms.get(go_id)
            parents = [p for p in (t.parents if t else ()) if p in cnt]
            if not parents:
                continue   # 根/孤立术语：IC=0（等价 log2(n/n)）
            min_n = min(cnt[p] for p in parents)
            ic[go_id] = math.log2(min_n / n) if min_n > n else 0.0
        return ic

    # ------------------------------------------------------------ 指标
    def fmax_smin_aupr(self, preds: Dict[str, Dict[str, float]],
                      truth: Dict[str, Set[str]],
                      ic: Dict[str, float]) -> Tuple[float, float, float, dict]:
        """蛋白质中心三指标（Fmax/Smin/AUPR），PSB26 阈值网格扫描。
        preds/truth 已限本体、已传播、已去根。"""
        targets = [p for p in truth if truth[p]]
        if not targets:
            return 0.0, 0.0, 0.0, {}
        n = len(targets)
        best_f, best_detail = 0.0, {}
        smin, smin_detail = math.inf, {}
        precisions: List[float] = []
        recalls: List[float] = []
        ic_get = ic.get
        for t in self.thresholds:
            p_sum, p_n = 0.0, 0
            r_sum = 0.0
            ru, mi = 0.0, 0.0
            for p in targets:
                tt = truth[p]
                pt = {g for g, s in preds.get(p, {}).items() if s >= t}
                inter = len(pt & tt)
                if pt:
                    p_sum += inter / len(pt)
                    p_n += 1
                    for g in pt - tt:
                        mi += ic_get(g, 0.0)
                for g in tt - pt:
                    ru += ic_get(g, 0.0)
                r_sum += inter / len(tt)
            precision = p_sum / p_n if p_n else 0.0
            recall = r_sum / n
            f = 2 * precision * recall / (precision + recall) \
                if precision + recall > 0 else 0.0
            if f > best_f:
                best_f = f
                best_detail = {"threshold": t, "precision": precision,
                               "recall": recall}
            ru_avg, mi_avg = ru / n, mi / n
            s = math.sqrt(ru_avg * ru_avg + mi_avg * mi_avg)
            if s < smin:
                smin = s
                smin_detail = {"threshold": t, "ru": round(ru_avg, 4),
                               "mi": round(mi_avg, 4)}
            precisions.append(precision)
            recalls.append(recall)
        # AUPR：按召回排序后的 (召回,精度) 梯形积分（PSB26 协议: np.trapz）
        order = sorted(range(len(recalls)), key=lambda i: recalls[i])
        aupr = 0.0
        for a, b in zip(order, order[1:]):
            dx = recalls[b] - recalls[a]
            if dx > 0:
                aupr += dx * (precisions[a] + precisions[b]) / 2.0
        return best_f, smin, aupr, {"fmax_detail": best_detail,
                                    "smin_detail": smin_detail}

    def _roc_auc(self, pos_scores: List[float], neg_scores: List[float]) -> float:
        """单类 ROC AUC（Mann-Whitney 秩法，并列中秩）。"""
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        srt = sorted(neg_scores)
        total = 0.0
        for s in pos_scores:
            lo = bisect.bisect_left(srt, s)
            hi = bisect.bisect_right(srt, s)
            total += lo + 0.5 * (hi - lo)
        return total / (n_pos * n_neg)

    def auc(self, preds: Dict[str, Dict[str, float]],
            truth: Dict[str, Set[str]], all_targets: List[str]) -> Tuple[float, int]:
        """类中心 AUC：对每个">=1正例且>=1负例"的GO类计算 ROC AUC 后取平均。
        类 = 评估真值（已传播）中出现的术语；全体目标蛋白构成样本空间。"""
        pos_by_term: Dict[str, List[str]] = defaultdict(list)
        for p in all_targets:
            for g in truth.get(p, ()):
                pos_by_term[g].append(p)
        n_all = len(all_targets)
        target_set = set(all_targets)
        # 预测倒排索引：term -> {protein: score}（仅限评估目标，防复用超集输出污染）
        pred_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        for p, gos in preds.items():
            if p not in target_set:
                continue
            for g, s in gos.items():
                pred_index[g][p] = s
        total_auc, n_classes = 0.0, 0
        for g, pos_ps in pos_by_term.items():
            n_pos = len(pos_ps)
            if n_pos >= n_all:
                continue   # 全正例类无判别意义
            pos_set = set(pos_ps)
            idx = pred_index.get(g, {})
            pos_scores = [idx.get(p, 0.0) for p in pos_ps]
            neg_scores = [s for p, s in idx.items() if p not in pos_set]
            n_zero_neg = n_all - n_pos - len(neg_scores)
            # 负例 = 有预测分的负例 + 其余零分负例
            srt = sorted(neg_scores)
            auc = 0.0
            for s in pos_scores:
                lo = bisect.bisect_left(srt, s)
                hi = bisect.bisect_right(srt, s)
                less = lo + (n_zero_neg if s > 0 else 0)
                eq = (hi - lo) + (n_zero_neg if s == 0 else 0)
                auc += less + 0.5 * eq
            total_auc += auc / (n_pos * (n_all - n_pos))
            n_classes += 1
        if n_classes == 0:
            return 0.5, 0
        return total_auc / n_classes, n_classes

    def coverage(self, preds: Dict[str, Dict[str, float]],
                 truth: Dict[str, Set[str]]) -> float:
        targets = [p for p in truth if truth[p]]
        if not targets:
            return 0.0
        have = sum(1 for p in targets if preds.get(p))
        return have / len(targets)

    # ------------------------------------------------------------ 主入口
    def evaluate(self, pred_tsv: str, truth_tsv: str, label: str,
                 train_tsv: Optional[str] = None,
                 out_json: Optional[str] = None,
                 target_ids: Optional[Set[str]] = None,
                 include_codes: Optional[set] = None) -> dict:
        preds_raw = load_predictions(pred_tsv)
        truth = load_truth(truth_tsv, exclude_iea=self.exclude_iea, dag=self.dag,
                           include_codes=include_codes)
        if target_ids is not None:
            # 演示模式/分批运行：只评估本次实际处理的目标蛋白（避免召回被稀释）
            truth = {p: g for p, g in truth.items() if p in target_ids}
        # 预测传播（祖先分数=蕴含它的最大预测分）
        preds = {p: self._propagate_preds(gos) for p, gos in preds_raw.items()}
        # 真值传播
        truth_p = {p: self._prop(gos) for p, gos in truth.items()}

        # IC 统计（修订版口径）：默认仅用训练库真值，评估真值不参与指标构建，
        # 杜绝测试信息泄漏；train_eval 为旧口径（训练+评估真值），仅敏感性对照。
        ic_sets: List[Set[str]] = []
        if train_tsv and Path(train_tsv).exists():
            train_truth = load_truth(train_tsv, exclude_iea=self.exclude_iea,
                                     dag=self.dag)
            ic_sets.extend(self._prop(gos) for gos in train_truth.values())
        if self.ic_source != "train_only":
            ic_sets.extend(truth_p.values())
        if not ic_sets:
            raise ValueError("ic_source=train_only 需要训练库注释 train_tsv")
        ic = self.compute_ic(ic_sets)

        all_targets = [p for p in truth_p if truth_p[p]]

        def _aspect_sets(aspect: str):
            preds_a = {p: {g: s for g, s in gos.items()
                           if self.dag.aspect(g) == aspect}
                       for p, gos in preds.items()}
            preds_a = {p: g for p, g in preds_a.items() if g}
            truth_a = {p: {g for g in gos if self.dag.aspect(g) == aspect}
                       for p, gos in truth_p.items()}
            truth_a = {p: g for p, g in truth_a.items() if g}
            return preds_a, truth_a

        # 整体（三本体合并；PSB26 以分本体结果为准，整体仅作参考）
        f, s, a, detail = self.fmax_smin_aupr(preds, truth_p, ic)
        auc_v, n_cls = self.auc(preds, truth_p, all_targets)
        report: dict = {
            "label": label,
            "protocol": "PSB26 / CAFA: Fmax, Smin, AUPR, AUC",
            "n_targets": len(truth),
            "n_predicted_proteins": len(preds),
            "fmax": round(f, 4),
            "smin": round(s, 4),
            "aupr": round(a, 4),
            "auc": round(auc_v, 4),
            "auc_n_classes": n_cls,
            "coverage": round(self.coverage(preds, truth_p), 4),
        }
        report.update({k: v for k, v in detail.items() if v})

        # 分子本体（PSB26 主报告口径：MFO/BPO/CCO）
        report["by_aspect"] = {}
        for aspect in ("F", "P", "C"):
            preds_a, truth_a = _aspect_sets(aspect)
            if not truth_a:
                continue
            fa, sa, aa, deta = self.fmax_smin_aupr(preds_a, truth_a, ic)
            auca, nclsa = self.auc(preds_a, truth_a, all_targets)
            report["by_aspect"][aspect] = {
                "ontology": ASPECT_ONTOLOGY[aspect],
                "fmax": round(fa, 4), "smin": round(sa, 4),
                "aupr": round(aa, 4), "auc": round(auca, 4),
                "auc_n_classes": nclsa,
                "coverage": round(self.coverage(preds_a, truth_a), 4),
                "n_targets": len(truth_a),
            }
            report["by_aspect"][aspect].update(
                {k: v for k, v in deta.items() if v})

        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as fo:
                json.dump(report, fo, ensure_ascii=False, indent=2)
            log.info("评估报告[%s] Fmax=%.4f Smin=%.4f AUPR=%.4f AUC=%.4f -> %s",
                     label, f, s, a, auc_v, out_json)
        return report

    def write_text_report(self, reports: List[dict], out_path: str):
        """汇总多组评估（如：单Channel消融 vs 多Channel）为文本报告。"""
        lines = ["=" * 72,
                 "评估报告（PSB26/CAFA协议：Fmax / Smin / AUPR / AUC）",
                 "确定性工具链单次运行（无随机性；对照方法才需多次运行取均值±标准差）",
                 "=" * 72, ""]
        for rep in reports:
            lines.append(f"## {rep['label']}")
            lines.append(f"  目标蛋白数: {rep['n_targets']}  有预测蛋白数: {rep['n_predicted_proteins']}")
            lines.append(f"  整体 Fmax={rep.get('fmax', 0):.4f}  Smin={rep.get('smin', 0):.4f}  "
                         f"AUPR={rep.get('aupr', 0):.4f}  AUC={rep.get('auc', 0):.4f}  "
                         f"Coverage={rep.get('coverage', 0):.4f}")
            if "fmax_detail" in rep:
                d = rep["fmax_detail"]
                lines.append(f"    (Fmax最优阈值={d.get('threshold', '-')} "
                             f"P={d.get('precision', 0):.4f} R={d.get('recall', 0):.4f})")
            for asp in ("F", "P", "C"):
                if asp in rep.get("by_aspect", {}):
                    b = rep["by_aspect"][asp]
                    lines.append(f"    [{b.get('ontology', asp)}] Fmax={b['fmax']:.4f} "
                                 f"Smin={b['smin']:.4f} AUPR={b['aupr']:.4f} "
                                 f"AUC={b['auc']:.4f} Cov={b['coverage']:.4f} "
                                 f"(n={b['n_targets']}, AUC类数={b.get('auc_n_classes', 0)})")
            lines.append("")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_path
