# -*- coding: utf-8 -*-
"""
Arbiter（显式规则仲裁器）
========================
白盒规则引擎，决策过程完全可追溯（JSONL审计日志）。

流程（每蛋白）：
  1. 汇聚三Channel候选，相同GO合并（无冲突），证据取各分量最优
  2. 冲突检测（基于GO-DAG拓扑）：
       - 父子关系 -> 层级冗余：取子节点，父节点证据并入子节点
       - 不同分支（互非祖先，含跨命名空间）-> 分支冲突：进入加权评分
  3. 加权评分（固定公式，分量均归一化到[0,1]）：
       score = 0.4*seq_conf + 0.35*struct_conf + 0.25*lit_conf
       seq_conf    = min(-log10(E-value), 10)/10        E<=1e-10 满置信
       struct_conf = 域匹配覆盖率                        （0-1）
       lit_conf    = min(log(citations+1), 5)/5          100篇引用满置信
     分支冲突双方按总分裁决：过线者保留（分数即优先级依据）
  4. 阈值过滤（min_score）+ 每蛋白上限（max_terms_per_protein）
  5. 未知处理：三者均低置信且无候选过线 -> 标记"功能未知"，不强行预测
  6. GO-DAG校验层：冗余父节点过滤 / 跨命名空间冲突检测 / 非法术语过滤
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .channels.base import read_tsv
from .go_dag import GODag

log = logging.getLogger("arbiter")


@dataclass
class Evidence:
    """单个GO候选的三方证据池（各分量取最优）。"""
    evalue: Optional[float] = None        # Seq-Channel 最优E-value（越小越好）
    identity: Optional[float] = None
    qcov: Optional[float] = None
    coverage: Optional[float] = None      # Struct-Channel 域覆盖率（越大越好）
    citations: Optional[int] = None       # Lit-Channel 引用篇数（越大越好）
    sources: Set[str] = field(default_factory=set)
    details: Dict[str, str] = field(default_factory=dict)  # channel -> detail

    def merge(self, other: "Evidence") -> "Evidence":
        """合并证据：数值分量取更优（evalue取小，其余取大），来源取并集。"""
        self.evalue = _min(self.evalue, other.evalue)
        self.identity = _max(self.identity, other.identity)
        self.qcov = _max(self.qcov, other.qcov)
        self.coverage = _max(self.coverage, other.coverage)
        self.citations = _max(self.citations, other.citations)
        self.sources |= other.sources
        self.details.update(other.details)
        return self


def _min(a, b):
    if a is None:
        return b
    return a if b is None else min(a, b)


def _max(a, b):
    if a is None:
        return b
    return a if b is None else max(a, b)


class Arbiter:
    def __init__(self, cfg: dict, go_dag: GODag):
        self.cfg = cfg
        self.dag = go_dag
        ac = cfg.get("arbiter", {})
        self.weights = ac.get("weights", {"seq": 0.40, "struct": 0.35, "lit": 0.25})
        # 可选：分本体权重覆写（默认关闭；启用后按 GO 本体取对应三元组）
        po = ac.get("per_ontology_weights")
        if po:
            missing = [k for k in ("MFO", "BPO", "CCO")
                       if not all(po.get(k, {}).get(c) is not None
                                  for c in ("seq", "struct", "lit"))]
            if missing:
                raise ValueError(f"per_ontology_weights 缺失或字段不全: {missing}")
        self.per_ont = po or None
        self.min_score = float(ac.get("min_score", 0.15))
        self.max_terms = int(ac.get("max_terms_per_protein", 15))
        self.low_conf = ac.get("low_conf", {"seq": 0.2, "struct": 0.2, "lit": 0.2})
        self.seq_logcap = float(cfg.get("seq_channel", {}).get("evalue_logcap", 10.0))
        self.lit_logcap = float(cfg.get("lit_channel", {}).get("citation_logcap", 5.0))
        # 可学习仲裁钩子：签名与 score() 一致的 (ev, go) -> float 回调。
        # 挂上后取代固定权重与分本体权重，min_score/上限/DAG校验/未知判定全部不变，
        # 因此"固定权重"与"学习式权重"两条路线在同一决策语义下可比。
        self.scorer = None
        # 冲突统计（训练集规则覆盖率验证）
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            "proteins": 0,
            "same_merges": 0,
            "parent_child_collapses": 0,
            "branch_conflicts": 0,
            "namespace_conflicts": 0,
            "illegal_terms": 0,
            "unknown_marked": 0,
            "proteins_with_predictions": 0,
            "branch_conflict_by_aspect": {"F": 0, "P": 0, "C": 0},
        }

    # ------------------------------------------------------------ 归一化
    def seq_conf(self, evalue: Optional[float]) -> float:
        if evalue is None or evalue <= 0:
            return 0.0
        return min(-math.log10(evalue), self.seq_logcap) / self.seq_logcap

    def struct_conf(self, coverage: Optional[float]) -> float:
        return coverage if coverage is not None else 0.0

    def lit_conf(self, citations: Optional[int]) -> float:
        if citations is None:
            return 0.0
        return min(math.log(citations + 1), self.lit_logcap) / self.lit_logcap

    def score(self, ev: Evidence, go: Optional[str] = None) -> float:
        if self.scorer is not None:
            return self.scorer(ev, go)
        w = self.weights
        if self.per_ont is not None and go is not None:
            aspect = self.dag.aspect(go)  # "F"/"P"/"C"
            w = self.per_ont.get({"F": "MFO", "P": "BPO", "C": "CCO"}.get(aspect, ""),
                                 self.weights)
        return (w.get("seq", 0.4) * self.seq_conf(ev.evalue)
                + w.get("struct", 0.35) * self.struct_conf(ev.coverage)
                + w.get("lit", 0.25) * self.lit_conf(ev.citations))

    # ------------------------------------------------------------ 主流程
    def arbitrate(self, seq_tsv: str, struct_tsv: str, lit_tsv: str,
                  out_pred_tsv: str, out_audit_jsonl: str) -> str:
        """读取三Channel输出，执行仲裁，写预测TSV与审计日志。"""
        self._reset_stats()   # 每个目标集独立统计（实例可能被复用）
        seq = self._load_seq(seq_tsv)
        struct = self._load_struct(struct_tsv)
        lit = self._load_lit(lit_tsv)

        all_proteins = set(seq) | set(struct) | set(lit)
        log.info("Arbiter输入: seq=%s struct=%s lit=%s 蛋白, 覆盖目标 %s",
                 f"{len(seq):,}", f"{len(struct):,}", f"{len(lit):,}", f"{len(all_proteins):,}")

        rows = []
        Path(out_pred_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_pred_tsv, "w", encoding="utf-8", newline="") as fo, \
                open(out_audit_jsonl, "w", encoding="utf-8") as fa:
            fo.write("protein_id\tgo_id\taspect\tscore\tseq_conf\tstruct_conf\tlit_conf\tsources\tdetail\n")
            for pid in sorted(all_proteins):
                result = self._arbitrate_protein(pid, seq.get(pid, {}), struct.get(pid, {}), lit.get(pid, {}))
                fa.write(json.dumps(result, ensure_ascii=False) + "\n")
                for t in result["final_terms"]:
                    fo.write("\t".join([
                        pid, t["go_id"], t["aspect"], f"{t['score']:.4f}",
                        f"{t['seq_conf']:.4f}", f"{t['struct_conf']:.4f}", f"{t['lit_conf']:.4f}",
                        "|".join(sorted(t["sources"])), t["detail"],
                    ]) + "\n")
                    rows.append(pid)

        log.info("Arbiter完成: %s 蛋白, 其中 %s 有预测输出, %s 标记功能未知; "
                 "合并=%s 层级冗余=%s 分支冲突=%s 跨命名空间=%s 非法术语=%s",
                 self.stats["proteins"], self.stats["proteins_with_predictions"],
                 self.stats["unknown_marked"], self.stats["same_merges"],
                 self.stats["parent_child_collapses"], self.stats["branch_conflicts"],
                 self.stats["namespace_conflicts"], self.stats["illegal_terms"])
        return out_pred_tsv

    # ------------------------------------------------------------ 单蛋白仲裁
    def collect(self, seq: Dict, struct: Dict, lit: Dict
                ) -> Tuple[Dict[str, Evidence], Dict[str, List[str]], int]:
        """仲裁步骤1–2：相同GO合并 + 父子层级坍缩。

        纯函数，不改动 self.stats，也不改动传入的证据字典，供特征抽取与
        正式仲裁共用同一条候选归并路径（学习式与固定权重两条路线因此特征一致）。
        返回 (候选GO->证据池, 父节点->被折叠子节点, 相同GO合并次数)。
        """
        cands: Dict[str, Evidence] = {}
        same_merges = 0
        for go, ev in seq.items():
            cands.setdefault(go, Evidence()).merge(ev)
        for go, ev in struct.items():
            if go in cands:
                same_merges += 1
            cands.setdefault(go, Evidence()).merge(ev)
        for go, ev in lit.items():
            if go in cands:
                same_merges += 1
            cands.setdefault(go, Evidence()).merge(ev)

        collapsed: Dict[str, List[str]] = {}
        go_list = list(cands.keys())
        for g in go_list:
            if g not in cands:
                continue
            is_redundant = False
            for other in go_list:
                if other == g or other not in cands:
                    continue
                if g in self.dag.ancestors(other):
                    is_redundant = True
                    collapsed.setdefault(g, []).append(other)
            if is_redundant and g not in collapsed:
                collapsed[g] = []
        for parent, children in collapsed.items():
            if parent not in cands:
                continue
            pev = cands.pop(parent)
            for ch in children:
                if ch in cands:
                    cands[ch].merge(pev)
                    cands[ch].details.setdefault("merged_from", parent)
        return cands, collapsed, same_merges

    def _arbitrate_protein(self, pid: str, seq: Dict, struct: Dict, lit: Dict) -> dict:
        self.stats["proteins"] += 1
        audit: dict = {"protein_id": pid}

        # ---- 1–2. 相同GO合并 + 父子层级坍缩 ----
        cands, collapsed, same_merges = self.collect(seq, struct, lit)
        self.stats["same_merges"] += same_merges
        audit["same_merges"] = same_merges

        if not cands:
            audit["unknown"] = True
            audit["final_terms"] = []
            self.stats["unknown_marked"] += 1
            return audit

        self.stats["parent_child_collapses"] += len(collapsed)
        audit["parent_child_collapses"] = {p: c for p, c in collapsed.items()}

        # ---- 3. 分支冲突检测（不同分支 -> 加权评分裁决） ----
        scored: Dict[str, dict] = {}
        branch_conflicts = []
        gos = sorted(cands.keys())
        for i, a in enumerate(gos):
            for b in gos[i + 1:]:
                if self.dag.relation(a, b) == "disconnected":
                    # 跨命名空间计入namespace冲突；同命名空间为分支冲突
                    nsa, nsb = self.dag.namespace(a), self.dag.namespace(b)
                    if nsa != nsb:
                        self.stats["namespace_conflicts"] += 1
                        branch_conflicts.append([a, b, "cross_namespace"])
                    else:
                        self.stats["branch_conflicts"] += 1
                        aspect = self.dag.aspect(a)
                        self.stats["branch_conflict_by_aspect"][aspect] = \
                            self.stats["branch_conflict_by_aspect"].get(aspect, 0) + 1
                        branch_conflicts.append([a, b, "branch"])
        audit["branch_conflicts"] = branch_conflicts

        # ---- 4. 加权评分 ----
        for go, ev in cands.items():
            sc = self.score(ev, go)
            scored[go] = {
                "go_id": go,
                "score": round(sc, 4),
                "seq_conf": round(self.seq_conf(ev.evalue), 4),
                "struct_conf": round(self.struct_conf(ev.coverage), 4),
                "lit_conf": round(self.lit_conf(ev.citations), 4),
                "sources": sorted(ev.sources),
                "evidence": {
                    "evalue": ev.evalue, "identity": ev.identity, "qcov": ev.qcov,
                    "coverage": ev.coverage, "citations": ev.citations,
                },
                "detail": "; ".join(f"{k}:{v}" for k, v in ev.details.items()),
            }
        audit["scored_terms"] = [scored[g] for g in sorted(scored, key=lambda x: -scored[x]["score"])]

        # ---- 5. 阈值过滤 + 截断 ----
        passed = [g for g in scored if scored[g]["score"] >= self.min_score]
        passed.sort(key=lambda g: (-scored[g]["score"], g))
        passed = passed[: self.max_terms]

        # ---- 未知判定 ----
        if not passed:
            all_low = True
            for go, ev in cands.items():
                if (self.seq_conf(ev.evalue) >= self.low_conf.get("seq", 0.2)
                        or self.struct_conf(ev.coverage) >= self.low_conf.get("struct", 0.2)
                        or self.lit_conf(ev.citations) >= self.low_conf.get("lit", 0.2)):
                    all_low = False
                    break
            audit["unknown"] = True
            audit["unknown_reason"] = "all_low_confidence" if all_low else "no_candidate_above_threshold"
            audit["final_terms"] = []
            self.stats["unknown_marked"] += 1
            return audit
        audit["unknown"] = False

        # ---- 6. GO-DAG校验层 ----
        final, removed = self._dag_validate(passed, scored)
        audit["dag_validation_removed"] = removed
        audit["final_terms"] = final
        if final:
            self.stats["proteins_with_predictions"] += 1
        else:
            audit["unknown"] = True
            audit["unknown_reason"] = "all_removed_by_dag_validation"
            self.stats["unknown_marked"] += 1
        return audit

    def _dag_validate(self, gos: List[str], scored: Dict[str, dict]) -> Tuple[List[dict], List[str]]:
        """校验层：非法术语过滤 -> 冗余父节点删除。"""
        removed: List[str] = []
        # 非法/废弃术语
        legal, illegal = self.dag.filter_illegal(gos)
        self.stats["illegal_terms"] += len(illegal)
        removed.extend(illegal)
        # 冗余父节点
        kept, redundant = self.dag.remove_redundant(legal)
        removed.extend([f"{g}(redundant_parent)" for g in redundant])
        # 命名空间一致性（aspect由DAG真实命名空间决定，天然一致；记录跨命名空间冲突即可）
        out = []
        for g in kept:
            item = dict(scored[g])
            item["aspect"] = self.dag.aspect(g) or "?"
            out.append(item)
        return out, removed

    # ------------------------------------------------------------ 载入
    def _load_seq(self, path: str) -> Dict[str, Dict[str, Evidence]]:
        out: Dict[str, Dict[str, Evidence]] = {}
        if not path or not Path(path).exists():
            return out
        for r in read_tsv(path):
            go = self.dag.normalize(r["go_id"])
            try:
                ev = float(r.get("evalue") or "nan")
            except ValueError:
                ev = None
            e = Evidence(evalue=ev,
                         identity=_f(r.get("identity")), qcov=_f(r.get("qcov")),
                         sources={"seq"}, details={"seq": f"hit={r.get('hit_acc','')} E={r.get('evalue','')}"})
            out.setdefault(r["protein_id"], {}).setdefault(go, Evidence()).merge(e)
        return out

    def _load_struct(self, path: str) -> Dict[str, Dict[str, Evidence]]:
        out: Dict[str, Dict[str, Evidence]] = {}
        if not path or not Path(path).exists():
            return out
        for r in read_tsv(path):
            go = self.dag.normalize(r["go_id"])
            e = Evidence(coverage=_f(r.get("coverage")),
                         sources={"struct"}, details={"struct": f"sig={r.get('signature','')} cov={r.get('coverage','')}"})
            out.setdefault(r["protein_id"], {}).setdefault(go, Evidence()).merge(e)
        return out

    def _load_lit(self, path: str) -> Dict[str, Dict[str, Evidence]]:
        out: Dict[str, Dict[str, Evidence]] = {}
        if not path or not Path(path).exists():
            return out
        for r in read_tsv(path):
            go = self.dag.normalize(r["go_id"])
            try:
                cit = int(float(r.get("citations") or 0))
            except ValueError:
                cit = None
            e = Evidence(citations=cit,
                         sources={"lit"}, details={"lit": f"citations={r.get('citations','')} pmids={r.get('pmids','')[:80]}"})
            out.setdefault(r["protein_id"], {}).setdefault(go, Evidence()).merge(e)
        return out


def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None
