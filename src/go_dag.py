# -*- coding: utf-8 -*-
"""
GO-DAG 解析与校验层
====================
- 解析 go-basic.obo：术语、命名空间、is_a / part_of 关系、obsolete、alt_id
- 供 Arbiter 使用的拓扑关系判定（相同/父子/不同分支）
- 供 GO-DAG 校验层使用：冗余父节点过滤、非法术语过滤、跨命名空间冲突检测
自研解析（networkx 可选加速未必需），满足"goatools 或 networkx 自定义解析"要求。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

ASPECT_OF_NAMESPACE = {
    "molecular_function": "F",
    "biological_process": "P",
    "cellular_component": "C",
}
NAMESPACE_OF_ASPECT = {v: k for k, v in ASPECT_OF_NAMESPACE.items()}


@dataclass
class GOTerm:
    go_id: str
    name: str = ""
    namespace: str = ""          # molecular_function / biological_process / cellular_component
    is_obsolete: bool = False
    parents: Set[str] = field(default_factory=set)      # 经 is_a/part_of 直接父节点
    children: Set[str] = field(default_factory=set)
    alt_ids: Set[str] = field(default_factory=set)
    replaced_by: Optional[str] = None


class GODag:
    """GO 有向无环图（is_a + part_of 合并视为层级关系，与CAFA评估惯例一致）。"""

    def __init__(self, obo_path: str, relations: Optional[List[str]] = None):
        self.relations = set(relations or ["is_a", "part_of"])
        self.terms: Dict[str, GOTerm] = {}
        self.alt2main: Dict[str, str] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._loaded_from = obo_path
        self._parse(obo_path)

    # ------------------------------------------------------------------ 解析
    def _parse(self, obo_path: str) -> None:
        in_term = False
        cur: Optional[GOTerm] = None
        with open(obo_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line == "[Term]":
                    in_term, cur = True, None
                    continue
                if line.startswith("["):
                    in_term = False
                    cur = None
                    continue
                if not in_term or cur is None:
                    if line.startswith("id: GO:") and cur is None and in_term:
                        cur = GOTerm(go_id=line.split(":", 1)[1].strip())
                        self.terms[cur.go_id] = cur
                    continue
                # ---- 属性行 ----
                if line.startswith("id: "):
                    gid = line.split(":", 1)[1].strip()
                    if cur.go_id is None:
                        cur.go_id = gid
                        self.terms[gid] = cur
                elif line.startswith("name: "):
                    cur.name = line.split(":", 1)[1].strip()
                elif line.startswith("namespace: "):
                    cur.namespace = line.split(":", 1)[1].strip()
                elif line.startswith("is_obsolete: true"):
                    cur.is_obsolete = True
                elif line.startswith("replaced_by: "):
                    cur.replaced_by = line.split(":", 1)[1].strip()
                elif line.startswith("alt_id: "):
                    aid = line.split(":", 1)[1].strip()
                    cur.alt_ids.add(aid)
                    self.alt2main[aid] = cur.go_id
                elif line.startswith("is_a: "):
                    p = line.split(":", 1)[1].split("!")[0].strip()
                    cur.parents.add(p)
                elif line.startswith("relationship: "):
                    parts = line.split(":", 1)[1].strip()
                    try:
                        rel, target = parts.split(None, 1)
                    except ValueError:
                        continue
                    if rel in self.relations:
                        cur.parents.add(target.split("!")[0].strip())
        # 子节点索引
        for t in self.terms.values():
            for p in t.parents:
                if p in self.terms:
                    self.terms[p].children.add(t.go_id)

    # ------------------------------------------------------------------ 基础查询
    def normalize(self, go_id: str) -> str:
        """alt_id -> 主ID；未知ID原样返回。"""
        return self.alt2main.get(go_id, go_id)

    def get(self, go_id: str) -> Optional[GOTerm]:
        return self.terms.get(self.normalize(go_id))

    def is_valid(self, go_id: str) -> bool:
        t = self.get(go_id)
        return t is not None and not t.is_obsolete

    def namespace(self, go_id: str) -> Optional[str]:
        t = self.get(go_id)
        return t.namespace if t else None

    def aspect(self, go_id: str) -> Optional[str]:
        ns = self.namespace(go_id)
        return ASPECT_OF_NAMESPACE.get(ns) if ns else None

    def name(self, go_id: str) -> str:
        t = self.get(go_id)
        return t.name if t else ""

    # ------------------------------------------------------------------ 拓扑
    def ancestors(self, go_id: str) -> Set[str]:
        """全部祖先（含is_a/part_of闭包），带缓存。"""
        gid = self.normalize(go_id)
        if gid in self._ancestor_cache:
            return self._ancestor_cache[gid]
        seen: Set[str] = set()
        stack = [gid]
        while stack:
            cur = stack.pop()
            t = self.terms.get(cur)
            if t is None:
                continue
            for p in t.parents:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        self._ancestor_cache[gid] = seen
        return seen

    def propagate(self, go_ids: Iterable[str]) -> Set[str]:
        """项集传播至全部祖先（评估用）。"""
        out: Set[str] = set()
        for g in go_ids:
            g = self.normalize(g)
            if g in self.terms:
                out.add(g)
                out |= self.ancestors(g)
        return {g for g in out if g in self.terms}

    def relation(self, go_a: str, go_b: str) -> str:
        """判定两GO拓扑关系（Arbiter冲突检测核心）。

        返回:
          "same"        - 相同术语（规范化后）
          "parent_of"   - a 是 b 的祖先（a更泛化）
          "child_of"    - a 是 b 的子孙（a更特化）
          "disconnected"- 不同分支（互非祖先；含跨命名空间情形）
        """
        a, b = self.normalize(go_a), self.normalize(go_b)
        if a == b:
            return "same"
        if a in self.ancestors(b):
            return "parent_of"
        if b in self.ancestors(a):
            return "child_of"
        return "disconnected"

    # ------------------------------------------------------------------ 校验层
    def remove_redundant(self, go_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
        """GO-DAG校验层步骤1：删除被集合内其他GO包含的父节点。

        返回 (保留的特化GO列表, 被删除的冗余父节点列表)。
        """
        norm = {self.normalize(g) for g in go_ids}
        norm = {g for g in norm if g in self.terms}
        redundant: Set[str] = set()
        # 若 g1 是 g2 的祖先（且 g1 != g2），则 g1 冗余
        ancestor_union: Set[str] = set()
        for g in norm:
            ancestor_union |= self.ancestors(g)
        redundant = norm & ancestor_union
        kept = sorted(norm - redundant)
        return kept, sorted(redundant)

    def validate_namespace_consistency(self, aspect_by_go: Dict[str, str]) -> List[str]:
        """校验层步骤2：检测跨命名空间冲突（声明的aspect与GO实际命名空间不符）。"""
        issues = []
        for go_id, aspect in aspect_by_go.items():
            real = self.aspect(go_id)
            if real is not None and aspect and real != aspect:
                issues.append(f"{go_id}: aspect={aspect} 但实际命名空间={real}")
        return issues

    def filter_illegal(self, go_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
        """校验层步骤3：过滤非法术语（不在DAG / obsolete）。

        返回 (合法列表, 非法列表)。
        """
        ok, bad = [], []
        for g in go_ids:
            t = self.get(g)
            if t is None:
                bad.append(f"{g}(未知术语)")
            elif t.is_obsolete:
                # 若有 replaced_by 则替换
                if t.replaced_by and t.replaced_by in self.terms:
                    ok.append(t.replaced_by)
                else:
                    bad.append(f"{g}(已废弃)")
            else:
                ok.append(self.normalize(g))
        return ok, bad

    def stats(self) -> dict:
        from collections import Counter
        ns = Counter(t.namespace for t in self.terms.values())
        return {
            "total_terms": len(self.terms),
            "obsolete": sum(1 for t in self.terms.values() if t.is_obsolete),
            "alt_ids": len(self.alt2main),
            "namespaces": dict(ns),
        }
