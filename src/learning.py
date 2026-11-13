# -*- coding: utf-8 -*-
"""可学习仲裁层（Learned Arbitration）
=====================================
把固定权重 score = 0.40·seq_conf + 0.35·struct_conf + 0.25·lit_conf 换成
候选级特征上的分本体 logistic 回归，其余决策语义（min_score、每蛋白上限、
DAG 校验、未知判定、传播）与固定权重路线完全一致，两条路线因此可配对比较。

设计约束：
  1. 只在验证窗候选上拟合、按蛋白分组的 K 折选正则强度，测试窗一次都不参与，
     与阈值/权重的"验证集选取—测试集冻结"纪律相同；
  2. 特征只允许使用训练库（<=T0）统计的量（术语 IC、直接注释蛋白数、本体深度）
     与该候选自身的通道置信度，不使用任何评估窗真值；
  3. 求解器为 numpy 实现的带阻尼 IRLS，不引入 sklearn 依赖，系数与标准化参数
     全量落盘，可逐条复算。
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ASPECTS = ("F", "P", "C")
FEATURES: Tuple[str, ...] = (
    "seq_conf", "struct_conf", "lit_conf",
    "n_src", "seq_x_struct", "seq_x_lit", "struct_x_lit",
    "ic_norm", "log_prevalence", "depth_norm", "collapsed",
)
LAMBDAS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def term_depth(dag) -> Dict[str, float]:
    """术语到本体根的最短 is_a/part_of 距离（根为0，不可达为 NaN）。"""
    roots = {"GO:0003674", "GO:0008150", "GO:0005575"}
    depth: Dict[str, float] = {r: 0.0 for r in roots}
    frontier = list(roots)
    while frontier:
        nxt = []
        for go in frontier:
            term = dag.terms.get(go)
            if term is None:
                continue
            for child in getattr(term, "children", ()) or ():
                if child in depth:
                    continue
                depth[child] = depth[go] + 1.0
                nxt.append(child)
        frontier = nxt
    return depth


class CandidateFeatures:
    """把 (Evidence, GO) 映射成特征向量；训练与推断共用同一实例。"""

    def __init__(self, arb, dag, ic: Dict[str, float], prevalence: Dict[str, int],
                 depth: Dict[str, float]):
        self.arb = arb
        self.dag = dag
        self.ic = ic
        self.prevalence = prevalence
        self.depth = depth
        self.ic_max = max(list(ic.values()) + [1.0])
        self.prev_max = math.log1p(max(list(prevalence.values()) + [1]))
        self.depth_max = max([d for d in depth.values() if d == d] + [1.0])

    def vector(self, ev, go: str) -> List[float]:
        sc = self.arb.seq_conf(ev.evalue)
        tc = self.arb.struct_conf(ev.coverage)
        lc = self.arb.lit_conf(ev.citations)
        n_src = len(ev.sources)
        collapsed = 1.0 if "merged_from" in ev.details else 0.0
        d = self.depth.get(go, float("nan"))
        return [
            sc, tc, lc,
            float(n_src), sc * tc, sc * lc, tc * lc,
            min(self.ic.get(go, 0.0) / self.ic_max, 1.0),
            math.log1p(self.prevalence.get(go, 0)) / self.prev_max,
            0.0 if d != d else min(d / self.depth_max, 1.0),
            collapsed,
        ]

    def matrix(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray(rows, dtype=float)


class AspectLogit:
    """单个命名空间的 L2 正则 logistic 回归（阻尼 Newton / IRLS）。"""

    def __init__(self):
        self.mean = None
        self.std = None
        self.beta = None
        self.lam = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), (X - self.mean) / self.std])

    def fit(self, X: np.ndarray, y: np.ndarray, lam: float = 1e-2,
            max_iter: int = 60, tol: float = 1e-8) -> "AspectLogit":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        D = self._design(X)
        y = y.astype(float)
        n, p = D.shape
        pen = np.ones(p) * lam
        pen[0] = 0.0                      # 不惩罚截距
        beta = np.zeros(p)

        def loss(b):
            eta = np.clip(D @ b, -30.0, 30.0)
            ll = np.logaddexp(0.0, eta) - y * eta
            return float(ll.sum()) + 0.5 * float((pen * b * b).sum())

        cur = loss(beta)
        for _ in range(max_iter):
            eta = np.clip(D @ beta, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            w = np.maximum(mu * (1.0 - mu), 1e-6)
            g = D.T @ (mu - y) + pen * beta
            H = (D * w[:, None]).T @ D + np.diag(pen)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            t = 1.0
            for _ls in range(20):          # 回溯阻尼，保证损失单调下降
                cand = beta - t * step
                new = loss(cand)
                if new <= cur + 1e-12:
                    break
                t *= 0.5
            if new > cur - tol:
                break
            beta, cur = cand, new
        self.beta = beta
        self.lam = lam
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self._design(X) @ self.beta, -30.0, 30.0)))

    def logloss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.proba(X), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def to_dict(self) -> dict:
        return {"lam": self.lam,
                "intercept": float(self.beta[0]),
                "coef": {f: float(v) for f, v in zip(FEATURES, self.beta[1:])},
                "mean": {f: float(v) for f, v in zip(FEATURES, self.mean)},
                "std": {f: float(v) for f, v in zip(FEATURES, self.std)}}


def group_kfold_keys(protein_ids: Sequence[str], k: int, seed: int) -> np.ndarray:
    """按蛋白分组的折号：同一蛋白的全部候选落在同一折。"""
    uniq = sorted(set(protein_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {p: i % k for i, p in enumerate(uniq)}
    return np.array([assign[p] for p in protein_ids])


class ArbitrationModel:
    """三个命名空间各一个 AspectLogit，对外提供 (ev, go) -> p 的仲裁打分。"""

    def __init__(self, feats: Optional[CandidateFeatures] = None,
                 seed: int = 20261104, n_folds: int = 5):
        self.feats = feats
        self.seed = seed
        self.n_folds = n_folds
        self.models: Dict[str, AspectLogit] = {}
        self.selection: Dict[str, dict] = {}
        self.rows: Dict[str, int] = {}
        self.oof: Dict[str, dict] = {}      # 出折预测 -> 校准诊断

    # ---------------------------------------------------------- 拟合
    def fit(self, samples: Iterable[Tuple[str, str, List[float], int]]
            ) -> "ArbitrationModel":
        """samples: (protein_id, aspect, feature_vector, label)。"""
        by: Dict[str, List[list]] = {a: [[], [], []] for a in ASPECTS}
        for pid, aspect, vec, lab in samples:
            if aspect not in by:
                continue
            by[aspect][0].append(pid)
            by[aspect][1].append(vec)
            by[aspect][2].append(lab)
        for a in ASPECTS:
            pids, X, y = by[a]
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=int)
            self.rows[a] = int(len(y))
            if len(np.unique(y)) < 2:
                raise ValueError(f"命名空间 {a} 的训练样本标签单一，无法拟合")
            folds = group_kfold_keys(pids, self.n_folds, self.seed)
            cv = {}
            oof = np.zeros(len(y))
            best = None
            for lam in LAMBDAS:
                errs, hold = [], np.zeros(len(y))
                for f in range(self.n_folds):
                    tr, te = folds != f, folds == f
                    m = AspectLogit().fit(X[tr], y[tr], lam)
                    errs.append(m.logloss(X[te], y[te]))
                    hold[te] = m.proba(X[te])
                ll = float(np.mean(errs))
                cv[f"{lam:g}"] = round(ll, 5)
                if best is None or ll < best[0]:
                    best, oof = (ll, lam, hold), hold
            self.selection[a] = {"cv_logloss": cv, "lambda": float(best[1])}
            self.oof[a] = {"logloss_round2": round(float(best[0]), 5),
                           "n_pos": int(y.sum()), "n_row": int(len(y))}
            self.models[a] = AspectLogit().fit(X, y, best[1])
            self.oof[a]["oof_prob"] = [round(float(v), 6) for v in oof]
            self.oof[a]["oof_label"] = [int(v) for v in y]
        return self

    # ---------------------------------------------------------- 推断
    def score(self, ev, go: str) -> float:
        aspect = self.feats.dag.aspect(go)
        m = self.models.get(aspect)
        if m is None:
            return 0.0
        return float(m.proba(self.feats.matrix([self.feats.vector(ev, go)]))[0])

    def __call__(self, ev, go: Optional[str]) -> float:
        if go is None:
            raise ValueError("学习式仲裁需要 GO 术语以确定命名空间")
        return self.score(ev, go)

    def to_dict(self) -> dict:
        return {
            "features": list(FEATURES),
            "lambdas_grid": [f"{v:g}" for v in LAMBDAS],
            "seed": self.seed, "n_folds": self.n_folds,
            "rows": self.rows,
            "selection": self.selection,
            "oof": {a: {"logloss_round2": self.oof[a]["logloss_round2"],
                        "n_pos": self.oof[a]["n_pos"],
                        "n_row": self.oof[a]["n_row"],
                        "ece": ece(self.oof[a]["oof_prob"], self.oof[a]["oof_label"])}
                    for a in ASPECTS},
            "models": {a: self.models[a].to_dict() for a in ASPECTS},
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")


def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> dict:
    """期望校准误差与分箱可靠度表。"""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table, total = [], 0.0
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] + (1e-12 if i == bins - 1 else 0))
        n = int(sel.sum())
        if n == 0:
            continue
        conf, acc = float(p[sel].mean()), float(y[sel].mean())
        table.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": n,
                      "mean_prob": round(conf, 4), "empirical": round(acc, 4)})
        total += n * abs(acc - conf)
    return {"ece": round(total / max(len(p), 1), 5), "n": int(len(p)),
            "reliability": table}
