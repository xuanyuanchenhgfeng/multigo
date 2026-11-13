# -*- coding: utf-8 -*-
"""
通路富集分析（苦荞预测的下游分析，可复现正式版）
================================================
对苦荞预测注释（results/tartary_go_annotation.tsv，取 score>=t 的 GO）做
GO 术语级超几何富集检验，背景为训练库注释（data/annotations/split1_go.tsv
传播后的注释集），BH-FDR 校正。用法:
    python3 -X utf8 scripts/enrichment.py [--t 0.15] [--top 60]
"""
import argparse
import sys
from collections import defaultdict
from math import lgamma
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import src.pipeline  # noqa: F401
from src.go_dag import GODag
ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}
import yaml


def log1p_exp(lx):
    import math
    m = max(lx)
    return m + math.log(sum(math.exp(v - m) for v in lx))


def hypergeom_sf(k, N, K, n):
    """P(X >= k)：超几何尾概率（对数空间稳定计算）。"""
    if k > min(K, n):
        return 0.0  # 不可能事件
    lx = []
    for i in range(k, min(K, n) + 1):
        # C(K,i)*C(N-K,n-i)/C(N,n)
        lx.append(lgamma(K + 1) - lgamma(i + 1) - lgamma(K - i + 1)
                  + lgamma(N - K + 1) - lgamma(n - i + 1) - lgamma(N - K - n + i + 1)
                  - (lgamma(N + 1) - lgamma(n + 1) - lgamma(N - n + 1)))
    return min(1.0, __import__("math").exp(log1p_exp(lx)))


def bh_fdr(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = min(1.0, prev)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.15, help="预测分数阈值")
    ap.add_argument("--top", type=int, default=60, help="输出前 N 行")
    args = ap.parse_args()

    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))

    # 背景库：训练库注释（传播）
    lib = defaultdict(set)
    for ln in (BASE / "data/annotations/split1_go.tsv").read_text(
            encoding="utf-8").splitlines():
        if ln.startswith("protein_id"):
            continue
        p, g = ln.split("\t")[:2]
        lib[p].add(g)
    lib_prop = {p: set(dag.propagate(gos)) - ROOT_TERMS
                for p, gos in lib.items()}
    lib_term2prot = defaultdict(set)
    for p, gos in lib_prop.items():
        for g in gos:
            lib_term2prot[g].add(p)
    N = len(lib_prop)

    # 苦荞预测
    tar = defaultdict(set)
    for ln in (BASE / "results/tartary_go_annotation.tsv").read_text(
            encoding="utf-8").splitlines():
        if ln.startswith("protein_id"):
            continue
        parts = ln.split("\t")
        p, gos_str, scores_str = parts[0], parts[2], parts[3]
        for g, sv in zip(gos_str.split(";"), scores_str.split(";")):
            if float(sv) >= args.t:
                tar[p].add(g)
    tar_prop = {p: set(dag.propagate(gos)) - ROOT_TERMS
                for p, gos in tar.items()}
    n = len(tar_prop)
    tar_term2prot = defaultdict(set)
    for p, gos in tar_prop.items():
        for g in gos:
            tar_term2prot[g].add(p)

    rows = []
    for g, genes in tar_term2prot.items():
        K = len(lib_term2prot.get(g, ()))
        if K == 0:
            continue
        k = len(genes)
        fe = (k / n) / (K / N) if K and n else 0.0
        pv = hypergeom_sf(k, N, K, n)
        rows.append((g, dag.name(g), k, K, n, N, fe, pv,
                     ";".join(sorted(genes)[:50])))
    rows.sort(key=lambda r: (r[7], -r[6]))

    pvals = [r[7] for r in rows]
    adj = bh_fdr(pvals)
    out = BASE / "results/tartary_pathway_enrichment.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("go_id\tpathway_name\ttartary_genes\tlibrary_genes\tN_tartary"
                "\tN_library\tfold_enrichment\tp_value\tfdr_bh\tgene_list(max50)\n")
        for r, q in zip(rows, adj):
            f.write("\t".join([r[0], r[1], str(r[2]), str(r[3]), str(r[4]),
                               str(r[5]), f"{r[6]:.3f}", f"{r[7]:.3e}",
                               f"{q:.3e}", r[8]]) + "\n")
    sig = sum(1 for q in adj if q < 0.05)
    print(f"富集完成: {len(rows)} 术语, FDR<0.05 共 {sig} 个, "
          f"背景库 {N:,} 蛋白 / 苦荞注释 {n:,} 蛋白 (t={args.t})")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
