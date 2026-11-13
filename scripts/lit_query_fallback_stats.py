# -*- coding: utf-8 -*-
"""文献通道检索式回退分布与命中文献篇数统计。

不重新访问网络：按 Lit-Channel 的确定性优先级（基因名+物种 > 基因名 > 蛋白名，
三级均附出版日期截断）重建每条查询的检索式字符串，再从本地 SQLite 缓存读取
当次运行记录的命中篇数，取第一条有命中的检索式作为该蛋白实际使用的级别。

用法:
    env -u LD_LIBRARY_PATH python3 -X utf8 scripts/lit_query_fallback_stats.py \
        [--out results/revision/lit_query_fallback.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.channels.lit_channel import LitChannel  # noqa: E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402

SETS = [("split2_val", "split2"), ("split3_test", "split3"),
        ("arabidopsis", "arabidopsis_val")]


def read_meta(path):
    with open(path, encoding="utf-8") as f:
        return {r["protein_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "results/revision/lit_query_fallback.json"))
    args = ap.parse_args()

    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ch = LitChannel(cfg, dag)
    cutoff = ch.pub_date_cutoff
    clause = (f' AND ("0001/01/01"[Date - Publication] : '
              f'"{cutoff}"[Date - Publication])') if cutoff else ""

    meta = read_meta(cfg["paths"]["meta_dir"] + "/sprot_targets_meta.tsv")
    lib_meta = read_meta(cfg["paths"]["meta_dir"] + "/sprot_library_meta.tsv")
    meta.update({k: v for k, v in lib_meta.items() if k not in meta})

    out = {"pub_date_cutoff": cutoff, "sets": {}}
    for tag, part in SETS:
        fasta = PROJECT / cfg["paths"]["partitions_dir"] / f"{part}.fasta"
        prots = [l[1:].split()[0] for l in open(fasta, encoding="utf-8")
                 if l.startswith(">")]
        level = {"gene_organism": 0, "gene_only": 0, "protein_name": 0,
                 "no_hit": 0, "no_meta": 0}
        pmids, cached_missing = set(), 0
        for pid in prots:
            m = meta.get(pid)
            if not m:
                level["no_meta"] += 1
                continue
            gene, organism = m.get("gene_name", ""), m.get("organism", "")
            pname = m.get("protein_name", "")
            attempts = []
            if gene:
                if organism:
                    attempts.append(f'"{gene}"[Gene Name] AND "{organism}"[Organism]')
                attempts.append(f'"{gene}"[Gene Name]')
            if pname:
                attempts.append(f'"{pname}"')
            attempts = [t + clause for t in attempts]
            chosen = None
            for term in attempts:
                got = ch._cache_get("pubmed_search", term)
                if got is None:
                    cached_missing += 1
                    continue
                if (got[1] or 0) > 0 and got[2]:
                    chosen = got
                    break
            if chosen is None:
                level["no_hit"] += 1
                continue
            pmids.update(chosen[2].split(";"))
            if "[Organism]" in chosen[0]:
                level["gene_organism"] += 1
            elif "[Gene Name]" in chosen[0]:
                level["gene_only"] += 1
            else:
                level["protein_name"] += 1
        n_hit = sum(level[k] for k in ("gene_organism", "gene_only", "protein_name"))
        out["sets"][tag] = {"queries": len(prots), **level,
                            "with_hits": n_hit, "distinct_pmids": len(pmids),
                            "uncached_lookups": cached_missing}
        print(f"{tag}: {out['sets'][tag]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
