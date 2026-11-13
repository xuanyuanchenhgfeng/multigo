# -*- coding: utf-8 -*-
"""复现 results/revision/truth_stats.json（正文表 1 与图 2 右侧的数字来源）。

口径：三个集合的真值 TSV 是 data_prep 按 13 码白名单与各自注释日期窗口筛过的直接注释，
本脚本只做三件事——把 GO 标识规范化并丢弃本体快照外的项、按 (蛋白, GO) 去重、按命名空间
计数。fasta_counts 直接数各划分 FASTA 的条头，用于核对"查询数"与"可评估蛋白数"的差。

用法: python scripts/revision_truth_stats.py [--config config_revision.yaml]
                                            [--out results/revision/truth_stats.json]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.go_dag import GODag                # noqa: E402

TRUTH_FILES = {                       # 划分标签 -> 相对 partitions_dir 的真值文件与 FASTA
    "train": ("split1_go.tsv",  "split1.fasta"),
    "val":   ("split2_go.tsv",  "split2.fasta"),
    "test":  ("split3_go.tsv",  "split3.fasta"),
}
ARABIDOPSIS_FASTA = {"ath_lib": "arabidopsis_lib.fasta", "ath_val": "arabidopsis_val.fasta"}
ASPECTS = ("F", "P", "C")             # 输出顺序固定，避免行的遇到顺序改变键序


def count_fasta(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def split_stats(rows, dag):
    """返回该集合的 (蛋白数, 注释条目数, 术语集合) 按命名空间拆分与去重后总蛋白数。"""
    seen = set()
    prot = {a: set() for a in ASPECTS}
    terms = {a: set() for a in ASPECTS}
    entries = {a: 0 for a in ASPECTS}
    for r in rows:
        go = dag.normalize(r["go_id"])
        if not dag.is_valid(go):
            continue
        asp = dag.aspect(go)
        if asp is None:
            continue
        key = (r["protein_id"], go)
        if key in seen:
            continue
        seen.add(key)
        prot[asp].add(r["protein_id"])
        terms[asp].add(go)
        entries[asp] += 1
    return (
        {a: len(v) for a, v in prot.items() if v},
        {a: v for a, v in entries.items() if v},
        terms,
        len(prot["F"] | prot["P"] | prot["C"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_revision.yaml")
    ap.add_argument("--out", default=str(ROOT / "results" / "revision" / "truth_stats.json"))
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    dag = GODag(cfg["paths"]["go_obo"], relations=cfg.get("go_dag", {}).get("relations"))
    adir = Path(cfg["paths"]["annotations_dir"])
    pdir = Path(cfg["paths"]["partitions_dir"])

    truth_proteins, truth_entries, dedup, fasta_counts = {}, {}, {}, {}
    union_terms: dict = defaultdict(set)
    for tag, (truth_name, fasta_name) in TRUTH_FILES.items():
        rows = list(csv.DictReader(open(adir / truth_name, encoding="utf-8"), delimiter="\t"))
        prot, entries, terms, n_all = split_stats(rows, dag)
        truth_proteins.update({f"{tag}|{a}": v for a, v in prot.items()})
        truth_entries.update({f"{tag}|{a}": v for a, v in entries.items()})
        dedup[tag] = n_all
        fasta_counts[tag] = count_fasta(pdir / fasta_name)
        for a, s in terms.items():
            union_terms[a] |= s

    for key, name in ARABIDOPSIS_FASTA.items():
        fasta_counts[key] = count_fasta(pdir / name)

    out = {
        "protocol": f"直接注释、13 种白名单证据码、{cfg['paths']['go_obo']}",
        "go_terms": len(dag.terms),
        "union_terms": {a: len(union_terms[a]) for a in ASPECTS},
        "truth_proteins": truth_proteins,
        "truth_entries": truth_entries,
        "dedup_proteins": dedup,
        "fasta_counts": fasta_counts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
