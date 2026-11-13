#!/usr/bin/env python3
"""把 data/partitions/*.fasta 的序列头导出为划分清单 splits/partitions.tsv.gz。

划分本身由 src/data_prep.py 依据逐条注释日期与证据码白名单算出，输入是 Swiss-Prot
与 GOA 的原始下载物，不随仓库分发；本脚本只导出结果名单，使正文表 1 的六个集合规模
（16 948 / 3 122 / 1 159 / 25 453 / 8 505 / 3 607）可以在不重跑流水线的情况下核对。

用法：python3 scripts/export_partition_manifest.py
"""
import gzip
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "data" / "partitions"
OUT = PROJ / "splits" / "partitions.tsv.gz"

SETS = ("split1", "split2", "split3", "library", "arabidopsis_lib", "arabidopsis_val")


def ids(path):
    """FASTA 头第一字段即 UniProt accession，OS=/OX= 等元数据不进清单。"""
    with open(path, encoding="utf-8") as fh:
        return [ln[1:].split()[0] for ln in fh if ln.startswith(">")]


def main():
    rows = [(name, pid) for name in SETS for pid in ids(SRC / f"{name}.fasta")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        fh.write("set\taccession\n")
        for name, pid in rows:
            fh.write(f"{name}\t{pid}\n")
    print(f"{OUT} 写出 {len(rows)} 行：",
          "，".join(f"{n} {sum(1 for s, _ in rows if s == n)}" for n in SETS))


if __name__ == "__main__":
    main()
