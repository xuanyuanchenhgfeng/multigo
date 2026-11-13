# -*- coding: utf-8 -*-
"""量化 DIAMOND 检索库口径对 Seq-Channel 的影响。

背景：data/diamond_db/sprot_library.dmnd 建于 2026-09-10，含 36 941 条序列，而修订版比对库
data/partitions/library.fasta（split1 16 948 + 拟南芥库侧 8 505）只有 25 453 条。
_transfer_go 先按 E-value 取前 top_hits 条命中再做 GO 转移，因此库外（晚于知识截止或目标物种）
的命中会占用 top-5 名额、挤掉可用转移。本脚本用按 library.fasta 重建的库重跑各查询集，
与在档通道产物逐行对照，给出差异规模。

用法: python scripts/seq_channel_dbcheck.py [--sets split3_test,split2_val]
"""
import argparse
import copy
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.go_dag import GODag                # noqa: E402
from src.channels.seq_channel import SeqChannel  # noqa: E402

SETS = {
    # tag: (查询 fasta, 在档通道产物, 正文规定的检索库, 库前缀)
    "split3_test": ("work/query_split3_test.fasta", "work/seq_channel_split3_test.tsv",
                    "data/partitions/split1.fasta", "data/diamond_db/split1_revision"),
    "split2_val": ("work/query_split2_val.fasta", "work/seq_channel_split2_val.tsv",
                   "data/partitions/split1.fasta", "data/diamond_db/split1_revision"),
    "arabidopsis": ("work/query_arabidopsis.fasta", "work/seq_channel_arabidopsis.tsv",
                    "data/partitions/library.fasta", "data/diamond_db/library_revision"),
}


def load_rows(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.add((r["protein_id"], r["go_id"], r["hit_acc"], r["evalue"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--apply", action="store_true",
                    help="把重跑结果写回 work/ 正式路径，原文件先备份到 work/backup_staledb_20261110/")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config_revision.yaml")
    dag = GODag(cfg["paths"]["go_obo"], relations=cfg.get("go_dag", {}).get("relations"))
    work = ROOT / "work" / "dbcheck"
    work.mkdir(parents=True, exist_ok=True)
    bak = ROOT / "work" / "backup_staledb_20261110"
    if args.apply:
        bak.mkdir(exist_ok=True)

    for tag in args.sets.split(","):
        qfa, old_tsv, lib_fa, db_pfx = SETS[tag]
        c = copy.deepcopy(cfg)
        c["paths"]["work_dir"] = str(work)
        c["seq_channel"]["diamond_db"] = str(ROOT / db_pfx)
        c["seq_channel"]["library_fasta"] = str(ROOT / lib_fa)
        new_tsv = work / f"seq_channel_{tag}.tsv"
        SeqChannel(c, dag).run(str(ROOT / qfa), str(new_tsv))
        old, new = load_rows(ROOT / old_tsv), load_rows(new_tsv)
        og, ng = {(a, b) for a, b, _, _ in old}, {(a, b) for a, b, _, _ in new}
        print(f"\n### {tag}  (检索库 {lib_fa.split('/')[-1]})")
        print(f"在档(旧库) 行 {len(old):,} / (蛋白,GO) {len(og):,} / 蛋白 {len({x[0] for x in old}):,}")
        print(f"重跑(新库) 行 {len(new):,} / (蛋白,GO) {len(ng):,} / 蛋白 {len({x[0] for x in new}):,}")
        print(f"新增 (蛋白,GO) {len(ng - og):,}  丢失 {len(og - ng):,}  交集 {len(og & ng):,}")
        only_new_acc = {h for _, _, h, _ in new} - {h for _, _, h, _ in old}
        print(f"新出现的命中蛋白 {len(only_new_acc):,}  样例 {sorted(only_new_acc)[:5]}")
        if args.apply:
            src = ROOT / old_tsv
            shutil.copy2(src, bak / src.name)
            shutil.copy2(new_tsv, src)
            print(f"[apply] 已写回 {old_tsv}，旧文件备份为 {bak.name}/{src.name}")


if __name__ == "__main__":
    main()
