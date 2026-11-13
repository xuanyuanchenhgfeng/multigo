# -*- coding: utf-8 -*-
"""修订版：文献通道知识截止重跑。

仅纳入出版日期 <= lit_channel.pub_date_cutoff（2023/11/08）的 PubMed 文献，
对 split2_val / split3_test / arabidopsis 三个评估目标集重建 lit 候选，
输出 work/lit_channel_{tag}_dcut.tsv（不覆盖旧文件）。
摘要缓存按 PMID 复用；esearch 因查询串含日期子句而全部重新请求。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.channels.lit_channel import LitChannel  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
TAGS = ["split2_val", "split3_test", "arabidopsis"]


def main():
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:
        k = sys.argv.index("--shard")
        shard_i, shard_n = int(sys.argv[k + 1]), int(sys.argv[k + 2])
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ch = LitChannel(cfg, dag)
    assert ch.pub_date_cutoff, "config_revision.yaml 未设置 pub_date_cutoff"
    suffix = "" if shard_n == 1 else f".s{shard_i}of{shard_n}"
    for tag in TAGS:
        meta_tsv = PROJECT / "work" / f"meta_{tag}.tsv"
        out_tsv = PROJECT / "work" / f"lit_channel_{tag}_dcut{suffix}.tsv"
        if out_tsv.exists():
            print(f"[skip] {out_tsv.name} 已存在")
            continue
        pids = sorted({r["protein_id"] for r in read_tsv(str(meta_tsv))})
        if shard_n > 1:
            pids = pids[shard_i::shard_n]
        targets = set(pids)
        print(f"[run ] {tag} shard {shard_i}/{shard_n}: {len(targets)} 蛋白, "
              f"cutoff={ch.pub_date_cutoff}", flush=True)
        ch.run(str(meta_tsv), targets, str(out_tsv))
        print(f"[done] -> {out_tsv}", flush=True)


if __name__ == "__main__":
    main()
