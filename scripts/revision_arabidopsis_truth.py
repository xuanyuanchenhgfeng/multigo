# -*- coding: utf-8 -*-
"""修订版：重建拟南芥验证集真值（严格 t0/t1 时间窗）。

初版 arabidopsis_go.tsv 直接取自 2026-06-17 下载的 GOA GAF，30% 留出蛋白
的真值未做日期封顶（含截至 2026-06 的注释），与其声称的 2024-11-27
评估窗不一致。本脚本对同一批留出蛋白（arabidopsis_val.fasta）重建真值：
  - 13 种人工/高通量实验及核对证据码（与主实验白名单一致，排除 IEA/IBA
    /ISS/ND 等）；
  - 注释日期 <= 2024-11-27（t1，与验证窗结束日一致）；
输出 data/annotations/arabidopsis_go_t1.tsv，并报告可评估蛋白数。
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.channels.base import iter_fasta  # noqa: E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WHITE = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
         "HTP", "HDA", "HMP", "HGI", "HEP"}
ASPECT = {"F", "P", "C"}
T1 = "20241127"


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"], relations=["is_a", "part_of"])
    val_ids = {r[0] for r in iter_fasta(
        str(PROJECT / "data/partitions/arabidopsis_val.fasta"))}
    gaf = PROJECT / "数据集/goa_arabidopsis.gaf.gz"
    out = PROJECT / "data/annotations/arabidopsis_go_t1.tsv"
    seen = set()
    rows = []
    n_raw = 0
    with gzip.open(gaf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            v = line.rstrip("\n").split("\t")
            if len(v) < 14:
                continue
            pid, go, ev, asp, date = v[1], v[4], v[6], v[8], v[13]
            n_raw += 1
            if pid not in val_ids:
                continue
            if ev not in WHITE or asp not in ASPECT or date > T1:
                continue
            g = dag.normalize(go)
            if not dag.is_valid(g):
                continue
            key = (pid, g)
            if key in seen:
                continue
            seen.add(key)
            rows.append((pid, g, asp, ev, date))
    with open(out, "w", encoding="utf-8") as f:
        f.write("protein_id\tgo_id\taspect\tevidence\tdate\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    prots = {r[0] for r in rows}
    by_a = {a: {r[0] for r in rows if r[2] == a} for a in ASPECT}
    print(f"GAF 原始行 {n_raw:,}；窗口内行（去重）{len(rows):,}；"
          f"可评估蛋白 {len(prots)} / {len(val_ids)}")
    print("分本体蛋白:", {a: len(s) for a, s in by_a.items()})
    print("->", out)


if __name__ == "__main__":
    main()
