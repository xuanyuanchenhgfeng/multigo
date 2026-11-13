# -*- coding: utf-8 -*-
"""Struct-Channel 缺口补跑（PSB26 时间基划分对齐）。

split2_val / split3_test 的 struct TSV 预填自旧运行子集（InterProScan 结果仅依赖
序列，可确定性复用）。本脚本找出目标集中尚无 struct 行的蛋白，单独运行
InterProScan（独立 gap 批次目录，不与旧批次冲突），把结果追加进主 TSV。

缺口含两类蛋白（无法从 TSV 区分，统一重跑兜底）：
  1) 新划分窗口中从未处理过的蛋白；
  2) 曾处理但无域-GO 结果的蛋白（确定性工具重跑结果仍为空）。
拟南芥 struct 无需补跑：新验证集 ⊆ 旧查询集，已全部处理过。

用法: nohup python3 -X utf8 scripts/fill_struct_gaps.py [--phase ips|append|all] \
      > logs/fill_struct_gaps.log 2>&1 &
  --phase ips    仅运行 InterProScan 缺口批次（写独立 gap TSV，不触碰主 TSV）
  --phase append 仅把已有 gap TSV 结果追加进主 TSV（需 ips 阶段已完成）
  --phase all    两者顺序执行（默认）
注意：Java 需在 PATH（如 PATH=/home/zsl/tools/jdk-11.0.2/bin:$PATH），
本机 JDK 11 不在默认 PATH 时 interproscan.sh 会瞬时失败。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402  先导入打破 data_prep<->seq_channel 循环导入
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.channels.struct_channel import StructChannel  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent


def fasta_records(path):
    """极简FASTA解析：[(pid, desc, seq), ...]。"""
    recs = []
    pid, desc, seq = None, "", []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if pid is not None:
                    recs.append((pid, desc, "".join(seq)))
                parts = line[1:].split(None, 1)
                pid = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
                seq = []
            elif line:
                seq.append(line)
    if pid is not None:
        recs.append((pid, desc, "".join(seq)))
    return recs


def append_rows(gap_tsv, main_tsv):
    """gap TSV 数据行（跳过表头）追加进主 TSV（整行去重保险）。"""
    have = set()
    with open(main_tsv, encoding="utf-8") as f:
        next(f)
        for line in f:
            have.add(line.rstrip("\n"))
    added = 0
    with open(main_tsv, "a", encoding="utf-8") as out, \
            open(gap_tsv, encoding="utf-8") as g:
        next(g)
        for line in g:
            line = line.rstrip("\n")
            if not line or line in have:
                continue
            out.write(line + "\n")
            have.add(line)
            added += 1
    return added


def main():
    ap = argparse.ArgumentParser(description="Struct-Channel 缺口补跑")
    ap.add_argument("--phase", choices=("ips", "append", "all"), default="all",
                    help="ips=仅运行缺口批次; append=仅追加结果; all=顺序执行")
    ap.add_argument("--config", default=str(PROJECT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    channel = StructChannel(cfg, dag)
    work = Path(cfg["paths"]["work_dir"])
    pdir = Path(cfg["paths"]["partitions_dir"])

    plan = {"split2_val": "split2.fasta", "split3_test": "split3.fasta"}
    for tag, part in plan.items():
        tgt = {r[0] for r in fasta_records(str(pdir / part))}
        main_tsv = work / f"struct_channel_{tag}.tsv"
        covered = {r["protein_id"] for r in read_tsv(str(main_tsv))} & tgt
        gap = tgt - covered
        gap_fa = work / f"struct_gap_{tag}.fasta"
        gap_out = work / f"struct_channel_gap_{tag}.tsv"
        log.info("=" * 50)
        log.info("[%s] 目标 %s | struct已覆盖 %s | 缺口 %s | phase=%s",
                 tag, len(tgt), len(covered), len(gap), args.phase)
        if not gap:
            log.info("[%s] 无缺口，跳过", tag)
            continue
        if args.phase in ("ips", "all"):
            # 缺口FASTA（幂等重建）
            with open(gap_fa, "w", encoding="utf-8", newline="") as f:
                for pid, desc, seq in fasta_records(str(pdir / part)):
                    if pid in gap:
                        f.write(f">{pid} {desc}\n")
                        for j in range(0, len(seq), 60):
                            f.write(seq[j:j + 60] + "\n")
            # 独立gap输出（批次目录 work/struct_channel/gap_<tag>/，不与旧批次冲突）
            channel.run(str(gap_fa), str(gap_out))
        if args.phase in ("append", "all"):
            if not gap_out.exists():
                log.warning("[%s] gap输出不存在（先运行 --phase ips），跳过追加", tag)
                continue
            n = append_rows(str(gap_out), str(main_tsv))
            covered2 = {r["protein_id"] for r in read_tsv(str(main_tsv))} & tgt
            log.info("[%s] 追加 %s 行 -> 最终覆盖 %s/%s（剩余 %s 为InterProScan无域GO结果）",
                     tag, n, len(covered2), len(tgt), len(tgt - covered2))
    log.info("struct 缺口补跑完成 (phase=%s)", args.phase)


if __name__ == "__main__":
    main()
