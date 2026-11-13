# -*- coding: utf-8 -*-
"""修订版：真正的跨物种验证（拟南芥）。

与原"70% 同物种蛋白在库"的留出实验不同，本脚本从检索库中剔除全部
taxid=3702（拟南芥）蛋白：既不并入 8 505 条拟南芥库侧蛋白，也从
16 948 条 Swiss-Prot 训练库中移除拟南芥条目，仅保留异种同源库。
结构域通道（签名与库无关）与文献通道（出版日期截断版）复用既有产物。
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.channels.base import (iter_fasta, write_fasta, write_candidates,  # noqa: E402
                               SEQ_CHANNEL_COLUMNS)
from src.channels.base import read_tsv  # noqa: E402
from src.channels.seq_channel import SeqChannel  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "work" / "revision"
DIAMOND = "/home/zsl/tools/diamond/diamond"


def build_noath_library(cfg):
    """library.fasta 剔除 taxid=3702 -> work/recision/cross_lib.fasta。"""
    outdir = OUT / "cross_species"
    outdir.mkdir(parents=True, exist_ok=True)
    lib_fasta = PROJECT / cfg["paths"]["partitions_dir"] / "library.fasta"
    meta = PROJECT / cfg["paths"]["meta_dir"] / "sprot_library_meta.tsv"
    ath = set()
    for r in read_tsv(str(meta)):
        tax = (r.get("taxid") or "").split()[0]
        if tax == "3702":
            ath.add(r["protein_id"])
    # GOA 来源的 8 505 条拟南芥库侧蛋白（ID 命名空间不同于 Swiss-Prot）
    ath_lib = PROJECT / cfg["paths"]["partitions_dir"] / "arabidopsis_lib.fasta"
    n_ath_goa = 0
    for rec in iter_fasta(str(ath_lib)):
        ath.add(rec[0]); n_ath_goa += 1
    print(f"拟南芥剔除集合: Swiss-Prot taxid3702 {len(ath) - n_ath_goa} + "
          f"GOA 库侧 {n_ath_goa}")
    kept, excl = [], []
    for rec in iter_fasta(str(lib_fasta)):
        pid = rec[0]
        (excl if pid in ath else kept).append(rec)
    out_fa = outdir / "library_noath.fasta"
    write_fasta(str(out_fa), kept)
    print(f"跨物种库: 保留 {len(kept)}，剔除拟南芥 {len(excl)}")
    return out_fa, len(kept), len(excl)


def diamond_search(cfg, lib_fasta, query_fasta, tag):
    db = OUT / "cross_species" / "noath_db"
    hits_tsv = OUT / "cross_species" / f"diamond_hits_{tag}.tsv"
    if not db.with_suffix(".dmnd").exists():
        r = subprocess.run([DIAMOND, "makedb", "--in", str(lib_fasta),
                            "-d", str(db)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
    cmd = [DIAMOND, "blastp", "-d", str(db), "-q", query_fasta,
           "-o", str(hits_tsv), "--outfmt", "6", "qseqid", "sseqid",
           "evalue", "pident", "qcovhsp", "--evalue", "1e-3",
           "--threads", "32", "--max-target-seqs", "300", "--sensitive"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    hits = []
    for line in hits_tsv.read_text().splitlines():
        v = line.split("\t")
        if len(v) < 5:
            continue
        hits.append((v[0], v[1], float(v[2]), float(v[3]) / 100,
                     float(v[4]) / 100))
    # 与 SeqChannel 相同的后过滤（E/id/qcov/top5 在 _transfer_go 内）
    hits = [h for h in hits if h[2] <= 1e-3 and h[3] >= 0.25 and h[4] >= 0.50]
    return hits


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    OUT.mkdir(parents=True, exist_ok=True)

    lib_fasta, n_keep, n_excl = build_noath_library(cfg)
    query_fasta = str(PROJECT / "data/partitions/arabidopsis_val.fasta")
    hits = diamond_search(cfg, lib_fasta, query_fasta, "arabidopsis")
    print(f"DIAMOND 过滤后命中: {len(hits):,}")

    ch = SeqChannel(cfg, dag)
    ch.load_library_go()
    rows = ch._transfer_go(hits)
    seq_tsv = OUT / "cross_species" / "seq_channel_arabidopsis_noath.tsv"
    write_candidates(str(seq_tsv), SEQ_CHANNEL_COLUMNS, rows)
    n_q = len({h[0] for h in hits})
    print(f"跨物种 seq 候选: {len(rows)} 行, 覆盖 {n_q} 查询蛋白 -> {seq_tsv.name}")

    arb = Arbiter(cfg, dag)
    truth = str(PROJECT / "data/annotations/arabidopsis_go_t1.tsv")
    train = str(PROJECT / "data/annotations/split1_go.tsv")
    struct_tsv = str(PROJECT / "work/struct_channel_arabidopsis.tsv")
    lit_tsv = str(PROJECT / "work/lit_channel_arabidopsis_dcut.tsv")

    # 仅异种序列（基线参照）
    pred_seq = OUT / "cross_species" / "pred_arabidopsis_noath_seq-only.tsv"
    arb.arbitrate(str(seq_tsv), "", "", str(pred_seq),
                  str(OUT / "cross_species" / "audit_seq.jsonl"))
    # 三通道（异种库 + 结构域 + 截断文献）；文献重跑未完成则跳过
    pred_multi = OUT / "cross_species" / "pred_arabidopsis_noath_multi.tsv"
    if Path(lit_tsv).exists():
        arb.arbitrate(str(seq_tsv), struct_tsv, lit_tsv, str(pred_multi),
                      str(OUT / "cross_species" / "audit_multi.jsonl"))
    else:
        print(f"[warn] 缺少 {lit_tsv}，跳过三通道跨物种评估")
        pred_multi = None

    ev = Evaluator(cfg, dag)
    import json
    results = {"n_library_kept": n_keep, "n_library_excluded_ath": n_excl,
               "n_queries_with_seq_hit": n_q}
    for name, pred in (("seq-only_noath", pred_seq), ("multi_noath", pred_multi)):
        if pred is None:
            continue
        rep = ev.evaluate(str(pred), truth, name, train_tsv=train)
        results[name] = rep
    out_json = PROJECT / "results/revision/eval_cross_species.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("->", out_json)


if __name__ == "__main__":
    main()
