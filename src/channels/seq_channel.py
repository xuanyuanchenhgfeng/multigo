# -*- coding: utf-8 -*-
"""
Channel 1: Seq-Channel（序列同源证据）
=====================================
- 工具：MMseqs2（本地）；不可用时自动降级 BLAST+（blastp）
- 输入：查询蛋白FASTA（划分2/3、拟南芥子集或苦荞蛋白组）
- 比对库：划分1（Swiss-Prot时间切割子集，与查询严格互斥）
- 输出：同源蛋白GO候选 + E-value/一致性/覆盖度原始证据（TSV，供Arbiter评分）
"""
from __future__ import annotations

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base import SEQ_CHANNEL_COLUMNS, iter_fasta, write_candidates
from ..data_prep import find_executable

log = logging.getLogger("seq_channel")


class SeqChannel:
    def __init__(self, cfg: dict, go_dag):
        self.cfg = cfg
        self.go_dag = go_dag
        self.ac = cfg.get("seq_channel", {})
        self.tools = cfg.get("tools", {})
        self.paths = cfg["paths"]
        # 库GO映射：hit_acc -> {go_id}
        self.lib_go: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------ 库映射
    def load_library_go(self) -> int:
        """加载划分1的GO映射（split1_go.tsv）。"""
        path = Path(self.paths["annotations_dir"]) / "split1_go.tsv"
        self.lib_go.clear()
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                vals = line.rstrip("\n").split("\t")
                if len(vals) < 2:
                    continue
                acc, go = vals[0], vals[1]
                go = self.go_dag.normalize(go)
                if self.go_dag.is_valid(go):
                    self.lib_go.setdefault(acc, set()).add(go)
        total = sum(len(v) for v in self.lib_go.values())
        log.info("库GO映射加载: %s 蛋白, %s (蛋白,GO)对", f"{len(self.lib_go):,}", f"{total:,}")
        return len(self.lib_go)

    # ------------------------------------------------------------ 检索库选择
    def use_library(self, set_tag: str) -> None:
        """按查询集选定比对库 FASTA 与 DIAMOND 索引前缀。

        db_by_set 缺省时沿用 accession 里的显式配置，再缺省回落到历史行为
        （library.fasta + sprot_library 索引）。用错库会把不该检索的序列
        放进 top-5 名额，见 scripts/seq_channel_dbcheck.py。
        """
        entry = self.ac.get("db_by_set", {}).get(set_tag)
        if not entry:
            return
        db_dir = Path(self.paths.get("partitions_dir", "data/partitions")).parent / "diamond_db"
        self.ac["library_fasta"] = str(Path(entry[0])
                                       if Path(entry[0]).is_absolute()
                                       else Path(self.paths["partitions_dir"]) / entry[0])
        self.ac["diamond_db"] = str(Path(entry[1])
                                    if Path(entry[1]).is_absolute()
                                    else db_dir / entry[1])
        log.info("[%s] 检索库 %s -> 索引 %s", set_tag, self.ac["library_fasta"],
                 self.ac["diamond_db"])

    # ------------------------------------------------------------ 主入口
    def run(self, query_fasta: str, output_tsv: str) -> str:
        """对查询FASTA执行同源搜索并转移GO。返回输出TSV路径。"""
        self.load_library_go()
        hits = self._search(query_fasta)
        rows = self._transfer_go(hits)
        n = write_candidates(output_tsv, SEQ_CHANNEL_COLUMNS, rows)
        log.info("Seq-Channel完成: %s 查询, %s 候选GO行 -> %s",
                 f"{len({h[0] for h in hits}):,}", f"{n:,}", Path(output_tsv).name)
        return output_tsv

    # ------------------------------------------------------------ 搜索
    def _search(self, query_fasta: str) -> List[Tuple[str, str, float, float, float]]:
        """执行比对，返回[(query, target, evalue, identity_frac, qcov_frac)]（已过阈值）。"""
        ev_thr = float(self.ac.get("evalue", 1e-3))
        min_id = float(self.ac.get("min_identity", 0.25))
        min_qcov = float(self.ac.get("min_qcov", 0.50))
        threads = int(self.ac.get("threads", 8))

        raw_hits: List[Tuple[str, str, float, float, float]] = []
        # 引擎优先级：DIAMOND(--sensitive) -> MMseqs2 -> BLAST+
        # 实测（36,941条库/7,474查询，8线程）：DIAMOND --sensitive 比 MMseqs2 默认快约3倍、
        # 省约7倍内存，且同源检出更多（GO覆盖蛋白 5,107 vs 4,936）。
        diamond = find_executable(self.cfg, "diamond")
        if diamond:
            try:
                raw_hits = self._run_diamond(diamond, query_fasta, ev_thr, threads)
            except Exception as e:  # noqa: BLE001
                log.error("DIAMOND 运行失败(%s)，尝试MMseqs2降级", e)
                raw_hits = []
        if not raw_hits:
            mmseqs = find_executable(self.cfg, "mmseqs")
            if mmseqs:
                try:
                    raw_hits = self._run_mmseqs(mmseqs, query_fasta, ev_thr, threads)
                except Exception as e:  # noqa: BLE001
                    log.error("MMseqs2 运行失败(%s)，尝试BLAST+降级", e)
                    raw_hits = []
        if not raw_hits:
            blastp = find_executable(self.cfg, "blastp")
            if blastp:
                raw_hits = self._run_blast(blastp, query_fasta, ev_thr, threads)
            else:
                log.warning("无可用比对工具（DIAMOND/MMseqs2/BLAST+均缺失），Seq-Channel返回空结果")

        hits = [h for h in raw_hits
                if h[2] <= ev_thr and h[3] >= min_id and h[4] >= min_qcov]
        log.info("原始命中 %s -> 过滤后(E<=%g, id>=%g, qcov>=%g) %s",
                 f"{len(raw_hits):,}", ev_thr, min_id, min_qcov, f"{len(hits):,}")
        return hits

    def _run_diamond(self, diamond: str, query_fasta: str, ev_thr: float,
                     threads: int) -> List[Tuple[str, str, float, float, float]]:
        """DIAMOND blastp --sensitive。返回与 _run_mmseqs 相同结构的命中。
        pident/qcovhsp 输出为 0-100，此处归一化为 0-1。"""
        pdir = Path(self.paths["partitions_dir"])
        lib_fasta = Path(self.ac.get("library_fasta") or (pdir / "library.fasta"))
        if not lib_fasta.exists():
            lib_fasta = pdir / "split1.fasta"
        db_dir = Path(self.cfg.get("paths", {}).get("partitions_dir", "data/partitions")).parent / "diamond_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_prefix = Path(self.ac.get("diamond_db") or (db_dir / "sprot_library"))
        if not Path(str(db_prefix) + ".dmnd").exists():
            log.info("DIAMOND 建库: %s makedb --in %s", diamond, lib_fasta)
            r = subprocess.run([diamond, "makedb", "--in", str(lib_fasta),
                                "-d", str(db_prefix)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"diamond makedb 失败: {(r.stderr or r.stdout)[-300:]}")
        out = Path(self.cfg["paths"]["work_dir"]) / "seq_channel_tmp" / "diamond_hits.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sensitive = [] if self.ac.get("diamond_fast", False) else ["--sensitive"]
        cmd = [diamond, "blastp", "-d", str(db_prefix), "-q", query_fasta,
               "-o", str(out), "--outfmt", "6", "qseqid", "sseqid", "evalue",
               "pident", "qcovhsp", "--evalue", str(ev_thr),
               "--threads", str(threads), "--max-target-seqs", "300"] + sensitive
        log.info("DIAMOND 搜索(%s): %s", "fast" if not sensitive else "sensitive",
                 " ".join(cmd[:6]) + " ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"diamond blastp 失败: {(r.stderr or r.stdout)[-300:]}")
        hits = []
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                vals = line.rstrip("\n").split("\t")
                if len(vals) < 5:
                    continue
                try:
                    e = float(vals[2])
                    pid = float(vals[3]) / 100.0   # pident: 0-100 -> 0-1
                    qc = float(vals[4]) / 100.0    # qcovhsp: 0-100 -> 0-1
                except ValueError:
                    continue
                hits.append((vals[0], vals[1], e, pid, qc))
        return hits

    def _run_mmseqs(self, mmseqs: str, query_fasta: str, ev_thr: float,
                    threads: int) -> List[Tuple[str, str, float, float, float]]:
        work = Path(self.cfg["paths"]["work_dir"]) / "seq_channel_tmp"
        work.mkdir(parents=True, exist_ok=True)
        qdb = str(work / "queryDB")
        libdb = str(Path(self.paths["mmseqs_db"]))
        resdb = str(work / "resultDB")
        # 清理旧结果
        for p in work.glob("resultDB*"):
            p.unlink()
        for p in work.glob("queryDB*"):
            p.unlink()

        def call(args: List[str], desc: str):
            log.info("MMseqs2 %s: %s", desc, " ".join(args))
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"mmseqs {desc} 失败: {(r.stderr or r.stdout)[-300:]}")

        call([mmseqs, "createdb", query_fasta, qdb], "createdb(查询)")
        call([mmseqs, "search", qdb, libdb, resdb, str(work / "tmp"),
              "-e", str(ev_thr), "--threads", str(threads)], "search")
        tsv = work / "hits.tsv"
        call([mmseqs, "convertalis", qdb, libdb, resdb, str(tsv),
              "--format-output", "query,target,evalue,pident,qcov"], "convertalis")

        hits = []
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                vals = line.rstrip("\n").split("\t")
                if len(vals) < 5:
                    continue
                q, t = vals[0], vals[1]
                try:
                    e = float(vals[2])
                    pid = float(vals[3]) / 100.0   # pident: 0-100
                    qc = float(vals[4])             # qcov: 0-1（实测cygwin构建）
                    if qc > 1.0:                    # 兼容输出百分比的版本
                        qc /= 100.0
                except ValueError:
                    continue
                hits.append((q, t, e, pid, qc))
        return hits

    def _run_blast(self, blastp: str, query_fasta: str, ev_thr: float,
                   threads: int) -> List[Tuple[str, str, float, float, float]]:
        dbp = Path(self.paths["mmseqs_db"])
        db = dbp.parent / (dbp.stem + "_blast")
        if not db.exists():
            # 兼容 makeblastdb 尚未建库的情形
            from ..data_prep import build_search_db
            build_search_db(self.cfg)
        out = Path(self.cfg["paths"]["work_dir"]) / "seq_channel_tmp" / "blast_hits.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [blastp, "-query", query_fasta, "-db", str(db),
               "-outfmt", "6 qseqid sseqid evalue pident qcovhsp",
               "-evalue", str(ev_thr), "-num_threads", str(threads),
               "-max_target_seqs", "50"]
        log.info("BLAST+ 搜索: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"blastp 失败: {(r.stderr or r.stdout)[-300:]}")
        hits = []
        with open(out, "w", encoding="utf-8") as fo:
            fo.write(r.stdout)
        for line in r.stdout.splitlines():
            vals = line.split("\t")
            if len(vals) < 5:
                continue
            try:
                e = float(vals[2])
                pid = float(vals[3]) / 100.0
                qc = float(vals[4]) / 100.0
            except ValueError:
                continue
            hits.append((vals[0], vals[1], e, pid, qc))
        return hits

    # ------------------------------------------------------------ GO转移
    def _transfer_go(self, hits) -> List[list]:
        """top-K命中 -> GO候选。每个(查询,GO)保留最优(最低E-value)命中的证据。"""
        top_k = int(self.ac.get("top_hits", 5))
        by_query: Dict[str, List] = {}
        for q, t, e, pid, qc in hits:
            by_query.setdefault(q, []).append((e, t, pid, qc))
        rows = []
        for q, hlist in by_query.items():
            hlist.sort(key=lambda x: x[0])
            best: Dict[str, Tuple] = {}   # go -> (evalue, identity, qcov, hit_acc)
            for e, t, pid, qc in hlist[:top_k]:
                for go in self.lib_go.get(t, ()):  # 无GO注释的命中无候选价值
                    cur = best.get(go)
                    if cur is None or e < cur[0]:
                        best[go] = (e, pid, qc, t)
            for go, (e, pid, qc, t) in sorted(best.items()):
                rows.append([q, go, f"{e:.3g}", f"{pid:.3f}", f"{qc:.3f}", t])
        return rows
