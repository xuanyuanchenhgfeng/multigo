# -*- coding: utf-8 -*-
"""
流水线编排器
============
- 三个Channel并行收集证据（线程池；Seq/Struct为子进程计算型，Lit为网络IO型）
- Arbiter显式规则仲裁 -> GO-DAG校验 -> 标准化输出（GAF + 简化TSV）
- 评估（PSB26/CAFA四指标：Fmax/Smin/AUPR/AUC，含单Channel/双Channel消融对照）
- 演示模式（demo_sample_size）与断点续跑（Channel输出已存在则跳过）
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set

from .channels.base import iter_fasta, write_fasta, read_tsv
from .channels.lit_channel import LitChannel
from .channels.seq_channel import SeqChannel
from .channels.struct_channel import StructChannel
from .arbiter import Arbiter
from .config import load_config, setup_logging
from .evaluator import Evaluator
from .go_dag import GODag

log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dag = GODag(cfg["paths"]["go_obo"],
                         relations=cfg.get("go_dag", {}).get("relations"))
        self.seq_channel = SeqChannel(cfg, self.dag)
        self.struct_channel = StructChannel(cfg, self.dag)
        self.lit_channel = LitChannel(cfg, self.dag)
        self.arbiter = Arbiter(cfg, self.dag)
        self.evaluator = Evaluator(cfg, self.dag)
        self.work = Path(cfg["paths"]["work_dir"])
        self.results = Path(cfg["paths"]["results_dir"])
        self.sample_size = int(cfg.get("runtime", {}).get("demo_sample_size", 0))
        self.force = bool(cfg.get("runtime", {}).get("force_rerun", False))
        # 元数据（Lit-Channel查询词）：目标蛋白 + 库蛋白（供苦荞等无元数据物种借用同源命中）
        self.meta: Dict[str, dict] = {}
        meta_dir = Path(cfg["paths"]["meta_dir"])
        for meta_file in ("sprot_targets_meta.tsv", "sprot_library_meta.tsv"):
            meta_path = meta_dir / meta_file
            if meta_path.exists():
                for r in read_tsv(str(meta_path)):
                    self.meta.setdefault(r["protein_id"], r)

    # ------------------------------------------------------------ 采样
    def _prepare_query(self, fasta: str, tag: str) -> str:
        """演示模式取前N条，返回实际查询FASTA路径与蛋白ID集合。"""
        records = list(iter_fasta(fasta))
        if self.sample_size and self.sample_size > 0:
            records = records[: self.sample_size]
        out = self.work / f"query_{tag}.fasta"
        write_fasta(str(out), records)
        return str(out), [r[0] for r in records]

    # ------------------------------------------------------------ Channel并行
    def _run_channels_parallel(self, tag: str, query_fasta: str,
                               protein_ids: List[str]) -> Dict[str, str]:
        """三Channel并行执行，返回 {channel_name: output_tsv}。"""
        self.seq_channel.use_library(tag)
        outs = {
            "seq": str(self.work / f"seq_channel_{tag}.tsv"),
            "struct": str(self.work / f"struct_channel_{tag}.tsv"),
            "lit": str(self.work / f"lit_channel_{tag}.tsv"),
        }
        jobs = []

        def job_seq():
            if Path(outs["seq"]).exists() and not self.force:
                log.info("[seq] 已存在，跳过: %s", outs["seq"])
                return "seq"
            self.seq_channel.run(query_fasta, outs["seq"])
            return "seq"

        def job_struct():
            if Path(outs["struct"]).exists() and not self.force:
                log.info("[struct] 已存在，跳过: %s", outs["struct"])
                return "struct"
            self.struct_channel.run(query_fasta, outs["struct"])
            return "struct"

        def job_lit():
            if Path(outs["lit"]).exists() and not self.force:
                log.info("[lit] 已存在，跳过: %s", outs["lit"])
                return "lit"
            target_ids = set(protein_ids)
            # 苦荞等无元数据蛋白：由调用方预填 meta 查询表
            meta_tsv = self._write_meta_subset(target_ids, tag)
            self.lit_channel.run(meta_tsv, target_ids, outs["lit"])
            return "lit"

        jobs = [job_seq, job_struct, job_lit]
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="channel") as ex:
            futures = {ex.submit(fn): fn.__name__ for fn in jobs}
            for fu in as_completed(futures):
                try:
                    fu.result()
                except Exception as e:  # noqa: BLE001
                    log.error("Channel %s 失败（降级为空结果）: %s", futures[fu], e)
                    name = futures[fu].replace("job_", "")
                    if not Path(outs[name]).exists():
                        Path(outs[name]).write_text(
                            "\t".join({"seq": "protein_id\tgo_id\tevalue\tidentity\tqcov\thit_acc",
                                       "struct": "protein_id\tgo_id\tcoverage\tsignature",
                                       "lit": "protein_id\tgo_id\tcitations\tpmids"}[name]) + "\n",
                            encoding="utf-8")
        return outs

    def _write_meta_subset(self, target_ids: Set[str], tag: str) -> str:
        """为目标蛋白写元数据子集TSV（Lit-Channel输入）。"""
        out = self.work / f"meta_{tag}.tsv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("protein_id\tentry_name\tgene_name\tprotein_name\ttaxid\torganism\tpubmed_ids\n")
            for pid in sorted(target_ids):
                m = self.meta.get(pid)
                if m:
                    f.write("\t".join(m.get(k, "") for k in
                                      ("protein_id", "entry_name", "gene_name",
                                       "protein_name", "taxid", "organism", "pubmed_ids")) + "\n")
                else:
                    # 无元数据（如苦荞蛋白）：用查询FASTA描述兜底
                    f.write(f"{pid}\t\t{pid}\t\t\t\t\n")
        return str(out)

    # ------------------------------------------------------------ 输出
    def write_gaf(self, pred_tsv: str, out_gaf: str, db: str = "UniProtKB",
                  taxon: str = "") -> str:
        """预测TSV -> 标准GAF 2.2。"""
        oc = self.cfg.get("output", {})
        ev_code = oc.get("gaf_evidence_code", "RCA")
        assigned_by = oc.get("assigned_by", "MultiChannelArbiter")
        db_ref = oc.get("gaf_db_reference", "GO_REF:0000002")
        today = date.today().strftime("%Y%m%d")
        symbol_by_prot = {}
        name_by_prot = {}
        for pid, m in self.meta.items():
            symbol_by_prot[pid] = m.get("gene_name") or pid
            name_by_prot[pid] = m.get("protein_name") or ""

        lines = ["!gaf-version: 2.2", "!Generated-by: MultiChannelGOAnnotationPipeline"]
        n = 0
        for r in read_tsv(pred_tsv):
            pid = r["protein_id"]
            go = r["go_id"]
            aspect = r.get("aspect") or self.dag.aspect(go) or "?"
            name = name_by_prot.get(pid, "")
            tax = taxon or ""
            row = [db, pid, symbol_by_prot.get(pid, pid), "", go, db_ref, ev_code,
                   f"multigo:{r.get('sources','')}", aspect, name, "", "protein",
                   f"taxon:{tax}" if tax else "", today, assigned_by, "", ""]
            lines.append("\t".join(row))
            n += 1
        Path(out_gaf).parent.mkdir(parents=True, exist_ok=True)
        with open(out_gaf, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
        log.info("GAF输出: %s 注释行 -> %s", f"{n:,}", out_gaf)
        return out_gaf

    def write_simple_tsv(self, pred_tsv: str, out_tsv: str) -> str:
        """简化版：蛋白ID + GO列表 + 置信度 + 证据来源（每蛋白一行）。"""
        by_prot: Dict[str, List[dict]] = {}
        for r in read_tsv(pred_tsv):
            by_prot.setdefault(r["protein_id"], []).append(r)
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_tsv, "w", encoding="utf-8", newline="") as f:
            f.write("protein_id\tn_go\tgo_list\tscores\tsources\n")
            for pid in sorted(by_prot):
                terms = sorted(by_prot[pid], key=lambda r: -float(r.get("score") or 0))
                f.write("\t".join([
                    pid, str(len(terms)),
                    ";".join(t["go_id"] for t in terms),
                    ";".join(f"{float(t.get('score') or 0):.3f}" for t in terms),
                    ";".join(t.get("sources", "") for t in terms),
                ]) + "\n")
        log.info("简化TSV输出: %s 蛋白 -> %s", f"{len(by_prot):,}", out_tsv)
        return out_tsv

    # ------------------------------------------------------------ 目标集处理
    def run_target_set(self, tag: str, fasta: str, truth_tsv: Optional[str],
                       meta_override: Optional[Dict[str, dict]] = None,
                       evaluate: bool = True, db: str = "UniProtKB",
                       taxon: str = "", ablations: bool = False,
                       train_tsv: Optional[str] = None) -> Optional[dict]:
        """处理一个目标集：并行Channel -> 仲裁 -> 输出 -> 评估。
        train_tsv: 训练库GO注释（Smin 的 IC 统计来源，PSB26协议）。"""
        log.info("=" * 60)
        log.info("目标集 [%s]: %s", tag, fasta)
        if self.sample_size:
            log.info("演示模式: 仅取前 %s 条蛋白", self.sample_size)
        if meta_override:
            self.meta.update(meta_override)

        query_fasta, protein_ids = self._prepare_query(fasta, tag)
        log.info("查询蛋白: %s", f"{len(protein_ids):,}")

        channel_outs = self._run_channels_parallel(tag, query_fasta, protein_ids)

        # ---- 仲裁（全部Channel） ----
        pred_tsv = str(self.work / f"predictions_{tag}.tsv")
        audit = str(self.work / f"arbiter_audit_{tag}.jsonl")
        self.arbiter.arbitrate(channel_outs["seq"], channel_outs["struct"],
                               channel_outs["lit"], pred_tsv, audit)

        # ---- 输出 ----
        self.write_gaf(pred_tsv, str(self.results / f"{tag}_go_annotation.gaf"),
                       db=db, taxon=taxon)
        self.write_simple_tsv(pred_tsv, str(self.results / f"{tag}_go_annotation.tsv"))

        # ---- 冲突统计报告（白盒可追溯） ----
        stats_path = self.results / f"{tag}_arbiter_conflict_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.arbiter.stats, f, ensure_ascii=False, indent=2)
        log.info("Arbiter冲突统计 -> %s", stats_path)

        # ---- 评估 ----
        report = None
        if evaluate and truth_tsv:
            reports = []
            tgt = set(protein_ids)
            rep = self.evaluator.evaluate(
                pred_tsv, truth_tsv, f"{tag}-multchannel(3)",
                train_tsv=train_tsv,
                out_json=str(self.results / f"eval_{tag}_multi.json"),
                target_ids=tgt)
            reports.append(rep)
            # 单Channel/双Channel消融（评估多Channel协同增益）
            if ablations:
                for combo_name, combo in (
                        ("seq-only", ("seq",)), ("seq+struct", ("seq", "struct")),
                        ("seq+lit", ("seq", "lit"))):
                    pred_ab = str(self.work / f"predictions_{tag}_{combo_name}.tsv")
                    audit_ab = str(self.work / f"arbiter_audit_{tag}_{combo_name}.jsonl")
                    arb = Arbiter(self.cfg, self.dag)
                    arb.arbitrate(
                        channel_outs["seq"] if "seq" in combo else "",
                        channel_outs["struct"] if "struct" in combo else "",
                        channel_outs["lit"] if "lit" in combo else "",
                        pred_ab, audit_ab)
                    r = self.evaluator.evaluate(
                        pred_ab, truth_tsv, f"{tag}-{combo_name}",
                        train_tsv=train_tsv,
                        out_json=str(self.results / f"eval_{tag}_{combo_name.replace('+','_')}.json"),
                        target_ids=tgt)
                    reports.append(r)
            report = reports
            self.evaluator.write_text_report(
                reports, str(self.results / f"eval_report_{tag}.txt"))
        return report

    # ------------------------------------------------------------ 阶段1
    def run_stage1(self, only: Optional[List[str]] = None) -> None:
        """PSB26 时间基训练验证：划分2（验证）+ 划分3（测试）+ 拟南芥（近缘跨物种）。
        only: 限定运行的目标集标签列表（None=全部）。"""
        pdir = Path(self.cfg["paths"]["partitions_dir"])
        adir = Path(self.cfg["paths"]["annotations_dir"])
        # Smin 的 IC 统计源：训练库GO注释（划分1，已按证据码白名单+时间截止过滤）
        train_go = str(adir / "split1_go.tsv")

        def want(tag: str) -> bool:
            return only is None or tag in only

        # 划分2：验证集（首次实验性注释 ∈ (train_end, val_end]）
        if want("split2_val") and (pdir / "split2.fasta").exists():
            self.run_target_set(
                "split2_val", str(pdir / "split2.fasta"),
                truth_tsv=str(adir / "split2_go.tsv"),
                train_tsv=train_go, evaluate=True)
        # 划分3：测试集（首次实验性注释 ∈ (val_end, test_end]，PSB26 主报告口径）
        if want("split3_test") and (pdir / "split3.fasta").exists():
            self.run_target_set(
                "split3_test", str(pdir / "split3.fasta"),
                truth_tsv=str(adir / "split3_go.tsv"),
                train_tsv=train_go, evaluate=True, ablations=True)
        # 拟南芥：近缘跨物种验证（70%同物种旁系同源在库、无自身）
        if want("arabidopsis") and (pdir / "arabidopsis_val.fasta").exists():
            self.run_target_set(
                "arabidopsis", str(pdir / "arabidopsis_val.fasta"),
                truth_tsv=str(adir / "arabidopsis_go.tsv"),
                train_tsv=train_go, evaluate=True)

    # ------------------------------------------------------------ 阶段2
    def run_stage2(self, lit_query_from_seq: bool = True) -> None:
        """苦荞预测：完整pipeline -> GAF/TSV输出。"""
        tartary_faa = self.cfg["paths"]["tartary_faa"]
        if not Path(tartary_faa).exists():
            log.error("苦荞蛋白组不存在: %s", tartary_faa)
            return
        if lit_query_from_seq:
            self._prefill_tartary_meta()
        self.run_target_set(
            "tartary", tartary_faa, truth_tsv=None, evaluate=False,
            db="GWH", taxon="62330")   # Fagopyrum tataricum NCBI taxid（Pinku1 装配，GWHBJBL00000000）
        log.info("苦荞预测完成。TBGP比对与通路富集见后续步骤。")

    def _prefill_tartary_meta(self):
        """苦荞蛋白无Swiss-Prot元数据：先跑Seq-Channel，用同源命中的基因名构建
        Lit-Channel查询表（确定性：取top命中的基因名+物种）。"""
        log.info("构建苦荞Lit-Channel查询表（来自Seq-Channel同源命中）...")
        # 先单独跑Seq-Channel
        query_fasta, protein_ids = self._prepare_query(self.cfg["paths"]["tartary_faa"], "tartary")
        seq_out = str(self.work / "seq_channel_tartary.tsv")
        if not Path(seq_out).exists() or self.force:
            self.seq_channel.use_library("tartary")
            self.seq_channel.run(query_fasta, seq_out)
        # 命中acc -> 元数据
        hit_meta = {}
        for r in read_tsv(seq_out):
            hit_acc = r.get("hit_acc", "")
            if hit_acc and hit_acc in self.meta:
                hit_meta.setdefault(r["protein_id"], self.meta[hit_acc])  # 首个=最优命中
        for pid, m in hit_meta.items():
            m_copy = dict(m)
            m_copy["protein_id"] = pid  # 覆盖为苦荞蛋白ID，供_write_meta_subset输出
            self.meta[pid] = m_copy
        log.info("苦荞元数据预填: 用Seq-Channel同源命中: %s/%s 蛋白获得同源查询词",
                 f"{len(hit_meta):,}", f"{len(protein_ids):,}")
