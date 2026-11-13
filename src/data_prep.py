# -*- coding: utf-8 -*-
"""
数据准备模块（P0核心）—— PSB26 时间基三段式划分
================================================
对齐 PSB26 (Pacific Symposium on Biocomputing 2026) 的 CAFA 时间基协议：
1. 解析 filtered GAF，仅保留实验性/人工审核证据码（PSB26 EXP_CODES 白名单）
2. 按蛋白首次实验性注释日期做时间基三段式切割（以 GAF 日期近似 Swiss-Prot
   各版本发布日的"新蛋白"边界）：
   - 划分1（训练库）：首次实验性注释 <= train_end（模拟 v2023_05 知识截止）
   - 划分2（验证集）：首次实验性注释 ∈ (train_end, val_end]（对应 v2024_06 新蛋白）
   - 划分3（测试集）：首次实验性注释 ∈ (val_end, test_end]（对应 v2025_03 新蛋白）
   - 更晚的蛋白剔除（PSB26 不含 v2025_03 之后蛋白）
3. 拟南芥近缘验证集：Swiss-Prot ∩ goa_arabidopsis（白名单证据码）按确定性70/30切分
   - 70% 入比对库（GO映射仅取 <= library_annotation_cutoff）
   - 30% 验证集（库内有同物种旁系近缘同源、无自身，模拟苦荞近缘检索实况）
   - 与划分1零重叠
4. 库GO映射：划分1蛋白仅保留 <= library_annotation_cutoff 的白名单注释（知识截止纪律）
5. 元数据：从 uniprot_sprot.dat 提取目标蛋白的基因名/蛋白名/物种/PubMed引用（供Lit-Channel）
6. 构建MMseqs2搜索库（不可用时自动降级BLAST+ makeblastdb）
7. 输出 PSB26 Table 1 风格统计（各子本体 terms/proteins/train/val/test）

注意：GAF证据码 ND（no biological data）为"功能未知"占位注释，
在库映射与真值中一律剔除。
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .channels.base import iter_fasta, write_fasta
from .config import load_config, setup_logging

log = logging.getLogger("data_prep")

# GAF 2.2 列索引（0-based）
GAF_DB, GAF_ACC, GAF_SYM, GAF_QUAL, GAF_GO, GAF_REF, GAF_EV, GAF_WITH = 0, 1, 2, 3, 4, 5, 6, 7
GAF_ASPECT = 8          # 第9列：F/P/C（旧代码误用第4列qualifier）
GAF_TAXON, GAF_DATE, GAF_ASSIGNED = 12, 13, 14

# PSB26 实验性/人工审核证据码白名单
PSB26_EXP_CODES = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
    "HTP", "HDA", "HMP", "HGI", "HEP"})


# ---------------------------------------------------------------- GAF 解析
def parse_gaf(gaf_path: str, taxid_filter: Optional[str] = None):
    """解析GAF。返回 (annos, taxids)：
    annos[acc] = [(go_id, evidence, date, aspect)]
    taxids[acc] = taxon id
    """
    import gzip
    annos: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    taxids: Dict[str, str] = {}
    opener = gzip.open if str(gaf_path).endswith(".gz") else open
    n_lines = 0
    with opener(gaf_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= GAF_DATE:
                continue
            acc = cols[GAF_ACC]
            go = cols[GAF_GO]
            ev = cols[GAF_EV]
            date = cols[GAF_DATE]
            aspect = cols[GAF_ASPECT]
            tax = cols[GAF_TAXON].replace("taxon:", "").strip()
            n_lines += 1
            annos[acc].append((go, ev, date, aspect))
            taxids[acc] = tax
            if taxid_filter and tax == taxid_filter:
                pass  # 不过滤，由调用方决定
    log.info("GAF解析完成: %s 注释行, %s 蛋白 (%s)", f"{n_lines:,}", f"{len(annos):,}", Path(gaf_path).name)
    return dict(annos), taxids


def filter_annotations(annos: List[Tuple[str, str, str, str]],
                       date_cutoff: Optional[str] = None,
                       allowed_codes: Optional[Set[str]] = None,
                       drop_nd: bool = True) -> List[Tuple[str, str, str, str]]:
    """过滤注释：可选限定证据码白名单（PSB26 EXP_CODES）；剔除ND；
    可选剔除日期 > cutoff 的注释。"""
    out = []
    for go, ev, date, aspect in annos:
        if drop_nd and ev == "ND":
            continue
        if allowed_codes is not None and ev not in allowed_codes:
            continue
        if date_cutoff and date > date_cutoff:
            continue
        out.append((go, ev, date, aspect))
    return out


# ---------------------------------------------------------------- 划分
def deterministic_split(accs: List[str], train_ratio: float, salt: str) -> Set[str]:
    """确定性哈希70/30切分：md5(acc+salt) 前8hex mod 100 < ratio*100 -> 训练集。"""
    train = set()
    thr = int(round(train_ratio * 100))
    for acc in accs:
        h = hashlib.md5((salt + acc).encode()).hexdigest()[:8]
        if (int(h, 16) % 100) < thr:
            train.add(acc)
    return train


def write_go_tsv(path: str, annos_by_acc: Dict[str, List[Tuple[str, str, str, str]]]) -> int:
    """写注释TSV: protein_id, go_id, aspect, evidence, date"""
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("protein_id\tgo_id\taspect\tevidence\tdate\n")
        for acc in sorted(annos_by_acc):
            for go, ev, date, aspect in annos_by_acc[acc]:
                f.write(f"{acc}\t{go}\t{aspect}\t{ev}\t{date}\n")
                n += 1
    return n


# ---------------------------------------------------------------- 主流程
def run_data_prep(cfg: dict) -> dict:
    paths = cfg["paths"]
    split_cfg = cfg["split"]
    train_end = split_cfg["train_end"]
    val_end = split_cfg["val_end"]
    test_end = split_cfg["test_end"]
    lib_cutoff = split_cfg.get("library_annotation_cutoff", train_end)
    salt = split_cfg["split_seed_salt"]
    ratio = split_cfg["train_ratio"]
    allowed = set(split_cfg.get("evidence_codes") or PSB26_EXP_CODES)

    pdir = Path(paths["partitions_dir"])
    adir = Path(paths["annotations_dir"])

    # ---- 1. Swiss-Prot accession 集合 ----
    log.info("读取 Swiss-Prot 序列索引 ...")
    sp_seqs: Dict[str, Tuple[str, str]] = {}   # acc -> (desc, seq)  仅目标+库需要
    for pid, desc, seq in iter_fasta(paths["sprot_fasta"]):
        # 头格式: sp|ACC|ENTRYNAME desc
        parts = pid.split("|")
        acc = parts[1] if len(parts) >= 2 else pid
        sp_seqs[acc] = (desc, seq)
    log.info("Swiss-Prot 序列: %s", f"{len(sp_seqs):,}")

    # ---- 2. PSB26 时间基三段式切割（按首次实验性注释日期） ----
    annos, taxids = parse_gaf(paths["goa_all_noiea"])
    in_sp = {a: v for a, v in annos.items() if a in sp_seqs}
    # 首次实验性注释日期（仅白名单证据码；无白名单注释的蛋白剔除，
    # 对齐 PSB26 数据集仅保留实验性注释蛋白）
    first_exp: Dict[str, str] = {}
    for a, v in in_sp.items():
        ds = [x[2] for x in v if x[1] in allowed and x[1] != "ND"]
        if ds:
            first_exp[a] = min(ds)
    p1 = sorted(a for a, d in first_exp.items() if d <= train_end)
    p2 = sorted(a for a, d in first_exp.items() if train_end < d <= val_end)
    p3 = sorted(a for a, d in first_exp.items() if val_end < d <= test_end)
    n_late = sum(1 for d in first_exp.values() if d > test_end)
    log.info("PSB26时间切割(证据码白名单=%d种): 训练库=%s, 验证=%s, 测试=%s, 剔除晚于%s=%s",
             len(allowed), f"{len(p1):,}", f"{len(p2):,}", f"{len(p3):,}",
             test_end, f"{n_late:,}")

    # ---- 3. 拟南芥近缘验证（70%入库 + 30%验证） ----
    ath_annos, _ = parse_gaf(paths["goa_arabidopsis"])
    ath_in_sp = {a: v for a, v in ath_annos.items() if a in sp_seqs}
    # 全池：>=1条白名单证据码注释；剔除已在划分1中的（防自命中）
    p1_set = set(p1)
    ath_pool = sorted(a for a, v in ath_in_sp.items()
                      if a not in p1_set and any(x[1] in allowed for x in v))
    # 70% 植物代表入库 / 30% 近缘验证（确定性哈希，独立盐；逐蛋白判定与池成员变化无关）
    ath_lib_set = deterministic_split(ath_pool, ratio, salt + "-arabidopsis")
    ath_lib = sorted(ath_lib_set)
    ath_val = sorted(set(ath_pool) - ath_lib_set)
    log.info("拟南芥近缘验证: 全池=%s -> 库(70%%)=%s + 验证(30%%)=%s (与split1重叠=0)",
             f"{len(ath_pool):,}", f"{len(ath_lib):,}", f"{len(ath_val):,}")

    # ---- 4. 写划分FASTA（以accession为干净ID）----
    def fasta_records(accs):
        for acc in accs:
            desc, seq = sp_seqs[acc]
            yield acc, desc, seq

    n1 = write_fasta(pdir / "split1.fasta", fasta_records(p1))
    n2 = write_fasta(pdir / "split2.fasta", fasta_records(p2))
    n3 = write_fasta(pdir / "split3.fasta", fasta_records(p3))
    nath_lib = write_fasta(pdir / "arabidopsis_lib.fasta", fasta_records(ath_lib))
    nath = write_fasta(pdir / "arabidopsis_val.fasta", fasta_records(ath_val))

    # ---- 5. 写注释TSV（全部限定白名单证据码） ----
    # 库映射：划分1 + 拟南芥库侧(70%)，仅保留 <= lib_cutoff 的注释（知识截止纪律）
    lib_annos = {a: filter_annotations(in_sp[a], date_cutoff=lib_cutoff,
                                      allowed_codes=allowed) for a in p1}
    ath_lib_annos = {a: filter_annotations(ath_in_sp[a], date_cutoff=lib_cutoff,
                                           allowed_codes=allowed) for a in ath_lib}
    lib_annos.update({a: v for a, v in ath_lib_annos.items() if v})
    lib_annos = {a: v for a, v in lib_annos.items() if v}
    n_lib = write_go_tsv(adir / "split1_go.tsv", lib_annos)
    # 验证真值：日期 <= val_end（对应 v2024_06 发布时的注释状态）
    n_truth2 = write_go_tsv(adir / "split2_go.tsv",
                            {a: filter_annotations(in_sp[a], date_cutoff=val_end,
                                                  allowed_codes=allowed) for a in p2})
    # 测试真值：日期 <= test_end（对应 v2025_03 发布时的注释状态）
    n_truth3 = write_go_tsv(adir / "split3_go.tsv",
                            {a: filter_annotations(in_sp[a], date_cutoff=test_end,
                                                  allowed_codes=allowed) for a in p3})
    # 拟南芥验证真值：白名单注释全量（近缘跨物种验证用"当前知识"，库侧时间截止已防泄漏）
    n_ath = write_go_tsv(adir / "arabidopsis_go.tsv",
                         {a: filter_annotations(ath_in_sp[a], allowed_codes=allowed)
                          for a in ath_val})

    # ---- 6. PSB26 Table 1 风格统计（各子本体） ----
    table1 = _psb26_table1(lib_annos, p2, p3, in_sp, allowed, val_end, test_end)

    # ---- 7. 元数据提取（dat -> 目标蛋白基因名/物种/文献）----
    target_accs = set(p2) | set(p3) | set(ath_val)
    meta_rows = extract_dat_metadata(paths["sprot_dat"], target_accs)
    meta_path = Path(paths["meta_dir"]) / "sprot_targets_meta.tsv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        f.write("protein_id\tentry_name\tgene_name\tprotein_name\ttaxid\torganism\tpubmed_ids\n")
        for r in meta_rows:
            f.write("\t".join(r) + "\n")
    log.info("目标蛋白元数据: %s -> %s", f"{len(meta_rows):,}", meta_path.name)

    # ---- 8. 统计报告 ----
    stats = {
        "sprot_total": len(sp_seqs),
        "gaf_proteins": len(annos),
        "gaf_in_sprot": len(in_sp),
        "evidence_codes": sorted(allowed),
        "train_end": train_end, "val_end": val_end, "test_end": test_end,
        "split1_training": len(p1),
        "split2_validation": len(p2),
        "split3_testing": len(p3),
        "excluded_after_test_end": n_late,
        "arabidopsis_pool": len(ath_pool),
        "arabidopsis_library": len(ath_lib),
        "arabidopsis_validation": len(ath_val),
        "split1_fasta": n1, "split2_fasta": n2, "split3_fasta": n3,
        "arabidopsis_lib_fasta": nath_lib,
        "arabidopsis_val_fasta": nath,
        "library_go_lines": n_lib,
        "truth2_lines": n_truth2, "truth3_lines": n_truth3,
        "arabidopsis_go_lines": n_ath,
        "metadata_rows": len(meta_rows),
        "psb26_table1": table1,
    }
    stats_path = Path(paths["results_dir"]) / "data_prep_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("统计已写入 %s", stats_path)

    # ---- 9. 构建比对库 ----
    build_search_db(cfg)

    return stats


def _psb26_table1(lib_annos: Dict[str, list], p2: List[str], p3: List[str],
                  in_sp: Dict[str, list], allowed: Set[str],
                  val_end: str, test_end: str) -> dict:
    """PSB26 Table 1 风格：各子本体的 terms / proteins / training / validation / testing。"""
    splits = {
        "training": lib_annos,
        "validation": {a: filter_annotations(in_sp[a], date_cutoff=val_end,
                                             allowed_codes=allowed) for a in p2},
        "testing": {a: filter_annotations(in_sp[a], date_cutoff=test_end,
                                           allowed_codes=allowed) for a in p3},
    }
    table = {}
    aspect_names = {"F": "MFO", "P": "BPO", "C": "CCO"}
    for aspect, onto in aspect_names.items():
        terms = set()
        proteins = set()
        counts = {}
        for split_name, annos_by_acc in splits.items():
            n = 0
            for acc, lst in annos_by_acc.items():
                gos = {go for go, ev, date, asp in lst if asp == aspect and ev in allowed}
                if gos:
                    proteins.add(acc)
                    terms |= gos
                    n += 1
            counts[split_name] = n
        table[onto] = {
            "terms": len(terms),
            "proteins": len(proteins),
            **counts,
        }
    return table


# ---------------------------------------------------------------- 元数据
def extract_dat_metadata(dat_path: str, target_accs: Set[str]) -> List[List[str]]:
    """流式解析 uniprot_sprot.dat，提取目标蛋白元数据。
    返回行: [protein_id, entry_name, gene_name, protein_name, taxid, organism, pubmed_ids]
    """
    import gzip
    rows: List[List[str]] = []
    keep = False
    acc = entry = gene = pname = taxid = organism = ""
    pubmeds: List[str] = []

    def flush():
        nonlocal acc, entry, gene, pname, taxid, organism, pubmeds
        if keep and acc:
            rows.append([acc, entry, gene, pname, taxid, organism, ";".join(pubmeds[:50])])
        acc = entry = gene = pname = taxid = organism = ""
        pubmeds = []

    n_entries = 0
    with gzip.open(dat_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            tag = line[:2]
            if tag == "ID":
                flush()
                keep = False
                parts = line.split()
                entry = parts[1] if len(parts) > 1 else ""
                n_entries += 1
                if n_entries % 100000 == 0:
                    log.info("  dat解析进度: %s 条目", f"{n_entries:,}")
            elif tag == "AC" and not acc:
                # 首个AC为主accession
                first = line[5:].split(";")[0].strip()
                if first in target_accs:
                    acc = first
                    keep = True
            elif tag == "GN" and not gene:
                # GN   Name=sina;  或 GN   Name=xxx {ECO...}; Synonyms=...
                seg = line[5:].strip()
                for kv in seg.split(";"):
                    kv = kv.strip()
                    if kv.startswith("Name="):
                        gene = kv[5:].split("{")[0].strip()
                        break
            elif tag == "DE" and not pname:
                seg = line[5:].strip()
                if seg.startswith("RecName: Full="):
                    pname = seg[len("RecName: Full="):].split("{")[0].split(";")[0].strip()
            elif tag == "OS":
                if not organism:
                    organism = line[5:].split(".")[0].strip()
            elif tag == "OX":
                seg = line[5:].strip()
                if seg.startswith("NCBI_TaxID="):
                    taxid = seg.split("=")[1].split(";")[0].strip()
            elif tag == "DR":
                seg = line[5:].strip()
                if seg.startswith("PubMed;"):
                    pmid = seg.split(";")[1].strip()
                    if pmid:
                        pubmeds.append(pmid)
            elif line.startswith("//"):
                flush()
    flush()  # 最后一个条目
    return rows


# ---------------------------------------------------------------- 建库
def find_executable(cfg: dict, name: str) -> Optional[str]:
    """在配置与PATH中查找可执行文件。"""
    import shutil
    cand = cfg.get("tools", {}).get(name, "")
    if cand and (Path(cand).exists() or shutil.which(cand)):
        return cand
    which = shutil.which(name)
    if which:
        return which
    # 常见备选名（Windows mmseqs.bat / mmseqs.exe）
    for alt in (name + ".bat", name + ".exe", name + ".sh"):
        w = shutil.which(alt)
        if w:
            return w
    return None


def build_search_db(cfg: dict) -> bool:
    """构建Seq-Channel搜索库：合并库(split1+拟南芥库侧70%)，
    优先MMseqs2 createdb，降级BLAST+ makeblastdb。"""
    paths = cfg["paths"]
    pdir = Path(paths["partitions_dir"])
    fasta = pdir / "split1.fasta"
    if not fasta.exists():
        log.error("split1.fasta 不存在，先完成数据划分")
        return False
    # 合并库 = split1 + 拟南芥库侧（统一供 MMseqs2 / BLAST+ 使用）
    ath_lib = pdir / "arabidopsis_lib.fasta"
    library = pdir / "library.fasta"
    if ath_lib.exists():
        with open(fasta, "r", encoding="utf-8") as fi, \
                open(ath_lib, "r", encoding="utf-8") as fa, \
                open(library, "w", encoding="utf-8", newline="") as fo:
            fo.write(fi.read())
            fo.write(fa.read())
        fasta = library
        log.info("合并搜索库(split1+拟南芥库侧): %s", library)

    mmseqs = find_executable(cfg, "mmseqs")
    if mmseqs:
        db_prefix = Path(paths["mmseqs_db"])
        db_prefix.parent.mkdir(parents=True, exist_ok=True)
        # 已存在则跳过
        if db_prefix.with_suffix(db_prefix.suffix + ".index").exists() and not cfg["runtime"].get("force_rerun"):
            log.info("MMseqs2 库已存在，跳过建库: %s", db_prefix)
            return True
        cmd = [mmseqs, "createdb", str(fasta), str(db_prefix)]
        log.info("MMseqs2 建库: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, shell=False,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            log.info("MMseqs2 建库完成")
            return True
        log.error("MMseqs2 建库失败: %s", (r.stderr or r.stdout)[-500:])

    # ---- BLAST+ 降级 ----
    makeblastdb = find_executable(cfg, "makeblastdb")
    if makeblastdb:
        db_mmseqs = Path(paths["mmseqs_db"])
        db_blast = db_mmseqs.parent / (db_mmseqs.stem + "_blast")
        db_blast.parent.mkdir(parents=True, exist_ok=True)
        cmd = [makeblastdb, "-in", str(fasta), "-dbtype", "prot", "-out", str(db_blast)]
        log.info("降级使用 BLAST+ 建库: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, shell=False,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            log.info("BLAST+ 库构建完成: %s", db_blast)
            return True
        log.error("makeblastdb 失败: %s", (r.stderr or r.stdout)[-500:])

    log.warning("无可用的本地比对工具（MMseqs2/BLAST+），建库跳过——请部署工具后重跑 data_prep 或建库步骤")
    return False


def main():
    cfg = load_config()
    setup_logging(cfg)
    stats = run_data_prep(cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
