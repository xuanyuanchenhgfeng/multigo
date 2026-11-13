# -*- coding: utf-8 -*-
"""提取库蛋白元数据（从Swiss-Prot dat文件），供苦荞Lit-Channel借用同源命中的基因名/PMID。
用法: python3 scripts/extract_library_meta.py
"""
import gzip
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            yield dict(zip(header, parts))


def extract_dat_metadata(dat_path, target_accs):
    """流式解析 uniprot_sprot.dat，提取目标蛋白元数据。"""
    rows = []
    keep = False
    acc = entry = gene = pname = taxid = organism = ""
    pubmeds = []
    n_entries = 0

    def flush():
        nonlocal acc, entry, gene, pname, taxid, organism, pubmeds
        if keep and acc:
            rows.append([acc, entry, gene, pname, taxid, organism, ";".join(pubmeds[:50])])
        acc = entry = gene = pname = taxid = organism = ""
        pubmeds = []

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
                    print(f"  dat解析进度: {n_entries:,} 条目", flush=True)
            elif tag == "AC" and not acc:
                first = line[5:].split(";")[0].strip()
                if first in target_accs:
                    acc = first
                    keep = True
            elif tag == "GN" and not gene:
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
    flush()
    return rows


def main():
    # 1. 读取tartary同源命中accessions
    seq_out = PROJECT / "work" / "seq_channel_tartary.tsv"
    hits = set()
    for r in read_tsv(str(seq_out)):
        acc = r.get("hit_acc", "")
        if acc:
            hits.add(acc)
    print(f"Total unique hit accessions: {len(hits)}", flush=True)

    # 2. 从Swiss-Prot dat提取元数据
    dat_path = PROJECT / "数据集" / "uniprot_sprot.dat.gz"
    rows = extract_dat_metadata(str(dat_path), hits)
    print(f"Extracted metadata for {len(rows)} library proteins", flush=True)

    # 3. 写入库蛋白元数据文件
    out_path = PROJECT / "data" / "meta" / "sprot_library_meta.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("protein_id\tentry_name\tgene_name\tprotein_name\ttaxid\torganism\tpubmed_ids\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"Written to {out_path}", flush=True)

    with_gene = sum(1 for r in rows if r[2])
    with_pmids = sum(1 for r in rows if r[6])
    print(f"With gene_name: {with_gene}/{len(rows)}", flush=True)
    print(f"With pubmed_ids: {with_pmids}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
