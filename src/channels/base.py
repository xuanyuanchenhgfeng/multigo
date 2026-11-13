# -*- coding: utf-8 -*-
"""Channel 公共数据结构与输出读写。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# 各Channel候选输出的统一TSV表头
SEQ_CHANNEL_COLUMNS = ["protein_id", "go_id", "evalue", "identity", "qcov", "hit_acc"]
STRUCT_CHANNEL_COLUMNS = ["protein_id", "go_id", "coverage", "signature"]
LIT_CHANNEL_COLUMNS = ["protein_id", "go_id", "citations", "pmids"]


def write_candidates(path: str, columns: List[str], rows: Iterable[list]) -> int:
    """写候选TSV（含表头），返回行数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
            n += 1
    return n


def read_tsv(path: str) -> List[dict]:
    """读取TSV为dict列表（首行表头）。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            vals = line.rstrip("\n").split("\t")
            row = {h: (vals[i] if i < len(vals) else "") for i, h in enumerate(header)}
            out.append(row)
    return out


def iter_fasta(path: str):
    """流式读取FASTA（支持.gz），yield (id, description, sequence)。"""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        pid, desc, seq = None, "", []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if pid is not None:
                    yield pid, desc, "".join(seq)
                head = line[1:].split(None, 1)
                pid = head[0] if head else ""
                desc = head[1] if len(head) > 1 else ""
                seq = []
            else:
                seq.append(line.strip())
        if pid is not None:
            yield pid, desc, "".join(seq)


def write_fasta(path: str, records: Iterable) -> int:
    """写FASTA，records为(id, description, sequence)三元组。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for pid, desc, seq in records:
            f.write(f">{pid} {desc}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
            n += 1
    return n
