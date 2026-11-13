# -*- coding: utf-8 -*-
"""文献通道证据抽检（审稿意见 6：co-mention 通道的精度必须给出抽检）。

文献通道的机制是"基因名检索 + GO 术语名在题录文本里整短语命中"，它只证明
"这篇文献同时谈到了这个基因和这个功能词"，不证明该功能属于该蛋白。审稿人要求
对该通道做精度抽检，本脚本把这个要求变成可复核的产出：

  1. 还原证据链。交付的 lit_channel_*.tsv 第 4 列名为 pmids，实际存的是
     match_gos() 返回的文本序号（"4;6" 这种），不是 PMID；序号 -> PMID 的映射
     可以离线复原：检索式按 lit_channel.py 的优先级与日期截断子句重新拼出，
     cache/pubmed_cache.sqlite 的 pubmed_search 表给出该检索式的前 20 条 PMID，
     再按 pubmed_abstract 表过滤"实际取回过摘要"的那批，顺序与 texts 一致，
     于是序号唯一对应到 PMID。
  2. 三条可机械判定的证据强度指标（不做主观打分）：
       title_hit            GO 短语出现在标题里（标题级别的功能断言）
       gene_in_sentence     命中短语的同一句里出现该蛋白的基因名/蛋白名词元
                            （即"这篇文献把该功能与该基因放在一句话里"）
                       两者都不满足 = 仅在同篇文献的不同位置各自出现，是最弱的一档；
       truth_hit            该 (蛋白, GO) 在评估真值（传播后）中成立。
  3. 分层统计 + 一份可人工复核的抽样表（含 PMID、标题、命中句），
     分本体、分引用数、分"是否为唯一证据来源"给出精度。

结论口径：文献通道的可用精度由 truth_hit 直接给出，与仲裁器无关；
title_hit / gene_in_sentence 用来解释它为什么低，不是用来替代它。
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402
from src.evaluator import Evaluator, load_truth  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
RES_DIR = PROJECT / "results" / "revision"
PRED_DIR = WORK / "revision" / "preds"
CONFIG = os.environ.get("MULTIGO_CONFIG", "full")
PRED_BASE = PRED_DIR / "_strict" if CONFIG == "strict" else PRED_DIR
SUFFIX = "" if CONFIG == "full" else "_strict"
SAMPLE_TSV = WORK / "revision" / f"lit_spotcheck_sample{SUFFIX}.tsv"
CUTOFF = "2023/11/08"
N_SAMPLE = 60
SEED = 20261104
WORD_SPLIT_RE = re.compile(r"[^a-z0-9\-']+")
SENT_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")


def tokens(text: str):
    return [t for t in WORD_SPLIT_RE.split(str(text).lower()) if t]


def match_doc(text: str, phrase):
    """在整篇题录的词元流上定位 GO 术语短语，返回 (是否命中, 命中句列表)。

    通道当初是在 title+" "+abstract 的整体词元流上做 n-gram 匹配的，逐句匹配会
    漏掉跨句短语、把真实证据误判为无证据，所以这里先切句、再拼回词元流并记录
    每个词元属于第几句。
    """
    sents = [s for s in SENT_SPLIT_RE.split(str(text)) if s.strip()]
    toks, sid = [], []
    for i, s in enumerate(sents):
        t = tokens(s)
        toks.extend(t)
        sid.extend([i] * len(t))
    k = len(phrase)
    found = set()
    for i in range(len(toks) - k + 1):
        if toks[i:i + k] == phrase:
            found.add(sid[i])
    return bool(found), [sents[j] for j in sorted(found)]


def queries(meta: dict, cutoff: str):
    """复刻 lit_channel.run() 的检索式优先级（基因名+物种 > 基因名 > 蛋白名）。"""
    gene, organism, pname = (meta.get("gene_name", ""),
                             meta.get("organism", ""), meta.get("protein_name", ""))
    clause = (' AND ("0001/01/01"[Date - Publication] : '
              f'"{cutoff}"[Date - Publication])')
    att = []
    if gene:
        if organism:
            att.append(f'"{gene}"[Gene Name] AND "{organism}"[Organism]')
        att.append(f'"{gene}"[Gene Name]')
    if pname:
        att.append(f'"{pname}"')
    return [a + clause for a in att]


class Cache:
    """只读打开 PubMed 缓存；抽检不复算检索，只复原当时用过的证据。"""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self._memo = {}

    def search(self, term: str):
        row = self.db.execute(
            "SELECT count, pmids FROM pubmed_search WHERE query=?", (term,)).fetchone()
        if not row:
            return None
        return int(row[0]), [p for p in row[1].split(";") if p]

    def doc(self, pmid: str):
        row = self.db.execute(
            "SELECT title, abstract FROM pubmed_abstract WHERE pmid=?",
            (pmid,)).fetchone()
        return (row[0] or "", row[1] or "") if row else None

    def resolved(self, meta_by_pid: dict, pid: str):
        """返回 (检索式, [(pmid, title, abstract), ...])，顺序与通道内 texts 一致。"""
        if pid in self._memo:
            return self._memo[pid]
        m = meta_by_pid.get(pid)
        if not m:
            self._memo[pid] = (None, [])
            return self._memo[pid]
        term, docs = None, []
        for q in queries(m, CUTOFF):
            hit = self.search(q)
            if hit is None:
                continue
            if hit[0] > 0:              # 与通道一致：首个有命中的检索式即定型
                term = q
                got = []
                for p in hit[1]:
                    d = self.doc(p)
                    if d:
                        got.append((p, d[0], d[1]))
                docs = got
                break
        self._memo[pid] = (term, docs)
        return term, docs


def emitted_sources(pred_tsv: Path):
    """从仲裁输出里取 (protein, go) -> sources，用于判定"文献是否为唯一证据"。"""
    out = {}
    for r in read_tsv(str(pred_tsv)):
        out[(r["protein_id"], r["go_id"])] = r.get("sources", "")
    return out


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    truth = load_truth(str(ANN / "split3_go.tsv"),
                       exclude_iea=ev.exclude_iea, dag=dag)
    tp = {p: ev._prop(g) for p, g in truth.items()}
    meta_by_pid = {r["protein_id"]: r for r in
                   read_tsv(str(WORK / "meta_split3_test.tsv"))}
    cache = Cache(PROJECT / "cache" / "pubmed_cache.sqlite")

    src_learn = emitted_sources(PRED_BASE / "predictions_split3_test_learn.tsv")
    src_fusion = emitted_sources(PRED_BASE /
                                 "predictions_split3_test_multi.tsv")

    rows = read_tsv(str(WORK / "lit_channel_split3_test_dcut.tsv"))
    records = []
    for r in rows:
        pid, go = r["protein_id"], r["go_id"]
        name = (dag.name(go) or "").lower().strip()
        phrase = tokens(name)
        if len(phrase) < 2:        # 与通道的 go_dict_min_words=2 同规则
            continue
        idx = [int(x) for x in str(r.get("pmids", "")).split(";") if str(x).strip()]
        term, docs = cache.resolved(meta_by_pid, pid)
        if not docs:
            continue
        m = meta_by_pid.get(pid, {})
        ident = m.get("gene_name", "") or m.get("protein_name", "")
        gene_tok = set(tokens(ident)) - {""}
        title_hit = sent_hit = False
        ev_lines = []
        for i in idx:
            if i >= len(docs):
                continue
            pmid, title, abstract = docs[i]
            for text, where in ((title, "title"), (abstract, "abstract")):
                hit, sents = match_doc(text, phrase)
                if not hit:
                    continue
                if where == "title":
                    title_hit = True
                good = [s for s in sents if gene_tok & set(tokens(s))]
                if good:
                    sent_hit = True
                pick = (good or sents)[0]
                ev_lines.append(f"[{where}] {pmid}: {pick[:220]}")
        records.append({
            "protein_id": pid, "go_id": go, "aspect": dag.aspect(go),
            "go_name": name, "citations": int(r["citations"]),
            "n_evidence_docs": len(idx), "query": term,
            "title_hit": title_hit, "gene_in_sentence": sent_hit,
            "truth_hit": go in tp.get(pid, set()),
            "source_learn": src_learn.get((pid, go), "(未输出)"),
            "source_fusion": src_fusion.get((pid, go), "(未输出)"),
            "evidence": " || ".join(ev_lines[:2]),
        })
    print("可复原证据链的文献通道行数:", len(records), "/", len(rows))

    def agg(sel):
        n = len(sel)
        if not n:
            return {"n": 0}
        return {"n": n,
                "title_hit": round(sum(x["title_hit"] for x in sel) / n, 4),
                "gene_in_sentence": round(
                    sum(x["gene_in_sentence"] for x in sel) / n, 4),
                "both": round(sum(x["title_hit"] or x["gene_in_sentence"]
                                  for x in sel) / n, 4),
                "truth_hit": round(sum(x["truth_hit"] for x in sel) / n, 4)}

    lit_only_learn = [x for x in records if x["source_learn"] == "lit"]
    out = {
        "T0": CUTOFF,
        "config": CONFIG,
        "column_semantics_note": "lit_channel_*.tsv 第4列 pmids 存的是 match_gos 的"
                                 "文本序号，PMID 由 cache/pubmed_cache.sqlite 按"
                                 "检索式复原（复原率见 recovery.n_rows/n_rows_all）",
        "recovery": {"n_rows_all": len(rows), "n_rows_resolved": len(records)},
        "all_lit_rows": agg(records),
        "by_citations": {f">={k}": agg([x for x in records
                                        if x["citations"] >= k])
                         for k in (1, 2, 3)},
        "by_aspect": {a: agg([x for x in records if x["aspect"] == a])
                      for a in ("F", "P", "C")},
        "emitted_by_learn": agg([x for x in records
                                 if x["source_learn"] != "(未输出)"]),
        "lit_only_emitted_by_learn": agg(lit_only_learn),
        "lit_only_emitted_by_fusion": agg(
            [x for x in records if x["source_fusion"] == "lit"]),
        "lit_only_truth_hit_note": "lit-only 行的 truth_hit 即文献通道的独立精度，"
                                   "与仲裁权重无关；样本量小于 30 时只作定性参考",
    }
    # 分层抽样表（人工复核用）：按引用数分层，随机种子固定
    rng = random.Random(SEED)
    hi = [x for x in records if x["citations"] >= 3]
    lo = [x for x in records if x["citations"] < 3]
    sample = (rng.sample(hi, min(20, len(hi))) +
              rng.sample(lo, max(0, min(N_SAMPLE - 20, len(lo)))))
    rng.shuffle(sample)
    cols = ["protein_id", "go_id", "aspect", "go_name", "citations",
            "title_hit", "gene_in_sentence", "truth_hit", "source_learn",
            "query", "evidence"]
    SAMPLE_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_TSV, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for x in sample:
            f.write("\t".join(str(bool_to_str(x[c])) for c in cols) + "\n")
    out["sample_n"] = len(sample)
    out["sample_file"] = str(SAMPLE_TSV.relative_to(PROJECT))
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"lit_spotcheck{SUFFIX}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("recovery", "all_lit_rows", "by_citations", "by_aspect",
                       "emitted_by_learn", "lit_only_emitted_by_learn",
                       "lit_only_emitted_by_fusion")},
                     ensure_ascii=False, indent=1))


def bool_to_str(v):
    return "1" if v is True else ("0" if v is False else v)


if __name__ == "__main__":
    main()
