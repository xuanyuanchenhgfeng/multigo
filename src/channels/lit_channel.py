# -*- coding: utf-8 -*-
"""
Channel 3: Lit-Channel（文献证据）
==================================
- 工具：NCBI E-utilities API（esearch/efetch），BioPython风格requests调用
- 限速：默认3次/秒（每次请求间隔>=0.34s），本地SQLite缓存避免重复检索
- 流程（全部确定性，可复现）：
    1. 逐蛋白用 基因名[Gene Name] AND 物种[Organism] 检索PubMed（无基因名则用蛋白名）
    2. 取回前N篇文献的标题+摘要（XML解析）
    3. GO词典短语匹配（词典=划分1库注释中出现过的GO术语名，>=N个词，
       规避"nucleus"等过泛单词）——词元n-gram精确匹配，无模糊推理
    4. 每个GO的"引用数"= 提及该术语的文献篇数 -> 引用数置信度
- 错误处理：API失败重试后降级为本地缓存/跳过，绝不中断流程
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from .base import LIT_CHANNEL_COLUMNS, write_candidates, read_tsv

log = logging.getLogger("lit_channel")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
WORD_SPLIT_RE = re.compile(r"[^a-z0-9\-']+")


class RateLimiter:
    """简单全局限速器：保证相邻请求间隔 >= interval 秒。"""

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        delta = now - self._last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self._last = time.monotonic()


class LitChannel:
    def __init__(self, cfg: dict, go_dag):
        self.cfg = cfg
        self.go_dag = go_dag
        self.ac = cfg.get("lit_channel", {})
        self.paths = cfg["paths"]
        self.enabled = bool(self.ac.get("enabled", True))
        self.rate = RateLimiter(float(self.ac.get("rate_limit_interval", 0.34)))
        self.timeout = int(self.ac.get("timeout", 30))
        self.max_retries = int(self.ac.get("max_retries", 2))
        self.max_abstracts = int(self.ac.get("max_abstracts", 20))
        self.email = self.ac.get("email", "")
        # 知识截止：仅纳入出版日期不晚于该日的文献（PubMed [Date - Publication]）
        # 格式 "YYYY/MM/DD"；为空则不限制（旧行为）
        self.pub_date_cutoff = self.ac.get("pub_date_cutoff", "") or ""
        # GO词典: phrase -> [go_id]
        self.go_dict: Dict[str, List[str]] = {}
        self._max_phrase_words = 8
        self._db = None

    # ------------------------------------------------------------ 缓存
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            path = Path(self.paths["cache_dir"]) / "pubmed_cache.sqlite"
            # check_same_thread=False：流水线中不同目标集在不同工作线程复用同一Channel实例
            self._db = sqlite3.connect(str(path), check_same_thread=False,
                                       timeout=30)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=30000")
            self._db.execute("""CREATE TABLE IF NOT EXISTS pubmed_search(
                query TEXT PRIMARY KEY, count INTEGER, pmids TEXT, fetched_at TEXT)""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS pubmed_abstract(
                pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT, fetched_at TEXT)""")
            self._db.commit()
        return self._db

    def _cache_get(self, table: str, key: str):
        pk = "query" if table == "pubmed_search" else "pmid"
        cur = self._conn().execute(f"SELECT * FROM {table} WHERE {pk} = ?", (key,))
        return cur.fetchone()

    def _cache_put_search(self, query: str, count: int, pmids: List[str]):
        self._conn().execute(
            "INSERT OR REPLACE INTO pubmed_search VALUES (?,?,?,datetime('now'))",
            (query, count, ";".join(pmids)))
        self._conn().commit()

    def _cache_put_abstract(self, pmid: str, title: str, abstract: str):
        self._conn().execute(
            "INSERT OR REPLACE INTO pubmed_abstract VALUES (?,?,?,datetime('now'))",
            (pmid, title, abstract))
        self._conn().commit()

    # ------------------------------------------------------------ GO词典
    def build_go_dictionary(self, library_go_tsv: str) -> int:
        """GO词典 = 库注释GO集合的术语名（>=min_words词）。"""
        min_words = int(self.ac.get("go_dict_min_words", 2))
        self.go_dict.clear()
        seen: Set[str] = set()
        for row in read_tsv(library_go_tsv):
            go = self.go_dag.normalize(row["go_id"])
            if go in seen or not self.go_dag.is_valid(go):
                continue
            seen.add(go)
            name = self.go_dag.name(go).lower().strip()
            if not name:
                continue
            words = name.split()
            if len(words) < min_words or len(words) > self._max_phrase_words:
                continue
            self.go_dict.setdefault(name, []).append(go)
        log.info("Lit-Channel GO词典: %s 个短语（来自库注释 %s 个GO）",
                 f"{len(self.go_dict):,}", f"{len(seen):,}")
        return len(self.go_dict)

    # ------------------------------------------------------------ PubMed API
    def _http_get(self, url: str, params: dict) -> Optional[requests.Response]:
        for attempt in range(self.max_retries + 1):
            try:
                self.rate.wait()
                r = requests.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r
                log.warning("HTTP %s (%s)", r.status_code, url)
            except requests.RequestException as e:
                log.warning("请求失败(%s/重试%s): %s", url.split('/')[-1], attempt, e)
        return None

    def esearch(self, term: str) -> Optional[Tuple[int, List[str]]]:
        """PubMed检索，返回 (总数, 前N条PMID)。失败返回None（调用方降级）。"""
        cached = self._cache_get("pubmed_search", term)
        if cached:
            return int(cached[1]), [p for p in cached[2].split(";") if p]
        params = {"db": "pubmed", "term": term, "retmode": "json",
                  "retmax": self.max_abstracts, "tool": "multigochannel"}
        if self.email:
            params["email"] = self.email
        r = self._http_get(f"{EUTILS}/esearch.fcgi", params)
        if r is None:
            return None
        try:
            data = r.json()["esearchresult"]
            count = int(data.get("count", "0"))
            pmids = list(data.get("idlist", []))[: self.max_abstracts]
        except (ValueError, KeyError) as e:
            log.warning("esearch解析失败: %s", e)
            return None
        self._cache_put_search(term, count, pmids)
        return count, pmids

    def efetch_abstracts(self, pmids: List[str]) -> Dict[str, Tuple[str, str]]:
        """批量取标题+摘要。返回 pmid -> (title, abstract)。"""
        out: Dict[str, Tuple[str, str]] = {}
        missing: List[str] = []
        for pmid in pmids:
            cached = self._cache_get("pubmed_abstract", pmid)
            if cached:
                out[pmid] = (cached[1] or "", cached[2] or "")
            else:
                missing.append(pmid)
        if not missing:
            return out
        params = {"db": "pubmed", "id": ",".join(missing), "retmode": "xml",
                  "tool": "multigochannel"}
        if self.email:
            params["email"] = self.email
        r = self._http_get(f"{EUTILS}/efetch.fcgi", params)
        if r is None:
            return out
        try:
            root = ET.fromstring(r.text)
            for art in root.iter("PubmedArticle"):
                cite = art.find("MedlineCitation")
                if cite is None:
                    continue
                pmid_el = cite.find("PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                article = cite.find("Article")
                if article is None:
                    continue
                title_el = article.find("ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                parts = []
                abs_el = article.find("Abstract")
                if abs_el is not None:
                    for at in abs_el.findall("AbstractText"):
                        parts.append("".join(at.itertext()))
                out[pmid] = (title, " ".join(parts))
                self._cache_put_abstract(pmid, title, " ".join(parts))
        except ET.ParseError as e:
            log.warning("efetch XML解析失败: %s", e)
        return out

    # ------------------------------------------------------------ 文本匹配
    @staticmethod
    def _tokens(text: str) -> List[str]:
        toks = []
        for w in WORD_SPLIT_RE.split(text.lower()):
            w = w.strip("-'").strip()
            if w:
                toks.append(w)
        return toks

    def match_gos(self, texts: List[str]) -> Dict[str, List[str]]:
        """对多篇文献文本做n-gram短语匹配。返回 go_id -> [pmid...]"""
        hits: Dict[str, Set[str]] = {}
        for idx, text in enumerate(texts):
            toks = self._tokens(text)
            n = len(toks)
            for i in range(n):
                for k in range(2, self._max_phrase_words + 1):
                    if i + k > n:
                        break
                    phrase = " ".join(toks[i:i + k])
                    for go in self.go_dict.get(phrase, ()):
                        hits.setdefault(go, set()).add(str(idx))
        return {go: sorted(v) for go, v in hits.items()}

    # ------------------------------------------------------------ 主入口
    def run(self, meta_tsv: str, target_ids: Set[str], output_tsv: str) -> str:
        if not self.enabled:
            log.warning("Lit-Channel 已禁用(enabled=false)，输出空结果")
            write_candidates(output_tsv, LIT_CHANNEL_COLUMNS, [])
            return output_tsv

        lib_go = Path(self.paths["annotations_dir"]) / "split1_go.tsv"
        self.build_go_dictionary(str(lib_go))

        meta = {}
        for row in read_tsv(meta_tsv):
            if row["protein_id"] in target_ids:
                meta[row["protein_id"]] = row

        rows: List[list] = []
        n_done = n_fail = 0
        for pid in sorted(target_ids):
            m = meta.get(pid)
            if not m:
                continue
            gene = m.get("gene_name", "")
            organism = m.get("organism", "")
            pname = m.get("protein_name", "")
            # 检索词优先级：基因名+物种 > 基因名 > 蛋白名（确定性顺序）
            attempts = []
            if gene:
                if organism:
                    attempts.append(f'"{gene}"[Gene Name] AND "{organism}"[Organism]')
                attempts.append(f'"{gene}"[Gene Name]')
            if pname:
                attempts.append(f'"{pname}"')
            if self.pub_date_cutoff:
                date_clause = (f' AND ("0001/01/01"[Date - Publication] : '
                               f'"{self.pub_date_cutoff}"[Date - Publication])')
                attempts = [t + date_clause for t in attempts]
            res = None
            for term in attempts:
                res = self.esearch(term)
                if res is None:
                    continue  # 网络失败，尝试下一检索式（也可直接降级）
                if res[0] > 0:
                    break
            if res is None:
                n_fail += 1
                continue  # 全部失败：跳过该蛋白（不中断）
            count, pmids = res
            if not pmids:
                continue
            abstracts = self.efetch_abstracts(pmids)
            cited = [p for p in pmids if p in abstracts]
            texts = [abstracts[p][0] + " " + abstracts[p][1] for p in cited]
            if not texts:
                continue
            go_hits = self.match_gos(texts)
            logcap = float(self.ac.get("citation_logcap", 5.0))
            for go, arts in sorted(go_hits.items()):
                # citations = 提及该GO的文献篇数（引用计数置信度）
                # arts 是 texts 下标，第4列写对应的真实 PMID（旧版交付文件里这一列
                # 存的是下标本身，复原方法见 scripts/lit_source_spotcheck.py）
                rows.append([pid, go, len(arts),
                             ";".join(cited[int(i)] for i in arts[:10])])
            n_done += 1
            if n_done % 100 == 0:
                log.info("Lit-Channel进度: %s/%s 蛋白已处理", f"{n_done:,}", f"{len(target_ids):,}")
        n = write_candidates(output_tsv, LIT_CHANNEL_COLUMNS, rows)
        log.info("Lit-Channel完成: 处理%s蛋白(失败跳过%s), 候选GO行 %s -> %s",
                 f"{n_done:,}", n_fail, f"{n:,}", Path(output_tsv).name)
        return output_tsv
