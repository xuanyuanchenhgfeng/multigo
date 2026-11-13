# -*- coding: utf-8 -*-
"""
Channel 2: Struct-Channel（结构功能域证据）
===========================================
- 工具：InterProScan（本地部署 / 远程API / 预计算结果）
- 约束：4线程 + 12GB堆内存，分批500条处理（内存优化）
- 五种模式（config: tools.interproscan_mode）：
    local       - 本地运行 interproscan.sh（仅Linux/WSL；Windows原生不支持）
    precomputed - 解析已有InterProScan输出TSV（在Linux服务器批量跑完拷回）
    ssh         - 远程服务器模式：scp上传查询 -> ssh逐批运行 -> scp拉回TSV解析
                  （需SSH免密登录；断点续跑：已完成的远程批次自动跳过）
    api         - 使用EBI InterProScan REST API批量查询（无需本地安装，需网络）
    skip        - 跳过该Channel（流水线以2个Channel继续，不中断）
- 输出：结构域GO候选 + 域匹配覆盖率（match_len/protein_len）
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .base import STRUCT_CHANNEL_COLUMNS, iter_fasta, write_candidates

log = logging.getLogger("struct_channel")

GO_FIELD_RE = re.compile(r"^(GO:\d{7})(\|GO:\d{7})*$")
# 真实 InterProScan TSV 的 GO 列格式为 "GO:0004332(InterPro)"，多个以 | 分隔，
# 无 GO 时为 "-"。用 token 正则提取 GO 号，剥离 (来源) 后缀。
GO_TOKEN_RE = re.compile(r"GO:\d{7}")

EBI_IPR_API = "https://www.ebi.ac.uk/interpro/api"


class StructChannel:
    def __init__(self, cfg: dict, go_dag):
        self.cfg = cfg
        self.go_dag = go_dag
        self.ac = cfg.get("struct_channel", {})
        self.tools = cfg.get("tools", {})
        self.paths = cfg["paths"]
        self.mode = self.tools.get("interproscan_mode", "skip")

    # ------------------------------------------------------------ 主入口
    def run(self, query_fasta: str, output_tsv: str) -> str:
        if self.mode == "skip":
            log.warning("Struct-Channel 模式=skip（本机为Windows/未部署InterProScan），"
                        "输出空结果，流水线继续")
            write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, [])
            return output_tsv

        if self.mode == "precomputed":
            return self._run_precomputed(query_fasta, output_tsv)

        if self.mode == "local":
            return self._run_local(query_fasta, output_tsv)

        if self.mode == "ssh":
            return self._run_ssh(query_fasta, output_tsv)

        if self.mode == "api":
            return self._run_api(query_fasta, output_tsv)

        raise ValueError(f"未知 interproscan_mode: {self.mode}")

    # ------------------------------------------------------------ 本地运行
    def _run_local(self, query_fasta: str, output_tsv: str) -> str:
        batch_size = int(self.ac.get("batch_size", 500))
        threads = int(self.ac.get("threads", 4))
        max_heap = str(self.ac.get("max_heap", "12G"))
        # 批次级并行：每个批次独立启动一个 interproscan.sh 进程（-cpu threads）。
        # InterProScan 单进程内加线程几乎不提速（瓶颈在线程级），多进程并行才有线性加速；
        # 各批次的 fasta/tmp/tsv 均按批次号独立命名，天然无冲突。
        # 总占用 ≈ parallel_batches × threads 核，parallel_batches × max_heap 内存。
        parallel_batches = int(self.ac.get("parallel_batches", 1))
        ips = self.tools.get("interproscan", "interproscan.sh")
        appl = self.ac.get("applications", "")

        # 按目标集(tag)隔离批次工作目录，避免不同目标集的 batch_0000.* 相互覆盖/误跳过
        tag = Path(output_tsv).stem.replace("struct_channel_", "")
        work = Path(self.paths["work_dir"]) / "struct_channel" / tag
        work.mkdir(parents=True, exist_ok=True)

        # 蛋白长度（覆盖率计算）
        prot_len: Dict[str, int] = {}
        records = list(iter_fasta(query_fasta))
        for pid, _, seq in records:
            prot_len[pid] = len(seq)

        n_batches = (len(records) + batch_size - 1) // batch_size

        # 1) 先写好所有批次 FASTA，并确定待运行批次
        pending: List[int] = []
        force = self.cfg["runtime"].get("force_rerun")
        for bi in range(n_batches):
            chunk = records[bi * batch_size:(bi + 1) * batch_size]
            batch_fa = work / f"batch_{bi:04d}.fasta"
            out_tsv = Path(str(work / f"batch_{bi:04d}") + ".tsv")
            if not batch_fa.exists():
                with open(batch_fa, "w", encoding="utf-8", newline="") as f:
                    for pid, desc, seq in chunk:
                        f.write(f">{pid} {desc}\n")
                        for j in range(0, len(seq), 60):
                            f.write(seq[j:j + 60] + "\n")
            if out_tsv.exists() and not force:
                log.info("批次 %s/%s 已完成，跳过", bi + 1, n_batches)
            else:
                pending.append(bi)

        # 2) 运行单批次（独立进程）
        def run_batch(bi: int) -> tuple:
            batch_fa = work / f"batch_{bi:04d}.fasta"
            out_base = work / f"batch_{bi:04d}"
            cmd = [ips, "-i", str(batch_fa), "-b", str(out_base),
                   "-goterms", "-cpu", str(threads), "-T", str(work / f"tmp_{bi:04d}")]
            if appl:
                cmd += ["-appl", appl]
            env = {"_JAVA_OPTIONS": f"-Xmx{max_heap}", **os.environ}
            java_bin = self.tools.get("java", "")
            if java_bin:
                java_home = str(Path(java_bin).parent)
                env["JAVA_HOME"] = java_home
                env["PATH"] = java_home + ":" + env.get("PATH", "")
            t0 = time.time()
            log.info("InterProScan 批次 %s/%s 启动（cpu=%s%s）",
                     bi + 1, n_batches, threads,
                     f", -appl {appl}" if appl else "")
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            ok = r.returncode == 0
            dt = (time.time() - t0) / 60
            if not ok:
                log.error("批次 %s 失败（跳过），耗时%.1f分钟: %s",
                          bi + 1, dt, (r.stderr or r.stdout)[-300:])
            else:
                log.info("InterProScan 批次 %s/%s 完成，耗时 %.1f 分钟", bi + 1, n_batches, dt)
            return bi, ok

        # 3) 批次级并行
        if pending:
            log.info("待运行批次 %s/%s，并行进程数=%s（每进程cpu=%s, heap=%s）",
                     len(pending), n_batches, parallel_batches, threads, max_heap)
            done_count = 0
            with ThreadPoolExecutor(max_workers=max(1, parallel_batches)) as ex:
                for bi, ok in ex.map(run_batch, pending):
                    done_count += 1
                    log.info("批次进度: %s/%s 完成", done_count, len(pending))

        # 4) 汇总解析所有批次 TSV
        all_rows: List[list] = []
        done_batches = 0
        for bi in range(n_batches):
            out_tsv = Path(str(work / f"batch_{bi:04d}") + ".tsv")
            if out_tsv.exists():
                all_rows.extend(self._parse_ips_tsv(out_tsv, prot_len))
                done_batches += 1

        # 防护：批次未全部完成时不得用残缺结果覆盖已有非空输出
        if done_batches < n_batches:
            outp = Path(output_tsv)
            if outp.exists() and sum(1 for _ in open(outp, encoding="utf-8")) > 1:
                log.error("批次完成 %s/%s，残缺结果将覆盖已有非空输出 %s —— 保留旧文件；"
                          "请修复失败的批次后重跑（断点续跑只补缺失批次）",
                          done_batches, n_batches, outp.name)
                return output_tsv
            log.warning("批次完成 %s/%s（原输出为空/缺失，先写出当前残缺结果）",
                        done_batches, n_batches)

        n = write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, all_rows)
        log.info("Struct-Channel完成: %s 候选GO行 -> %s", f"{n:,}", Path(output_tsv).name)
        return output_tsv

    # ------------------------------------------------------------ 预计算结果
    def _run_precomputed(self, query_fasta: str, output_tsv: str) -> str:
        """解析已有InterProScan TSV（tools.interproscan 指向文件或目录）。"""
        src = Path(self.tools.get("interproscan", ""))
        if not src.exists():
            log.error("precomputed 模式但路径不存在: %s，Struct-Channel输出空结果", src)
            write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, [])
            return output_tsv
        files = [src] if src.is_file() else sorted(src.glob("*.tsv"))
        prot_len = {pid: len(seq) for pid, _, seq in iter_fasta(query_fasta)}
        query_ids = set(prot_len)
        rows: List[list] = []
        for fp in files:
            rows.extend(self._parse_ips_tsv(fp, prot_len, only_ids=query_ids))
        n = write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, rows)
        log.info("Struct-Channel(precomputed)完成: %s 文件, %s 候选GO行 -> %s",
                 len(files), f"{n:,}", Path(output_tsv).name)
        return output_tsv

    # ------------------------------------------------------------ SSH远程运行
    def _run_ssh(self, query_fasta: str, output_tsv: str) -> str:
        """远程服务器模式：本地分批 -> scp上传 -> ssh逐批运行InterProScan -> scp拉回TSV解析。
        需SSH免密登录（公钥认证）；BatchMode=yes 避免交互式密码输入挂起流水线。
        断点续跑：远程已存在的批次TSV自动跳过，中断后重跑只补缺失批次。
        """
        import shutil
        import time

        sshc = self.ac.get("ssh", {})
        host = sshc.get("host", "")
        if not host:
            log.error("ssh 模式但 struct_channel.ssh.host 未配置，Struct-Channel输出空结果")
            write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, [])
            return output_tsv
        if shutil.which("ssh") is None or shutil.which("scp") is None:
            log.error("本机未找到 ssh/scp（需OpenSSH客户端），Struct-Channel输出空结果")
            write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, [])
            return output_tsv

        port = str(sshc.get("port", 22))
        remote_dir = sshc.get("remote_dir", "~/ips_jobs").rstrip("/")
        key_file = sshc.get("ssh_key", "")
        keep_remote = bool(sshc.get("keep_remote", True))
        ips = self.tools.get("interproscan", "/opt/interproscan/interproscan.sh")

        batch_size = int(self.ac.get("batch_size", 500))
        threads = int(self.ac.get("threads", 4))
        max_heap = str(self.ac.get("max_heap", "12G"))
        appl = self.ac.get("applications", "")

        tag = Path(output_tsv).stem.replace("struct_channel_", "")
        work = Path(self.paths["work_dir"]) / "struct_channel_ssh" / tag
        batches_dir = work / "batches"
        local_tsv = work / "tsv"
        batches_dir.mkdir(parents=True, exist_ok=True)
        local_tsv.mkdir(parents=True, exist_ok=True)

        def ssh_base() -> List[str]:
            cmd = ["ssh", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=accept-new", "-p", port]
            if key_file:
                cmd += ["-i", key_file]
            return cmd

        def scp_base() -> List[str]:
            cmd = ["scp", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=accept-new", "-P", port]
            if key_file:
                cmd += ["-i", key_file]
            return cmd

        def fail(msg: str) -> str:
            log.error("%s，Struct-Channel输出空结果", msg)
            write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, [])
            return output_tsv

        # 1) 连通性与远程InterProScan可执行检查
        r = subprocess.run(ssh_base() + [host, f"test -x {ips} && echo IPS_OK || echo IPS_MISSING"],
                           capture_output=True, text=True)
        if r.returncode != 0 or "IPS_OK" not in r.stdout:
            return fail(f"SSH连接失败或远程InterProScan不可用({ips}): "
                        f"{(r.stderr or r.stdout).strip()[:200]}")
        log.info("SSH远程InterProScan可用: %s @ %s:%s", ips, host, port)

        # 2) 远程目录准备
        r = subprocess.run(ssh_base() + [host,
                           f"mkdir -p {remote_dir}/batches {remote_dir}/tsv {remote_dir}/tmp"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return fail(f"远程目录创建失败: {r.stderr.strip()[:200]}")

        # 3) 本地分批（与local模式同一切分逻辑，保证可复现）
        records = list(iter_fasta(query_fasta))
        prot_len: Dict[str, int] = {pid: len(seq) for pid, _, seq in records}
        n_batches = (len(records) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            chunk = records[bi * batch_size:(bi + 1) * batch_size]
            batch_fa = batches_dir / f"batch_{bi:04d}.fasta"
            if batch_fa.exists() and batch_fa.stat().st_size > 0:
                continue
            with open(batch_fa, "w", encoding="utf-8", newline="") as f:
                for pid, desc, seq in chunk:
                    f.write(f">{pid} {desc}\n")
                    for j in range(0, len(seq), 60):
                        f.write(seq[j:j + 60] + "\n")

        # 4) 上传全部批次FASTA
        log.info("上传 %s 个批次FASTA -> %s:%s/batches/", n_batches, host, remote_dir)
        r = subprocess.run(scp_base() + [str(p) for p in sorted(batches_dir.glob("batch_*.fasta"))]
                           + [f"{host}:{remote_dir}/batches/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return fail(f"批次上传失败: {r.stderr.strip()[:300]}")

        # 5) 查询远程已完成批次（断点续跑）
        r = subprocess.run(ssh_base() + [host,
                           f"ls {remote_dir}/tsv/*.tsv 2>/dev/null || true"],
                           capture_output=True, text=True)
        done = {Path(x).name for x in r.stdout.split() if x.strip()}
        if done:
            log.info("远程已有 %s 个完成批次，将跳过", len(done))

        # 6) 逐批远程运行（4线程/12GB堆约束由远程命令显式指定）
        for bi in range(n_batches):
            name = f"batch_{bi:04d}"
            if f"{name}.tsv" in done:
                log.info("远程批次 %s/%s 已存在，跳过", bi + 1, n_batches)
                continue
            remote_cmd = (f"cd {remote_dir} && _JAVA_OPTIONS=-Xmx{max_heap} {ips} "
                          f"-i batches/{name}.fasta -b tsv/{name} -goterms "
                          f"-cpu {threads} -T tmp/{name}")
            if appl:
                remote_cmd += f" -appl {appl}"
            t0 = time.time()
            log.info("InterProScan远程批次 %s/%s: %s", bi + 1, n_batches, remote_cmd)
            r = subprocess.run(ssh_base() + [host, remote_cmd],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log.error("远程批次 %s 失败（跳过）: %s", name, (r.stderr or r.stdout)[-300:])
                continue
            log.info("远程批次 %s 完成，耗时 %.1f 分钟", name, (time.time() - t0) / 60)

        # 7) 拉回全部TSV
        log.info("下载远程TSV -> %s", local_tsv)
        r = subprocess.run(scp_base() + [f"{host}:{remote_dir}/tsv/*.tsv", f"{local_tsv}/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.error("TSV下载失败: %s", r.stderr.strip()[:300])

        # 8) 解析（仅保留当前目标集蛋白，复用precomputed解析器）
        rows: List[list] = []
        files = sorted(local_tsv.glob("*.tsv"))
        for fp in files:
            rows.extend(self._parse_ips_tsv(fp, prot_len, only_ids=set(prot_len)))
        n = write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, rows)
        log.info("Struct-Channel(ssh)完成: %s 文件, %s 候选GO行 -> %s",
                 len(files), f"{n:,}", Path(output_tsv).name)

        # 9) 可选清理远程中间文件（保留tsv以便断点续跑）
        if not keep_remote:
            subprocess.run(ssh_base() + [host,
                           f"rm -rf {remote_dir}/batches {remote_dir}/tmp"],
                           capture_output=True, text=True)
        return output_tsv

    # ------------------------------------------------------------ EBI REST API
    def _run_api(self, query_fasta: str, output_tsv: str) -> str:
        """使用 EBI InterProScan REST API 批量查询 UniProt 蛋白的 InterPro 注释。
        优点：无需本地安装 InterProScan；缺点：依赖网络、速率限制、仅对 UniProt 收录蛋白有效。
        """
        prot_len: Dict[str, int] = {}
        records = list(iter_fasta(query_fasta))
        for pid, _, seq in records:
            prot_len[pid] = len(seq)

        work = Path(self.paths["work_dir"]) / "struct_channel_api"
        work.mkdir(parents=True, exist_ok=True)

        # 断点续跑：已缓存的蛋白跳过
        cache_file = work / "ipr_cache.json"
        cache: Dict[str, List[dict]] = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                log.info("加载API缓存: %s 蛋白", len(cache))
            except json.JSONDecodeError:
                cache = {}

        def save_cache():
            cache_file.write_text(json.dumps(cache))

        min_cov = float(self.ac.get("min_coverage", 0.20))
        max_workers = int(self.ac.get("api_workers", 4))

        # 限速器（每秒最多请求数）
        rate_limit = float(self.ac.get("api_rate_limit", 2.0))
        lock = threading.Lock()
        last_call = [0.0]

        def rate_wait():
            with lock:
                elapsed = time.time() - last_call[0]
                wait = 1.0 / rate_limit - elapsed
                if wait > 0:
                    time.sleep(wait)
                last_call[0] = time.time()

        def fetch_protein(pid: str) -> Tuple[str, List[dict]]:
            """查询单个蛋白的 InterPro 条目。"""
            if pid in cache:
                return pid, cache[pid]
            rate_wait()
            url = f"{EBI_IPR_API}/entry/interpro/protein/uniprot/{pid}/?format=json"
            try:
                r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
                if r.status_code == 204:
                    return pid, []
                if r.status_code != 200:
                    log.debug("API %s -> HTTP %s", pid, r.status_code)
                    return pid, []
                data = r.json()
            except Exception as e:  # noqa: BLE001
                log.debug("API %s -> %s", pid, e)
                return pid, []

            results = data.get("results", [])
            entries: List[dict] = []
            for entry in results:
                meta = entry.get("metadata", {})
                go_terms = [g["identifier"] for g in meta.get("go_terms", [])]
                locs = entry.get("proteins", [{}])[0].get("entry_protein_locations", [])
                # 取该 InterPro 条目的最佳覆盖度
                best_cov = 0.0
                for loc in locs:
                    for frag in loc.get("fragments", []):
                        try:
                            start = int(frag["start"])
                            end = int(frag["end"])
                            plen = prot_len.get(pid, end - start + 1)
                            cov = min((end - start + 1) / plen, 1.0)
                            best_cov = max(best_cov, cov)
                        except (KeyError, ValueError, ZeroDivisionError):
                            continue
                if go_terms and best_cov >= min_cov:
                    signature = f"InterPro:{meta.get('accession', '')}"
                    entries.append({
                        "go_terms": go_terms,
                        "coverage": best_cov,
                        "signature": signature,
                    })
            cache[pid] = entries
            return pid, entries

        log.info("Struct-Channel(api): 查询 %s 蛋白 (workers=%s, rate=%s/s)",
                 len(prot_len), max_workers, rate_limit)
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_protein, pid): pid for pid in prot_len}
            for fu in as_completed(futures):
                pid = futures[fu]
                done += 1
                if done % 100 == 0:
                    log.info("API进度: %s/%s", done, len(prot_len))
                    save_cache()

        save_cache()

        # 解析缓存结果为 TSV 行
        rows: List[list] = []
        for pid in sorted(prot_len):
            entries = cache.get(pid, [])
            best: Dict[str, Tuple[float, str]] = {}  # go -> (cov, signature)
            for entry in entries:
                for go in entry["go_terms"]:
                    go_norm = self.go_dag.normalize(go)
                    if not self.go_dag.is_valid(go_norm):
                        continue
                    cov = entry["coverage"]
                    sig = entry["signature"]
                    cur = best.get(go_norm)
                    if cur is None or cov > cur[0]:
                        best[go_norm] = (cov, sig)
            for go, (cov, sig) in sorted(best.items()):
                rows.append([pid, go, f"{cov:.3f}", sig])

        n = write_candidates(output_tsv, STRUCT_CHANNEL_COLUMNS, rows)
        log.info("Struct-Channel(api)完成: %s 候选GO行 -> %s", f"{n:,}", Path(output_tsv).name)
        return output_tsv

    # ------------------------------------------------------------ TSV解析
    def _parse_ips_tsv(self, tsv_path: Path, prot_len: Dict[str, int],
                       only_ids: Optional[set] = None) -> List[list]:
        """解析InterProScan TSV（TSV 5 格式，15列）。列布局跨版本稳健解析：
        col0=蛋白ID, col2=长度, col3=分析库(Pfam/CDD...), col4=签名ID,
        col6/7=start/stop, col13=GO列("GO:0004332(InterPro)"，多个以 | 分隔，无则"-")。
        GO 号用 GO_TOKEN_RE 从含 "GO:" 的字段中提取（兼容 (来源) 后缀）。
        """
        min_cov = float(self.ac.get("min_coverage", 0.20))
        # go -> (max_coverage, signature)
        best: Dict[str, Dict[str, tuple]] = {}
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = line.rstrip("\n").split("\t")
                if len(vals) < 8:
                    continue
                pid = vals[0]
                if only_ids is not None and pid not in only_ids:
                    continue
                try:
                    start, stop = int(vals[6]), int(vals[7])
                except ValueError:
                    continue
                plen = prot_len.get(pid)
                if plen is None:
                    try:
                        plen = int(vals[2])
                    except (ValueError, IndexError):
                        continue
                    prot_len[pid] = plen
                # GO字段：扫描含 "GO:" 的列，提取所有 GO:\d{7} token（剥离 (InterPro) 等后缀）
                gos: List[str] = []
                for v in vals[3:]:
                    if v and "GO:" in v:
                        found = GO_TOKEN_RE.findall(v)
                        if found:
                            gos = found
                            break
                if not gos:
                    continue
                cov = min((stop - start + 1) / plen, 1.0)
                if cov < min_cov:
                    continue
                analysis = vals[3] if len(vals) > 3 else ""
                sig = vals[4] if len(vals) > 4 else ""
                signature = f"{analysis}:{sig}" if sig else analysis
                for go in gos:
                    go = self.go_dag.normalize(go)
                    if not self.go_dag.is_valid(go):
                        continue
                    d = best.setdefault(pid, {})
                    cur = d.get(go)
                    if cur is None or cov > cur[0]:
                        d[go] = (cov, signature)
        rows = []
        for pid in sorted(best):
            for go, (cov, sig) in sorted(best[pid].items()):
                rows.append([pid, go, f"{cov:.3f}", sig])
        return rows
