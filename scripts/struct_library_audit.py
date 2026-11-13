# -*- coding: utf-8 -*-
"""结构通道签名库时间审计（审稿意见 2：时间纪律/泄漏）。

背景：初稿把"序列与结构域通道产物不依赖本体版本之外的未来信息"当作既定事实写入
方法学，这句话站不住。本脚本把它变成可核验的三条结论：

  1. 清点：逐个目标集解析 InterProScan 批次 XML，抽出实际使用的软件版本与
     成员库版本（版本串来自文件本身，不靠记忆或引用）；并按 signature 前缀
     统计结构通道 TSV 里每个库贡献的行数与蛋白数。
  2. 对齐：把成员库版本所属的 InterProScan data release 发布日期与知识截止
     T0=2023-11-08 对齐。发布序列取自官方 GitHub releases（来源与检索日期见
     RELEASES_NOTE），T0 之前最后一次发布是 5.64-96.0（2023-09-14），实际使用
     的 5.72-103.0 发布于 2024-11-28，即整个签名快照在截止之后 386 天。
  3. 定代价：结构通道分别限制为 (a) 只保留版本串自带日期且不晚于 T0 的库、
     (b) 完全置空，固定权重路线与学习式路线各重跑一次（学习式的 λ、分数下限
     都在受限后的验证窗重新标定，不沿用全量结构窗标出的参数），报告网格
     Fmax/AUPR/Smin 与验证窗冻结阈的可部署 F1。

结论口径：泄漏不是"可能有影响"，而是可测量的——受限后 Fmax/AUPR 掉多少，
正文与补充材料直接引用这里的数字。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402
from src.evaluator import Evaluator, load_truth  # noqa: E402
from src.learning import ArbitrationModel, CandidateFeatures, term_depth  # noqa: E402
import revision_evaluate as rev  # noqa: E402
import revision_learning as rl  # noqa: E402

WORK = PROJECT / "work"
ANN = PROJECT / "data" / "annotations"
OUT_DIR = WORK / "revision" / "preds" / "_struct_audit"
RES_DIR = PROJECT / "results" / "revision"
SEED = rl.SEED
B = 2000                                      # 1-SE 标准误用的验证窗重采样次数
GATES = rl.GATES
T0 = "2023-11-08"

# --- 官方发布序列（https://api.github.com/repos/ebi-pf-team/interproscan/releases，
#     检索日期 2026-11-11；只记录到月日，不额外推断成员库自身发布日期） ---
IPS_RELEASES = {
    "5.61-93.0": "2023-03-06", "5.62-94.0": "2023-05-11",
    "5.63-95.0": "2023-06-29", "5.64-96.0": "2023-09-14",
    "5.65-97.0": "2023-11-09", "5.66-98.0": "2024-01-25",
    "5.67-99.0": "2024-03-29", "5.68-100.0": "2024-05-30",
    "5.69-101.0": "2024-07-25", "5.70-102.0": "2024-10-03",
    "5.71-102.0": "2024-10-31", "5.72-103.0": "2024-11-28",
}

# XML 目录名 -> 结构通道 TSV 名（split2_val 的 TSV 由 gap 补跑结果组装，
# 其批次 XML 落在 gap_split2_val/；split3_test 另有一份 gap 补跑）
TSV_OF = {
    "split2_train": "split2_train", "split3_val": "split3_val",
    "split3_test": "split3_test", "gap_split2_val": "split2_val",
    "gap_split3_test": None,        # 补跑批次，其行已并入 split3_val/split2_val
    "arabidopsis": "arabidopsis", "tartary": "tartary",
}

LIB_VER_RE = re.compile(rb'library="([A-Za-z0-9_]+)"\s+version="([^"]+)"')
IPS_VER_RE = re.compile(rb'interproscan-version="([^"]+)"')
DATE_CODED = re.compile(r"^20(\d{2})_(\d{2})$")      # PROSITE/HAMAP/PIRSR 版本串


def norm_lib(name: str) -> str:
    """XML 的 library 属性（PROSITE_PATTERNS）与 TSV signature 前缀
    （ProSitePatterns）是同一库的两种写法，去分隔符+大写后才可对账。"""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def scan_xml(path: Path) -> tuple:
    """返回 (ipr 版本集合, 库版本计数 Counter, protein 元素计数)。"""
    versions, libs, nprot = set(), Counter(), 0
    text = path.read_bytes()
    versions.update(v.decode() for v in IPS_VER_RE.findall(text))
    for lib, ver in LIB_VER_RE.findall(text):
        libs[(lib.decode(), ver.decode())] += 1
    nprot = text.count(b"<protein>")
    return versions, libs, nprot


def xml_census(tag: str) -> dict:
    d = WORK / "struct_channel" / tag
    if not d.is_dir():
        return {"missing": True}
    files = sorted(d.glob("*.xml"))
    versions, libs, nprot = set(), Counter(), 0
    for f in files:
        v, l, n = scan_xml(f)
        versions |= v
        libs += l
        nprot += n
    per_lib = defaultdict(lambda: {"version": None, "n_matches": 0,
                                   "xml_spelling": set()})
    for (lib, ver), cnt in libs.items():
        slot = per_lib[norm_lib(lib)]
        slot["xml_spelling"].add(lib)
        if slot["version"] and slot["version"] != ver:
            slot["version"] = f"{slot['version']}|{ver}"
        elif ver:
            slot["version"] = ver
        slot["n_matches"] += cnt
    return {"n_xml_files": len(files), "n_proteins_in_xml": nprot,
            "interproscan_versions": sorted(versions),
            "release_dates": {v: IPS_RELEASES.get(v, "未收录")
                              for v in sorted(versions)},
            "libraries": {k: {"version": v["version"],
                              "n_matches": v["n_matches"],
                              "xml_spelling": sorted(v["xml_spelling"])}
                          for k, v in sorted(
                              per_lib.items(),
                              key=lambda kv: -kv[1]["n_matches"])}}


def tsv_census(path: Path) -> dict:
    """按 signature 前缀统计行数、蛋白数，以及"只有该库给出结构证据"的蛋白数。"""
    rows = read_tsv(str(path))
    by_lib, prot_libs = Counter(), defaultdict(set)
    for r in rows:
        sig = str(r.get("signature", ""))
        lib = norm_lib(sig.split(":")[0] if ":" in sig else "(空)")
        by_lib[lib] += 1
        prot_libs[r["protein_id"]].add(lib)
    uniq = Counter()
    for libs in prot_libs.values():
        if len(libs) == 1:
            uniq[next(iter(libs))] += 1
    return {"n_rows": len(rows), "n_proteins": len(prot_libs),
            "rows_by_library": dict(by_lib.most_common()),
            "proteins_by_library": {
                lib: sum(1 for s in prot_libs.values() if lib in s)
                for lib in by_lib},
            "proteins_only_from_library": dict(uniq.most_common())}


def pre_t0_libraries(census: dict) -> set:
    """版本串自带日期且不晚于 T0 的库（PROSITE/PIRSR/HAMAP 的 2023_05 一类）。"""
    keep = set()
    for lib, info in census["libraries"].items():
        m = DATE_CODED.match(str(info["version"]))
        if m and (2000 + int(m.group(1)), int(m.group(2))) <= (2023, 11):
            keep.add(lib)
    return keep


def restrict_struct(src: Path, dst: Path, keep) -> dict:
    """按库前缀过滤结构通道 TSV；keep=None 表示原样复制，keep=set() 表示置空。"""
    rows = read_tsv(str(src))
    out, kept = [], 0
    for r in rows:
        sig = str(r.get("signature", ""))
        lib = norm_lib(sig.split(":")[0] if ":" in sig else "(空)")
        if keep is None or lib in keep:
            kept += 1
            out.append(r)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("protein_id\tgo_id\tcoverage\tsignature\n")
        for r in out:
            f.write(f"{r['protein_id']}\t{r['go_id']}\t{r['coverage']}\t"
                    f"{r.get('signature', '')}\n")
    return {"n_rows_in": len(rows), "n_rows_kept": kept,
            "n_proteins_kept": len({r["protein_id"] for r in out})}


def run_pair(cfg, dag, ev, tp2, tg2, tp3, tg3, scorer, gate, tag,
             struct2, struct3, rng):
    """两个窗口各跑一次仲裁，返回网格指标 + 验证窗冻结阈的可部署 F1。"""
    v2, v3 = rl.channels("split2_val")[0], rl.channels("split3_test")[0]
    l2, l3 = rl.channels("split2_val")[2], rl.channels("split3_test")[2]
    arb = Arbiter(cfg, dag)
    if scorer is not None:
        arb.scorer = scorer
    if gate is not None:
        arb.min_score = gate
    p2 = str(OUT_DIR / f"pred_{tag}_split2_val.tsv")
    p3 = str(OUT_DIR / f"pred_{tag}_split3_test.tsv")
    arb.arbitrate(v2, struct2, l2, p2, str(OUT_DIR / f"audit_{tag}_val.jsonl"))
    arb.arbitrate(v3, struct3, l3, p3, str(OUT_DIR / f"audit_{tag}_test.jsonl"))
    m2, preds2, _, _ = rl.metrics_of(ev, tp2, tg2, p2)
    m3, preds3, _, _ = rl.metrics_of(ev, tp3, tg3, p3)
    out = {"val": {k: m2[k] for k in ("fmax", "aupr", "smin", "mean_terms_raw",
                                      "covered_propagated")},
           "test_grid": {k: m3[k] for k in ("fmax", "aupr", "smin", "auc",
                                           "mean_terms_raw", "mean_terms_at_fmax",
                                           "covered_propagated",
                                           "distinct_terms_at_fmax")}}
    out.update(rl.fixed_f1(ev, tp2, tg2, preds2, tp3, tg3, preds3, rng))
    return out


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"],
               relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    assert ev.ic_source == "train_only"
    train_truth = load_truth(str(ANN / "split1_go.tsv"),
                             exclude_iea=ev.exclude_iea, dag=dag)
    ev.ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])
    prevalence = Counter(g for gos in train_truth.values() for g in gos)
    depth = term_depth(dag)

    truth2 = load_truth(str(ANN / "split2_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    truth3 = load_truth(str(ANN / "split3_go.tsv"),
                        exclude_iea=ev.exclude_iea, dag=dag)
    tp2 = {p: ev._prop(g) for p, g in truth2.items()}
    tp3 = {p: ev._prop(g) for p, g in truth3.items()}
    tg2 = [p for p in tp2 if tp2[p]]
    tg3 = [p for p in tp3 if tp3[p]]

    # ---- 1) 清点 ----
    census = {tag: xml_census(tag) for tag in
              ("split2_train", "split3_val", "split3_test", "gap_split2_val",
               "gap_split3_test", "arabidopsis", "tartary")}
    tsvs = {}
    for name in ("split2_train", "split2_val", "split3_val", "split3_test",
                 "arabidopsis", "tartary"):
        p = WORK / f"struct_channel_{name}.tsv"
        if p.exists():
            tsvs[name] = tsv_census(p)
    all_ipr = sorted({v for c in census.values()
                      for v in c.get("interproscan_versions", [])})
    permissible = sorted((v for v, d in IPS_RELEASES.items() if d <= T0))
    keep_pre = pre_t0_libraries(census["split3_val"])

    # 库级时间分类：只有版本串自带日期的库能独立判定，其余一律记为"随 103.0 发布"
    lib_table = {}
    for lib, info in census["split3_val"]["libraries"].items():
        m = DATE_CODED.match(str(info["version"]))
        lib_table[lib] = {
            "version": info["version"], "xml_spelling": info["xml_spelling"],
            "bundled_in": all_ipr[-1],
            "bundled_release_date": IPS_RELEASES.get(all_ipr[-1]),
            "date_coded": bool(m),
            "temporal_class": ("pre_T0" if m and (2000 + int(m.group(1)),
                                                  int(m.group(2))) <= (2023, 11)
                               else ("post_T0" if m else "undated_bundled_103.0"))}


    # 结构证据的独占性：只靠结构通道（序列/文献都无候选）的目标蛋白
    arb0 = Arbiter(cfg, dag)
    s2, t2, l2 = rl.channels("split2_val")
    s3, t3, l3 = rl.channels("split3_test")
    seq3, stru3, lit3 = (arb0._load_seq(s3), arb0._load_struct(t3),
                         arb0._load_lit(l3))
    struct_only3 = [p for p in tg3 if p in stru3 and p not in seq3 and p not in lit3]
    struct_supported3 = sum(1 for p in tg3 if p in stru3)

    audit = {
        "T0": T0,
        "interproscan_releases_used": all_ipr,
        "release_dates": {v: IPS_RELEASES[v] for v in all_ipr},
        "last_release_not_later_than_T0": permissible[-1],
        "gap_days_after_T0": _days(T0, IPS_RELEASES[all_ipr[-1]]),
        "gap_n_releases_after_T0": sum(1 for v, d in IPS_RELEASES.items()
                                       if d > T0),
        "per_target_set_xml": census,
        "per_target_set_tsv": tsvs,
        "library_temporal_table": lib_table,
        "pre_T0_date_coded_libraries": sorted(keep_pre),
        "share_struct_rows_from_pre_T0_datecoded": {
            tag: round(sum(v for k, v in tsvs[tag]["rows_by_library"].items()
                           if k in keep_pre) / max(tsvs[tag]["n_rows"], 1), 4)
            for tag in ("split2_val", "split3_test") if tag in tsvs},
        "test_window_struct_dependence": {
            "n_targets": len(tg3),
            "targets_with_struct_evidence": struct_supported3,
            "targets_with_struct_as_only_channel": len(struct_only3)},
    }
    print(json.dumps({k: audit[k] for k in (
        "interproscan_releases_used", "release_dates",
        "last_release_not_later_than_T0", "gap_days_after_T0",
        "test_window_struct_dependence")}, ensure_ascii=False, indent=1),
        flush=True)

    # ---- 2) 受限重跑 ----
    variants = {"full": None, "datecoded_pre_T0": keep_pre, "dropped": set()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, keep in variants.items():
        paths[name] = (
            str(OUT_DIR / f"struct_{name}_val.tsv"),
            str(OUT_DIR / f"struct_{name}_test.tsv"))
        audit.setdefault("restriction", {})[name] = {
            "keep_libraries": sorted(keep) if keep is not None else "全部",
            "val": restrict_struct(Path(t2), Path(paths[name][0]), keep),
            "test": restrict_struct(Path(t3), Path(paths[name][1]), keep)}
    print("受限后的结构行数:",
          json.dumps(audit["restriction"], ensure_ascii=False), flush=True)

    rng = np.random.default_rng(SEED)
    sens = {}
    for name, (_sp, _tp) in paths.items():
        sens[f"fusion_{name}"] = run_pair(cfg, dag, ev, tp2, tg2, tp3, tg3,
                                         None, None, f"fusion_{name}",
                                         _sp, _tp, rng)
        arb = Arbiter(cfg, dag)
        feats = CandidateFeatures(arb, dag, ev.ic, dict(prevalence), depth)
        samples, _ = build_samples_restricted(arb, feats, _sp, l2, tp2, tg2, dag)
        model = ArbitrationModel(feats, seed=SEED).fit(samples)
        gate, grid = gate_tune_restricted(ev, cfg, dag, model, _sp, l2,
                                          tp2, tg2, name)
        audit.setdefault("learn_gate", {})[name] = {"gate": gate, "grid": grid}
        sens[f"learn_{name}"] = run_pair(cfg, dag, ev, tp2, tg2, tp3, tg3,
                                        model, gate, f"learn_{name}",
                                        _sp, _tp, rng)
        print(f"[{name}] fusion Fmax={sens['fusion_' + name]['test_grid']['fmax']:.4f}"
              f" F1={sens['fusion_' + name]['test_F1_one_se']:.4f} |"
              f" learn Fmax={sens['learn_' + name]['test_grid']['fmax']:.4f}"
              f" F1={sens['learn_' + name]['test_F1_one_se']:.4f}", flush=True)
    audit["sensitivity"] = sens
    audit["keep_pre_T0_libraries"] = sorted(keep_pre)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "struct_library_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出", RES_DIR / "struct_library_audit.json")


def _days(a: str, b: str) -> int:
    import datetime
    return (datetime.date.fromisoformat(b)
            - datetime.date.fromisoformat(a)).days


def build_samples_restricted(arb, feats, struct_p, lit_p, truth_prop, targets, dag):
    """受限结构通道下的验证窗候选样本（与 rl.build_samples 同特征、同样标签）。"""
    seq = arb._load_seq(rl.channels("split2_val")[0])
    struct = arb._load_struct(struct_p)
    lit = arb._load_lit(lit_p)
    rows, fixed = [], []
    for pid in targets:
        cands, _, _ = arb.collect(seq.get(pid, {}), struct.get(pid, {}),
                                  lit.get(pid, {}))
        for go, evi in cands.items():
            aspect = dag.aspect(go)
            if aspect not in ("F", "P", "C"):
                continue
            lab = 1 if go in truth_prop.get(pid, set()) else 0
            rows.append((pid, aspect, feats.vector(evi, go), lab))
            fixed.append((aspect, arb.score(evi, go), lab))
    return rows, fixed


def gate_tune_restricted(ev, cfg, dag, model, struct_p, lit_p, tp2, tg2, prefix):
    rows, best = {}, None
    for g in GATES:
        arb = Arbiter(cfg, dag)
        arb.scorer, arb.min_score = model, g
        p = str(OUT_DIR / f"{prefix}_val_gate_{g:g}.tsv")
        arb.arbitrate(rl.channels("split2_val")[0], struct_p, lit_p, p,
                      str(OUT_DIR / f"{prefix}_audit_gate_{g:g}.jsonl"))
        m2, _, _, _ = rl.metrics_of(ev, tp2, tg2, p)
        rows[f"{g:g}"] = {"val_fmax": m2["fmax"], "val_aupr": m2["aupr"],
                          "val_mean_terms_raw": m2["mean_terms_raw"],
                          "val_covered": m2["covered_propagated"]}
        if best is None or m2["fmax"] > best[1]:
            best = (g, m2["fmax"])
    return best[0], rows


if __name__ == "__main__":
    main()
