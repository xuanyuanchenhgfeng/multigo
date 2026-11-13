# -*- coding: utf-8 -*-
"""把修订版仲裁预测 TSV 转成交付用 GAF 2.2 与简化注释表。

2026-11-04 的修订版主流程（revision_evaluate.py）只做仲裁与评估，不写 GAF；
本脚本按 src/pipeline.py 中 write_gaf / write_simple_tsv 的列布局补出交付文件，
差别仅在于 taxon 列取各蛋白自己的 Swiss-Prot 分类号（混合物种集合）。

用法:
    env -u LD_LIBRARY_PATH python3 -X utf8 scripts/make_delivery_annotation.py \
        [--out 数据/修订版注释]
"""
import argparse
import collections
import csv
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT / "work" / "revision" / "preds"
META_DIR = PROJECT / "data" / "meta"

# (目标集, 配置, 固定 taxid；空串=按蛋白各自分类号)
SETS = [
    ("split2_val", "multi", ""),
    ("split3_test", "multi", ""),
    ("split3_test", "seq-only", ""),
    ("split3_test", "seq+struct", ""),
    ("split3_test", "seq+lit", ""),
    ("arabidopsis", "multi", "3702"),
]


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_meta():
    meta = {}
    for fn in ("sprot_targets_meta.tsv", "sprot_library_meta.tsv"):
        p = META_DIR / fn
        if p.exists():
            for r in read_tsv(p):
                meta.setdefault(r["protein_id"], r)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "work" / "revision" / "annotation"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    today = date.today().strftime("%Y%m%d")

    for tag, combo, fixed_taxon in SETS:
        src = PRED_DIR / f"predictions_{tag}_{combo}.tsv"
        if not src.exists():
            raise SystemExit(f"缺少预测文件: {src}")
        rows = read_tsv(src)
        name = f"{tag}_{combo}"

        gaf = ["!gaf-version: 2.2", "!Generated-by: MultiChannelGOAnnotationPipeline"]
        for r in rows:
            pid = r["protein_id"]
            m = meta.get(pid, {})
            tax = fixed_taxon or (m.get("taxid") or "")
            gaf.append("\t".join([
                "UniProtKB", pid, m.get("gene_name") or pid, "", r["go_id"],
                "GO_REF:0000002", "RCA", f"multigo:{r.get('sources', '')}",
                r.get("aspect") or "?", m.get("protein_name") or "", "", "protein",
                f"taxon:{tax}" if tax else "", today, "MultiChannelArbiter", "", "",
            ]))
        (out_dir / f"{name}.gaf").write_text("\n".join(gaf) + "\n", encoding="utf-8")

        by_prot = collections.defaultdict(list)
        for r in rows:
            by_prot[r["protein_id"]].append(r)
        lines = ["protein_id\tn_go\tgo_list\tscores\tsources"]
        for pid in sorted(by_prot):
            terms = sorted(by_prot[pid], key=lambda r: -float(r.get("score") or 0))
            lines.append("\t".join([
                pid, str(len(terms)),
                ";".join(t["go_id"] for t in terms),
                ";".join(f"{float(t.get('score') or 0):.3f}" for t in terms),
                ";".join(t.get("sources", "") for t in terms),
            ]))
        (out_dir / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"{name}: {len(rows):,} 注释行 / {len(by_prot):,} 蛋白 -> {name}.gaf, {name}.tsv")


if __name__ == "__main__":
    main()
