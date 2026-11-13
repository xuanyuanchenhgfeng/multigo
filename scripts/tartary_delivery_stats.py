# -*- coding: utf-8 -*-
"""补充表 S16（苦荞全蛋白组注释交付产物明细）的确定性重算。

只读在档产物，不重新仲裁：
  work/query_tartary.fasta、work/{seq,struct,lit}_channel_tartary.tsv、
  results/tartary_go_annotation.gaf、results/tartary_arbiter_conflict_stats.json。
多路证据口径：GAF 第 8 列 With/From 的 `multigo:` 通道标记在蛋白层面取并集。
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
RES = PROJECT / "results"

CHANNELS = ("seq", "struct", "lit")


def prots(tsv):
    with open(tsv, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = header.index("protein_id")
        return {line.split("\t", idx + 1)[idx] for line in fh if line.strip()}


def main():
    gaf = RES / "tartary_go_annotation.gaf"
    n_input = sum(1 for line in open(WORK / "query_tartary.fasta", encoding="utf-8")
                  if line.startswith(">"))
    hits = {c: len(prots(WORK / f"{c}_channel_tartary.tsv")) for c in CHANNELS}

    by_prot = defaultdict(set)
    sources = defaultdict(set)
    namespace = Counter()
    combos = Counter()
    terms = set()
    with open(gaf, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            pid, go, aspect, withfrom = f[1], f[4], f[8], f[7]
            if not withfrom.startswith("multigo:"):
                raise ValueError(f"unexpected With/From: {withfrom!r}")
            src = withfrom[len("multigo:"):]
            by_prot[pid].add(go)
            terms.add(go)
            namespace[aspect] += 1
            combos[src] += 1
            for c in src.split("|"):
                sources[pid].add(c)

    counts = [len(g) for g in by_prot.values()]
    support = {c: sum(1 for s in sources.values() if c in s) for c in CHANNELS}
    multi = Counter(len(s) for s in sources.values())
    out_prots = len(by_prot)
    audit = json.load(open(RES / "tartary_arbiter_conflict_stats.json", encoding="utf-8"))

    stats = {
        "input_sequences": n_input,
        "channel_hit_proteins": hits,
        "arbiter_intake_proteins": audit["proteins"],
        "proteins_with_output": out_prots,
        "output_fraction_of_input": round(out_prots / n_input, 4),
        "unknown_marked": audit["unknown_marked"],
        "illegal_terms": audit["illegal_terms"],
        "gaf_rows": sum(combos.values()),
        "go_terms_covered": len(terms),
        "terms_per_protein_mean": round(sum(counts) / len(counts), 2),
        "terms_per_protein_median": statistics.median(counts),
        "terms_per_protein_max": max(counts),
        "proteins_at_term_cap": sum(1 for c in counts if c == max(counts)),
        "namespace_rows": {"BPO": namespace["P"], "CCO": namespace["C"],
                           "MFO": namespace["F"]},
        "source_combo_rows": dict(combos),
        "channel_supported_proteins": support,
        "proteins_ge_2_channels": sum(v for k, v in multi.items() if k >= 2),
        "proteins_3_channels": multi[3],
        "topology_events": {k: audit[k] for k in
                            ("same_merges", "parent_child_collapses",
                             "branch_conflicts", "namespace_conflicts")},
    }
    stats["ge2_fraction_of_output"] = round(
        stats["proteins_ge_2_channels"] / out_prots, 4)
    stats["3ch_fraction_of_output"] = round(
        stats["proteins_3_channels"] / out_prots, 4)

    dest = RES / "revision" / "tartary_delivery_stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"-> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
