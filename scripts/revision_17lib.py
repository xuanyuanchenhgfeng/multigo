# -*- coding: utf-8 -*-
"""修订版：17 签名库包含/排除敏感性分析（写入正文 3.5）。

用 2026-10-22 的 17 库完整产物（backup_17lib_20261027，含 PANTHER/Gene3D/
FunFam 等）重建：fusion17 / seq+struct17 / interpro17 / union17，
本体=2023-10-09 快照、IC 仅训练库、文献为日期截断版，保证唯一变化因子
为结构域签名库范围。NN 基线只依赖序列通道，直接沿用既有结果。
另给 fusion17 相对 interpro17/union17 的配对 bootstrap 差值 CI。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline  # noqa: F401,E402
from src.config import load_config, setup_logging  # noqa: E402
from src.go_dag import GODag  # noqa: E402
from src.arbiter import Arbiter  # noqa: E402
from src.evaluator import Evaluator, load_predictions, load_truth  # noqa: E402
from src.channels.base import read_tsv  # noqa: E402
import revision_bootstrap as rb  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
WORK = PROJECT / "work"
PRED = WORK / "revision" / "preds" / "lib17"
RES = PROJECT / "results" / "revision"
BACKUP = PROJECT / "backup_17lib_20261027"


def main():
    cfg = load_config(str(PROJECT / "config_revision.yaml"))
    setup_logging(cfg)
    dag = GODag(cfg["paths"]["go_obo"], relations=["is_a", "part_of"])
    ev = Evaluator(cfg, dag)
    PRED.mkdir(parents=True, exist_ok=True)
    tag = "split3_test"
    struct17 = str(BACKUP / "struct_channel_split3_test.tsv.17lib")
    n_struct_prot = len({r["protein_id"] for r in read_tsv(struct17)})
    print("17库 struct 覆盖蛋白:", n_struct_prot)

    # ---- 基线：interpro17 / union17（nn1/nn3 与库范围无关，沿用）----
    ip = defaultdict(dict)
    for r in read_tsv(struct17):
        cov = float(r.get("coverage") or 0)
        ip[r["protein_id"]][r["go_id"]] = max(
            ip[r["protein_id"]].get(r["go_id"], 0.0), cov)
    nn3 = load_predictions(str(WORK / "baseline_nn_top3_split3_test.tsv"))
    union17 = {p: dict(d) for p, d in ip.items()}
    for p, d in nn3.items():
        union17.setdefault(p, {})
        for g, s in d.items():
            union17[p][g] = max(union17[p].get(g, 0.0), s)

    def dump(name, d):
        fp = PRED / f"baseline_{name}_{tag}.tsv"
        with open(fp, "w", encoding="utf-8") as f:
            f.write("protein_id\tgo_id\tscore\n")
            for p, terms in d.items():
                for g, s in terms.items():
                    f.write(f"{p}\t{g}\t{s:.6f}\n")
        return str(fp)

    paths = {"interpro17": dump("interpro17", ip),
             "union17": dump("union17", union17)}

    # ---- 仲裁：fusion17（seq+17库struct+dcut lit）与 seq+struct17 ----
    seq = str(WORK / "seq_channel_split3_test.tsv")
    lit = str(WORK / "lit_channel_split3_test_dcut.tsv")
    f17 = PRED / "predictions_fusion17.tsv"
    s17 = PRED / "predictions_seq_struct17.tsv"
    if not f17.exists():
        Arbiter(cfg, dag).arbitrate(seq, struct17, lit, str(f17),
                                    str(PRED / "audit_fusion17.jsonl"))
    if not s17.exists():
        Arbiter(cfg, dag).arbitrate(seq, struct17, "", str(s17),
                                    str(PRED / "audit_seqstruct17.jsonl"))
    paths["fusion17"] = str(f17)
    paths["seq+struct17"] = str(s17)

    truth = str(PROJECT / "data/annotations/split3_go.tsv")
    train = str(PROJECT / "data/annotations/split1_go.tsv")
    summary = {}
    for name, p in paths.items():
        r = ev.evaluate(p, truth, name, train_tsv=train)
        summary[name] = {k: r[k] for k in
                         ("fmax", "smin", "aupr", "auc", "coverage")}
    (RES / "lib17_sensitivity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k:14s} Fmax={v['fmax']:.4f} Smin={v['smin']:.4f} "
              f"AUPR={v['aupr']:.4f} AUC={v['auc']:.4f} cov={v['coverage']:.3f}")

    # ---- 配对 bootstrap：fusion17 与基线 ----
    truth_p = {p: ev._prop(g) for p, g in
               load_truth(truth, dag=dag).items()}
    train_truth = load_truth(train, dag=dag)
    ev.ic = ev.compute_ic([ev._prop(g) for g in train_truth.values()])
    targets = [p for p in truth_p if truth_p[p]]
    arrs = {}
    for name, p in paths.items():
        preds = {pid: ev._propagate_preds(g)
                 for pid, g in load_predictions(p).items()}
        arrs[name] = rb.precompute(ev, preds, truth_p, targets)
    rng = np.random.default_rng(20261104)
    n = len(targets)
    pairs = [("fusion17", "interpro17"), ("fusion17", "union17"),
             ("fusion17", "seq+struct17")]
    deltas = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(rb.B):
        idx = rng.integers(0, n, size=n)
        vals = {k: rb.metrics_on(idx, a) for k, a in arrs.items()}
        for a, b in pairs:
            deltas[f"{a}-{b}"].append(
                (vals[a][0] - vals[b][0], vals[a][1] - vals[b][1]))
    ci = {}
    for k, lst in deltas.items():
        a = np.array(lst)
        ci[k] = {"dFmax": round(float(a[:, 0].mean()), 4),
                 "dFmax_ci": [round(float(np.percentile(a[:, 0], 2.5)), 4),
                              round(float(np.percentile(a[:, 0], 97.5)), 4)],
                 "dSmin": round(float(a[:, 1].mean()), 4),
                 "dSmin_ci": [round(float(np.percentile(a[:, 1], 2.5)), 4),
                              round(float(np.percentile(a[:, 1], 97.5)), 4)]}
    (RES / "lib17_bootstrap.json").write_text(
        json.dumps(ci, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ci, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
