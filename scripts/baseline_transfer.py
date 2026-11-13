# -*- coding: utf-8 -*-
"""
转移基线对比实验（投稿补强：横向对比）
======================================
在同一时间基测试集（split3_test）上复现 4 个经典基线：
  1. NN-top1   : 最近邻 BLAST 转移（top-1 同源命中注释整体转移，CAFA 标准基线）
  2. NN-top3   : top-3 命中注释并集（Blast2GO 简化式）
  3. InterPro : 结构域签名直接映射 InterPro2GO（无置信度公式）
  4. Union    : NN-top3 ∪ InterPro（Blast2GO 式双证据）
评估协议与主表完全一致（传播/根剔除/四指标/IC来源），输出 eval_baseline_*.json。
"""
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import src.pipeline  # noqa: F401  解循环导入
from src.channels.base import read_tsv
from src.evaluator import Evaluator
from src.go_dag import GODag
import yaml

TRUTH = {"split3_test": "split3_go.tsv", "split2_val": "split2_go.tsv"}


def build(seq_rows, struct_rows):
    """4 个转移基线的 (蛋白 -> {GO: 分数})；分数分别取序列一致性与域覆盖率。

    只做通道产物到预测字典的构造，不依赖任何评估配置，修订版脚本因此直接复用
    这一实现取验证窗基线，避免两处定义漂移。
    """
    hits = defaultdict(lambda: {"ident": 0.0, "gos": set()})
    for r in seq_rows:
        key = (r["protein_id"], r["hit_acc"])
        h = hits[key]
        ident = float(r.get("identity") or 0)
        if ident > h["ident"]:
            h["ident"] = ident
        h["gos"].add(r["go_id"])
    best_per_prot = defaultdict(list)
    for (pid, _acc), h in hits.items():
        best_per_prot[pid].append((h["ident"], _acc, h["gos"]))

    def nn_pred(topn):
        out = {}
        for pid, lst in best_per_prot.items():
            lst.sort(key=lambda x: -x[0])
            d = {}
            for ident, _acc, gos in lst[:topn]:
                for g in gos:
                    d[g] = max(d.get(g, 0.0), ident)   # 分数=序列一致性
            out[pid] = d
        return out

    ip_pred = defaultdict(dict)
    for r in struct_rows:
        pid, g = r["protein_id"], r["go_id"]
        cov = float(r.get("coverage") or 0)
        ip_pred[pid][g] = max(ip_pred[pid].get(g, 0.0), cov)

    nn3 = nn_pred(3)
    union = {pid: dict(ip_pred.get(pid, {})) for pid in
             set(nn3) | set(ip_pred)}
    for pid, d in nn3.items():
        for g, s in d.items():
            union[pid][g] = max(union[pid].get(g, 0.0), s)

    return {"nn_top1": nn_pred(1), "nn_top3": nn3,
            "interpro": dict(ip_pred), "union": union}


def main():
    cfg = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    work = BASE / "work"
    data = BASE / "data"
    res = BASE / "results"
    tag = sys.argv[1] if len(sys.argv) > 1 else "split3_test"

    # ---- 读 seq/struct 中间产物 ----
    seq_rows = list(read_tsv(work / f"seq_channel_{tag}.tsv"))
    struct_rows = list(read_tsv(work / f"struct_channel_{tag}.tsv"))
    baselines = build(seq_rows, struct_rows)

    # ---- 写 TSV + 评估（与主表同协议）----
    dag = GODag(cfg["paths"]["go_obo"],
                 relations=cfg.get("go_dag", {}).get("relations"))
    ev = Evaluator(cfg, dag)
    truth = str(data / "annotations" / TRUTH[tag])
    train = str(data / "annotations" / "split1_go.tsv")

    report = {}
    (res / "legacy_eval").mkdir(exist_ok=True)
    jpaths = {}
    for name, preds in baselines.items():
        out_tsv = work / f"baseline_{name}_{tag}.tsv"
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("protein_id\tgo_id\tscore\n")
            for p, d in preds.items():
                for g, s in d.items():
                    f.write(f"{p}\t{g}\t{s:.6f}\n")
        # 正文只引用测试窗基线；其余窗口的旧协议评估落在 legacy_eval/，避免与
        # results/revision/ 的修订版产物混在同一目录
        out_json = (res / f"eval_baseline_{name}_{tag}.json"
                    if tag == "split3_test"
                    else res / "legacy_eval" / f"eval_baseline_{name}_{tag}.json")
        jpaths[name] = out_json
        r = ev.evaluate(str(out_tsv), truth, f"baseline-{name}",
                        train_tsv=train, out_json=str(out_json))
        report[name] = {k: r[k] for k in ("fmax", "smin", "aupr", "auc")}
        print(name, "Fmax=%.4f AUPR=%.4f Smin=%.3f AUC=%.4f"
              % (r["fmax"], r["aupr"], r["smin"], r["auc"]), flush=True)

    # 分本体（Fmax/AUPR）摘要
    print("\n=== 分本体 Fmax / AUPR ===")
    for name in baselines:
        j = __import__("json").loads(jpaths[name].read_text())
        row = []
        for asp in ("F", "P", "C"):
            a = j["by_aspect"][asp]
            row.append(f"{a['ontology']} {a['fmax']:.4f}/{a['aupr']:.4f}")
        print(f"{name:10s}", " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
