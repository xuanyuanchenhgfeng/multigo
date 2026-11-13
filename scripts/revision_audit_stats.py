# -*- coding: utf-8 -*-
"""补充表 S6：从仲裁审计 JSONL 汇总各配置的拓扑冲突计数。

审计记录逐蛋白给出 same_merges、parent_child_collapses、branch_conflicts
（每条为 [术语A, 术语B, "branch"|"cross_namespace"]）与 dag_validation_removed，
本脚本只做计数还原，不重新仲裁。非法术语 = 被剔除且不带 redundant_parent 标记的条目。
分支冲突的分命名空间拆分按 arbiter._arbitrate 的口径复现（取编号较小术语的 aspect）。
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import src.pipeline  # noqa: F401,E402
from src.config import load_config  # noqa: E402
from src.go_dag import GODag  # noqa: E402

PRED = BASE / "work" / "revision" / "preds"
OUT = BASE / "results" / "revision" / "audit_stats.json"

CONFIGS = {
    "测试集·三通道融合": PRED / "audit_split3_test_multi.jsonl",
    "测试集·seq+struct": PRED / "audit_split3_test_seq+struct.jsonl",
    "测试集·seq+lit": PRED / "audit_split3_test_seq+lit.jsonl",
    "测试集·seq-only": PRED / "audit_split3_test_seq-only.jsonl",
    "验证集·三通道融合": PRED / "audit_split2_val_multi.jsonl",
    "拟南芥·三通道融合": PRED / "audit_arabidopsis_multi.jsonl",
    "测试集·fusion17（17 库）": PRED / "lib17" / "audit_fusion17.jsonl",
    "真跨物种·三通道融合": BASE / "work" / "revision" / "cross_species" / "audit_multi.jsonl",
}


def tally(path, dag=None):
    st = {"proteins": 0, "with_output": 0, "unknown": 0, "same_merges": 0,
          "parent_child_collapses": 0, "branch_conflicts": 0,
          "namespace_conflicts": 0, "illegal_terms": 0}
    by_aspect = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            st["proteins"] += 1
            if r.get("unknown"):
                st["unknown"] += 1
            elif r.get("final_terms"):
                st["with_output"] += 1
            st["same_merges"] += r.get("same_merges", 0)
            st["parent_child_collapses"] += len(r.get("parent_child_collapses", {}))
            for a, _b, kind in r.get("branch_conflicts", []):
                if kind == "branch":
                    st["branch_conflicts"] += 1
                    if dag is not None:
                        asp = dag.aspect(a) or "?"
                        by_aspect[asp] = by_aspect.get(asp, 0) + 1
                else:
                    st["namespace_conflicts"] += 1
            for x in r.get("dag_validation_removed", []):
                if "redundant_parent" not in x:
                    st["illegal_terms"] += 1
    if dag is not None:
        st["branch_conflict_by_aspect"] = by_aspect
    return st


def main():
    cfg = load_config(str(BASE / "config_revision.yaml"))
    dag = GODag(cfg["paths"]["go_obo"],
                relations=cfg.get("go_dag", {}).get("relations"))
    out = {}
    for name, path in CONFIGS.items():
        out[name] = tally(path, dag if "三通道融合" in name else None)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, st in out.items():
        print(f"{name:22s} 汇入{st['proteins']:5d} 有输出{st['with_output']:5d} "
              f"未知{st['unknown']:5d} 合并{st['same_merges']:6d} "
              f"折叠{st['parent_child_collapses']:6d} 分支{st['branch_conflicts']:7d} "
              f"跨命名空间{st['namespace_conflicts']:7d} 非法{st['illegal_terms']:2d} "
              f"{st.get('branch_conflict_by_aspect', '')}")


if __name__ == "__main__":
    main()
