#!/usr/bin/env python3
"""把 make_supplementary_tables.py 渲染的严格口径表格合并进补充材料正文。"""
import re
from pathlib import Path

SRC = Path("/home/zsl/project/results/revision/supp_tables.md")
DST = Path("/home/zsl/Materials/补充材料/supplementary.md")
HEAD = re.compile(r"(?m)^## .*")
KEY = re.compile(r"^## 补充(?:表|说明) (S\d+)\b")


def blocks(text):
    """返回 [(key|None, start, end)]，覆盖全部 ## 章节；补充图等无语义编号记为 None。"""
    hits = list(HEAD.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        k = KEY.match(m.group(0))
        out.append((k.group(1) if k else None, m.start(), end))
    return out


new = SRC.read_text(encoding="utf-8")
nb = {k: new[s:e] for k, s, e in blocks(new) if k}
assert set(nb) == {"S3", "S4"} | {f"S{i}" for i in range(17, 25)}, sorted(nb)

old = DST.read_text(encoding="utf-8")
have = {k for k, _, _ in blocks(old) if k}
missing = sorted([k for k in nb if k not in have], key=lambda x: int(x[1:]))

for k, s, e in sorted(blocks(old), key=lambda t: t[1], reverse=True):
    if k in nb:
        old = old[:s] + nb[k] + old[e:]

if missing:
    anchor = [e for k, _, e in blocks(old) if k == "S23"]
    assert anchor, "找不到 S23 作为插入锚点"
    tail = old[anchor[0]:]
    m = re.search(r"(?m)^## 补充图", tail)
    assert m, "找不到尾部 补充图 章节"
    old = old[: anchor[0]] + "".join(nb[k] for k in missing) + tail

old = re.sub(r"([^\n])\n(## )", r"\1\n\n\2", old)   # 渲染块末尾不留空行，这里补齐
DST.write_text(old, encoding="utf-8")
print(f"已更新 {DST}：{len(old.splitlines())} 行，替换 {len(nb) - len(missing)} 块，新增 {missing}")
