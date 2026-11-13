# 基于时间划分与可追溯证据仲裁的蛋白质功能自动注释方法 —— 工程目录

纯工具链的蛋白质 GO 功能注释管线。三条证据通道（序列同源、结构域签名、文献共现）
并行收集候选，经显式规则仲裁层汇合并做 DAG 合法性校验，输出 GAF 2.2 与逐蛋白审计日志。
仲裁层的打分有两种模式，决策规则完全相同：默认是固定权重 0.40/0.35/0.25 的规则路线
（正文主表与本文以下数字的口径），另可挂一个候选级学习式打分器（分命名空间 logistic
回归，只在验证窗拟合、系数冻结后应用测试窗，见 §1 与 §8.2）作对照。评估把知识截止
对齐到本体快照、文献日期与指标统计三个层面，用按时间的三段划分代替随机划分。

论文、补充材料与交付产物在 `../Materials/`：正文 `论文/manuscript.md`（及 `manuscript.docx`）、
补充表 `补充材料/supplementary.md`、数据交付包 `数据/`、代码快照 `代码/`。本目录是产生这些
数字的工程现场，两者的协议参数必须一致；本文以下所有数字取自 `results/revision/*.json`。
`../Materials/` 这个相对路径只在作者的本地工程目录里成立：克隆本仓库时不会带上稿件与补充材料，
它们随投稿系统提交；仓库内的代码、配置与 `results/` 指标 JSON 已足以重算并机检正文的每个数字
（`python3 scripts/check_manuscript_numbers.py`，只读核对，全部一致时退出码 0）。
原始下载物（UniProt、GO、GOA、TAIR、NCBI 与 CNCB-NGDC GWH 的 Pinku1 蛋白组）、比对库索引、
全量注释表与逐蛋白审计流不入版本库，来源与版本逐项见补充表 S1，重建命令见 §8 与补充说明 S7。
本仓库以 MIT 许可发布，见 `LICENSE`。

---

## 1. 系统架构

```
                          查询蛋白序列 (FASTA)
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  ┌────────────┐         ┌────────────┐          ┌────────────┐
  │  序列通道  │         │ 结构域通道 │          │  文献通道  │
  │  DIAMOND   │         │InterProScan│          │ PubMed API │
  │ 同源转移   │         │ InterPro2GO│          │ 术语共现   │
  │ E 值归一   │         │ 匹配覆盖率 │          │ 共现篇数   │
  └─────┬──────┘         └─────┬──────┘          └─────┬──────┘
        │ 候选GO+置信度          │                      │
        └──────────────┬─────────┴──────────────────────┘
                       ▼
              ┌─────────────────┐     GO-DAG (is_a + part_of)
              │     Arbiter     │◄────────────────────────────┐
              │ 同术语合并      │  权重 0.40/0.35/0.25         │
              │ 祖-父折叠       │  最低总分 0.15               │
              │ 分支竞争记录    │  每蛋白上限 15 术语          │
              │ 功能未知判定    │  三通道均 <0.20 判未知       │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  DAG 合法性校验  │  冗余父节点删除 / 快照外术语过滤
              └────────┬────────┘
                       ▼
        GAF 2.2 + 简化TSV + 逐蛋白审计JSONL + 冲突统计
                       ▼
              ┌─────────────────┐
              │    Evaluator    │  Fmax / Smin / AUPR / AUC（101 点网格，蛋白中心微平均）
              └─────────────────┘  祖先传播后阈值化；Smin 的信息内容只取训练库真值
```

归一化与评分：

```
score = 0.40 * seq_conf + 0.35 * struct_conf + 0.25 * lit_conf

  seq_conf    = min(-log10(E-value), 10) / 10        E ≤ 1e-10 即满置信
  struct_conf = 签名匹配覆盖率（丢弃 <0.20）
  lit_conf    = 共现文献数的对数归一化（每蛋白最多 20 篇摘要）
```

冲突处理规则：

| 检测结果（GO-DAG 拓扑） | 处理 |
|---|---|
| 相同 GO（多通道给出） | 合并证据，取各通道置信度 |
| 父子关系 | 层级冗余，折叠到子节点 |
| 同分支不同节点 | 分支竞争，进入加权评分排序并计数 |
| 跨命名空间（F/P/C） | 不视为冲突，各命名空间分别保留（只计数不取舍） |
| 快照外/非法术语 | 过滤并计数 |
| 三通道均低置信 | 标记"功能未知"，不强行预测 |

两种仲裁打分模式：上面的公式是**规则路线**（正文主表口径）。`src/arbiter.py` 另留了一个
可插拔钩子 `Arbiter.scorer`，缺省为 `None`（走上式的固定权重求和）；挂上
签名为 `(ev, go) -> float` 的回调后，它取代 `score()` 里的加权求和与可选的分本体权重，
而候选合并、祖-父折叠、分支冲突记录、`min_score`、每蛋白 15 术语上限、DAG 合法性校验与
功能未知判定一条都不改，两条路线因此共用同一套决策语义、可在同一批蛋白上配对比较。
学习式打分器由 `src/learning.py` 提供（§8.2）：11 个候选级特征上的分命名空间 L2 正则
logistic 回归，只在验证窗候选上拟合、按蛋白分组 5 折选 L2 强度，系数冻结后应用测试窗。
`config_revision.yaml` 里没有学习模式的开关，钩子只由脚本注入（`arb.scorer = model`）。

---

## 2. 目录结构

```
project/
├── 数据集/                        # 原始数据（已下载，勿改）
│   ├── go-basic-2023-10-09.obo    # 正文固定的本体快照，47 637 术语，MD5 b6691e54…
│   ├── go-basic.obo               # 最新版，被 scripts/enrichment.py 与 scripts/baseline_transfer.py
│                                  #   自带的旧口径评估使用（见 §8、§9）
│   ├── uniprot_sprot.fasta.gz     # Swiss-Prot 序列
│   ├── uniprot_sprot.dat.gz       # Swiss-Prot 完整注释（基因名/物种/文献）
│   ├── filtered_goa_uniprot_all_noiea.gaf.gz  # GOA 注释，含逐条注释日期
│   ├── goa_arabidopsis.gaf.gz     # 拟南芥注释
│   └── GWHBJBL00000000.Protein.faa.gz          # 苦荞蛋白组 34 045 条（Pinku1 装配，CNCB-NGDC GWH，下载 2026-07-17）
├── data/
│   ├── partitions/                # split1/2/3、library.fasta、拟南芥库侧与验证集 FASTA
│   ├── annotations/               # split1_go.tsv（库 GO 映射）与各集合真值 TSV
│   ├── meta/                      # Swiss-Prot 元数据（基因名/物种/PMID）
│   ├── diamond_db/                # split1_revision / library_revision / sprot_library(旧)
│   └── mmseqs_db/                 # MMseqs2 库（DIAMOND 不可用时的降级路径）
├── src/
│   ├── config.py                  # YAML 配置加载与日志
│   ├── data_prep.py               # 解压、解析、时间划分、检索库与 DIAMOND 建库
│   ├── go_dag.py                  # DAG 解析（祖先/传播/命名空间/合法性）
│   ├── arbiter.py                 # 显式规则仲裁层（含逐蛋白审计日志、可插拔 scorer 钩子）
│   ├── learning.py                # 学习式仲裁打分器（分本体 logistic，numpy 阻尼 IRLS）
│   ├── evaluator.py               # 传播后阈值化与四指标
│   ├── pipeline.py                # 通道并行调度、目标集流程、GAF 输出
│   └── channels/
│       ├── base.py                # TSV/FASTA 读写
│       ├── seq_channel.py         # DIAMOND --sensitive；use_library() 按集合切换检索库
│       ├── struct_channel.py      # InterProScan（skip/ssh/precomputed/local 四种模式）
│       └── lit_channel.py         # PubMed 受控词典共现，限速 + SQLite 缓存；第 4 列 pmids
│                                  #   写真实 PMID（2026-11-11 修正，见 §8.2）
├── scripts/
│   ├── run_train_val.py           # 阶段1：三目标集全流程
│   ├── run_tartary.py             # 阶段2：苦荞全蛋白组（旧配置，见 §9）
│   ├── revision_*.py              # 修订版协议的分析脚本（见 §8；第二轮的 revision_learning /
│                                  #   revision_power / revision_operating_point 见 §8.2）
│   ├── baseline_transfer.py       # 同协议转移基线 NN-top1/NN-top3/InterPro/Union（见 §8）
│   ├── struct_library_audit.py    # 结构通道签名库时间审计（见 §8.2）
│   ├── lit_source_spotcheck.py    # 文献通道证据链复原与抽检（见 §8.2）
│   └── seq_channel_dbcheck.py     # 检索库对照校验（见 §7）
├── results/revision/              # 指标 JSON：正文与补充材料引用的那批，加 §8.2 第二轮新增的
│                                  #   8 份（arbiter_model / learn_eval / bootstrap_learn /
│                                  #   source_precision / struct_library_audit / lit_spotcheck /
│                                  #   power_analysis / operating_point）；另有一份被
│                                  #   audit_stats.json 取代的旧冲突统计，已改名留档
├── work/                          # 通道中间产物、仲裁预测 TSV、审计日志
├── cache/pubmed_cache.sqlite      # PubMed 检索与摘要缓存，557 MB
└── config_revision.yaml           # 修订版协议配置（正文数字全部出自这套配置）
    config.yaml                    # 修订前配置，只用于苦荞演示与旧脚本
```

---

## 3. 环境

| 项 | 版本 | 说明 |
|---|---|---|
| 系统 | Linux x86_64，内核 6.8.0-124-generic | |
| Python | 3.12.3 | 依赖 PyYAML、requests；§8.2 的第二轮脚本另需 numpy（学习式仲裁与重采样用，求解器为 numpy 自带的线性代数，不引入 sklearn） |
| DIAMOND | 2.1.10 | `/home/zsl/tools/diamond/diamond` |
| InterProScan | 5.72-103.0 | 本机 `local` 模式部署 |
| Java | OpenJDK 11.0.2 | 每进程 12 GB 堆、4 线程 |
| 网络 | 可选 | 仅文献通道冷启动需要；缓存命中后不访问 |

所有 Python 调用统一用 `env -u LD_LIBRARY_PATH python3 -X utf8 …`：InterProScan 与
DIAMOND 会设置 `LD_LIBRARY_PATH`，不清掉会让 Python 的 SQLite 与 zlib 加载到不兼容的
so；不加 `-X utf8` 时中文注释与日志在非 UTF-8 locale 下会乱码。

---

## 4. 快速开始

```bash
cd <工程目录>
# 1) 时间划分、真值切分与检索库构建
env -u LD_LIBRARY_PATH python3 -X utf8 scripts/run_train_val.py --config config_revision.yaml
# 2) 三通道 + 仲裁 + GAF（split2_val / split3_test / arabidopsis）
env -u LD_LIBRARY_PATH python3 -X utf8 scripts/run_train_val.py --config config_revision.yaml --skip-prep
```

之后是分析与评估脚本，完整顺序（14 步）与每步依赖见补充说明 S7。最短的复现链路是
步骤 1–2 之后依次跑 `revision_evaluate.py`、`revision_bootstrap.py`、`baseline_transfer.py`、
`revision_17lib.py`。步骤 1–2 不访问任何公开数据库；文献通道只在缓存未命中时才发请求。
学习式仲裁路线（§8.2）在这四步之后跑 `scripts/revision_learning.py`，第二轮的其余四项分析
或直接读它写出的预测文件，或复用它的指标与工作点函数。

---

## 5. 数据划分与协议

时间边界 T0 = 2023-11-08（知识截止，对应 Swiss-Prot v2023_05）、T1 = 2024-11-27、
T2 = 2025-06-18；库侧注释一律 ≤T0，真值按各自窗口断开。证据码只保留 13 个白名单码
（EXP、IDA、IPI、IMP、IGI、IEP、TAS、IC、HTP、HDA、HMP、HGI、HEP）。

| 集合 | 定义 | 序列数 / 查询数 | 可评估蛋白 | 真值窗口 |
|---|---|---:|---:|---|
| split1（训练库） | 首次实验性注释 ≤T0 | 16 948 | — | ≤2023-11-08 |
| split2_val | 首次注释 ∈(T0, T1] | 3 122 | 3 104 | ≤2024-11-27 |
| split3_test | 首次注释 ∈(T1, T2] | 1 159 | 1 130 | ≤2025-06-18 |
| 拟南芥同物种留出 | 70/30 蛋白级 MD5 哈希（盐 `go-agent-split-v1`），库侧 8 505 条并入训练库 | 3 607 | 3 527 | ≤2024-11-27 |
| 真跨物种压力测试 | 检索库剔除全部拟南芥（16 948 条） | 3 607 | 3 527 | ≤2024-11-27 |
| 苦荞 | 无金标准，仅交付形态演示 | 34 045 | 不适用 | 不适用 |

拟南芥全池 12 112 条中 8 505 条在库侧、3 607 条作查询，两者按蛋白 ID 哈希互斥，
查询序列绝不出现在自己的比对库里。

评估口径：预测与真值都先沿 `is_a`/`part_of` 传播再阈值化（先传播后阈值），剔除三个
子本体根术语；Fmax/Smin/AUPR 为蛋白中心微平均，阈值网格 0.00–1.00 步长 0.01 共 101 点；
AUC 为类中心平均（每个至少一正一负的 GO 类做 Mann-Whitney ROC AUC 后取均值）；
Smin 的信息内容只由训练库真值统计（`ic_source: train_only`），评估集真值不参与 IC 计算。
配对 bootstrap 以蛋白为重采样单位，B = 2 000，种子 20261104，8 个方法同批配对。

第二轮分析另用一种可部署口径：阈值由验证窗 F1 曲线的 1-SE 规则选出（在与峰值相差不足
一个重采样标准误的网格点里取最高、即最保守的一档，验证窗重采样 200 次），冻结后应用
测试窗报 P/R/F1，全程不使用测试窗信息选阈值；`revision_learning.py` 在同一批蛋白上把含
学习式路线的 10 条路线做配对重采样（B 与种子同上，另存 `bootstrap_learn.json`）。

---

## 6. 主要结果

测试集整体口径（n = 1 130，`results/revision/eval_summary.json` 与 `bootstrap_ci.json`）：

| 方法 | Fmax | Smin | AUPR | AUC |
|---|---:|---:|---:|---:|
| 三通道融合 | 0.4321 | 7.5926 | 0.2947 | 0.6581 |
| seq+struct | 0.4240 | 7.5889 | 0.2793 | 0.6550 |
| seq+lit | 0.3490 | 8.1797 | 0.1920 | 0.6359 |
| seq-only | 0.3497 | 8.6464 | 0.0772 | 0.6339 |
| Union 朴素并集 | 0.4416 | 8.1438 | 0.2485 | 0.6496 |
| InterPro 直接映射 | 0.4243 | 7.8724 | 0.1790 | 0.5741 |
| NN-top1 / NN-top3 | 0.4161 / 0.3712 | 8.3326 / 8.5118 | 0.2105 / 0.2190 | 0.5791 / 0.6233 |

融合模型分本体 Fmax：MFO 0.5416、BPO 0.3091、CCO 0.7362。验证集整体 Fmax 0.3354。
配对差值（融合 − 对照，95% CI）：相对 seq-only 为 +0.0824（0.0568, 0.1087），相对
seq+struct 为 +0.0088（0.0035, 0.0152），相对 InterPro 为 +0.0084（−0.0063, +0.0223）跨零，
相对 Union 为 −0.0094（−0.0290, +0.0110）跨零。三者在 Fmax 上统计不可区分，融合模型的
显著优势在 AUPR（对 InterPro +0.1154、对 Union +0.0459）与 Smin（−0.28、−0.55）。
完整区间与逐本体、逐设置的数字见正文表 2–8 与补充表 S3–S6、S9。

上表全部是规则路线（固定权重 0.40/0.35/0.25）的口径。学习式路线在同一决策语义下的
对照——含候选级 logistic 仲裁、Vote-≥2 规则基线、共同覆盖子集重算与分证据来源精度——
由 `scripts/revision_learning.py` 产出，落在 `results/revision/learn_eval.json` 与
`results/revision/bootstrap_learn.json`，见 §8.2。

一句话结论：结构签名贡献 Fmax 的绝大部分增益，文献证据只在序列与结构两路都已给出
候选时才改善点值（17 库配置下这一增量缩到 +0.003 且不再显著），仲裁层的价值体现在
排序质量与语义距离，而不是 Fmax 本身。

---

## 7. 检索库与 2026-11-10 的索引修正

序列通道按 E 值取每条查询的前 5 个命中做 GO 转移，因此**用哪个检索库会直接改变结果**：
库外序列（拟南芥、或注释晚于知识截止的条目）本身没有可用 GO，却会挤占前 5 个名额。
2026-09-10 建的 `sprot_library.dmnd` 含 36 941 条序列，早于 09-17 的检索库重切，主流程
一度沿用了它。修正后每个查询集用哪个库由 `config_revision.yaml` 写死：

```yaml
seq_channel:
  db_by_set:
    split2_val:  ["split1.fasta",  "split1_revision"]   # 16 948 条
    split3_test: ["split1.fasta",  "split1_revision"]
    arabidopsis: ["library.fasta", "library_revision"]  # 25 453 条
    tartary:     ["library.fasta", "library_revision"]
```

`SeqChannel.use_library(set_tag)` 在 `Pipeline._run_channels_parallel()` 与苦荞流程里于每个
集合开跑前调用。`scripts/seq_channel_dbcheck.py` 用正文规定的库把各集合重跑一遍并与在档
产物逐行对照，`--apply` 只回写受影响集合（旧产物备份在 `work/backup_staledb_20261110/`）。
按当前配置重放 split3 与拟南芥的序列通道，产物与交付文件逐条相同（4 319 行 / 821 个蛋白、
29 051 行 / 2 817 个蛋白）。

该修正引起的数字位移：测试集序列命中 797 → 821 条，拟南芥 2 751 → 2 817 条，融合模型
测试集 Fmax 0.4130 → 0.4321，并有三处结论改写（详见 `../Materials/论文/修订说明.md` §11）。
苦荞产物保留修正前口径，量化对照见 §9 与 `results/revision/tartary_dbcheck.json`。

---

## 8. 脚本分类

### 8.1 第一轮：正文与补充材料的口径来源

`config_revision.yaml` + `revision_*` 系列是正文与补充材料的口径来源：

| 脚本 | 产物 | 支撑 |
|---|---|---|
| `revision_evaluate.py` | `eval_summary.json`、`calibration.json`、`coverage_stats.json`、`exp6_sensitivity.json` | 表 2–5、7 |
| `revision_bootstrap.py` | `bootstrap_ci.json` | 表 3、补充表 S3、S4 |
| `baseline_transfer.py` | `work/baseline_<arm>_<tag>.tsv` 四份同协议转移基线预测；`results/eval_baseline_<arm>_split3_test.json`（其余窗口落 `results/legacy_eval/`） | 供 `revision_evaluate.py`、`revision_bootstrap.py`、`revision_learning.py`、`revision_power.py` 读取 |
| `revision_17lib.py` | `lib17_sensitivity.json`、`lib17_bootstrap.json` | 表 6、补充表 S9 |
| `revision_weights.py` | `weight_search_val.json`、`eval_peront_weights.json` | 表 7、补充表 S11 |
| `bpo_analysis.py` | `bpo_analysis.json` | 表 5、图 5 |
| `revision_arabidopsis_truth.py`、`revision_cross_species.py` | `truth_stats.json`、`eval_cross_species.json` | 表 1、表 8、补充表 S5 |
| `revision_arbiter_sensitivity.py` | `arbiter_sensitivity.json` | 补充表 S11 |
| `lit_query_fallback_stats.py` | `lit_query_fallback.json` | 补充表 S8（只读缓存，不访问网络） |
| `revision_table4_stats.py`、`revision_audit_stats.py` | `table4_stats.json`、`audit_stats.json` | 表 4、补充表 S6（只读产物，不重新仲裁） |
| `revision_truth_stats.py` | `truth_stats.json` | 表 1、图 2（从三个真值 TSV 去重后重算） |

`weight_search.py`、`apply_weights.py`、`bootstrap_ci.py`、`run_full.py` 与 `enrichment.py`
读取旧配置 `config.yaml`，属修订前协议，其数字不进正文。`revision_lit_rerun.py` 与
`fill_struct_gaps.py` 生成通道产物（`*_dcut.tsv` 与结构域缺口补跑文件）。

`baseline_transfer.py` 是四条同协议转移基线在仓库里的唯一实现：`build()` 返回
`nn_top1` / `nn_top3` / `interpro` / `union` 四个预测字典（最近邻 Top1、Top-3 命中注释并集、
InterPro2GO 直接映射、两者朴素并集；分数分别取序列一致性与签名覆盖率），窗口标签由
命令行参数给（缺省 `split3_test`），`build()` 本身不读任何评估配置，因此验证窗与测试窗
共用同一份定义。`revision_evaluate.py`、`revision_bootstrap.py`、`revision_learning.py` 与
`revision_power.py` 取的都是它写出的 `work/baseline_<arm>_<tag>.tsv`。只有它自带的
`eval_baseline_<arm>_<tag>.json` 是旧口径（按 `config.yaml` 的 `paths.go_obo`，即最新版
`go-basic.obo` 评估，不是正文固定的 2023-10-09 快照），不进正文；非测试窗口的这份 JSON
自 2026-11-11 起改落 `results/legacy_eval/`，避免与修订版产物混目录。

### 8.2 第二轮（2026-11-11）：学习式仲裁与四项审稿补充分析

模块 `src/learning.py`（被上述脚本 import，不单独运行）实现仲裁层的学习式打分器：
`FEATURES` 是 11 个候选级特征（三个通道置信度、证据来源数、两两乘积三项、术语 IC、
训练库直接注释数取对数、本体深度、是否由祖-父折叠而来），特征只允许取训练库（≤T0）
统计的量与该候选自身的通道置信度，不碰任何评估窗真值；`AspectLogit` 是 F/P/C 各一个的
L2 正则 logistic 回归，求解器为 numpy 写的带回溯阻尼 Newton/IRLS（不引入 sklearn），
`ArbitrationModel.fit()` 用按蛋白分组的 5 折（`group_kfold_keys`，种子 20261104）在
`LAMBDAS` 六个候选值上选出折间对数损失最小的正则强度，再在全验证样本上重拟合；
`to_dict()`/`save()` 把系数、截距、标准化用的均值与标准差、逐 λ 的折间损失和出折 ECE
全部落盘，可逐条复算。它通过 §1 说的 `Arbiter.scorer` 钩子进入仲裁层，不改任何决策规则。

五条命令都在工程根目录执行、都读 `config_revision.yaml`（脚本内写死路径，不接受配置
参数），前缀统一为 `env -u LD_LIBRARY_PATH python3 -X utf8`（§3，下表命令列里的 `…` 即此
前缀）。三份脚本的重采样次数
可用环境变量 `MULTIGO_BOOTSTRAP` 覆盖以做冒烟测试（`MULTIGO_BOOTSTRAP=20`）：
`revision_learning.py` 缺省取 `revision_bootstrap.B`（2000），`revision_power.py` 与
`revision_operating_point.py` 缺省 2000。`struct_library_audit.py` 与
`lit_source_spotcheck.py` 不读该变量。

依赖顺序：`revision_learning.py` 需要 `revision_evaluate.py` 写出的 `predictions_<窗口>_multi.tsv`
与各消融预测，以及 `baseline_transfer.py` 在两个窗口下写出的 `work/baseline_<arm>_<tag>.tsv`；
`revision_power.py`（读 `predictions_<窗口>_{multi,learn,vote2}.tsv` 与
`work/baseline_union_<窗口>.tsv`，并复用 `bpo_analysis.py` 的 `depth_of`/`precompute`/`pr_f1`）
与 `lit_source_spotcheck.py`（读 `predictions_split3_test_learn.tsv` 与
`predictions_split3_test_multi.tsv`）排在它之后；`revision_operating_point.py` 与
`struct_library_audit.py` 只 import `revision_learning` 的通道路径与指标函数、自行重跑仲裁。

| 脚本 | 回答什么 | 命令 | 产物 |
|---|---|---|---|
| `revision_learning.py` | 把固定权重换成在验证窗学出的权重，同一套决策规则下指标是否更好、差值区间是否跨零 | `… scripts/revision_learning.py` | `results/revision/arbiter_model.json`、`learn_eval.json`、`bootstrap_learn.json`、`source_precision.json` |
| `struct_library_audit.py` | 签名库快照晚于知识截止这一件事值多少指标，扣除后还剩多少 | `… scripts/struct_library_audit.py` | `results/revision/struct_library_audit.json`（中间预测与审计在 `work/revision/preds/_struct_audit/`） |
| `lit_source_spotcheck.py` | 文献共现通道的候选有多少经得起逐条回读 | `… scripts/lit_source_spotcheck.py` | `results/revision/lit_spotcheck.json`、抽样表 `work/revision/lit_spotcheck_sample.tsv` |
| `revision_power.py` | 测试窗与 BPO 这一层能分辨多大的 F1 差值 | `… scripts/revision_power.py` | `results/revision/power_analysis.json` |
| `revision_operating_point.py` | 分命名空间的概率门限与一根全局分数下限哪个更可部署 | `… scripts/revision_operating_point.py` | `results/revision/operating_point.json`（中间预测在 `work/revision/preds/_oppoint/`） |

逐条说明：

1. `revision_learning.py`：训练样本取自验证窗候选——用 `Arbiter.collect()`（同术语合并 +
   祖-父折叠后的候选池）逐 (蛋白, GO) 抽特征，标签是验证窗传播后真值；跑 `fusion`（固定
   权重对照）、`learn`（分本体 logistic）、`vote2`（规则基线：只有 ≥2 个通道同时给出该候选
   才保留，保留者的排序仍用融合分）三条路线，两个窗口各重跑一次仲裁。学习式路线的分数
   下限在 `GATES`（0.05/0.10/0.15/0.25/0.35）上按验证窗 Fmax 选（logistic 概率与加权分数量纲
   不同，沿用 0.15 会把候选砍得过稀）。评估侧另给：验证窗 F1 曲线 1-SE 规则选阈值、冻结
   应用测试窗的可部署 F1（`pick_threshold`，同时报 argmax 口径）、分命名空间 ECE 与可靠度
   分箱（含把固定权重分数当概率用的对照）、逐 GO 类 AUC 及其"发射规模 vs 排序质量"分解
   （`per_term_auc`/`auc_breakdown`，并只在 `learn` 与 `fusion` 都发射的类上重算匹配口径 AUC）、
   三条仲裁路线的共同覆盖子集重算（回答"AUPR 优势是否只来自覆盖蛋白数与每蛋白术语数"）、
   连同三条通道消融与四条转移基线共 10 个配置的同协议对比与配对区间、按证据来源组合
   与分本体的经验精度，以及用冻结系数外推到拟南芥 T1 严格窗的结果。配对区间为蛋白级
   重采样 B = 2000、种子 20261104，写 `bootstrap_learn.json`。预测与审计写
   `work/revision/preds/predictions_<窗口>_<路线>.tsv`；验证窗的通道消融预测只用于定阈值，
   落 `work/revision/preds/_val_ablation/`，不进交付包。
2. `struct_library_audit.py`：逐批次读 `work/struct_channel/<tag>/*.xml`，从文件本身抽出
   `interproscan-version` 与每个成员库的 `library`/`version` 属性（不靠记忆或引用），按
   signature 前缀与结构通道 TSV 对账（`norm_lib()` 统一 `PROSITE_PATTERNS` 与
   `ProSitePatterns` 两种写法），统计每库行数、蛋白数与"只有该库给出结构证据"的蛋白数；
   再与脚本内硬编码的 InterProScan 发布表 `IPS_RELEASES`（来源为官方 GitHub releases，
   检索日期 2026-11-11）对齐到 T0 = 2023-11-08，给出 T0 之前最后一次发布、其后的发布数与
   间隔天数。然后把结构通道限制成 (a) 只保留版本串自带日期且不晚于 T0 的库（`DATE_CODED`
   匹配 `2023_05` 一类 PROSITE/HAMAP/PIRSR 版本串）与 (b) 完全置空两种形态，规则路线与
   学习式路线各重跑一次（学习式的 λ 与分数下限都在受限后的验证窗重新标定，不沿用全量
   结构窗的参数），报告网格 Fmax/AUPR/Smin/AUC 与验证窗冻结阈的可部署 F1。全程不访问
   网络，产物 `results/revision/struct_library_audit.json`。
3. `lit_source_spotcheck.py`：只读打开 `cache/pubmed_cache.sqlite`（`mode=ro`），按
   `lit_channel.py` 的检索式优先级（基因名+物种 > 基因名 > 蛋白名，统一附加 T0 出版日期
   截断子句）复原每行文献候选当时用过的那批 PMID，再对每行算三条可机械判定的强度指标：
   GO 术语是否出现在标题（`title_hit`）、命中短语所在同一句里是否出现基因名/蛋白名词元
   （`gene_in_sentence`）、该 (蛋白, GO) 是否在传播后真值里成立（`truth_hit`，即与仲裁权重
   无关的通道独立精度），并按本体、引用数（≥1/≥2/≥3）、是否为唯一证据来源分层。抽样表
   用固定种子 20261104，规模 `N_SAMPLE = 60`（引用数 ≥3 的一档最多取 20 行，其余补足）。
   该脚本不复算检索、不访问网络。
   与它对应的列语义修正：交付的 `work/lit_channel_*_dcut.tsv` 第 4 列名为 `pmids`，旧版
   实际写的是 `match_gos()` 返回的题录下标（`"4;6"` 这种）；`src/channels/lit_channel.py`
   已在写出前把下标映射回取回过摘要的那批 PMID（每行至多 10 个），修复后该列名副其实。
   在档的通道产物（`work/` 下与 `../Materials/数据/修订版通道产物/`）仍是修复前生成的下标版，
   没有重跑文献通道，抽检脚本因此按下标复原。仲裁层的 `_load_lit()` 只用第 3 列 `citations`
   计分，`pmids` 仅进审计日志的 detail 串，因此这一修复不改变任何已发表指标，只恢复证据链
   的可回溯性。
4. `revision_power.py`：先按"直接注释 / 传播后注释"两套口径重算测试真值规模（每本体蛋白数、
   注释条数、每蛋白注释数分布），纠正被引用过的错误规模；再建 7 个层——overall、MFO、
   BPO、CCO 与 BPO 按深度的三分位（浅/中/深，深度定义与 `bpo_analysis.py` 一致，即术语到
   GO:0008150 的 `is_a`/`part_of` 最短步数）；每层用同一批蛋白做配对重采样（B = 2000，
   种子 20261104），报 `learn` 相对 `fusion`/`union`/`vote2` 与 `fusion` 相对 `union` 的冻结阈
   F1 差值 95% CI、CI 半宽、标准误，并按 `MDD_80 = 2.80 × SE(Δ配对)`（z_0.025 + z_0.20，
   双侧 5%、80% 功效）给出最小可分辨差值。阈值 t* 一律用全验证窗 + 1-SE 规则选定后冻结，
   分层只作用于测试窗，因此各层用的是同一个可部署工作点、层间可比；分层本身只描述规模，
   不参与任何调参。
5. `revision_operating_point.py`：固定权重与学习式两条路线各自在验证窗按命名空间扫同一组
   分数下限网格 `GRID`（0.005–0.35 共 8 档），用 1-SE 规则（重采样 200 次估峰值处标准误）
   选点、冻结后应用测试窗，与现行交付下限（`GLOBAL_FLOOR`：fusion 0.15、learn 0.05）比较，
   整体与分命名空间都给 P/R/F1/覆盖蛋白数，并在同一 gate 约定内做四条配对比较
   （B = 2000，种子 20261104）。口径声明随产物落盘：本脚本一律"筛选后传播"（gate 规则），
   主表是"传播后阈值"（t* 规则，见 `bootstrap_learn.json`），两套部署约定数值不可互比。

---

## 9. 已知局限

1. **苦荞案例的口径与正文不同**。那批产物生成于 2026-10-19/22：用修订前主配置、文献通道
   没有出版日期截断、富集检验用最新 `go-basic.obo` 与未经证据码过滤的背景，序列通道也
   仍在 36 941 条复合索引上。按现行检索库重放序列通道会把有序列证据的蛋白从 17 322 提到
   17 637、候选行从 162 861 提到 187 226；但苦荞文献通道的查询词取自最优同源命中的基因名，
   序列通道一变就要重建全部检索式并触发大规模新的 E-utilities 访问，因此保留原产物、
   把差异量化写进正文表 9 注与补充表 S12，不做半截重算。苦荞只作交付形态演示，不与任何
   协议指标并列。
2. **签名库版本与知识截止无法对齐**。InterProScan 5.72-103.0 自带的 InterPro2GO 与 PANTHER
   19.0 映射晚于 2023-10-09 快照，可能给出快照外的术语。修订版主配置的六个结构域产物
   （12 库与 17 库，5 627 / 1 712 / 5 419 / 2 528 行）逐条回读越界数为 0，但苦荞 13 库产物含
   GO:0141166 与 GO:0180047、旧协议产物含 GO:0180047 与 GO:0170057，都不在快照内、会被 DAG
   合法性校验剔除。未触发不等于已消除。2026-11-11 起这一条从"声明"改为"可量化结论"：
   `scripts/struct_library_audit.py` 逐批次回读结果 XML 里的实际软件与成员库版本串、与脚本内
   硬编码的 InterProScan 发布表对齐到 T0，再在"只保留版本串自带日期且不晚于 T0 的库"与
   "结构通道整通道置空"两种受限下重跑规则路线与学习式路线（§8.2），数字见
   `results/revision/struct_library_audit.json`。
3. **17 库敏感性不是同环境产物**。PANTHER、Gene3D/FunFam、ProSiteProfiles、Hamap 依赖
   epang、cath-funfam、pfsearchV3 等原生二进制，在主运行环境触发段错误，17 库数字来自
   2026-10-22 在完整环境跑通的文件经 `revision_17lib.py` 重放仲裁。
4. **外部报告值未本地复现**。正文引用的 PSB26 分本体基线（DiamondScore、DeepGOPlus、
   SPROF-GO、MLP+DiamondScore 等）转录自其 Table 2，2026-11-11 已对照出版版式逐项核对无误；
   不可比之处在于样本来源（该文按 Swiss-Prot 逐版本快照取 84 748 条已审阅蛋白，本文用注释
   日期近似边界）与指标实现，时间锚点与 13 码证据码白名单两边相同。交付产物中没有对应的
   可机读文件。
5. `rutin_pathway_check.tsv`（芦丁途径逐步同源核对，224 行配对）没有生成脚本，逻辑为本地
   blastp 输出加参考酶名称人工指定，属一次性分析产物；配套的 `enrichment_summary.txt` 是
   控制台留存文本，同样没有写出它的脚本（`enrichment.py` 只输出 `tartary_pathway_enrichment.tsv`）。
   2026-11-11 起两者不再作为交付依据：正文 3.9 节末段与补充表 S12 的核对部分已删除，文件移入
   `results/superseded_no_provenance/` 本地留档。
6. **学习式路线的边界**。按蛋白分组 5 折的折间对数损失在 F 与 P 上都把 L2 强度选到网格
   上界 λ = 10（C 选 1），`revision_learning.py` 随产物声明记录了后果：λ=10 把 BPO 的概率
   整体压到分数下限以下，验证窗 BPO 的 F1 曲线恒为 0，1-SE 规则在该本体退回网格上界；
   分本体合并用的又是全局微平均，与蛋白中心的 Fmax/F1 口径不同、数值系统性偏低。因此
   分命名空间工作点只作敏感性检查，部署口径是三本体共用一个全局阈值。还要区分两套部署
   约定：`revision_operating_point.py` 是"筛选后传播"（gate），主表与 `revision_power.py`
   是"传播后阈值"（t*），数值不可互比。学习式与规则两条路线的差值只在同一决策语义内
   解释，模型不使用测试窗信息。

---

## 10. 复现性声明

- 数据划分：MD5(蛋白 ID + 盐 `go-agent-split-v1`)，与运行环境无关。
- 序列通道：DIAMOND 2.1.10 `--sensitive`，固定 E 值/一致性/覆盖度阈值与 top-5，检索库由
  `seq_channel.db_by_set` 按集合锁定（§7）。
- 结构域通道：InterProScan 5.72-103.0，12 个签名库固定 `-appl` 列表，批次并行不改变结果
  （每批输入与命中判定相互独立）。
- 文献通道：检索式统一附加 `("0001-01-01"[Date - Publication] : "2023-11-08"[Date - Publication])`，
  首次检索结果固化于 `cache/pubmed_cache.sqlite`（102 725 次检索、247 180 篇摘要），之后重跑
  只读缓存；缓存外的新请求会命中 T0 之后的文献，由 `pub_date_cutoff` 截断。候选 TSV 的第 4 列
  `pmids` 自 2026-11-11 起写真实 PMID（此前写的是题录下标，见 §8.2），计分只用第 3 列
  `citations`，这一列的语义修正不改变任何指标。
- 仲裁层（规则路线）：纯规则、无随机性；权重 0.40/0.35/0.25 与 `min_score` 0.15 写在
  `config_revision.yaml`，评估网格与 bootstrap 种子固定（20261104）。
- 仲裁层（学习式路线，§8.2）：只在验证窗候选上拟合，随机性只来自固定种子 20261104 的
  按蛋白分组 5 折与 B = 2000 配对重采样；L2 强度由折间对数损失确定，系数、截距与标准化
  用的均值/标准差全量写进 `results/revision/arbiter_model.json`，可逐条手工复算；测试窗一次
  都不参与拟合。该文件只用于披露参数，脚本里没有从它回读的加载器，重跑即重新拟合。
  冒烟测试可用 `MULTIGO_BOOTSTRAP` 缩短重采样。
- 结构通道的时间纪律：签名库与 InterPro2GO 映射的实际版本以结果 XML 自带的版本串为准，
  由 `scripts/struct_library_audit.py` 机读清点，不靠人工登记。
- 唯一非确定项：GAF 输出的 `Date` 列写导出当天日期，不是证据日期。

注：`split_seed_salt` 的字面值 `go-agent-split-v1` 是历史盐值，保留是为了让已固化的划分与
评估结果可复现；该字面不代表系统含 LLM 智能体——本管线不含任何智能体，两条仲裁路线在
固定种子下都是确定性的工具链，改名会导致划分不可复现。
