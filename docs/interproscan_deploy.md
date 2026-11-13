# InterProScan 服务器部署与远程使用指南

> Struct-Channel（Channel 2）依赖 InterProScan 官方发行包，**官方仅支持 64 位 Linux**。
> 本机为 Windows，因此需要 Linux 服务器承载 InterProScan，本地流水线远程调用或离线解析。
> 本文档覆盖：服务器部署 → SSH 免密配置 → 两种接入方式（`ssh` 全自动 / `precomputed` 手动）→ 断点续跑 → 常见问题。

---

## 1. 服务器要求

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | 64位 Linux（Ubuntu 20.04+ / Debian 11+ / CentOS 8+） | glibc ≥ 2.28 |
| Java | OpenJDK 11（`openjdk-11-jre`） | InterProScan 5.x 要求 |
| Python | 3.x | 激活脚本需要 |
| 内存 | ≥ 16 GB | 12GB JVM 堆 + 系统/数据库缓存 |
| 磁盘 | ≥ 40 GB 空闲 | 发行包约 20GB，解压+临时文件翻倍 |
| 网络 | 能访问 EBI FTP 下载 | 仅安装时需要 |

## 2. 部署 InterProScan（服务器上执行）

### 2.1 安装依赖

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y openjdk-11-jre python3 wget

# CentOS / Rocky
sudo yum install -y java-11-openjdk python3 wget
```

### 2.2 下载并解压

```bash
cd /tmp
# 版本号固定（论文可复现性）；可到 https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/ 查最新
wget https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.72-103.0/interproscan.tar.gz

sudo mkdir -p /opt/interproscan
sudo tar -pxzf interproscan.tar.gz -C /opt/interproscan --strip-components=1
sudo chown -R $USER:$USER /opt/interproscan     # 或单独授权给运行用户
```

> `-p` 参数保留可执行权限，**不要省略**。

### 2.3 首次激活

```bash
cd /opt/interproscan
python3 interproscan.sh        # 生成 python 预编译文件、报告二进制可用性
```

激活日志末尾若有 `-- Strings are switched to single byte` 之外的 `FAILED` 行，
通常是缺少对应预编译器（如 `tmhmm`/`phobius` 需单独许可），不影响核心库（Pfam 等）。

### 2.4 验证

```bash
cd /opt/interproscan
./interproscan.sh -i test_proteins.fasta -goterms -cpu 4 -T /tmp/ipstest
# 成功标志: /tmp 下生成 test_proteins.fasta.tsv（默认输出到输入旁），含 GO:xxxxxxx 列
```

## 3. 配置 SSH 免密登录（本地 Windows 执行）

流水线 `ssh` 模式使用 `BatchMode=yes`（禁止交互输密码），**必须**配置公钥认证：

```powershell
# 1) 生成本机密钥（已有 ~/.ssh/id_rsa 可跳过）
ssh-keygen -t rsa -b 4096

# 2) 上传公钥到服务器（Windows 无 ssh-copy-id，用 type 管道）
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh user@服务器IP "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# 3) 验证免密（不应提示输密码）
ssh user@服务器IP "hostname && test -x /opt/interproscan/interproscan.sh && echo IPS_READY"
```

> 首次连接会询问指纹，输入 `yes`。之后流水线用的
> `StrictHostKeyChecking=accept-new` 会自动接受新主机指纹。
> 校园网/内网穿透（frp 等）场景注意保持端口稳定，config.yaml 中 `ssh.port` 可改。

## 4. 方式A：`ssh` 全自动模式（推荐）

流水线自动完成：本地按 500 条/批切分 → `scp` 上传 → 逐批 `ssh` 远程运行
（自动附加 `-Xmx12G -cpu 4 -goterms`）→ `scp` 拉回 TSV → 本地解析。

### 4.1 修改 config.yaml

```yaml
tools:
  interproscan_mode: "ssh"
  interproscan: "/opt/interproscan/interproscan.sh"   # 远程服务器上的路径

struct_channel:
  ssh:
    host: "user@192.168.1.100"   # 改成实际 用户@IP
    port: 22
    remote_dir: "~/ips_jobs"     # 远程工作目录（自动创建）
    ssh_key: ""                  # 留空=默认 ~/.ssh/id_rsa
    keep_remote: true            # 保留远程结果，支持断点续跑
```

### 4.2 重跑流水线

```powershell
# 先删除 skip 模式留下的空结果（否则流水线检测到已存在会跳过 Struct-Channel）
Remove-Item work\struct_channel_split2_train.tsv, work\struct_channel_split3_val.tsv, work\struct_channel_arabidopsis.tsv -ErrorAction SilentlyContinue

# seq/lit 中间文件保留不动，只补 struct + Arbiter + 评估
python scripts/run_train_val.py --skip-prep
# 或只跑某个目标集: python scripts/run_train_val.py --skip-prep --only arabidopsis
```

### 4.3 运行行为说明

- **断点续跑**：每个目标集独立目录（`work/struct_channel_ssh/<tag>/`）；远程已生成的
  `tsv/batch_XXXX.tsv` 自动跳过，中断后直接重跑同一命令即可续传。
- **资源约束**：远程命令显式携带 `_JAVA_OPTIONS=-Xmx12G` 与 `-cpu 4`，
  与论文约束一致；批大小 500 由 `struct_channel.batch_size` 控制。
- **子集库提速**：`struct_channel.applications: "Pfam,PROSITE,SMART"` 只跑核心库，
  速度提升约 10 倍（全应用库 4 线程下 500 条约需数小时；论文需注明所用应用集）。
- **失败降级**：SSH 不通/远程未部署时输出空结果并记 ERROR，流水线不中断（等同 skip）。

## 5. 方式B：`precomputed` 手动模式（服务器上跑完拷回）

适合：不方便开 SSH 免密、或由他人代跑服务器的场景。TSV 文件本身就是审计证据。

### 5.1 上传查询序列

```powershell
# 本地已生成的查询文件（按需）
scp work\query_split2_train.fasta work\query_split3_val.fasta work\query_arabidopsis.fasta user@服务器IP:~/ips_jobs/
```

### 5.2 服务器批量运行（保存为 run_ips.sh）

```bash
#!/bin/bash
# 用法: ./run_ips.sh query.fasta   （严格遵守 4线程/12GB堆/每批500条）
set -euo pipefail
IPS=/opt/interproscan/interproscan.sh
IN=$1; OUT=ips_out
mkdir -p batches "$OUT"

# 1) 按500条切分
python3 - <<'EOF'
import sys
recs = []
for line in open(sys.argv[1]):
    if line.startswith('>'): recs.append([line, []])
    else: recs[-1][1].append(line)
for i in range(0, len(recs), 500):
    with open(f'batches/batch_{i//500:04d}.fasta', 'w') as f:
        f.writelines(h + ''.join(s) for h, s in recs[i:i+500])
print(f'{len(recs)} 条 -> {(len(recs)+499)//500} 批')
EOF

# 2) 逐批运行（已存在的批次自动跳过 => 断点续跑）
export _JAVA_OPTIONS="-Xmx12G"
for fa in batches/*.fasta; do
    base="$OUT/$(basename "$fa" .fasta)"
    [ -f "$base.tsv" ] && continue
    echo "=== $fa ==="
    "$IPS" -i "$fa" -b "$base" -goterms -cpu 4 -T "tmp_$(basename "$fa" .fasta)"
done
echo "完成 -> $OUT/*.tsv"
```

```bash
chmod +x run_ips.sh
tmux new -s ips          # 挂 tmux 防断线
cd ~/ips_jobs
./run_ips.sh query_split2_train.fasta      # 35批
./run_ips.sh query_split3_val.fasta        # 15批
./run_ips.sh query_arabidopsis.fasta       # 10批
```

### 5.3 拷回结果并切换模式

```powershell
mkdir work\struct_channel_server
scp user@服务器IP:~/ips_jobs/ips_out/*.tsv work\struct_channel_server\
```

```yaml
tools:
  interproscan_mode: "precomputed"
  interproscan: "work/struct_channel_server"   # 指向TSV文件或目录
```

重跑前同样删除 `work/struct_channel_*.tsv` 空结果（见 4.2）。
解析器按当前目标集蛋白 ID 过滤（`only_ids`），多个目标集的 TSV 混放同一目录亦可。

## 6. 与评估的衔接

Struct-Channel 接入后无需改其他代码：

- Arbiter 自动获得 struct 证据（输入统计从 `struct=0` 变为真实数字）；
- 消融评估组 `seq+struct` 不再等同于 `seq-only`；
- 三个目标集（split2_train / split3_val / arabidopsis）各自独立完成 struct 阶段后，
  `eval_*.json` 与 `*_arbiter_conflict_stats.json` 自动更新。

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `SSH连接失败或远程InterProScan不可用` | 免密未配置 / IP端口错 / `tools.interproscan` 路径与服务器不一致。先用第3节命令手工验证 |
| `Host key verification failed` | 首次连接未接受指纹：手工 `ssh user@host` 一次，输入 yes |
| 远程批次失败（跳过） | 看 `logs/pipeline.log` 中该批 stderr 尾部；常见为内存不足（服务器同时跑其他任务）或磁盘满 |
| 下载 TSV 后候选为 0 | 检查 TSV 是否含 GO 列（须带 `-goterms`）；或 `min_coverage` 过滤过严 |
| 想只跑部分成员库 | config: `struct_channel.applications: "Pfam,PROSITE,SMART"`（ssh 模式会透传 `-appl`） |
| Windows 找不到 ssh/scp | 设置→应用→可选功能→添加"OpenSSH 客户端"，或升级 Win10 1809+ |

## 8. 审计与可复现性

- 服务器 InterProScan **版本号写入论文**（发行包目录名，如 5.72-103.0）；
- 成员库版本随发行包固定（`-appl` 指定子集时注明）；
- 拉回的 `work/struct_channel_ssh/<tag>/tsv/*.tsv` 建议归档（zip），
  与 `*_arbiter_conflict_stats.json` 一同作为白盒证据链。
