# CalibGuard (CG)

**低成本文件安全审计系统** —— 本地小模型门控 + 云端大模型按需复核。

面向轻度开发者、CS 学生与 AI 初学者：把可疑文件（脚本 / 代码 / 文本 / 压缩包）丢进去，
得到一个四档结论：**放行 / 可疑 / 拦截 / 转人工**。

---

## 核心思路

```
                     ┌──────────── 云端 LLM 复核（可选，仅灰区）────────────┐
                     ▼                                                      │
文件 → 预处理 → 特征提取 → CodeBERT 滑窗推理 → Platt 校准 → 三桶门控 ──────┘
                                                                  │
                          p < 下限  → 放行        （不联网）
                          p > 上限  → 拦截        （不联网）
                          灰区之间  → 提交云端复核（p>0.5 兜底 / 可关闭）
```

三个关键设计：

1. **概率校准**：微调模型输出的原始 logit 经 Platt scaling 校准为可解释的"恶意概率"，
   阈值判定的可靠性显著优于裸 logit（校准前后 F1 差 1.4pt，但概率可解释性大幅提升）。
2. **三桶门控**：绝大多数文件（低危 / 高危）完全不联网即可判定，只有灰区才付费调用云端大模型，
   兼顾成本与召回。
3. **全文滑窗**：推理按 512-token 窗口、256 步长覆盖全文并取 `max(窗口 logit)`，
   不受单次 512 token 截断限制。

---

## 效果（test 段 25,049 条，统一 `p > 0.5` 判定）

| 底座 | 裸测 F1 | 校准后 F1 | 校准后 P | 校准后 R | 训练耗时 |
|---|---|---|---|---|---|
| **CodeBERT (batch 16，推荐)** | 0.8734 | **0.8600** | 0.8740 | 0.8464 | 2.43h |
| CodeBERT (batch 8) | 0.8691 | 0.8672 | 0.8200 | 0.9201 | 2.54h |
| Qwen2.5-Coder-0.5B-Instruct | 0.8685 | 0.8396 | 0.8601 | 0.8202 | 5.69h |
| Qwen3-0.6B | 0.8572 | 0.8295 | 0.8440 | 0.8156 | 5.7h |
| Qwen3-CoderSmall | 0.8582 | 0.8276 | 0.8620 | 0.7958 | 7.8h |

对照：规则基线 F1 0.5025；历史版本 v0.2.2 裸测 0.5953。
**加云端 LLM 灰区复核后**：F1 0.8600 → **0.8689**、精确率 0.8739 → **0.9017**。

> 选型结论：CodeBERT 推理 **22.4 条/s**（约为 0.5B 级 Qwen 的 3–4.3 倍），训练快 2.3–3.2 倍，
> 体量小 4–5 倍，综合最优，故作为默认部署底座。

---

## 仓库结构

```
CG/
├── CG_GUI/                 桌面壳：启动器 + 后端 + 前端单页（用户直接使用）
│   ├── launcher.py         启动器（纯标准库；首启自动装环境）
│   ├── server.py           FastAPI 后端
│   ├── audit_service.py    审计任务封装（单并发）
│   ├── calibguard/         引擎副本
│   ├── web/index.html      前端单页（自包含，离线可用）
│   ├── samples/            演示样本
│   ├── tests/              端到端验收
│   ├── tools/              验收脚本
│   ├── README.txt          面向最终用户的使用说明
│   └── requirements-gui.txt
└── CG_Bench/               基准设施：训练 / 评测 / 消融（产出模型）
    ├── harness.py          七步流水线编排（HPO→train→eval→calibrate→…）
    ├── train/              训练、评测、校准、HPO 引擎
    ├── adapters/           LLM / 混合 / 规则 / 零样本等对照适配器
    ├── calibguard/         引擎副本
    ├── tests/
    ├── requirements.txt
    └── setup.sh / setup.bat
```

> 两个子项目各自带一份 `calibguard/` 引擎副本，是为了让它们能独立运行/打包。

---

## 快速开始

### 方式一：直接下载使用（推荐，无需 Python 环境）

1. 到本仓库的 [**Releases**](../../releases) 页面，下载 `CalibGuard-GUI.zip`
2. 解压到任意目录（**整个文件夹一起**，不要只留 exe）
3. 双击 `CalibGuard.exe` —— 首次会自动准备运行环境（约 2–10 分钟，需联网）
4. 准备完成后浏览器自动打开，拖入文件即可审计

详见压缩包内的 `README.txt`。模型权重已随包提供，**无需自行训练**。

### 方式二：源码运行（开发者）

**桌面壳**：

```bash
cd CG_GUI
python -m venv .venv
.venv/Scripts/pip install -r requirements-gui.txt      # Linux/macOS 用 .venv/bin/pip
.venv/Scripts/python launcher.py
```

> 需要 `models/` 目录（模型 + 校准参数）。见下方「模型与数据」。

**基准设施（训练 / 评测）**：

```bash
cd CG_Bench
bash setup.sh                 # Linux；Windows 用 setup.bat
bash train-bench.sh --dry-run # 先看将执行的命令序列
bash train-bench.sh           # 按 experiments.yaml 顺序串行训练
```

---

## 配置

桌面壳的设置界面（右上角「设置」）可配置：

| 项 | 说明 |
|---|---|
| 使用云端 LLM 复核 | 总开关。关闭后灰区按本地模型 50% 处判定 |
| API Key / Base URL / 模型名称 | OpenAI 兼容接口（默认 DeepSeek） |
| 送审内容 | 精炼证据 / 文件原文 |
| 投递方式 | 截断（前 N 字符）/ 滑窗（分片，取最严重）/ 一次性全部上传 |
| 灰区范围 | 默认 0.15 ~ 0.85 |

配置存本机 `gui-settings.json`，**不入库**。

基准设施的 Key 走环境变量或项目根 `.env`（`DEEPSEEK_API_KEY`），同样**不入库**。

---

## 模型与数据

为保证仓库轻量，以下内容**不随仓库分发**：

| 内容 | 说明 |
|---|---|
| 训练/评测语料 | 约 26 GB，含第三方包内容，不适合公开分发 |
| 训练产物 `results/` | 中间产物约 8 GB |
| 模型权重 | 已打包进 Releases 里的 `CalibGuard-GUI.zip`，开箱即用 |

如需自行复现训练：准备语料后按 `CG_Bench/experiments.yaml` 执行，
数据划分、HPO 网格与校验流程都在 `harness.py` 与 `train/` 中。

---

## 已知限制

- 语料以 Python 生态（PyPI / npm）为主，其他语言生态的召回会下降。
- 云端复核为**可选**：不配置 API Key 时，灰区文件统一转人工，不影响其余判定。
- 无独立显卡时可运行（自动使用 CPU 版），但推理较慢。
- 算力低于 7.5 的旧显卡（如 GTX 10 系）会自动切换 CPU 版——新版加速库已不支持其算力。

---

## 许可证

见 [LICENSE](LICENSE)。
