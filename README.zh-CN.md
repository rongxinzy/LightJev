# LightJev · 轻量决策

**把语言模型底座训练成可输出候选概率的决策模型。**

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-LightJev--0.6B-yellow)](https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1)

[English](README.md) · [数据格式](docs/data.md) · [架构](docs/design.md) · [评测](docs/evaluation.md)

标准预训练底座采用 **Qwen3-0.6B**。输入上下文、问题和候选描述，输出完整候选分布与选中值。支持 Choice、Boolean 和有序 Score，使用交叉熵或 Brier loss 训练底座与共享评分头。

## 发布进展

**2026-09-18：** 已完成 Qwen3-0.6B 全参数 CE/Brier 两组训练，首个模型 checkpoint 已在 Hugging Face 公开。配套 [v0.1.1 代码发布版](https://github.com/rongxinzy/LightJev/releases/tag/v0.1.1) 通过 46 项测试和 CI，公开模型文件的发布哈希及权重匿名访问已核验。

**[下载 LightJev-0.6B-v0.1](https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1)**：包含训练后的 Qwen3-0.6B 评分权重、tokenizer、冻结程序真值数据、训练记录及 CE/Brier 两组评测。发布 CE 的第 250 步 checkpoint，在查看最终测试结果前按开发集 CE 选定。

原始概率的硬标签准确率为 **test 79.57%（656 题）/ 来源定义 OOD 72.44%（352 题）**；精确软分布的平方 L2 分别为 **0.003132 / 0.003964**。这是合成任务、单种子研究结果。目录检索和智能家居规则较强，网格与井字棋状态判断较弱，尚不能称为通用业务决策模型。[完整结果与限制](docs/results-v0.1.md)。

**2026-09-24：Typed Decisions 迁移实验。** LightJev-0.6B 零样本准确率为 0.396，低于该数据集 0.470 的先验准确率。Laya 专家模型的 soft-CE 对照为 0.781，但 specialist 与 generalist 不可直接比较；且该对照是在查看 RLCD 测试结果后启动，因此仅作探索性结果，不宣称榜单突破。[可复现报告与产物](research/laya-typed-decisions-2026-09-24/REPORT.zh-CN.md)包含切分、代码和完整逐题概率。约 804 MB 的权重未发布。

**Workflow 留一验证已完成：** 四折留一 workflow 共评估 6,000 个决策，RLCD 准确率 45.3%、配对 soft-CE 44.7%、Laya 基座 36.1%、LightJev 零样本 35.7%。[验证方案、复现脚本和完整结果](research/laya-typed-decisions-2026-09-24/workflow-generalization/README.md)未使用官方 public test split。此前实验查看过这些 workflow 的汇总分数，因此本轮仍属探索性结果。

## 使用训练权重

### 安装与首次运行

使用 Python 3.10 或更新版本（实际验证使用 Python 3.12）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'git+https://github.com/rongxinzy/LightJev.git@v0.1.1' huggingface_hub
```

Windows 使用 `.venv\Scripts\activate` 激活环境。CPU 推理无需 CUDA；NVIDIA GPU 推理需要与驱动兼容的 CUDA 版 PyTorch。先运行 `python -c "import torch; print(torch.cuda.is_available())"`，确认后再使用 `device="cuda"`。

公开权重无需 HF token。首次下载包含约 **2.38 GB 的 FP32 权重**及 tokenizer、元数据；后续复用 Hugging Face 磁盘缓存。运行时内存需求高于权重文件大小，目前没有测定最低内存/显存要求。将下方 Python 示例保存为 `example.py`，执行 `python example.py`。

```python
from huggingface_hub import snapshot_download
from lightjev.inference import predict

checkpoint = snapshot_download("rongxinzy/LightJev-0.6B-v0.1")
records = [{
    "id": "example-1", "group_id": "example",
    "state": "Home log: user wants the dining room fan set to on; access=yes; occupants=2; clock=20:00.",
    "question": "Select exactly the room and device named in the request. Ignore authorization and occupancy for this question.",
    "kind": "choice", "candidates": ["balcony speaker", "dining room fan"],
}]
print(predict(checkpoint, records, device="cpu"))
```

示例取自冻结 test 的第一题，保留原始任务格式，不作为新的泛化证明。另一个手写降温规则抽查失败，已记录在[局限](docs/results-v0.1.md#additional-manual-probe)中。

这是 **LightJev 自定义评分 checkpoint**，不能当普通聊天模型用 `AutoModelForCausalLM` 加载。GPU 推理可改为 `device="cuda"`。默认输出未经温度缩放的概率；温度拟合在选中 CE 模型的最终测试 CE/ECE 上没有带来改善，因此不自动应用。每个候选超过 256 tokens 会明确报错。[复现训练](docs/training-release.md) · [数据归因](docs/data.md)。

### 输入、输出与多题调用

每条输入必须包含非空字符串 `id`、`group_id`、`state`、`question`，以及 `kind` 和 `candidates`。推理无需 `target`，自己的请求可直接省略。候选必须为 2–255 个互不重复的非空字符串。

| `kind` | 候选格式 | 额外输出 |
|---|---|---|
| `choice` | 自定义候选描述 | — |
| `boolean` | 严格按顺序填写 `["false", "true"]` | — |
| `score` | 按等级从低到高排列描述 | `expectation`：概率加权的等级索引 |

`predict()` 按输入顺序返回列表。上方示例已在 CPU 验证，选择 `dining room fan`，概率约为 `[0.000000076, 0.99999988]`；不同设备可能有少量数值差异。每条输出包含 `id`、`kind`、`candidates`、`probabilities`、`selected`、从 0 开始的 `selected_index` 和 `temperature`。

多道题可放入同一次 `predict(checkpoint, records, device="cuda")` 调用，共用一次模型加载。**每次调用都会重新加载模型，调用内部仍逐题执行**，并非跨题 GPU 批处理或常驻服务。磁盘缓存避免重复下载，但不会让模型常驻内存。常驻服务需自行保留 `load_checkpoint()` 返回的 model/tokenizer，并实现相同的编码、评分与每题 softmax。

256-token 限制针对**每个候选的完整格式化输入**，包括 state、question、候选和提示文本，超长需主动缩短。默认 `temperature=1`；已发布的温度拟合会使选中模型的最终测试 CE/ECE 变差。

### JSONL 命令行推理

执行上方 Python 示例、得到 `checkpoint` 和 `records` 后，可保存为本地文件：

```python
import json
from pathlib import Path

Path("checkpoint-path.txt").write_text(checkpoint, encoding="utf-8")
Path("input.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
    encoding="utf-8",
)
```

在 POSIX shell 中运行：

```bash
lightjev predict --checkpoint "$(cat checkpoint-path.txt)" \
  --input input.jsonl --output predictions.json --device cpu
```

GPU 可改为 `--device cuda`。输入每行一个 JSON 对象、ID 不重复；输出为 JSON 数组。离线使用已下载的本地 checkpoint 路径，需要保留 `manifest.json`、`model.safetensors`、`backbone/` 和 `tokenizer/`；加载器不会联网补下载缺失文件。

**推理后端状态：** 当前使用 PyTorch/Transformers，尚未实现或验证直接 `vllm serve` 加载，也不能作为普通聊天权重通过 `AutoModelForCausalLM` 或 chat-completions API 使用。

## 快速运行

```bash
git clone https://github.com/rongxinzy/LightJev.git
cd LightJev
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
lightjev demo --output-dir runs/demo --steps 8
```

安装依赖后，示例不再访问网络：本地创建随机初始化的极小 BERT 与 tokenizer，使用合成传感器数据训练，再生成 `report.json` 和 `predictions.json`。数据模板在不同 split 间相似，因此它只验证工程链路，不证明泛化。重复运行请使用新输出目录。

## 已实现

- 动态候选评分，支持 2–255 个候选；Boolean 固定 false/true。
- backbone 隐藏表示经过 LayerNorm 和共享标量头，不逐 token 生成答案。
- 全参数训练，支持 CE 与 Brier，每题等权。
- 数据校验、train/dev 源组与 ID 隔离、超长输入明确报错。
- 本地 safetensors checkpoint、tokenizer、配置和训练来源记录。
- 概率误差、硬标签准确率、可靠性分箱、自动处理覆盖率与错误率。
- 独立校准集上的温度拟合，CPU CI 和端到端离线示例。

## 用自己的数据训练

按[数据契约](docs/data.md)准备 JSONL。训练/开发/校准/测试应按原始来源分组隔离，不能靠给同一条内容换 ID 实现隔离。

```bash
lightjev train \
  --model Qwen/Qwen3-0.6B \
  --train data/train.jsonl --dev data/dev.jsonl \
  --output-dir runs/qwen-ce \
  --steps 200 --batch-size 2 --lr 2e-5 \
  --loss ce --max-length 512 --device cuda
```

默认底座固定为 `Qwen/Qwen3-0.6B`，版本 `c1899de289a04d12100db370d81485cdf75e47ca`。训练会下载模型并进行全参更新；可加 `--revision` 覆盖版本，其他模型或本地路径不会被套用这一版本。训练命令不代表已经验证模型能力。显存取决于候选数、长度和 batch，当前不承诺具体显存下限。

```bash
lightjev predict --checkpoint runs/qwen-ce --input data/test.jsonl \
  --output runs/predictions.json --device cuda
lightjev evaluate --input data/test.jsonl --predictions runs/predictions.json
```

`calibrate` 可在独立校准集预测上拟合温度，再用 `predict --temperature` 应用于测试集。不要把测试集用于拟合。

## 验证真实 Qwen 底座

```bash
python scripts/verify_qwen.py --output-dir runs/qwen-smoke --device cpu
```

这项接入检查会下载真实权重，完成一次全参数更新、checkpoint 保存和重新加载。使用六条手写样本，仅验证接入与训练链路，不证明能力或校准效果。它与极小随机模型的离线示例分开，需要足够内存及首次下载。

## 当前边界

每个候选仍重复编码完整上下文，尚未实现共享前缀推理，也没有等质量加速成绩。Boolean 目前使用两条候选路径。Score 输出有序等级下标的期望，并非任意数字回归。softmax 分数不自动代表真实正确率。

首版重点是可训练、可验证的基础实现。后续再加入真实领域数据、多种子实验、候选集合交互、共享前缀和推理服务。

## 来源与许可

研究方向参考 [TypeSafe Jev](https://typesafe.ai/) 与 [NanoJev](https://github.com/TianyuCodings/NanoJev)。本项目代码独立实现，Qwen 训练实验使用 [C-Tianyu/NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) 的转换子集；每条来源记录声明 CC0-1.0，仅保留程序真值，排除 Jev 教师输出。未使用 NanoJev 权重与实现文件，独立代码不意味着训练数据也是自建。本项目与 TypeSafe 无隶属关系，也不宣称复现其未公开的 RLCD 配方。

代码使用 Apache-2.0。外部底座和数据各自遵循原许可证。
