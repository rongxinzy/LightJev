# LightJev · 轻量决策

**把语言模型底座训练成可输出候选概率的决策模型。**

[English](README.md) · [数据格式](docs/data.md) · [架构](docs/design.md) · [评测](docs/evaluation.md)

标准预训练底座采用 **Qwen3-0.6B**。输入上下文、问题和候选描述，输出完整候选分布与选中值。支持 Choice、Boolean 和有序 Score，使用交叉熵或 Brier loss 训练底座与共享评分头。

**v0.1 是研究工具包，尚未发布经过广泛任务训练的决策权重。** 自带 CPU 离线示例验证真实训练、checkpoint 保存、重新加载、推理和评测；不把小型示例结果作为通用能力证明。

真实 Qwen3-0.6B 的 CE/Brier 对照训练**正在进行**，使用固定版本 NanoJev-Data 的程序真值子集。当前尚无已完成的能力结果或已发布训练权重声明。详见[复现协议](docs/training-release.md)与[数据来源](docs/data.md)。

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
