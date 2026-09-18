# LightJev · 轻量决策

**把语言模型底座训练成可输出候选概率的决策模型。**

[English](README.md) · [数据格式](docs/data.md) · [架构](docs/design.md) · [评测](docs/evaluation.md)

输入上下文、问题和候选描述，输出完整候选分布与选中值。支持 Choice、Boolean 和有序 Score，使用交叉熵或 Brier loss 训练底座与共享评分头。

**v0.1 是研究工具包，尚未发布经过广泛任务训练的决策权重。** 自带 CPU 离线示例验证真实训练、checkpoint 保存、重新加载、推理和评测；不把小型示例结果作为通用能力证明。

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

这是预训练底座接入方法，尚不是已经公布的 Qwen 实验成绩。训练会下载模型并进行全参更新；可加 `--revision` 固定模型版本。显存取决于候选数、长度和 batch，当前不承诺具体显存下限。

```bash
lightjev predict --checkpoint runs/qwen-ce --input data/test.jsonl \
  --output runs/predictions.json --device cuda
lightjev evaluate --input data/test.jsonl --predictions runs/predictions.json
```

`calibrate` 可在独立校准集预测上拟合温度，再用 `predict --temperature` 应用于测试集。不要把测试集用于拟合。

## 当前边界

每个候选仍重复编码完整上下文，尚未实现共享前缀推理，也没有等质量加速成绩。Boolean 目前使用两条候选路径。Score 输出有序等级下标的期望，并非任意数字回归。softmax 分数不自动代表真实正确率。

首版重点是可训练、可验证的基础实现。后续再加入真实领域数据、多种子实验、候选集合交互、共享前缀和推理服务。

## 来源与许可

研究方向参考 [TypeSafe Jev](https://typesafe.ai/) 与 [NanoJev](https://github.com/TianyuCodings/NanoJev)。本项目独立实现，未重新发布 NanoJev 代码、数据、权重或实验成绩；与 TypeSafe 无隶属关系，也不宣称复现其未公开的 RLCD 配方。

代码使用 Apache-2.0。外部底座和数据各自遵循原许可证。
