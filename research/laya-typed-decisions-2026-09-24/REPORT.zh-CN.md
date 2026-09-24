# Laya / LightJev Typed Decisions 研究记录

日期：2026-09-24。数据集、模型和源码 revision 已写入 [`manifest.json`](manifest.json)。

## 榜单、数据与提交

这里的基准是 Hugging Face 上由 LocalLLaMA 维护的 [`typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)，不是 TypeSafe/Laya 的官方竞赛，也不是隐藏测试集。它使用与 System One 相似的 `noul`、`choice`、`score` 接口，但数据集说明明确表示它独立于 TypeSafe，也不复现 Jev。

数据共四个合成 workflow：agent trace observability、customer service、invoice processing、security incidents。每个 workflow 有 300 个 train case、100 个 test case；每个 case 对同一个 state 提出五个 typed question，所以总计 1,200/400 个 case、6,000/2,000 项决策。每个选项的 `criteria` 描述属于输入。gold 是一个约 4B teacher 对每个 case 以 temperature 0.7 采样三次、再平均完整概率分布得到的值。因此测试的是对这位 teacher 分布的拟合，不是现实任务的客观正确率。

数据卡要求在公开 `test` split 上计算完整概率分布，并在数据集 Discussion 中发布指标及运行模式 `specialist` 或 `generalist`。截至本次核查，generalist榜首 meraGPT Decider 1 的 accuracy/KL/Brier 为 0.768/0.096/0.052；TypeSafe Jev 1.13.0 为 0.727/1.442/0.148。Laya typed-decisions 模型卡报告 specialist 0.766/0.062 Brier。卡片明确说 specialist 与 generalist 不可直接比较：specialist 用这四个 workflow 的训练集拟合固定问题；generalist 在其他 workflow 上训练后零样本作答。

## Laya 训练思路与本次实现

公开 Laya 专家模型以 421M ModernBERT-large 为底座，使用这 1,200 个训练 case。训练时在每个问题的候选 logits 上加四组零均值 Gaussian 探索噪声；用 log score、spherical score 和有序 score 的 RPS 组成 proper-score reward；按同组 reward 的均值做 baseline、标准化 advantage，以 REINFORCE 更新策略，并并行保留 soft-label cross-entropy。其公开 notebook 是 Kaggle 2×T4 配方。

我保留了这个目标，但把 train case 按 workflow 分层切成 train/dev/calibration = 960/120/120 case，测试集 400 case 始终独立。这样同一 state 上的五道题不会被拆到训练和验证两侧；每轮用 dev soft NLL 选 checkpoint，并仅用单独 calibration split 拟合温度。两次训练均使用 8× RTX 4060 Ti 16 GiB、BF16、DDP、global effective batch 64、4 epochs，AdamW + cosine learning-rate schedule（encoder 2.5e-5、head 1e-4），每次约 16 分钟、峰值显存约 9.34 GiB/GPU。RLCD 组大小为 4、噪声标准差从 0.4 退火至 0.1。

另外跑了 `--rl-weight 0` 的纯 soft-CE 对照。这个对照是在看到 RLCD 公共 test 结果后才决定启动的；它没有用 test 更新参数或选 epoch，但“是否跑这个训练目标”受到已见 test 的影响，所以这条成绩作为探索性结果记录，不应包装成独立预注册的榜单胜利。

## 同一公开 test 上的结果

所有本地行均覆盖 400 cases / 2,000 decisions。Laya 行用其官方 Agent API；LightJev 使用候选打分 API，完整保留 state、问题说明和候选 criteria。延迟的计量方式不同，不应与榜单客户端延迟直接比较。

| 模型 | 模式 | Accuracy ↑ | Soft accuracy ↑ | KL ↓ | Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| meraGPT Decider 1（数据卡） | generalist | 0.768 | 0.608 | 0.096 | 0.052 | 0.180 |
| TypeSafe Jev 1.13.0（数据卡） | generalist | 0.727 | 0.580 | 1.442 | 0.148 | 0.144 |
| Laya 官方 specialist（数据卡） | specialist | 0.766 | 0.471 | — | 0.062 | 0.213 |
| Laya 官方 checkpoint（本地复测） | specialist | 0.769 | 0.471 | 0.117 | 0.061 | 0.216 |
| Laya RLCD，seed 17 | specialist | 0.744 | 0.505 | 0.101 | 0.057 | 0.128 |
| Laya soft-CE，seed 19（探索性） | specialist | **0.781** | **0.510** | **0.088** | **0.049** | 0.162 |
| LightJev 0.6B，zero-shot | generalist | 0.396 | 0.325 | 0.470 | 0.254 | 0.066 |

soft accuracy 对每项决策将预测完整分布与 gold 完整分布相乘后平均。LightJev 低于数据卡的 0.470 prior accuracy，表明它的现有训练分布不能直接迁移到这批决策场景；这只说明对本基准的零样本表现，不代表它在原训练域的结果。

按 question type，soft-CE 的 choice/noul/score accuracy 为 0.752/0.855/0.748，RLCD 为 0.713/0.845/0.691，LightJev 为 0.348/0.540/0.323。soft-CE 的 score MAE 是 0.216（within one level 为 0.995）；RLCD MAE 是 0.243。按 workflow，soft-CE accuracy 为 observability 0.748、customer service 0.774、invoice processing 0.824、security incidents 0.778。完整切片在每个模型的 `*_report.json` 及 `*_predictions.jsonl` 中。

纯 soft-CE specialist 的原始 accuracy 高于榜单 generalist 榜首公开值，但两种模式不可直接比较，而且此对照是在第一次 test 结果后添加的。因此目前结论是：在这四个已知 synthetic workflow 上，Laya + soft-CE 值得继续做；还没有可严谨宣称的 generalist 榜单突破。相对 RLCD，纯 CE 在该次 test 上 accuracy、KL、Brier 更好，RLCD 的 ECE 更低。这说明 RLCD 不能只凭训练理念假定优于软标签 CE。

复测官方 Laya checkpoint 时，运行库发出 temperature 警告：继承的 `choice:11+` 温度值 0.1006 被限制为 0.5。数据卡本身也记录该 checkpoint 继承了 option-bucket temperature，覆盖新拟合的 per-type temperature。accuracy 的 argmax 不受统一温度改变影响，但其置信度/校准指标需按该限制解释。

LightJev 的发布配置用 256-token 训练窗口；为了保留整个测试输入，本次将推理上限显式扩至 640 tokens，最长输入 525 tokens、没有截断。长输入超过其训练窗口，所以该 zero-shot 结果包含上下文长度外推因素。

## 文件和复现

- [`manifest.json`](manifest.json)：dataset/model/Laya notebook/LightJev revisions 与权重 SHA256。
- [`prepare_data.py`](prepare_data.py)：检查数据并产生 case-level splits。
- [`train_rlcd_ddp.py`](train_rlcd_ddp.py)：完整 RLCD 与 soft-CE 对照训练入口。
- [`evaluate_benchmark.py`](evaluate_benchmark.py)、[`evaluate_lightjev.py`](evaluate_lightjev.py)：Laya 与 LightJev 评测入口。
- [`results/`](results/)：四个模型逐题完整标签分布、gold 分布和聚合报告。
- [`runs/`](runs/)：dev 选模数据、calibration 温度和训练日志。两个 safetensors 权重各约 804 MB，未纳入本 GitHub 分支；SHA256 记录在对应的 `selected.json`。

完整权重各约 804 MB，保留在训练环境中，没有随本次 GitHub 分支发布，也尚未上传 Hugging Face；对应 SHA256 记录在 `runs/*/selected.json`。本次公开代码、切分数据、训练日志、逐题预测和评测报告，不代表新模型权重已可下载；也没有向榜单提交结果。
