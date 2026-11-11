# PyTorch 神经调度器

## 1. 目标与方法

神经策略通过离线模仿学习拟合 TUS 在真实仿真队列状态中的优先级分数。它用于研究“数据驱动调度器能否学习任务与网络状态之间的映射”，不是强化学习智能体，也不在训练过程中反向修改仿真环境。

模型为 16→64→64→1 的 MLP，使用 GELU 和 LayerNorm。训练标签为 `TUS score / 4`，推理时乘以 4 并取相反数作为 SimPy 优先级。

## 2. 特征与防泄漏

16 维特征覆盖剩余路径、松弛时间、紧迫度、关键性、语义密度、事件类别、消息大小、网络/计算队列、模态与事件 one-hot 编码。训练和推理共用 `extract_features()`；checkpoint 同时保存特征名称和 TUS 参数，避免自定义参数下发生特征漂移。

数据按完整仿真种子进行 60/20/20 划分，而不是随机拆分单条记录。这样同一任务轨迹不会同时出现在训练和测试中。种子必须唯一，至少需要 5 个种子。默认训练种子为 `base_seed + 1000 + i`，评估使用原实验种子，两者保持分离。

## 3. 训练与评估

```bash
agentcomsim neural train \
  -c configs/quickstart.yaml \
  -o results/neural/scheduler.pt \
  --seeds 10 --epochs 40 --device cpu

agentcomsim neural evaluate \
  -c configs/quickstart.yaml \
  --checkpoint results/neural/scheduler.pt \
  -o results/neural-evaluation
```

训练使用 AdamW、梯度裁剪、验证集早停和最佳权重恢复。输出包括 `.pt` checkpoint 和 `.metrics.json`。checkpoint 含模型权重、特征模式、TUS 配置、种子清单和逐轮历史；加载使用 PyTorch `weights_only=True`，并检查格式版本与特征模式。

也可在本地 Web 的“ns-3 与神经调度”页训练，再把保存路径填入左侧神经策略配置。

## 4. 如何解释结果

- 测试 MSE 只衡量对教师分数的拟合误差，不等于任务成功率提升。
- 神经策略默认不启用有损压缩，因此与 TUS 的任务指标差异同时反映排序和压缩机制差异。
- 若研究重点是排序网络本身，应补充“TUS 不压缩”和“神经策略＋相同压缩规则”的消融实验。
- 当前训练样本来自默认任务图和配置分布。改变模态比例、带宽范围或任务图后，应重新采样并做分布外评估。
- 不建议把小型 MLP 的模仿结果表述为已具备泛化到真实 6G 网络的能力。
