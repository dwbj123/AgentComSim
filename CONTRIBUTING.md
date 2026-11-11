# Contributing to AgentComSim

感谢贡献研究代码、实验配置、文档或问题复现材料。

## 开发流程

1. Fork 仓库并创建聚焦单一目标的分支。
2. 安装开发依赖：`python -m pip install -e '.[all,dev]'`。
3. 为行为变化添加测试和文档；不要通过删除测试或放宽统计断言掩盖回归。
4. 提交前运行：

```bash
ruff format --check .
ruff check .
pytest -q
python -m build
```

5. 在 Pull Request 中说明动机、方法、验证命令、输出变化和剩余限制。实验数值变化应附配置、种子和逐次结果。

## 添加调度策略

继承 `SchedulingPolicy`，实现 `priority(context)`，并保持“数值越小越先服务”的约定。若策略修改压缩状态，必须说明语义质量与字节数如何变化。新增策略应进入统一实验矩阵，复用共同随机数，不另写不可比较的实验路径。

## 研究完整性

不伪造 ns-3 运行、训练结果或置信区间。区分真实执行、解析器测试、模拟外部进程和待验证功能。新增外部依赖时给出安装方式、失败行为和最小测试。

## 行为与安全

参与讨论时遵守 `CODE_OF_CONDUCT.md`。安全漏洞不要提交到公开 Issue，按 `SECURITY.md` 处理。
