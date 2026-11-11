# Security Policy


## 报告漏洞

请在 GitHub 仓库启用并使用 **Private vulnerability reporting / Security Advisory** 私密报告漏洞；不要先在公开 Issue 中披露可利用细节。报告应包含受影响版本、环境、复现步骤、潜在影响和可行缓解措施。维护者确认后再协调修复与公开说明。

## 使用注意事项

- YAML 使用安全加载；不要绕过 `load_config()` 执行不受信任的配置。
- PyTorch checkpoint 使用 `weights_only=True` 并检查特征模式，但仍应只使用来源可信的模型文件。
- Web 工作台设计为本机研究工具，不包含身份认证；不要直接暴露到公共网络。
- ns-3 桥接会复制 C++ 文件并启动外部进程。仅对可信的 ns-3 树和场景目录授权，审查日志中的路径信息。
- 实验输出可能包含本地路径和研究数据，公开归档前应检查。
