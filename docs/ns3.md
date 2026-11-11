# ns-3 数据包级桥接

## 1. 功能定位

桥接器将 Python 任务轨迹展开为 sense、tool、plan、control 四类消息，导出 CSV，并调用外部 ns-3 中的 C++ UDP 点对点程序。接收端按消息累计负载字节，输出逐消息完成时间和汇总 JSON。

这是**单向数据包级验证与时延校准**：ns-3 结果不会回写到正在运行的 Python 离散事件内核。导出的四类消息使用任务到达时间作为发送时间，不重建 Python 任务 DAG 的实际排队与汇合时刻。因此不能把桥接输出直接当作完整任务级联合仿真结果。

## 2. 环境与执行

接口目标为支持 C++17 的 ns-3.43 或兼容版本。用户需要先自行安装 ns-3，并保证其根目录存在 `ns3` 启动器。

```bash
agentcomsim ns3 export \
  -c configs/quickstart.yaml \
  -o results/ns3/scenario.csv \
  --arrival-rate 0.8 --seed 202600

agentcomsim ns3 run \
  --scenario results/ns3/scenario.csv \
  --output results/ns3/run \
  --ns3-dir /absolute/path/to/ns-3 \
  --bandwidth 25 --delay-ms 2
```

也可设置 `NS3_HOME` 代替 `--ns3-dir`。桥接器把包内 `agentcomsim_bridge.cc` 复制到目标树的 `scratch/`，使用参数数组调用外部进程，不经过 shell，并保存 stdout 与 stderr。

已有结果可重新解析：

```bash
agentcomsim ns3 calibrate --results results/ns3/run
```

## 3. 数据格式

输入 CSV 字段：

```text
message_id,task_id,kind,send_time_s,size_bytes
```

每条消息被拆为不超过 1024 字节的原始负载。每个 UDP 数据报在线路负载前 4 字节以大端序携带内部消息编号；不能用 ns-3 本地 Packet Tag 代替，因为本地标签不会自动跨节点传输。

输出 `messages.csv` 增加 `finish_time_s` 和 `received_bytes`。未完成消息的完成时间为 -1，接收字节数可小于原始大小。`summary.json` 包含消息数、完成数、接收数据包数、带宽和链路时延。

Python 解析器检查必需列、接收字节上界、完成消息时序和汇总消息数一致性。校准时延仅使用完整消息，`suggested_base_delay_s` 是当前样本的最小完成时延，只是后续参数拟合的起点。

## 4. 验证范围

自动化测试覆盖场景导出、合法与非法结果解析、未完成消息、外部命令构造、源码安装和日志保存。当前开发环境没有 ns-3，因此本版本**未在真实 ns-3 中编译或执行 C++ 程序**。发布者不将模拟子进程测试描述为真实 ns-3 验证。

在真实环境中首次运行时，请保留构建日志、ns-3 版本和输出目录；若 API 兼容性导致编译失败，应将其作为集成问题报告，而不是修改结果文件绕过校验。
