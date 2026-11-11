"""Local Streamlit workbench for AgentComSim.

Run with: agentcomsim web
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from agentcomsim.config import ExperimentConfig
from agentcomsim.experiment import default_policies, run_experiment
from agentcomsim.ns3.bridge import export_scenario
from agentcomsim.reporting import plot_summary
from agentcomsim.trace import generate_trace

st.set_page_config(page_title="AgentComSim · 智能体通信仿真", page_icon="◉", layout="wide")

st.markdown(
    """
<style>
:root {
  --acs-ink: #10252b;
  --acs-muted: #49646b;
  --acs-bg: #c4dbe0;
  --acs-panel: #edf5f6;
  --acs-line: #9fbcc3;
  --acs-accent: #146e75;
  --acs-warm: #a56735;
}
.stApp { background: var(--acs-bg); color: var(--acs-ink); font-family: "JetBrains Mono", "Noto Sans CJK SC", "Microsoft YaHei", monospace; }
[data-testid="stHeader"] { background: var(--acs-bg); }
[data-testid="stSidebar"] { background: #ddedf0; border-right: 1px solid var(--acs-line); }
[data-testid="stMainBlockContainer"] { max-width: 1540px; padding-top: 1.4rem; }
h1, h2, h3 { color: var(--acs-ink); letter-spacing: -0.025em; }
h1 { font-size: 2.15rem !important; }
p, label, .stMarkdown { font-size: 0.95rem; line-height: 1.65; }
.acs-header { border-bottom: 1px solid var(--acs-line); padding: 0 0 1rem 0; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: end; }
.acs-kicker { color: var(--acs-accent); font-size: .76rem; letter-spacing: .16em; font-weight: 700; }
.acs-status { color: var(--acs-muted); font-size: .78rem; white-space: nowrap; }
.acs-status::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #2f7d68; margin-right: 7px; }
[data-testid="stMetric"] { background: var(--acs-panel); border: 1px solid var(--acs-line); padding: 12px 14px; border-radius: 3px; }
[data-testid="stMetricLabel"] { color: var(--acs-muted); }
.stButton button, .stDownloadButton button { background: var(--acs-accent); color: #f4fafa; border: 1px solid var(--acs-accent); border-radius: 3px; min-height: 44px; }
.stButton button:hover, .stDownloadButton button:hover { background: #0e5960; border-color: #0e5960; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 3px solid #a56735; outline-offset: 2px; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--acs-line); gap: 20px; }
.stTabs [data-baseweb="tab"] { color: var(--acs-muted); }
.stTabs [aria-selected="true"] { color: var(--acs-accent); }
[data-testid="stDataFrame"] { background: var(--acs-panel); }
code { color: #824e29; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media (max-width: 640px) { .acs-status { display: none; } h1 { font-size: 1.65rem !important; } }
</style>
<div class="acs-header">
  <div><div class="acs-kicker">TASK-CENTRIC NETWORK RESEARCH</div><h1>AgentComSim</h1></div>
  <div class="acs-status">本地研究环境 · 数据不离开当前设备</div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("实验配置")
duration = st.sidebar.number_input("仿真时长（秒）", 20, 600, 45, 5)
warmup = st.sidebar.number_input(
    "预热时长（秒）", 0, int(duration) - 1, min(5, int(duration) - 1), 1
)
runs = st.sidebar.number_input("独立运行次数", 1, 100, 5, 1)
rates_text = st.sidebar.text_input("任务到达率（逗号分隔）", "0.8, 1.4")
bandwidth = st.sidebar.number_input("瓶颈带宽（Mbit/s）", 1.0, 1000.0, 25.0, 1.0)
policies_selected = st.sidebar.multiselect(
    "调度策略", ["fifo", "static", "tus", "neural"], ["fifo", "static", "tus"]
)
record_events = st.sidebar.checkbox("记录因果事件轨迹", False)
checkpoint_path = ""
if "neural" in policies_selected:
    checkpoint_path = st.sidebar.text_input(
        "神经调度器 checkpoint", st.session_state.get("checkpoint_path", "")
    )

try:
    rates = tuple(float(item.strip()) for item in rates_text.split(",") if item.strip())
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError
except ValueError:
    st.sidebar.error("到达率必须是正数，例如 0.8, 1.2, 1.6")
    st.stop()

config = ExperimentConfig(
    duration_s=float(duration),
    warmup_s=float(warmup),
    runs=int(runs),
    arrival_rates=rates,
    policies=tuple(policies_selected),
    record_events=record_events,
)
config.network.bandwidth_mbps = float(bandwidth)

run_clicked = st.sidebar.button("运行对照实验", type="primary", use_container_width=True)
if run_clicked:
    if not policies_selected:
        st.error("请至少选择一种调度策略。")
    else:
        try:
            policy_factories = default_policies(config)
            if "neural" in policies_selected:
                if not checkpoint_path or not Path(checkpoint_path).is_file():
                    raise ValueError("请提供有效的神经调度器 checkpoint 文件。")
                from agentcomsim.policies.neural import NeuralPolicy

                policy_factories["neural"] = lambda: NeuralPolicy(checkpoint_path)
            with st.spinner("正在执行共享任务轨迹上的对照实验……"):
                raw, summary, results = run_experiment(config, policy_factories)
            st.session_state.update(raw=raw, summary=summary, results=results, config=config)
            st.success(f"已完成 {len(raw)} 次仿真。")
        except Exception as exc:
            st.error(f"实验失败：{exc}")

experiment_tab, trace_tab, integration_tab, about_tab = st.tabs(
    ["实验结果", "任务与因果轨迹", "ns-3 与神经调度", "模型说明"]
)

with experiment_tab:
    if "summary" not in st.session_state:
        st.info("在左侧设置负载与调度策略，然后运行实验。首次使用可先保持默认参数。")
    else:
        summary = st.session_state.summary
        raw = st.session_state.raw
        best = summary.sort_values("task_success_rate_mean", ascending=False).iloc[0]
        high_load = summary[summary["arrival_rate"] == summary["arrival_rate"].max()]
        best_high = high_load.sort_values("task_success_rate_mean", ascending=False).iloc[0]
        columns = st.columns(4)
        columns[0].metric(
            "最佳任务成功率", f"{best['task_success_rate_mean'] * 100:.2f}%", best["policy"]
        )
        columns[1].metric("高负载最佳95分位时延", f"{best_high['p95_latency_s_mean']:.3f} s")
        columns[2].metric("高负载最佳传输量", f"{best_high['total_bytes_mean'] / 1e6:.1f} MB")
        columns[3].metric("独立仿真运行", len(raw))
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "performance.png"
            plot_summary(summary, figure)
            st.image(str(figure), use_container_width=True)
        display = summary.copy()
        display["task_success_rate_mean"] *= 100
        display["task_success_rate_ci95"] *= 100
        display["total_bytes_mean"] /= 1e6
        st.dataframe(
            display[
                [
                    "arrival_rate",
                    "policy",
                    "task_success_rate_mean",
                    "task_success_rate_ci95",
                    "p95_latency_s_mean",
                    "mean_latency_s_mean",
                    "total_bytes_mean",
                    "mean_semantic_quality_mean",
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True,
            column_config={
                "arrival_rate": "到达率/个·s⁻¹",
                "policy": "策略",
                "task_success_rate_mean": "成功率/%",
                "task_success_rate_ci95": "95%CI",
                "p95_latency_s_mean": "P95时延/s",
                "mean_latency_s_mean": "平均时延/s",
                "total_bytes_mean": "传输量/MB",
                "mean_semantic_quality_mean": "语义质量",
            },
        )
        st.subheader("导出实验材料")
        download_columns = st.columns(4)
        download_columns[0].download_button(
            "原始运行 CSV", raw.to_csv(index=False), "agentcomsim-runs.csv", "text/csv"
        )
        download_columns[1].download_button(
            "汇总结果 JSON", summary.to_json(orient="records", indent=2), "agentcomsim-summary.json"
        )
        download_columns[2].download_button(
            "解析后配置 YAML",
            yaml.safe_dump(asdict(st.session_state.config), sort_keys=False),
            "agentcomsim-config.yaml",
        )
        if record_events:
            event_text = "\n".join(
                json.dumps(asdict(event), ensure_ascii=False)
                for result in st.session_state.results
                for event in result.events
            )
            download_columns[3].download_button(
                "因果事件 JSONL", event_text, "agentcomsim-events.jsonl"
            )

with trace_tab:
    st.subheader("参考任务图")
    st.markdown(
        """
每项任务包含两个并行分支，汇合后才能进入规划与控制。平台记录的不是孤立数据包，而是任务、网络与计算之间的因果关系。

```text
感知上传 ─→ 感知计算 ─┐
                      ├─→ 规划传输 ─→ 规划计算 ─→ 控制下发
工具传输 ─→ 工具计算 ─┘
```

**事件字段**：`task_id` 标识任务，`cause_id` 标识当前事件，`parent_cause_id` 指向触发原因。这样可区分链路拥塞、计算排队和上游迟滞。
"""
    )
    if "results" in st.session_state and record_events:
        rows = [asdict(event) for result in st.session_state.results for event in result.events]
        if rows:
            events = pd.DataFrame(rows)
            selected_task = st.selectbox("查看任务编号", sorted(events["task_id"].unique()))
            st.dataframe(
                events[events["task_id"] == selected_task],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("勾选“记录因果事件轨迹”并运行实验后，可在这里逐任务检查事件。")

with integration_tab:
    ns3_column, neural_column = st.columns(2, gap="large")
    with ns3_column:
        st.subheader("ns-3 数据包级桥接")
        st.write("将当前首个到达率对应的任务轨迹导出为CSV，再由外部ns-3执行逐数据包传输。")
        trace = generate_trace(config, config.arrival_rates[0], config.base_seed)
        with tempfile.TemporaryDirectory() as directory:
            scenario = export_scenario(trace, Path(directory) / "scenario.csv")
            st.download_button(
                "导出ns-3场景 CSV", Path(scenario).read_bytes(), "agentcomsim-ns3.csv", "text/csv"
            )
        st.code(
            "agentcomsim ns3 run --scenario agentcomsim-ns3.csv --output results/ns3 --ns3-dir /path/to/ns-3"
        )
        st.caption("当前为单向数据包级验证与校准桥接，不是ns-3与Python内核的闭环同步。")
    with neural_column:
        st.subheader("PyTorch 神经调度器")
        st.write("从TUS实际队列状态采样，按仿真种子划分训练、验证和测试集。")
        neural_seeds = st.number_input("训练种子数量", 5, 50, 5, 1)
        neural_epochs = st.number_input("最大训练轮数", 1, 200, 8, 1)
        neural_output = st.text_input("模型保存路径", "results/neural/scheduler.pt")
        if st.button("训练神经调度器"):
            try:
                from agentcomsim.neural.train import train_neural_scheduler

                with st.spinner("正在采样并训练，CPU环境也可运行……"):
                    metrics = train_neural_scheduler(
                        config,
                        neural_output,
                        seeds=[config.base_seed + 1000 + i for i in range(int(neural_seeds))],
                        epochs=int(neural_epochs),
                    )
                st.session_state["checkpoint_path"] = neural_output
                st.json(metrics)
                st.success("模型已保存，可在左侧选择 neural 策略并填入该路径。")
            except Exception as exc:
                st.error(f"训练失败：{exc}")

with about_tab:
    st.subheader("模型边界与复现原则")
    st.markdown(
        """
- **任务成功**：同时满足截止时间和最低语义质量，而不是仅要求消息到达。
- **公平对照**：同一负载与种子下，各策略复用完整预生成任务轨迹。
- **不确定性**：表格中的95%置信区间描述独立运行均值，不是单次任务分布。
- **TUS压缩**：仅在队列积压或期限紧张时启用；控制与结构化信息不做有损压缩。
- **当前边界**：默认内核为消息级模型；语义质量由配置系数抽象；真实无线信道和协议在环需要外部扩展。
"""
    )
    st.code(
        "pip install -e '.[all]'\nagentcomsim quickstart\nagentcomsim run -c configs/paper.yaml -o results/paper"
    )
