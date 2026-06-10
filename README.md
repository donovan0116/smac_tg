# TrustGraph + SMAC 集成骨架

这个目录提供一套用于把 TrustGraph 本体论/Graph RAG 能力接入 SMAC 训练流程的接口文档和 Python 骨架。

设计目标不是让 TrustGraph 替代 SMAC 的底层动作策略，而是把 TrustGraph 作为“慢速顶层战术推理层”：

- SMAC/MARL 算法继续负责逐 step 动作选择、训练、回放、优化。
- TrustGraph 每隔若干 step 接收一次结构化战场状态。
- TrustGraph 基于本体图谱和战术知识生成高层战术建议。
- 训练程序把高层战术作为额外输入、奖励 shaping 信号或 rule-based override 信号。

## 目录结构

```text
smac_tg/
  README.md
  docs/
    interface.md
  examples/
    train_qmix_smac.py
    train_with_trustgraph_stub.py
  smac_tg/
    __init__.py
    config.py
    qmix.py
    models.py
    state_encoder.py
    trustgraph_client.py
    tactic_planner.py
    training_adapter.py
```

## 快速使用

1. 确认 TrustGraph 已运行，并已有 `smac-onto-rag` 或类似 ontology flow。

2. 设置环境变量：

```bash
export TRUSTGRAPH_URL="http://localhost:8088/"
export TRUSTGRAPH_TOKEN="$IAM_BOOTSTRAP_TOKEN"
export TRUSTGRAPH_WORKSPACE="default"
```

3. 在你的 SMAC 训练代码里引入：

```python
from smac_tg.config import TrustGraphSMACConfig
from smac_tg.training_adapter import TrustGraphTrainingAdapter

tg_adapter = TrustGraphTrainingAdapter(
    TrustGraphSMACConfig(
        flow_id="smac-onto-rag",
        collection="smac",
        decision_interval=40,
    )
)
```

4. 在训练循环中每个 step 调用：

```python
tactical_command = tg_adapter.maybe_update_tactic(
    episode_id=episode_id,
    step=step,
    obs_summary=obs_summary,
)
```

`obs_summary` 是你从 SMAC observation 中整理出的结构化状态。格式见 [docs/interface.md](docs/interface.md)。

## 独立 QMIX 训练

当前仓库提供了一个不接入 TrustGraph 的 QMIX 训练入口，用于先完成 SMAC 底层 MARL 训练闭环：

```bash
python3 examples/train_qmix_smac.py --map-name 3m --total-timesteps 200000
```

可选参数示例：

```bash
python3 examples/train_qmix_smac.py \
  --map-name 8m \
  --total-timesteps 1000000 \
  --batch-size 32 \
  --device auto \
  --checkpoint-dir runs/qmix
```

训练实现位于 `smac_tg/qmix.py`，包含：

- RNN agent 网络，输入为局部 observation、上一动作 one-hot 和 agent id。
- QMIX monotonic mixing network，基于全局 state 生成超网络权重。
- episode replay buffer、double Q target、epsilon-greedy exploration。
- 定期 evaluation、日志输出和 checkpoint 保存。

运行前需要单独安装并配置 OxWhirl SMAC 和 StarCraft II 环境。不要用 PyPI 上同名 `smac` 自动配置库替代 OxWhirl 的 `smac.env.StarCraft2Env`。

## 重要约束

TrustGraph 不适合放在每一帧动作控制内环。推荐频率：

- 小地图：每 20 到 50 个 environment step 查询一次。
- 大地图或长 episode：每 50 到 100 个 step 查询一次。
- 训练早期：可以关闭 TrustGraph，只收集轨迹和状态摘要。
- 训练中后期：把 TrustGraph 作为高层条件输入或课程学习信号。

