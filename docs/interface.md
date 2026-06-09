# TrustGraph 与 SMAC 接口文档

## 1. 系统定位

SMAC 的常规训练流程通常是：

```text
SMAC env observation -> policy network -> action -> env step -> replay/update
```

接入 TrustGraph 后建议变成：

```text
SMAC env observation
  -> 状态摘要器
  -> TrustGraph triples / Graph RAG
  -> 高层战术 JSON
  -> policy condition / reward shaping / action mask hint
  -> policy network
  -> action
```

TrustGraph 只负责顶层战术推理，不直接输出每个 agent 的底层动作。

## 2. 需要的 TrustGraph 功能

### Ontology Management

用于定义 SMAC 本体，例如：

- `BattleState`
- `AllyUnit`
- `EnemyUnit`
- `UnitType`
- `MapRegion`
- `Threat`
- `TacticalOption`
- `FocusFireAction`
- `RetreatAction`
- `KiteAction`
- `RegroupAction`

推荐关系：

- `hasAllyUnit`
- `hasEnemyUnit`
- `nearEnemy`
- `threatens`
- `hasLowHealthUnit`
- `hasPriorityTarget`
- `suggestsTactic`
- `controlsRegion`

推荐数据属性：

- `timeStep`
- `health`
- `shield`
- `energy`
- `cooldown`
- `x`
- `y`
- `distance`
- `weaponRange`
- `unitType`
- `threatScore`

### Graph Storage / Triples

用于保存当前战场状态和战术知识图谱。

在线训练时，建议由 Python 程序直接把 observation 转成 triples，再通过 `api.bulk().import_triples(...)` 写入 TrustGraph。

### Graph RAG

用于回答顶层战术问题，例如：

```text
Given the current battle state, should allied units focus fire, retreat, kite, regroup, flank, or hold position?
Return the tactical option, priority target, controlled region, and reason.
```

### Text Completion

用于把 Graph RAG 的自然语言回答规整成训练程序可消费的 JSON。

### Library + Ontology Flow

用于导入长期战术文档，例如：

- SMAC 地图经验
- 兵种克制关系
- 集火规则
- 撤退规则
- 风筝规则
- 编队规则

这些材料可以用 `onto-rag` flow 抽取成图谱知识。

## 3. 推荐 CLI 初始化命令

设置 CLI 环境：

```bash
cd /Users/mac/trustgraph-compose/runtime
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
set -a; . ./.env; set +a
export TRUSTGRAPH_TOKEN="$IAM_BOOTSTRAP_TOKEN"
export TRUSTGRAPH_URL="http://localhost:8088/"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
```

导入 SMAC 本体：

```bash
cli-env/bin/tg-put-config-item \
  --type ontology \
  --key smac \
  --stdin < smac_ontology_trustgraph.json
```

启动 SMAC ontology flow：

```bash
cli-env/bin/tg-start-flow \
  -n ontology \
  -i smac-onto-rag \
  -d "SMAC Ontology RAG" \
  --param llm-model=deepseek-v4-flash \
  --param llm-rag-model=deepseek-v4-flash
```

导入战术文档：

```bash
cli-env/bin/tg-add-library-document \
  --identifier "smac://docs/tactics-v1" \
  --name "SMAC Tactical Doctrine" \
  --description "High-level SMAC tactical rules" \
  -k text/markdown \
  --tags smac,tactics,ontology \
  smac_tactics.md
```

处理战术文档：

```bash
cli-env/bin/tg-start-library-processing \
  -i smac-onto-rag \
  -d "smac://docs/tactics-v1" \
  --id "smac-tactics-processing-001" \
  --collection smac \
  --tags smac,tactics
```

手动查询 Graph RAG：

```bash
cli-env/bin/tg-invoke-graph-rag \
  -f smac-onto-rag \
  -C smac \
  -q "Given the current battle state, what high-level tactic should allied units execute next?" \
  --no-streaming \
  -e 20 \
  --triple-limit 10 \
  -s 80 \
  -p 2 \
  --edge-limit 10
```

## 4. Python 输入接口

训练程序需要提供 `obs_summary`，推荐格式：

```python
obs_summary = {
    "map_name": "3m",
    "allies": [
        {
            "id": "ally_0",
            "unit_type": "Marine",
            "health": 35.0,
            "shield": 0.0,
            "x": 12.4,
            "y": 8.2,
            "weapon_cooldown": 3.0,
            "is_alive": True,
        }
    ],
    "enemies": [
        {
            "id": "enemy_0",
            "unit_type": "Marine",
            "health": 18.0,
            "shield": 0.0,
            "x": 14.1,
            "y": 8.8,
            "is_alive": True,
        }
    ],
    "relations": [
        {
            "subject": "ally_0",
            "predicate": "nearEnemy",
            "object": "enemy_0",
            "distance": 1.8,
        }
    ],
}
```

## 5. Python 输出接口

TrustGraph 适配器输出 `TacticalCommand`：

```python
{
    "tactic": "focus_fire",
    "priority_target": "enemy_0",
    "confidence": 0.73,
    "reason": "Enemy enemy_0 is low health and close to allied units.",
    "raw_answer": "..."
}
```

推荐战术枚举：

- `focus_fire`
- `kite`
- `retreat`
- `regroup`
- `flank`
- `hold_position`
- `push`

## 6. 与训练算法的结合方式

### 方式 A：作为 policy condition

把 `tactic` 编码成 one-hot 或 embedding，拼接到每个 agent 的 observation。

### 方式 B：作为 reward shaping

当 policy 行为符合高层战术时增加小奖励，例如集火 priority target。

### 方式 C：作为 action mask hint

不强制动作，只降低明显违背战术的动作概率。

### 方式 D：作为 curriculum signal

训练早期使用 TrustGraph 建议更多；后期逐步降低权重。

## 7. 延迟建议

不要每 step 查询 TrustGraph。推荐：

- `decision_interval=40`
- 每次只上传当前窗口摘要，不上传完整历史。
- Graph RAG 参数保持小规模：`entity_limit=20`、`triple_limit=10`、`max_subgraph_size=80`、`edge_limit=10`。

