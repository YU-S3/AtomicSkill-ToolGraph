## 一、当前系统结构

当前系统不是多个能够互相发消息、讨论和协商的 Agent，也不是 Planner Agent、Executor Agent、Summarizer Agent 三个独立进程。

当前实际结构是：

1. 一个 `AtomicSkillGraphSystem` 主控制器。
2. 一个可选调用 LLM 的 Atomic Planner。
3. 一个执行任务的 Benchmark Adapter。
4. 一个独立 LLM 会话形式的 Trace Extractor。
5. 一组确定性代码模块，负责 Skill、Tool、图结构、验证、统计和版本管理。
6. 一个失败分支管理器，负责隔离修改、严格 replay 和合并。

所谓“Planner Agent”“Trace Extractor Agent”“Composite Graph Proposal Agent”，主要是对不同 LLM调用职责的命名。它们不共享聊天历史，也不直接通信。

它们之间的“协作”实际是代码框架传递结构化数据：

- Planner 输出候选 Atomic Skill 顺序。
- 执行器输出 Trace。
- Trace Extractor 读取 Trace，提出语义分段。
- 代码验证分段。
- Tool Miner 读取验证后的分段。
- Composite Builder 读取验证后的 Atomic occurrence。
- Failure Branch Manager 读取失败 attempt 和成功补救轨迹。

因此更准确的说法是：

> 当前是单控制器编排多个模块，其中部分模块调用同一个模型的独立会话。模块之间通过 JSON、Trace、Registry 和文件存储协作，不存在 Agent 间自由通信。

主控制器见 [system.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/system.py:43)。

---

# 二、正式实验启动后，第一步做什么

完整小规模实验入口会依次执行三个 Benchmark：

1. ALFWorld。
2. HumanEval。
3. GSM8K。

每个 Benchmark 都运行五个条件：

- `baseline_dynamic`
- `flowevo`
- `atomic_graph_only`
- `tool_repo_only`
- `atomic_skillgraph_full`

其中前两个不是 AtomicSkillGraph 主链。

`baseline_dynamic` 和 `flowevo` 会启动 vendored FlowEvo 子进程，运行 FlowEvo 自己的执行框架。它们的结果只作为基线，不会进入我方 SkillGraph。

后三个才进入 `AtomicSkillGraphSystem`：

- `atomic_graph_only`：保留 Atomic SkillGraph，关闭独立 Tool 进化。
- `tool_repo_only`：保留 Atomic Skill 和 Tool Repository，但关闭 Composite 与 Layer-3 Insight。
- `atomic_skillgraph_full`：打开当前完整功能。

每个条件使用独立的 `data` 目录，因此不同实验条件不会互相共享 Skill 或 Tool。

这部分由 [common.py](/D:/T3S_exp/AtomicSkill-ToolGraph/experiments/common.py:27) 和 [run_all_small.py](/D:/T3S_exp/AtomicSkill-ToolGraph/experiments/run_all_small.py) 完成，不需要 LLM决定。

---

# 三、每个任务开始时，代码框架先做什么

## 第一步：把 Benchmark 样本转成统一 Task

适配器把原始任务转成统一结构：

- `task_id`
- `benchmark`
- `task_type`
- `goal`
- `context`
- `state`
- `target_effects`

例如 ALFWorld 的 heat-then-place 任务会有类似目标效果：

- `object.heated`
- `object.at_location`

HumanEval 和 GSM8K 会使用：

```text
callable.returns_expected
```

这一步不调用 LLM。

ALFWorld 的目标类型、对象、目标位置和目标 Effect 都由规则解析；HumanEval/GSM8K 的入口函数、公开测试和答案也由数据加载器提供。

---

## 第二步：从 SkillGraph 检索已有能力

Planner 把任务转成检索条件：

- 任务目标文本。
- `task_type`。
- 当前状态。
- 已知输入。
- 目标 Effect。
- 现有 Composite 和 Abstract Atomic Skill。

Registry 使用确定性评分召回候选节点。评分会考虑：

- 文本和标签匹配。
- Effect 覆盖。
- 输入是否可以绑定。
- Skill 状态。
- 历史 utility。
- task type 只作为软加分，默认不是硬限制。

如果库为空，或者没有候选达到 `planning_min_score`，就返回：

```text
start_mode = cold
no_capability_retrieved
```

此时直接走 Dynamic。

这一步主要不是 LLM做的。

---

# 四、Planner 的 LLM 到底什么时候调用

当前 Planner 只有在以下条件同时成立时才调用 LLM：

1. 已召回 Abstract Atomic Skills。
2. 任务没有提供明确的 `target_effects`。
3. 不能使用 Composite 直接形成计划。

正式的 ALFWorld、HumanEval 和 GSM8K 都提供 `target_effects`，所以这三个正式 Benchmark 的普通运行中，Planner 基本不会调用 LLM。

它们使用代码完成：

- Effect 覆盖。
- 前置能力闭包。
- Effect→Precondition 拓扑排序。
- Dynamic gap 补齐。
- 参数绑定。

因此，当前正式实验不是“让 LLM自由选择 Skill”。

Planner LLM只是自定义任务或缺少目标 Effect 时的备用路径。

## Planner 的输入

输入内容类似：

```text
Task goal: <任务目标>
Task type: <任务类型>
Available atomic skills:
- logical_id: ... | inputs: [...] | summary: ...
- logical_id: ... | inputs: [...] | summary: ...
```

## Planner 的原始英文提示词

```text
You are an atomic task planner. Given the task goal and a list of reusable atomic skills (each with summary and inputs), select the minimal sufficient ordered subset of skills that achieves the goal, and bind any known parameters from the goal text. Output ONLY a JSON object with the form {"skills": [{"logical_id": "...", "params": {...}}]} (no extra text).
```

## 中文翻译

```text
你是一个原子任务规划器。给定任务目标和一组可复用的原子技能，每个技能都包含摘要和输入，请选择能够实现目标的最小充分技能子集，并按照执行顺序排列。同时绑定能够从任务目标文本中识别出的已知参数。

只能输出下面格式的 JSON 对象，不要输出其他文字：

{"skills": [{"logical_id": "...", "params": {...}}]}
```

## Planner 输出后代码怎么处理

LLM返回：

```json
{
  "skills": [
    {
      "logical_id": "alfworld.acquire_object",
      "params": {
        "object": "mug"
      }
    }
  ]
}
```

但当前代码实际上只读取 `logical_id` 和顺序，LLM返回的 `params` 不会直接采用。

参数仍由代码的 `_bind_params()` 从 `task.context.params` 和任务文本重新绑定。这是为了防止 LLM虚构对象或错误绑定位置。

这是一个需要明确指出的实现细节：提示词要求 LLM返回参数，但当前 Planner 解析器只使用 Skill ID。

代码见 [atomic_planner.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/runtime/atomic_planner.py:27)。

---

# 五、得到计划以后，代码怎样选择 Implementation 和 Tool

Planner 只选择 Abstract Atomic Skill，不直接选择 Tool。

之后由代码执行两层解析。

第一层是 Implementation Selector。它根据：

- Atomic Skill 固定版本。
- 当前 Benchmark Harness。
- Implementation 状态。
- 参数兼容性。
- Implementation utility。

选出实现这个 Abstract Skill 的 Implementation Atom。

第二层是 Tool Resolver。它读取 Implementation 中的 `tool_bindings`，完成：

- 固定 Tool 版本解析。
- 参数映射。
- 必填参数检查。
- Tool 状态检查。

LLM看不到整个 Tool Repository，也不在所有 Tool 中自由搜索。

这解决了两个问题：

1. Tool 数量增长后，不能每次把整个仓库塞进提示词。
2. 不能让 LLM用相似名称把错误 Tool 绑定到当前 Atomic Skill。

---

# 六、执行时的三层路线

每个 Atomic 节点按照下面的顺序执行：

1. Direct。
2. Seeded。
3. Dynamic。

这不是一次整题统一决定。ALFWorld 支持在同一个环境 episode 中逐节点选择不同模式。

例如：

- Acquire 用 Direct。
- Heat 的 Direct 失败，改用 Seeded。
- Place 没有可用 Skill，使用 Dynamic。

这三个节点可以出现在同一任务中。

---

## 第一层：Direct

Direct 完全不调用 LLM。

代码先检查：

- Implementation 是否 active。
- Tool 是否已经 Admission。
- Tool 是否为 candidate/active/preferred。
- Tool 是否被 suppressed 或 retired。
- 当前输入是否能完整绑定。
- Atomic preconditions 是否满足。
- Tool utility 是否达到门槛。
- 是否有足够的 Direct 成功证据，或者 Tool 已 active 且有 Admission replay 成功证据。

通过后：

- Python Tool 直接进入 Sandbox 和官方 verifier。
- ALFWorld action-template 直接实例化为环境动作。

例如 Tool 保存：

```text
go to {heating_station}
open {heating_station}
heat {object} with {heating_station}
```

当前绑定为：

```text
object = mug 1
heating_station = microwave 1
```

代码直接生成并执行：

```text
go to microwave 1
open microwave 1
heat mug 1 with microwave 1
```

执行完成后，Node Validator 检查真正的状态 Effect，而不是只看动作有没有执行。

如果 `object.heated(mug_1)` 没有出现，Direct 就算失败。

---

## 第二层：Seeded

Seeded 会调用 LLM。

但 Seeded 不是执行 Tool。它只是把当前 Atomic Skill 的 summary、guideline 和最多两个已绑定 Tool 的正文放进 LLM上下文。

Seed context 由代码生成，格式是：

```text
[Atomic Skill] <Atomic 摘要>
  - <guideline 规则一>
  - <guideline 规则二>
[Tool] <Tool 摘要>
```text
<Tool 动作模板>
```
```

对于 Python Tool，围栏类型是 `python`；对于环境动作模板，围栏类型是 `text`。

LLM拿到这些经验后，仍然自己生成代码或选择动作。

因此：

- Seeded 成功表示“LLM在看到 Skill/Tool 经验后解决了任务”。
- 不表示 Tool artifact 被成功执行。
- 不能给这个 Tool 记录 Direct 成功。
- 只能把它作为提示经验的效果统计或后续成功轨迹来源。

这正是之前统计问题修复后的口径。

---

## 第三层：Dynamic

Dynamic 也调用 LLM，但不提供 Skill 或 Tool 上下文。

它只接收：

- 当前任务。
- 当前环境观察。
- 合法动作，或者程序任务描述。
- 先前失败反馈。

Dynamic 是冷启动和最终兜底。

---

# 七、HumanEval 和 GSM8K 中，LLM具体做什么

Code/Math 执行器让 LLM生成完整 Python 程序。

默认配置：

```yaml
max_repairs: 2
```

循环使用：

```python
for index in range(max_repairs + 1)
```

因此总预算是三次：

1. 第一次 draft。
2. 第一次 repair。
3. 第二次 repair。

这与参考文本中“每轮给三次 attempt”的想法一致，但当前只在 Code/Math 代码修复层实现，不是所有 Agent 都统一三次。

## Code/Math 系统提示词英文原文

```text
You are an expert Python programmer. Output ONLY raw Python code (no markdown fences, no explanation) that satisfies the task.
```

## 中文翻译

```text
你是一名专业的 Python 程序员。只输出能够满足任务要求的原始 Python 代码，不要使用 Markdown 代码围栏，也不要进行解释。
```

## HumanEval 的用户输入

优先直接使用 FlowEvo 任务加载器提供的原始 HumanEval prompt：

```text
<HumanEval 原始函数说明、函数签名和示例>
```

如果缺失，才使用：

```text
Task: <任务目标>
```

如果是 Seeded，则变成：

```text
<Atomic Skill 和 Tool 上下文>

Now solve this task:
<HumanEval 原始任务>
```

中文含义是：

```text
<相关 Atomic Skill 和 Tool 经验>

现在解决下面的任务：
<原始任务>
```

## GSM8K 的用户输入英文原文

```text
Question: <question>
Write raw Python only. Implement `def solve():`. Return only the final numeric answer.
```

## 中文翻译

```text
问题：<问题内容>

只编写原始 Python 代码。实现 `def solve():`。函数只返回最终数值答案。
```

## Verifier 失败后的修复提示词英文原文

```text
Previous attempt failed (<verifier feedback>). Output ONLY the corrected raw Python code.
```

## 中文翻译

```text
上一次尝试失败了，验证器反馈为：<失败信息>。只输出修正后的原始 Python 代码。
```

这里的 `<verifier feedback>` 来自代码框架：

- HumanEval 官方测试错误。
- Python 异常。
- 超时。
- GSM8K 最终数值不匹配。

LLM只负责根据错误修改代码。

是否正确，由 Sandbox 和 verifier 决定，不由 LLM自己判断。

代码见 [code_math.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/adapters/code_math.py:181)。

---

# 八、ALFWorld 中，LLM具体做什么

ALFWorld 的 Seeded 和 Dynamic 都是逐步选择动作。

每一步 LLM输入：

- 当前局部目标。
- Skill 上下文，Dynamic 时为空。
- 已经检查过的位置。
- 最近十步动作和观察。
- 当前 observation。
- 当前全部 admissible commands。

LLM必须从合法动作列表中选一个。

## ALFWorld 系统提示词英文原文

```text
You are an expert household robot completing tasks in a virtual home. You will be given a task goal, the current observation, and a list of valid actions.

At each step:
1. Think about what you need to do next and why.
2. Choose exactly ONE action from the valid actions list.

Common task patterns:
- pick_and_place: go to object location -> take it -> go to destination -> put it
- pick_clean_then_place: go to object -> take -> go to sinkbasin -> clean -> go to dest -> put
- pick_heat_then_place: go to object -> take -> go to microwave -> heat -> go to dest -> put
- pick_cool_then_place: go to object -> take -> go to fridge -> cool -> go to dest -> put
- examine_in_light: go to object -> take -> go to lamp -> use lamp
- pick_two: find first object -> take -> go to dest -> put -> find second -> take -> go to dest -> put

Format your response as:
Think: <your step-by-step reasoning>
Act: <the exact action from the valid actions list>
```

## 中文翻译

```text
你是一名专业的家务机器人，需要在虚拟家庭环境中完成任务。系统会提供任务目标、当前观察以及合法动作列表。

每一步：
1. 思考下一步需要做什么以及为什么。
2. 从合法动作列表中准确选择一个动作。

常见任务模式：
- 拿取并放置：前往物品位置 -> 拿取 -> 前往目标位置 -> 放置
- 拿取、清洗并放置：前往物品 -> 拿取 -> 前往水槽 -> 清洗 -> 前往目标位置 -> 放置
- 拿取、加热并放置：前往物品 -> 拿取 -> 前往微波炉 -> 加热 -> 前往目标位置 -> 放置
- 拿取、冷却并放置：前往物品 -> 拿取 -> 前往冰箱 -> 冷却 -> 前往目标位置 -> 放置
- 在灯光下检查：前往物品 -> 拿取 -> 前往灯 -> 使用灯
- 拿取两个物品：找到第一个物品 -> 拿取 -> 前往目标位置 -> 放置 -> 找到第二个物品 -> 拿取 -> 前往目标位置 -> 放置

输出格式：
Think: <逐步思考>
Act: <合法动作列表中的准确动作>
```

## 每一步用户提示词模板英文原文

```text
Task: <task goal>

Relevant experience:
<Atomic Skill / Tool seed context>

Structured search state:
Already checked: <locations>. Do not navigate to these receptacles again while searching.

Recent actions:
  > <action>
    <observation>

Current observation:
<current observation>

Valid actions (<count>):
  <action 1>
  <action 2>
  ...

Choose ONE action from the valid actions list above:
```

其中 `Relevant experience` 只在 Seeded 时出现；`Structured search state` 只有已经检查过位置时出现。

## 中文翻译

```text
任务：<当前任务或当前 Atomic 节点目标>

相关经验：
<Atomic Skill 和 Tool 上下文>

结构化搜索状态：
已经检查过：<位置列表>。搜索期间不要再次前往这些容器。

最近动作：
  > <动作>
    <环境观察>

当前观察：
<当前环境观察>

合法动作，共 <数量> 个：
  <动作一>
  <动作二>
  ...

从上面的合法动作中选择一个：
```

## LLM输出后代码还会做什么

代码按下面的顺序解析：

1. 找 `Act:` 行。
2. 尝试与合法动作精确匹配。
3. 尝试从 LLM输出中寻找合法动作子串。
4. 尝试逐行匹配。
5. 如果都失败，选择合法动作列表中的第一个；没有合法动作时使用 `look`。

因此，环境真正执行的动作一定经过代码映射，不会直接执行 LLM任意文本。

此外还有确定性防循环设计：

- 连续三次 `Nothing happens` 后结束当前尝试。
- 12 步以后检测 `ABCABCABC` 动作循环。
- Acquire 搜索时记录已经检查过的位置。
- 如果 LLM又选择已检查位置，代码优先换成另一个未检查位置。
- 每个动作后立即更新状态并检查 Atomic Effect。
- Effect 满足后立即结束当前节点，不继续浪费动作。

代码见 [alfworld.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/adapters/alfworld.py:925)。

---

# 九、任务成功后，LLM还会做什么

任务成功之后，不是简单把完整轨迹直接保存成一个 Skill。

当前增加了一个独立的 Trace Extractor LLM会话。

这个 Extractor 与任务执行 LLM使用相同模型和 API 配置，但会通过 `fork()` 创建独立客户端：

- 独立会话。
- 独立上下文。
- 温度 `0.1`。
- `thinking=enabled`。
- `reasoning_effort=low`。
- 结构化 JSON 输出。
- 最长读取超时 600 秒。
- 不继承任务执行时的聊天历史。

它的职责不是决定最终 Skill，而是提出：

- 哪些动作属于同一个高层 Atomic occurrence。
- 哪些动作是探索、循环或恢复。
- occurrence 的语义名称。
- 参数的语义角色。
- 可能的核心 Effect 名称。

---

## Trace Extractor 的输入

代码先把成功 Trace 转成结构化事件，每个事件包含：

- `event_index`
- 动作
- 动作参数
- before state
- after state
- positive effects
- negative effects
- mode
- node ref
- tool ref
- accepted
- state_changed

还会输入当前已经验证过的 Atomic catalog：

```json
{
  "canonical_name": "acquire_object",
  "logical_id": "alfworld.acquire_object",
  "summary": "...",
  "inputs": ["object", "object_location"],
  "effects": ["agent.holds"],
  "semantic_aliases": [],
  "support_count": 2
}
```

这使后面的独立 Extractor 会话可以知道已有能力，但这不是 Agent 间通信，而是框架显式把当前 Registry 摘要放进输入。

---

## Trace Extractor 英文提示词原文

```text
You are the Trace Extractor Agent in a reusable capability-learning system.
Your input is a successful task trace already normalized into structured events. Each event has an
action, parameters, before/after state, positive/negative effects, execution mode, and acceptance.

Infer a minimal set of discrete high-level Atomic capability occurrences that explains task success.
Do not copy every state change into the workflow. Group low-level actions and intermediate state
changes into the capability whose stable effect they implement. Remove exploration, repeated checks,
failed attempts, loops, recovery detours, and duplicate occurrences unless they are causally required.
An Atomic occurrence must have one coherent intent, a contiguous event range, explicit parameter
roles, and one or more stable effects observed at the end of its range. Preserve evidence: never invent
an action, entity, parameter, state, effect, tool, or skill.

Name intent by reusable capability semantics, never by the concrete object, receptacle, task wording,
or incidental navigation (use acquire_object, heat_object, clean_object, cool_object, place_object,
open_container, etc.). If known_atomic_contracts contains an equivalent validated Effect contract,
reuse its canonical_name. A phase may contain internal navigation or container operations, but expose
only the phase's externally meaningful core Effect. In effect_predicates return predicate names only,
without arguments (for example "agent.holds", not "agent.holds(mug_1)").

Generalization is mandatory. Describe the operation shared by different entities and environments,
not the concrete episode. The intent must still be correct after every observed object and location is
replaced by another value of the same semantic role. Good: acquire_object (拿取物品), heat_object
(加热物品), place_object (放置物品). Bad: take_out_a_banana, acquire_mug_from_cabinet,
transport_mug_to_microwave, place_apple_in_fridge. Do not include object classes, instance numbers,
source/destination names, appliance names, or a sequence of multiple capabilities in intent. Keep such
details only in parameter_roles. Generalize at the stable Effect level rather than using a vague label
such as handle_object: different core Effects remain different Atomic capabilities.

Return ONLY JSON:
{
  "phases": [
    {
      "phase_id": "phase_000",
      "intent": "snake_case_capability_name",
      "event_start": 0,
      "event_end": 3,
      "parameter_roles": {"object": "observed value"},
      "effect_predicates": ["object.heated"],
      "rationale": "why this range is one capability"
    }
  ],
  "discarded_event_indices": [4, 5],
  "discard_reasons": {"4": "exploration|duplicate|loop|failed_attempt|recovery"},
  "workflow_summary": "short semantic summary"
}

Requirements: ranges must be valid, non-overlapping and ordered; prefer the fewest sufficient phases;
intermediate placement inside heating/cleaning/cooling belongs inside that transformation phase; the
final delivery placement is a separate phase; do not infer correctness merely from action wording.
```

## 中文翻译

```text
你是一个可复用能力学习系统中的轨迹提取 Agent。

输入是一条已经标准化为结构化事件的成功任务轨迹。每个事件都包含动作、参数、动作前后状态、正负状态变化、执行模式以及动作是否被接受。

请推断出能够解释任务成功的最小离散高层 Atomic 能力实例集合。

不要把每一个状态变化都复制进工作流。应把低层动作和中间状态变化归入它们最终实现的稳定能力 Effect。除非在因果上确实必要，否则应删除探索、重复检查、失败尝试、循环、恢复绕路和重复能力实例。

一个 Atomic occurrence 必须具有一个连贯意图、连续事件区间、明确参数角色，以及在区间末尾真实观察到的一个或多个稳定 Effect。必须保留证据，不能虚构动作、实体、参数、状态、Effect、Tool 或 Skill。

应使用可复用能力语义命名意图，不能使用具体物品、容器、任务原句或偶然导航。例如使用 acquire_object、heat_object、clean_object、cool_object、place_object、open_container。

如果 known_atomic_contracts 中已经有等价且验证过的 Effect Contract，应复用它的 canonical_name。

一个 phase 可以包含内部导航或容器操作，但对外只暴露这个 phase 真正有意义的核心 Effect。effect_predicates 只返回谓词名称，不带参数。例如返回 agent.holds，而不是 agent.holds(mug_1)。

必须进行泛化。描述不同实体和环境共享的操作，而不是本次具体 episode。把观察到的对象和位置替换为相同语义角色的其他值后，intent 仍然必须成立。

正确示例：acquire_object、heat_object、place_object。
错误示例：take_out_a_banana、acquire_mug_from_cabinet、transport_mug_to_microwave、place_apple_in_fridge。

intent 中不能包含对象类别、实例编号、源位置、目标位置、设备名称，也不能把多个能力序列塞进同一个 intent。这些具体信息只能保存在 parameter_roles 中。

必须在稳定 Effect 层泛化，不能使用 handle_object 这类过于模糊的名称。不同核心 Effect 必须保持为不同 Atomic 能力。

只能输出指定 JSON。

区间必须合法、互不重叠并保持顺序。优先选择最少的充分 phase。加热、清洗、冷却过程中的中间放置属于相应转换能力；最终交付放置必须是单独 phase。不能仅根据动作文字判断执行正确。
```

提示词见 [semantic_extractor.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/atomicizer/semantic_extractor.py:27)。

---

# 十、Trace Extractor 输出后，不是直接生成 Skill

这是当前设计最关键的限制之一。

LLM输出只是提案。代码随后重新检查：

1. `event_start/event_end` 是否在真实 Trace 范围内。
2. phase 是否重叠。
3. 参数值是否真实出现在任务或 Trace 中。
4. before/after 是否真的产生了稳定 Effect。
5. LLM声明的 Effect 是否和状态差分一致。
6. 哪一个事件才是真正的 Effect producer。
7. 是否存在多余导航、循环和恢复动作。
8. 是否能够从原始事件形成安全 replay slice。
9. phase 是否真的在任务目标的后向因果链中。

最终 Atomic Skill 的 logical ID 不采用 LLM自由生成的名称，而是根据代码验证后的 Effect family 得到。

例如 LLM提议：

```text
acquire_and_transport_mug_to_microwave
```

但状态证据只证明：

```text
agent.holds(object)
```

最终 canonical name 会是：

```text
acquire_object
```

LLM原始命名只作为 alias 和审计证据保存。

这是为了解决 LLM容易把：

- 具体对象写进 Skill 名。
- 把导航和拿取混成一个能力。
- 把多个稳定 Effect 混成一个节点。
- 根据动作文字虚构状态变化。

---

# 十一、Atomic Skill 是代码怎样建立的

代码根据验证后的 Effect 构造：

- `logical_id`
- summary
- inputs
- outputs
- preconditions
- effects
- validator
- guideline
- metadata
- statistics
- status

例如：

```text
alfworld.heat_object@1.0.0
```

Atomic Skill 的身份标准是稳定 Effect，不是动作数量。

一个 Atomic Skill 可以包含：

```text
go to microwave
open microwave
heat mug
```

因为这三个动作共同实现一个主要 Effect：

```text
object.heated
```

但不能把：

```text
拿取 + 加热 + 最终放置
```

全部放进一个 Atomic Skill，因为它们有不同的稳定主要 Effect，可以独立验证和失败归因。

如果一个候选真的包含多个独立正 Effect，SplitScore 达到阈值后，代码会拆成多个 Atomic 节点。

这一部分不由 LLM直接决定。

---

# 十二、Tool 是怎样产生的

Tool 不是由专门的 Tool Agent 重新编写。

Tool 的原料来自成功执行产物。

## ALFWorld Tool

来源是成功 Atomic occurrence 中的真实动作。

代码执行：

1. 取经过因果切片的真实动作。
2. 删除探索动作和无信息动作。
3. 删除连续重复。
4. 把具体实体替换为 slot。
5. 构造 replay case。
6. 保存来源 Trace 和原任务位置。
7. 生成 `action_template` Tool Skeleton。

例如真实轨迹：

```text
go to microwave 1
open microwave 1
heat mug 1 with microwave 1
```

转换为：

```text
go to {heating_station}
open {heating_station}
heat {object} with {heating_station}
```

“动作由谁产生”与“Tool 由谁生成”要分开：

- Dynamic/Seeded 动作通常是 LLM生成的。
- 哪些动作属于 Tool、怎样切片、怎样参数化、是否删除探索、Tool ID 和接口是什么，由代码决定。

## Code Tool

设计上，成功 Python 代码会经过 AST：

- 完整入口函数形成主 Tool。
- 从调用图中最多提取两个可达 helper。
- 函数参数形成 Tool signature。
- 成功测试形成 replay tests。

这部分也是代码完成，不是 LLM再次总结。

代码见 [compiler_adapter.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/tools/compiler_adapter.py:89)。

---

# 十三、当前 Code/Math 学习链有一个必须说明的实际限制

当前真实配置中，Trace Extractor 采用 fail-closed。

HumanEval/GSM8K 的成功 Trace 目前主要保存：

- `candidate_code`
- `attempts`
- 两个合成 state snapshot

但没有像 ALFWorld 那样保存逐动作 `trace.actions`。

`SemanticExtractorAgent.build_structured_events()` 只从 `trace.actions` 构造事件。因此真实 Code/Math 成功后通常得到：

```text
no_events_or_llm
```

真实配置又不允许回退到旧的 `detect_code_boundaries()`，所以结果是：

- Code/Math 成功任务本身正常保存。
- 但本次成功不会形成 Atomic segments。
- 没有 Atomic segments 就不会继续挖掘长期 Code Tool。
- 也不会建立新的 Code/Math Composite。

Mock 测试模式允许 legacy fallback，所以 toy code smoke 能形成 Code Skill/Tool；真实 HumanEval/GSM8K 当前主链则会 fail-closed。

因此，当前代码中“Code Tool AST Miner 已实现”和“真实 HumanEval/GSM8K 已经能从零持续进化 Code Tool”不是同一个结论。

前者存在，后者目前被真实 Extractor 的事件输入缺口阻断。

这也是当前实际实现中最需要在论文解释或后续修复的误差之一。

---

# 十四、Tool Admission 是怎么做的

Tool Skeleton 不能直接用于 Direct。

代码执行 Admission：

## Python Tool

检查：

1. AST 语法。
2. 禁止危险 import 和调用。
3. 检查空函数和常量返回。
4. 入口函数是否存在。
5. Sandbox replay tests。
6. 第二次运行，检查确定性。
7. 结构哈希去重。

## ALFWorld action-template

检查：

1. 参数是否存在。
2. Tool 使用的 slot 是否全部声明。
3. 声明参数是否都实际使用。
4. 模板中是否残留 `mug 1`、`cabinet_2` 等具体实例。
5. 动作是否非空、长度是否合法。
6. 在来源任务上执行环境 replay。
7. 最终是否满足预期 Effect。

Admission 通过后是 `candidate`，不是立即 active。

默认需要：

```yaml
candidate_min_support: 3
```

即达到三条支持证据，并具有实际成功或 Admission replay 证据后，才晋升 active。

Direct Gate 默认要求：

```yaml
direct_min_utility: 0.5
direct_min_success: 3
```

Admission 和 Lifecycle 全部是代码，不调用 LLM。

---

# 十五、Composite Graph 是怎样产生的

至少存在两个经过验证的 Atomic occurrence 时，代码才尝试建立 Composite。

这时会调用第二个语义 LLM步骤，即 Composite Graph Proposal。

## 输入

输入包括：

- 任务目标。
- task type。
- target effects。
- 每个经过验证的 occurrence：
  - phase ID
  - Skill ref
  - 事件范围
  - 参数
  - preconditions
  - effects
  - 实际 Tool refs

## Composite Graph Prompt 英文原文

```text
You are the Composite Graph Proposal Agent. You receive a successful task goal and
code-validated Atomic occurrences. Each occurrence includes its validated skill reference, event range,
parameters, preconditions, effects, and available tool references.

Propose the minimal reusable capability/tool-reference composition that achieves the task. Use only the
provided phase IDs, skill refs and tool refs. Preserve parameter roles. Prefer causal dependencies over
raw temporal adjacency. Do not include exploration, loops, retries, recovery, or implementation-internal
operations. Add an implicit dependency only when it is semantically necessary but not already explicit
from Effect-to-Precondition or parameter data flow.

Return ONLY JSON:
{
  "ordered_phase_ids": ["phase_000", "phase_001"],
  "summary": "what reusable method this Composite represents",
  "implicit_dependencies": [
    {"source_phase_id": "phase_000", "target_phase_id": "phase_001",
     "relation": "requires_skill", "reason": "short reason"}
  ],
  "tool_plan": [
    {"phase_id": "phase_000", "skill_ref": "skill://...@...", "tool_refs": ["tool://...@..."]}
  ]
}
```

## 中文翻译

```text
你是 Composite Graph 提案 Agent。输入是一条成功任务目标和经过代码验证的 Atomic occurrence。每个 occurrence 都包含已经验证的 Skill 引用、事件范围、参数、前置条件、Effect 和可用 Tool 引用。

提出能够完成任务的最小可复用能力与 Tool 引用组合。

只能使用输入中提供的 phase ID、Skill ref 和 Tool ref。保留参数的语义角色。优先使用因果依赖，而不是仅依据时间相邻关系。

不要加入探索、循环、重试、恢复过程或 Implementation 内部操作。

只有在某项依赖在语义上确实必要，而且不能从 Effect→Precondition 或参数数据流中直接得到时，才能增加隐式依赖。

只能输出指定 JSON。
```

## LLM输出不会直接成为图

代码随后限制：

- LLM提出的顺序只有与真实执行顺序一致时才采用。
- LLM不能把未执行节点加入图。
- LLM不能凭空修改 Atomic ref。
- `summary` 只作为审计证据。
- Composite 的公开 summary 由 Atomic ref 顺序生成。
- `tool_plan` 不负责最终 Tool Binding。
- 隐式依赖必须连接真实 phase，且方向符合执行顺序。

代码自动建立：

- `contains`
- `next`
- 可验证的 `data_flow`
- Effect→Precondition 的 `requires_skill`
- Implementation→Abstract 的 `implements`
- Implementation→环境的 `requires_environment`

单条成功轨迹生成的 Composite 是 `draft`。默认至少两条独立成功 Trace 支持后才成为 `active`。

---

# 十六、成功之后哪些步骤完全不是 LLM做的

下面这些全部由代码框架完成：

1. 状态 before/after 差分。
2. Effect 提取和规范化。
3. 参数值是否真实出现的检查。
4. 因果事件切片。
5. exploration、loop、duplicate 的删除验证。
6. Atomic I/O 构造。
7. precondition 构造。
8. validator 构造。
9. SplitScore。
10. Atomic 对齐、复用或新增。
11. Tool 动作参数化。
12. Python AST Tool 提取。
13. Tool Admission。
14. Tool 生命周期。
15. Implementation Binding。
16. Composite `contains`、`next`、`data_flow` 等边。
17. 图校验。
18. utility 和 support 统计。
19. Tool/Skill 版本号。
20. suppressed、retired 和 recommended pointer。
21. Layer-3 Insight 聚合。

当前 Layer-3 Insight 不是 LLM总结，而是规则统计：

- 常见位置词频。
- station 词频。
- 重复动作。
- 搜索优先级。
- 常见 pitfall。

---

# 十七、失败后现在如何处理

失败分为两类。

## 第一类：基础设施失败

例如：

- API 超时。
- 网络错误。
- Provider 空响应。
- 连续 LLM错误。

这类错误标记为：

```text
failure_type = llm_error
```

它不会：

- 给 Tool 记失败。
- 给 Atomic 记失败。
- 触发 Failure Branch。
- 产生负迁移证据。
- 生成失败 Skill。

因为 API故障不是方法失败。

当前 Provider 层最多：

- 初次请求。
- 两次短暂故障重试。

如果仍失败，决策层停止当前环境 attempt。

然后任务级最多从初始状态重跑两次，所以环境任务最多有三个完整 episode attempt。

这不是“工具调用轮次超限”。它区分了：

- API基础设施重试。
- 任务执行动作预算。
- Code verifier 修复预算。
- Atomic 节点 fallback。

ALFWorld 的正常动作受 `max_steps` 控制；达到步数预算后记任务失败并进入下一任务，不会让整个实验异常退出。

---

## 第二类：任务或节点失败

代码首先按实际 attempt 记账。

例如 Direct Tool 失败：

- 旧 Tool 的 `call_count + 1`。
- `failure_count + 1`。
- `direct_failure_count + 1`。
- `consecutive_failures + 1`。

之后 Seeded 或 Dynamic 即使成功，也不能覆盖旧 Tool 的失败，更不能给旧 Tool 记成功。

Atomic 有两套统计：

- 节点级 `failure_count`。
- 任务级 `task_failure_count`。

同一个任务中，同一 logical Skill 即使出现多次，只产生一次任务级失败证据，避免重复计数。

---

# 十八、Direct 失败但 Seeded/Dynamic 成功后怎样进化 Tool

这部分主要不是 LLM完成的。

代码根据真实失败和补救轨迹执行：

1. 找到失败 Direct attempt。
2. 找到后续第一个通过当前 Atomic Effect 的 Seeded/Dynamic attempt。
3. 只截取成功补救 attempt 对应的动作范围。
4. 根据核心 Effect 做因果切片。
5. 删除探索、循环和无关动作。
6. 参数化具体实体。
7. 生成 Tool patch。
8. 比较旧 Tool 和新 Tool 的步骤差异。
9. 复制 main SkillGraph 和 Tool Repository 到独立分支。
10. 在分支中注册新 Tool 和新 Implementation。
11. 在来源失败任务上严格 Direct replay。
12. 禁止 Seeded fallback。
13. 禁止 Dynamic fallback。
14. 运行 Tool Admission。
15. 验证分支 SkillGraph。
16. 确认 main bank 在 replay 期间没有被其他操作修改。
17. 全部通过后才合并 main。
18. 建立新旧 Implementation 的 `supersedes`。
19. main 图再次验证。
20. 失败时恢复推荐指针。

环境 Tool 修复的 executable body 是从成功补救动作中确定性提取的，不是让 LLM根据一句错误描述重新发明。

---

# 十九、Seeded 失败但 Dynamic 成功后怎样进化 Atomic Skill

这一路会再次调用 LLM，但不是让 LLM自由改写 Atomic Contract。

代码先从 Dynamic 成功动作中提取参数化补救模板，然后生成 guideline：

```text
仅当常规执行失败且核心 Effect 尚未满足时，使用经原任务严格重放验证的参数化补救模板：
<step 1> -> <step 2> -> ...
```

接着构造 Seeded 上下文：

```text
[Candidate Atomic Skill] <candidate summary>
  - <原 guideline>
  - <新补救 guideline>
[Current Task Bindings]
  - $inputs.object=<value>
  - $inputs.target_location=<value>
```

随后仍使用 ALFWorld 的同一套系统提示词和逐步动作提示词，让 LLM在候选 guideline 条件下重放原任务。

合并条件不仅是任务成功，还要求：

- 当前 Atomic Effect 真正满足。
- 只使用 Seeded，没有转 Dynamic。
- replay 动作确实遵循候选补救模板。
- 分支图合法。
- main bank 在 replay 期间未变化。

因此这里 LLM的职责是“在候选经验条件下执行一次”，不是判断是否合并。

是否合并由代码门禁决定。

---

# 二十、完全失败、没有成功补救轨迹时做什么

此时系统不会让 LLM凭失败文本生成一个可合并 Tool。

代码仍会建立独立 branch，并生成 shadow 候选副本：

- Atomic 候选加入失败观察和 guard guideline。
- Tool 候选加入失败回归案例。
- Composite 候选记录顺序失败和待验证规则。
- manifest 标记 `awaiting_success_evidence`。
- `merge_allowed = false`。

main 不会改变。

原因是失败轨迹只能证明“原方法不行”，不能证明“某个修改一定正确”。

这遵守：

```text
Failure proposes; successful replay admits.
```

即：

```text
失败只提出候选；成功 replay 才允许进入主版本。
```

失败分支见 [branch_repair.py](/D:/T3S_exp/AtomicSkill-ToolGraph/src/atomic_skillgraph/evolution/branch_repair.py:31)。

---

# 二十一、SkillGraph 是怎样从零长出来的

初始时：

- 没有 Abstract Atomic Skill。
- 没有 Implementation。
- 没有 Composite。
- 没有 Tool。
- 图中没有能力节点和能力边。

第一个 ALFWorld 任务：

1. Planner 检索为空。
2. 使用 Dynamic LLM逐步完成任务。
3. 成功 Trace 保存。
4. Trace Extractor LLM提出语义 phase。
5. 代码验证并做因果切片。
6. 创建 Abstract Atomic Skill。
7. 代码从成功动作中产生 Tool Skeleton。
8. Tool 通过 Admission 后成为 candidate。
9. 创建 Implementation，绑定 Abstract 与 Tool。
10. 两个以上 Atomic occurrence 形成 draft Composite。

后续独立成功 Trace：

1. 新提取的 Atomic Contract 与已有节点对齐。
2. 等价时复用原节点，增加 support。
3. 同结构 Tool 增加 support。
4. Tool 达到支持门槛后 active。
5. Composite 达到两条独立轨迹后 active。
6. 后续任务才开始出现稳定 Warm/Seeded/Direct。

所以它不是：

```text
LLM先设计 SkillGraph，然后执行。
```

而是：

```text
先执行成功任务
→ LLM提出语义分段
→ 代码验证
→ 代码建立 Atomic/Tool/Implementation/Composite
→ 后续任务复用
→ 失败后分支修复
→ 严格 replay 后再合并
```

---

# 二十二、冻结 replay 时发生什么

在线进化完成后，冻结评估会：

1. 复制每个条件的 `skill_graph` 和 `tools`。
2. 恢复该条件自己的 feature flags。
3. 设置：

```text
freeze_skills = true
```

4. replay 前计算整个 bank 的 SHA-256。
5. 使用冻结库重新运行 train 或 hold-out test。
6. 允许 Planner、Direct、Seeded、Dynamic 正常执行。
7. 禁止：
   - 成功学习。
   - 失败学习。
   - Tool 统计更新。
   - Skill 统计更新。
   - Failure Branch。
   - Lifecycle 修改。
8. replay 后重新计算 SHA-256。
9. 哈希不同则评估直接报错。
10. replay 前还会运行 Graph Validator。

冻结模式仍可能调用 LLM，因为 Seeded/Dynamic 仍需要模型；冻结的是 Skill/Tool bank，不是模型。

代码见 [run_evolve_eval.py](/D:/T3S_exp/AtomicSkill-ToolGraph/experiments/run_evolve_eval.py:111)。

---

# 二十三、把所有 LLM代劳部分最终归纳起来

当前我方主链中，LLM只负责下面五类事情。

1. 可选 Atomic 规划。

   只在没有明确 target Effect 时使用。正式三个 Benchmark 基本不触发。

2. HumanEval/GSM8K 完整代码生成和 verifier 反馈修复。

3. ALFWorld Seeded/Dynamic 的逐步动作选择。

4. 成功 ALFWorld Trace 的高层语义 phase 提案。

5. 多 Atomic occurrence 的 Composite 语义和隐式依赖提案。

另外，Seeded 失败、Dynamic 成功后的 Atomic guideline 严格 replay 会再次调用第3类 ALFWorld执行 LLM，但不会增加新的自由改写提示词。

LLM不负责：

- 最终参数绑定。
- Effect 的真实性。
- Atomic identity。
- Tool 切片。
- Tool 参数化。
- Tool Admission。
- 是否记录 Tool 成功。
- 是否合并失败补丁。
- Graph 边的最终合法性。
- 版本号。
- 生命周期。
- 统计。
- 冻结审计。

---

# 二十四、当前实际设计最准确的结论

当前系统可以理解为：

> 一个由确定性代码控制的集中式执行与进化框架。LLM负责生成程序、选择环境动作，以及对成功轨迹提出语义抽象建议；代码框架负责验证事实、建立 SkillGraph、生成 Tool、管理统计、隔离失败修改、执行严格 replay 并决定是否合并。

它不是三个互相通信的 Agent，也没有一个自由的 Summarizer Agent 根据成功或失败自行执行 `add/delete/keep/edit`。

当前进化操作实际是：

- 成功轨迹：确定性 add/reuse/split。
- 独立支持增加：更新 evidence。
- Tool 结构变化：新版本。
- Direct 失败但补救成功：分支 Tool patch。
- Seeded 失败但 Dynamic 成功：分支 Atomic guideline patch。
- 无正证据：shadow 候选，不合并。
- 图或 replay 失败：reject/rollback。
- 冻结评估：完全禁止写入。

当前最大的实现限制是：

> ALFWorld 的“执行—语义提取—Atomic—Tool—Composite—失败修复”主链已经接通；HumanEval/GSM8K 虽然有代码生成、verifier、AST Tool Miner 和 Direct 执行模块，但真实成功 Trace 尚未转换为 Extractor 可消费的结构化代码事件，因此真实 Code/Math 的长期 Skill/Tool 从零进化目前会被 fail-closed 阻断。

这部分如果用于论文，必须明确说明或继续修复，不能把 toy/mock 中的 Code Tool 进化结果直接当成真实 HumanEval/GSM8K 已经完成的实现。