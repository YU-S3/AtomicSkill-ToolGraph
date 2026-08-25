# AtomicSkillGraph：面向集中式 Skill/Tool 联合自进化的原子化模块完整设计文档

**文档版本：** v2.0 Design Freeze  
**日期：** 2026-08-21  
**前序版本：** `FedAtomicSkill_原子化模块完整设计文档_v1.0.md`  
**研究主线：** 集中式 Self-Evolving Agent；原子 SkillGraph 与可执行 Tool 联合进化  
**首选实现底座：** FlowEvo（尽可能复用原始 Harness、Compiler、Admission、Governance 与 Benchmark Setting）  
**文档性质：** 研究方案 + 软件工程详细设计 + 实验实现规范  
**状态：** 核心概念、对象边界、生命周期与接口冻结，可据此进入原型实现与代码适配阶段

---

# 0. 执行摘要

AtomicSkillGraph v2.0 将研究重点从 v1.0 的“面向联邦 Skill 聚合的原子化表示”转为：

> **在单一集中式长期运行 Agent 中，将成功执行轨迹持续蒸馏为可复用的原子能力、可执行 Tool 和复合 Skill，并通过 SkillGraph、Tool Repository、节点级验证和生命周期治理，使能力能够被细粒度复用、泛化、合并、拆分和优胜劣汰。**

系统不再以整份 `SKILL.md` 作为 Skill 本体，也不再以“生成一份更好的 Markdown Skill”为唯一进化目标。

v2.0 的长期知识由两套相互连接但语义不同的持久化结构组成：

```text
                         ┌─────────────────────────────┐
                         │       Global SkillGraph      │
                         │                             │
Task / Trace ───────────►│ Abstract Atomic Skill       │
                         │ Implementation Atom          │
                         │ Composite Skill              │
                         └──────────────┬──────────────┘
                                        │ tool_ref / bindings
                                        ▼
                         ┌─────────────────────────────┐
                         │      Global Tool Repository  │
                         │                             │
                         │ Executable Tool Artifacts    │
                         │ Interface / Tests / Safety   │
                         │ Provenance / Utility         │
                         │ Versions / Lifecycle         │
                         └─────────────────────────────┘
```

其中：

1. **Abstract Atomic Skill** 描述稳定、可复用、可验证的最小核心状态转移，即“做什么”；
2. **Implementation Atom** 描述当前 Harness 中“如何实现该状态转移”，自身不保存大段代码，而保存 `tool_ref + bindings + execution_policy`；
3. **Composite Skill** 将多个 Atomic Skill 组织为高层可复用能力，并承载由多轨迹聚合形成的 Layer-3 insight；
4. **Tool Asset** 是独立于 SkillGraph 的全局可执行资产，可以被多个 Skill 引用，一个 Skill 也可以组合多个 Tool，因此 Skill 与 Tool 是 **N:M** 关系；
5. **Runtime Task Execution Instance** 只记录一次任务的实际执行图、Tool 绑定、状态变化、验证结果和完整 Trace，不作为长期 Skill 节点。

系统保留 FlowEvo 的核心思想：

```text
successful trace
  → executable skill/tool compilation
  → admission
  → direct reuse / skill-conditioned generation / dynamic planning
  → downstream utility & negative-transfer governance
```

但在其上增加三个关键结构化层：

```text
1. State-Effect Atomicization
2. SkillGraph / Composite Skill
3. Independent Global Tool Repository with N:M Skill Bindings
```

核心闭环为：

```text
已有能力前置规划
      │
      ▼
SkillGraph Retrieval
      │
      ▼
Atomic Runtime Graph
      │
      ▼
Implementation Atom
      │
      ▼
Tool Resolution & Execution
      │
      ▼
Node / Composite / Benchmark Validation
      │
      ▼
Execution Trace
      │
      ├─────────────── success ───────────────┐
      │                                        ▼
      │                           Atomic Skill / Tool Compile
      │                                        │
      │                                        ▼
      │                           Admission & Generalization
      │                                        │
      │                                        ▼
      │                           SkillGraph + ToolRepo Update
      │
      └─────────────── failure ───────────────┐
                                               ▼
                                Failure Localization / Repair Candidate
                                               │
                                               ▼
                                必须经成功 replay 后才能进入可调用状态
```

v2.0 **不研究联邦、不定义联邦 Patch、不进行个性化分发、不做 Composite Tool、不做跨 Harness 迁移实验**。跨 Harness Tool 适配仅保留数据结构扩展位，正式研究放到 v2.1。

---

# 1. v1.0 → v2.0 的设计变化

| v1.0 | v2.0 |
|---|---|
| 研究主线是 Federated Skill Evolution | 研究主线改为 Centralized Skill/Tool Co-Evolution |
| SkillGraph 面向客户端/服务器抽象与聚合 | 只有一份长期 Global SkillGraph |
| Patch 是客户端上传与审计协议 | 删除 Patch 体系，不再存在联邦 Patch |
| `SKILL.md` 作为兼容外壳 | 不再持久化 `SKILL.md`；Skill 被拆成机器可读语义部件 |
| Implementation Atom 可直接保存执行代码 | Implementation Atom 只保存 Tool 引用、参数绑定与执行策略 |
| Tool 主要是 Implementation 的字段 | Tool 变成独立 Global Tool Repository 中的一等可执行资产 |
| Skill 与 Implementation 主要是 1:N | Skill 与 Tool 为 N:M |
| 工具生成不是独立长期进化对象 | Tool 拥有完整 discover/parameterize/generalize/specialize/merge/split 生命周期 |
| 联邦聚合负责重复节点统一 | 集中式维护器负责全局去重、合并、泛化和优胜劣汰 |
| Layer-3 未与 Composite 明确绑定 | Layer-3 insight 作为 Composite Skill 的高层经验部分 |
| 跨 Harness 是当前共享目标之一 | v2.0 暂不研究跨 Harness；v2.1 再加入 |
| task family/client 是重要聚合维度 | `task_type` 只作为标签/检索/统计依据，不构成能力边界 |

v2.0 仍然保留 v1.0 最重要的原则：

> **原子性由稳定状态转移、输入输出、验证能力和错误归因边界决定，而不是由文本长度、函数行数、Tool Call 数量或 Benchmark Task Type 决定。**

---

# 2. 研究问题与核心假设

## 2.1 核心研究问题

### RQ1：如何自动发现真正可复用的原子能力边界？

给定 Agent 的长轨迹：

```text
Observation → Reasoning → Action → Observation → ... → Success
```

需要识别其中哪些片段应成为：

```text
Abstract Atomic Skill
```

而不是简单将：

```text
每个 Tool Call = 一个 Skill
每个代码函数 = 一个 Skill
每个计划步骤 = 一个 Skill
```

### RQ2：如何把轨迹中的重复可执行行为沉淀成真正可调用的 Tool？

例如多个任务中出现：

```python
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["ShipDate"] = pd.to_datetime(df["ShipDate"])
```

系统应逐渐形成：

```python
normalize_date_column(table, target_column, ...)
```

而不是生成三个 task-specific Tool。

### RQ3：Skill 与 Tool 应如何解耦又保持可执行关联？

一个 Tool 可能实现多个能力；一个 Atomic Skill 也可能依赖多个 Tool，因此必须支持：

\[
Skill \leftrightarrow Tool
\]

的 N:M 关系。

### RQ4：如何让文本经验、可执行 Tool 和 Composite Skill 协同进化？

需要同时保留：

- Layer-2 风格的抽象文本规则；
- Layer-1 风格的可执行模板/函数/Primitive；
- Layer-3 风格的多轨迹高层 insight；
- Atomic/Composite 的图结构关系。

### RQ5：如何在可靠性、安全和成本之间进行长期路由？

系统不能因为已经存在 Tool 就强制使用 Tool，也不能因为动态规划更灵活就永远重新生成。

目标延续 FlowEvo：

> **在安全与可靠性足够的前提下，优先选择成本最低的可行执行路径。**

---

## 2.2 设计假设

- 使用冻结或外部 API LLM，不要求梯度训练；
- Agent Harness 能记录完整执行轨迹；
- Benchmark 或环境能够给出最终成功/失败反馈；
- 可执行 Tool 必须经过 admission 后才能成为可调用资产；
- 新 Tool 的有效性必须最终由成功执行/replay 支撑；
- 失败轨迹可以发现问题和提出修复，但不能单独证明新 Tool 正确；
- 第一阶段尽量复用 FlowEvo 原论文代码、Harness、数据与评测设置；
- `task_type` 可用于 grouping，但禁止作为 Tool/Skill 的硬作用域限制；
- v2.0 只有一个集中式长期 SkillGraph 和一个集中式 Tool Repository。

---

# 3. 设计目标与非目标

## 3.1 首要目标

1. **能力原子化**：将长 Skill/Workflow 拆为稳定状态效果的 Atomic Skill；
2. **可执行沉淀**：将成功轨迹中的重复执行模式沉淀为可调用 Tool Asset；
3. **高层组合**：将 Atomic Skill 重新组合成 Composite Skill；
4. **跨 Task Type 复用**：能力和 Tool 不被 `task_type` 锁死；
5. **节点级验证**：任务失败时定位到 Atomic Skill、Implementation Atom、Tool 或连接关系；
6. **Tool 长期进化**：支持泛化、特化、合并、拆分、测试增强、退役和回滚；
7. **可靠性-成本路由**：复用 FlowEvo cheapest-yet-reliable 路由原则；
8. **可复现实验**：保持原 Benchmark/Harness/模型设置，直接比较 FlowEvo 与本方案；
9. **低侵入复用**：尽量复用 FlowEvo Compiler、Admission、GateKeeper、Sandbox、Governance 与 Benchmark runner；
10. **结构化长期知识**：不再让一份 `SKILL.md` 同时承担语义、代码、测试、状态和版本等所有职责。

## 3.2 次要目标

- SkillGraph 可视化；
- Tool 复用关系可视化；
- Graph/Tool 生命周期可审计；
- 支持后续 v2.1 跨 Harness；
- 支持未来重新扩展成分布式/联邦版本，但 v2.0 完全不依赖该假设。

## 3.3 v2.0 明确不做

- Federated aggregation；
- client/server Patch；
- 个性化 bundle 分发；
- 通信量优化；
- Composite Tool；
- 强制统一所有 Benchmark 的 Tool 类型；
- 跨 Harness Tool transfer 实验；
- 用 task_type 限制 Tool 复用；
- 将每条代码语句建成 Skill 节点；
- 将每个原生 Harness Action 无条件视为一个 Atomic Skill；
- 让 LLM 在 Prompt 中看到不断增长的完整 Tool 列表；
- 让失败轨迹产生的未经成功验证代码直接进入 active Tool Bank。

---

# 4. 核心概念与术语

## 4.1 Skill

**Skill** 是可在多个任务实例中复用的能力表示，包含：

- 语义目标；
- 输入输出；
- 前置条件；
- 核心 Effect；
- 验证规则；
- 失败模式；
- 文本 guideline；
- 实现引用；
- 图结构关系；
- 历史证据与版本。

Skill 本身不等同于 Markdown，也不等同于 Tool。

## 4.2 Atomic Skill

在当前 Harness 和环境抽象层下：

> **能够独立调用、拥有稳定输入输出、产生一个主要可描述状态效果、可在任务结束前独立验证，并能将失败归因到该能力边界的最小可复用状态转移单元。**

## 4.3 Tool Asset

Tool 是：

> **能够被 Runtime 直接调用或重放的可执行资产。**

Tool 的具体形式由当前 Benchmark/Harness 决定，可包括：

- 参数化 Action Template；
- Python callable；
- helper primitive；
- Harness macro；
- 其他 FlowEvo 原始执行器可直接消费的 executable artifact。

v2.0 不要求 Tool 一定是 Python 函数。

## 4.4 Implementation Atom

Implementation Atom 是 Abstract Atomic Skill 与 Tool Repository 之间的执行绑定层。

它回答：

```text
在当前 Harness 中，这个 Atomic Skill 应调用哪些 Tool、如何绑定参数、按什么策略执行？
```

自身不保存大段 executable code。

## 4.5 Composite Skill

Composite Skill 是多个 Atomic Skill 或其他可引用 Skill 节点形成的高层复用方法。

它同时承载：

- 高层目标；
- 子图；
- 控制/数据关系；
- 高层验证；
- Layer-3 insight。

v2.0 不将 Composite Skill 编译成 Composite Tool。

## 4.6 Task Type

`task_type` 是：

- 统计标签；
- 检索信号；
- 初始模板匹配信号；
- Layer-3 多轨迹聚合的默认 grouping key。

但不是能力作用域。

同一个 Atomic Skill / Tool 可以被多个 task type 复用。

## 4.7 Runtime Execution Graph

某次任务实际被执行的最小充分 Skill 子图，包含：

- 当前任务选中的 Atomic/Composite 版本；
- Implementation Atom；
- Tool 绑定；
- 参数；
- 动态分支；
- 验证点；
- fallback。

只保存为执行记录，不进入长期 SkillGraph。

---

# 5. 形式化定义

## 5.1 Abstract Atomic Skill

定义：

\[
A=\langle I,O,P,E,V,F,G,M,R\rangle
\]

其中：

- \(I\)：输入语义与 Schema；
- \(O\)：输出语义与 Schema；
- \(P\)：前置条件；
- \(E\)：核心状态 Effect；
- \(V\)：节点级 Validator；
- \(F\)：Failure Modes；
- \(G\)：Layer-2 semantic guideline；
- \(M\)：metadata / provenance / lifecycle；
- \(R\)：Implementation Atom 引用集合。

给定状态 \(s\)：

\[
r(s,I)\rightarrow(s',O,m),\quad r\in R
\]

成功至少满足：

\[
P(s,I)=true
\land E(s,s')=true
\land V(s,I,s',O)=pass
\]

## 5.2 Implementation Atom

定义：

\[
R=\langle A_{ref},B,\pi,C,Q\rangle
\]

其中：

- \(A_{ref}\)：实现的 Abstract Atomic Skill 固定版本；
- \(B\)：一个或多个 Tool Binding；
- \(\pi\)：execution policy；
- \(C\)：当前 Harness/环境兼容约束；
- \(Q\)：历史质量统计。

重要：

\[
|B|\ge1
\]

一个 Implementation Atom 可以组合多个 Tool，但这一组合仍然只实现一个 Atomic Skill 的核心 Effect。

## 5.3 Tool Asset

定义：

\[
T=\langle \Sigma,X,T_s,S_f,P_v,Q,L\rangle
\]

其中：

- \(\Sigma\)：接口/signature/parameters；
- \(X\)：executable artifact；
- \(T_s\)：Tool-level tests / replay cases；
- \(S_f\)：safety constraints；
- \(P_v\)：provenance；
- \(Q\)：utility / success / failure / cost；
- \(L\)：lifecycle / version / lineage。

## 5.4 Composite Skill

定义：

\[
C=\langle V_C,E_C,P_C,EFF_C,V_C^{check},G_C,INS_C,M_C\rangle
\]

其中：

- \(V_C\)：引用的 Skill 节点；
- \(E_C\)：控制/数据/结构关系；
- \(P_C\)：高层前置条件；
- \(EFF_C\)：复合效果；
- \(V_C^{check}\)：复合验证；
- \(G_C\)：高层 guideline；
- \(INS_C\)：Layer-3 multi-trace insight；
- \(M_C\)：metadata / evidence / utility。

---

# 6. 原子性判定规范

## 6.1 Atomic Skill 必须满足

1. 可以作为独立执行意图被规划；
2. 输入输出在同类使用场景中稳定；
3. 只有一个主要核心 Effect；
4. Effect 可以在整个任务结束前验证；
5. 失败能够归因到该节点、Implementation 或 Tool；
6. 至少具有潜在跨实例复用价值；
7. 内部固定循环、辅助动作、机械步骤可以封装；
8. 不要求内部所有语句/动作都独立可观测。

## 6.2 Atomic Tool 的边界

Atomic Tool 与 Atomic Skill 不要求 1:1。

Tool 允许包含多个机械内部步骤，只要：

- 整体服务于一个稳定 Skill Effect；
- 接口稳定；
- 可以整体重放；
- 可以整体测试；
- 发生错误时可以明确判断“该 Tool/该 Atomic Skill 执行失败”。

例如：

```text
AcquireObject
```

可以由 Tool 内部执行：

```text
go to object_location
→ take object from object_location
```

不要求继续拆成 `GoTo` 和 `Take` 两个长期 Skill，除非它们在实际轨迹中表现出独立复用、验证、失败簇或替代实现价值。

## 6.3 应继续拆分 Atomic Skill 的情况

- 一个节点产生两个彼此独立的核心 Effect；
- 子过程被多个其他 Composite Skill 独立复用；
- 子过程需要独立 fallback/retry；
- 子过程有独立输入输出；
- 子过程有不同 Tool 实现；
- 失败长期形成两个稳定且互不相干的 failure cluster；
- 一个节点经常部分成功；
- 当前节点太大导致无法判断核心责任。

## 6.4 SplitScore

保留 v1.0 的规则 + LLM 混合评分作为辅助，而不是唯一标准：

\[
SplitScore=
0.20R_{reuse}
+0.20R_{validation}
+0.15R_{failure}
+0.15R_{io}
+0.10R_{retry}
+0.10R_{executor}
+0.10R_{effect}
\]

默认：

- `>= 0.70`：强制形成 split candidate；
- `0.45–0.70`：通过 replay/复用证据决定；
- `< 0.45`：保持封装。

阈值必须配置化。

---

# 7. 核心对象模型

长期持久化 SkillGraph 只包含三类 Skill 节点：

```text
1. Abstract Atomic Skill
2. Implementation Atom
3. Composite Skill
```

Runtime 另有：

```text
4. Task Execution Instance
```

但它不是长期能力节点。

Tool Repository 中保存：

```text
Tool Asset
```

Tool Asset **不是 SkillGraph 核心节点类型**。

在 UI 的 Implementation View 中可以将 Tool 显示成外部资产节点：

```text
Abstract Skill
   │ implements
   ▼
Implementation Atom
   │ uses
   ├────────► Tool A
   └────────► Tool B
```

---

# 8. Skill 与 Tool 的 N:M 关系

系统必须允许：

```text
一个 Tool → 多个 Skill
一个 Skill → 多个 Tool
```

## 8.1 一个 Tool 被多个 Skill 使用

例如：

```text
Tool: acquire_object(object, object_location)
```

可以被：

```text
pick_and_place
pick_heat_then_place
pick_clean_then_place
pick_cool_then_place
```

多个 Composite/Atomic 能力引用。

## 8.2 一个 Skill 使用多个 Tool

例如一个 Atomic Skill：

```text
ValidateAndNormalizeInput
```

可能绑定：

```text
Tool A: infer_schema
Tool B: normalize_values
Tool C: validate_output
```

只要整体核心 Effect 仍然唯一，并且这些 Tool 的固定组合不值得单独成为 Composite Skill，即可保留在一个 Implementation Atom 中。

## 8.3 禁止把 Tool 与 Skill 强行等同

错误设计：

```text
Skill == Python Function
```

正确设计：

```text
Skill = semantic capability
Implementation = execution binding
Tool = executable reusable asset
```

---

# 9. 不再存在持久化 SKILL.md

v2.0 明确删除：

```text
skills/<name>/SKILL.md
```

作为事实来源或长期资产的设计。

传统 `SKILL.md` 中可能存在的内容被拆到：

```text
Summary                 → Abstract/Composite metadata
Task Strategy           → Layer-2 guideline
Inputs / Outputs        → Contract
Preconditions            → Contract
Expected Effects         → Effect predicates
Execution Method         → Implementation Atom
Executable Code          → Tool Repository
Warnings / Attention     → FailureMode / constraints
Examples                 → evidence / example_bindings
Tests                    → Tool tests / Node validators
High-level Experience    → Composite Layer-3 insight
Version                   → Registry version
```

因此：

> **完整 Skill 是多个结构化部件和关系的组合，而不是一个 Markdown 文件。**

可以在调试 UI 中临时渲染“human-readable skill view”，但该视图：

- 不持久化为事实源；
- 不参与进化；
- 不拥有独立版本；
- 不允许反向覆盖 IR。

---

# 10. Layer-1 / Layer-2 / Layer-3 在 v2.0 中的映射

## 10.1 Layer-1：Executable Tool

沿用 FlowEvo：

- ALFWorld：parameterized action template；
- Code/Math：callable code skill；
- helper-level：PrimitiveCompiler 提取的 executable primitive；
- 其他 Benchmark：由对应 Compiler/Harness 决定。

进入 v2.0 后，Layer-1 不再直接等于完整 Skill，而是进入 Tool Repository。

## 10.2 Layer-2：Semantic Guideline

Layer-2 文本规则成为：

```text
Abstract Atomic Skill.guideline
或
Composite Skill.guideline
```

它不是独立 Markdown 文件。

Layer-2 主要用于：

- skill-conditioned generation；
- planning hint；
- failure avoidance；
- semantic retrieval；
- 人类可解释展示。

## 10.3 Layer-3：Composite Skill Insight

Layer-3 被定义为 Composite Skill 的多轨迹高层 insight。

默认规则：

```text
同一 task_type 的 trace 累积
→ sample_count >= 3
→ maintenance/compiler 聚合
→ common locations / pitfalls / environment facts / search priority
→ 更新 Composite Skill insight
```

这里的 `task_type` 只是 **生成 insight 时的默认 trace grouping key**。

生成后的 insight 不被永久锁定在该 task type。

如果另一个 task type 检索到同一 Composite/Atomic 子图，并满足语义与结构条件，同样允许使用这些 insight。

---

# 11. 总体系统架构

```text
┌───────────────────────────────────────────────────────────────┐
│                       Benchmark / Environment                  │
│       ALFWorld / HumanEval / MBPP / GSM8K / MATH / ...       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    Benchmark Adapter / Harness                │
│       优先原样复用 FlowEvo 当前 executor / verifier          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                         Runtime Agent                         │
│                                                               │
│ Task Parser → Skill Retriever → Atomic Planner                │
│      │                │               │                       │
│      │                │               ▼                       │
│      │                │       Runtime Skill Subgraph          │
│      │                │               │                       │
│      │                │               ▼                       │
│      │                │      Implementation Selector          │
│      │                │               │                       │
│      │                │               ▼                       │
│      │                │          Tool Resolver                │
│      │                │               │                       │
│      │                │               ▼                       │
│      │                └──────► Direct / Seeded / Dynamic      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│             Tool Executor + Node Validator + Verifier         │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                         Trace Store                           │
└───────────────┬───────────────────────────────┬───────────────┘
                │ success                       │ failure
                ▼                               ▼
┌────────────────────────────┐      ┌────────────────────────────┐
│ Trace Atomicizer           │      │ Failure Localizer          │
│ FlowEvo Compiler           │      │ Repair Candidate Generator │
│ Tool Candidate Miner       │      └──────────────┬─────────────┘
└───────────────┬────────────┘                     │
                ▼                                  │
┌────────────────────────────┐                     │
│ Admission / GateKeeper     │◄────────────────────┘
└───────────────┬────────────┘
                ▼
┌───────────────────────────────────────────────────────────────┐
│ SkillGraph Registry + Tool Repository + Governance/Maintenance│
└───────────────────────────────────────────────────────────────┘
```

---

# 12. Cold Start 与 Warm Start

## 12.1 Cold Start

当系统没有相关 Skill/Tool 时：

```text
Task
→ 原始 FlowEvo dynamic planning / generation
→ Harness 正常执行
→ Benchmark/Environment 验证
→ 若成功：记录完整 Trace
→ 从成功 Trace 事后学习 Atomic Skill / Tool / Composite
```

关键原则：

> **第一次遇到未知任务时，不要求 LLM 先凭空构造正确 Atomic SkillGraph。**

Skill 是从真实成功经验中学习，而不是先验幻想出来。

## 12.2 Warm Start

当存在相关能力时：

```text
Task
→ retrieve Composite / Atomic Skill
→ 构建最小充分 Runtime Graph
→ Implementation Selector
→ Tool Resolver
→ direct execution（满足可靠性门槛）
→ 若不适合 direct：skill-conditioned generation
→ 再失败：pure dynamic fallback
```

## 12.3 已有能力前置规划 + 新能力事后学习

这是 v2.0 的固定运行范式：

```text
Known Capability     → Plan Before Execution
Unknown Capability   → Learn After Successful Execution
```

---

# 13. LLM 与 Tool Repository 的隔离原则

LLM **不直接看到不断增长的 Tool 列表**。

错误方式：

```text
Prompt / tool schema 中塞入 500、1000、5000 个长期 Tool
```

正确方式：

```text
LLM / Planner：判断“现在要做哪个 Atomic Skill”
       ↓
Implementation Selector：选择实现
       ↓
Tool Resolver：解析 tool_ref
       ↓
Runtime：调用 Tool
```

例如：

```text
LLM 只规划：AcquireObject
```

而不是让 LLM 从：

```text
tool_001 ... tool_1000
```

中直接选择。

这样做的目的：

- 避免 Tool 数量增长导致上下文爆炸；
- 避免 LLM 依赖实现细节而不是能力语义；
- 允许 Tool Repository 独立去重和版本升级；
- 允许一个 Atomic Skill 的 Tool 实现发生变化而不影响上层规划语义。

---

# 14. SkillGraph 图模型

## 14.1 持久化节点

```text
abstract_atomic
implementation_atomic
composite
```

## 14.2 Runtime 节点

```text
task_instance
runtime_atomic_instance
validator_instance
```

只存在于执行记录。

## 14.3 Tool 不属于核心 SkillGraph

Tool 通过引用绑定：

```text
Implementation Atom.tool_bindings[].tool_ref
```

UI 可渲染虚拟 `uses_tool` 边。

---

# 15. SkillGraph 边类型

## 15.1 structural

```text
contains
implements
```

## 15.2 control

```text
next
branch
parallel
retry
fallback
loop
```

除显式 loop 外，单次编译 Runtime Graph 默认应为 DAG。

## 15.3 data

描述：

```text
source_output → target_input
```

包括：

- schema mapping；
- parameter binding；
- artifact mapping。

## 15.4 dependency

```text
requires_skill
requires_permission
requires_environment
requires_schema
```

Tool 依赖主要放在 Implementation Atom 内的 `tool_bindings`，不强制作为 SkillGraph 核心边。

## 15.5 semantic

```text
equivalent
similar
alternative
conflict
```

## 15.6 evolution

```text
derived_from
supersedes
split_from
merged_from
generalized_from
specialized_from
```

这些边用于 Skill 版本/结构演化，不替代 Tool Repository 自身 lineage。

---

# 16. Abstract Atomic Skill 数据结构

```yaml
kind: abstract_atomic
id: alfworld.acquire-object
version: 1.0.0
status: active

summary: >
  获取目标对象，使环境状态从“目标对象未被 Agent 持有”转变为“Agent 持有目标对象”，
  适用于需要后续放置、清洗、加热、冷却或观察该对象的多类任务。

inputs:
  - name: object
    semantic_type: object_ref
  - name: object_location
    semantic_type: location_ref

outputs:
  - name: held_object
    semantic_type: object_ref

preconditions:
  - predicate: object.exists
  - predicate: object.is_accessible

effects:
  - predicate: agent.holds
    args:
      object: $object

validator:
  pre_checks:
    - object_exists
  post_checks:
    - inventory_contains_object

failure_modes:
  - object_not_found
  - location_incorrect
  - take_action_rejected

guideline:
  layer: 2
  rules:
    - 在获取对象前先确认其当前位置，避免重复搜索已经检查过的位置。

implementation_refs:
  - impl.alfworld.acquire-object@1.0.0

metadata:
  task_type_labels:
    - pick_and_place_simple
    - pick_heat_then_place_in_recep
    - pick_clean_then_place_in_recep
  source_trace_ids: []
```

注意：

`task_type_labels` 只是历史标签，不限制其他类型使用。

---

# 17. Implementation Atom 数据结构

```yaml
kind: implementation_atomic
id: impl.alfworld.acquire-object
version: 1.0.0
status: active

implements:
  id: alfworld.acquire-object
  version: 1.0.0

compatibility:
  harness: flowevo_alfworld
  runtime: alfworld_text_env

tool_bindings:
  - tool_ref: tool://alfworld/acquire-object-template@1.2.0
    role: primary
    parameter_mapping:
      object: $inputs.object
      object_location: $inputs.object_location

execution_policy:
  mode: direct_if_eligible
  on_failure: fallback_to_skill_conditioned_generation
  max_direct_retries: 0

validator_overrides: []

quality:
  use_count: 0
  success_count: 0
  failure_count: 0
  utility: 0.5
```

Implementation Atom 只保存：

- Tool 引用；
- 参数映射；
- 执行策略；
- 兼容条件；
- 质量统计。

不保存 Tool executable body。

---

# 18. Tool Asset 数据结构

```yaml
tool_id: alfworld.acquire-object-template
version: 1.2.0
status: active
artifact_kind: action_template

summary: >
  在已知目标对象当前位置时，导航到该位置并获取对象，使对象进入 Agent inventory。

signature:
  parameters:
    - name: object
      semantic_type: object_ref
      required: true
    - name: object_location
      semantic_type: location_ref
      required: true

artifact:
  format: alfworld_action_template
  content_ref: artifact://sha256/...

interface:
  inputs:
    object: object_ref
    object_location: location_ref
  outputs:
    held_object: object_ref

tests:
  - test_ref: tool-test://...

safety:
  direct_execution_allowed: true
  checks_passed:
    - interface
    - replay
    - allowed_action_template

provenance:
  source_trace_ids: []
  source_task_types: []
  extraction_method: flowevo_compiler_plus_atomicizer

statistics:
  support_count: 3
  call_count: 12
  success_count: 11
  failure_count: 1
  utility: 0.91

lineage:
  generalized_from: []
  specialized_from: []
  supersedes: null
```

对于 Code/Math：

```yaml
artifact_kind: python_callable
artifact:
  code_ref: code://sha256/...
  entry_point: normalize_date_column
```

具体形式跟随原 FlowEvo 编译器，不由 v2.0 强行统一成 Python。

---

# 19. Composite Skill 数据结构

```yaml
kind: composite
id: alfworld.pick-heat-then-place
version: 2.0.0
status: active

summary: >
  获取目标对象、完成加热并将其放置到目标位置的复合能力。

task_type_labels:
  - pick_heat_then_place_in_recep

graph:
  nodes:
    - alfworld.acquire-object@1.0.0
    - alfworld.heat-object@1.1.0
    - alfworld.place-object@1.0.0
  control:
    - [acquire-object, heat-object]
    - [heat-object, place-object]

guideline:
  layer: 2
  rules:
    - 加热必须发生在最终放置之前。

insight:
  layer: 3
  sample_count: 7
  common_locations:
    - countertop
    - cabinet
  search_priority:
    - countertop
    - cabinet
    - microwave
  common_pitfalls:
    - 避免重复搜索已经检查过的位置。
  environment_facts:
    - 加热站通常是 microwave 类实体。

validator:
  - object_heated
  - object_at_target

metadata:
  source_trace_ids: []
```

---

# 20. Task Execution Instance

```yaml
execution_id: run-uuid
task_id: ...
task_type: ...
start_mode: cold | warm

runtime_graph:
  composite_refs: []
  atomic_refs: []
  implementation_refs: []
  tool_refs: []

node_results: []
benchmark_result: {}
trace_ref: ...

metrics:
  llm_tokens: 0
  tool_calls: 0
  latency_ms: 0
  direct_reuse_count: 0
  seeded_generation_count: 0
  dynamic_generation_count: 0
```

运行实例可以完整保存，不进入长期 SkillGraph。

---

# 21. 任务规划算法

## 21.1 输入

- 当前任务描述/Goal；
- 当前 Observation；
- SkillGraph；
- Tool Repository metadata；
- 当前 Harness Profile；
- 历史 Utility；
- task_type（若 Benchmark 提供）；
- 当前可用环境约束。

## 21.2 检索顺序

```text
1. 解析目标状态
2. 使用 task_type 做弱召回信号
3. 检索 Composite Skill
4. 检索 Abstract Atomic Skill
5. I/O / Preconditions / Effect 硬过滤
6. 结构与历史 Utility 重排
7. 生成最小充分 Runtime Graph
8. 选择 Implementation Atom
9. Runtime 解析 Tool 引用
```

## 21.3 task_type 禁止作为硬过滤

禁止：

```text
candidate.task_type != current.task_type → reject
```

允许：

```text
task_type match → retrieval bonus
```

最终是否能使用由：

- Semantic Match；
- Preconditions；
- I/O；
- Effect；
- Implementation Compatibility；
- Validation evidence；
- Utility；

共同决定。

---

# 22. Execution Routing：复用 FlowEvo cheapest-yet-reliable 原则

v2.0 不重新训练一个 Router。

首版尽量保持 FlowEvo 原始路由逻辑：

```text
Route 1: Direct Skill/Tool Execution
        ↓ 不够可靠或执行失败
Route 2: Skill-Conditioned Generation
        ↓ 仍失败/无匹配能力
Route 3: Pure Dynamic Planning / Generation
```

## 22.1 硬门槛优先

Direct execution 前必须满足：

- Tool admission passed；
- Tool 未 suppressed/retired；
- Skill preconditions 满足；
- Implementation compatible；
- Interface 可绑定；
- 历史可靠性达到阈值；
- 无已知严重 negative transfer。

## 22.2 成本优化

在足够可靠的可行路径中优先选择：

```text
成本更低 / token 更少 / 重试更少 / 执行更直接
```

不单独最大化成功率，也不单独最小化 token。

---

# 23. Tool 参数绑定

沿用 FlowEvo：

```text
成功 Trace
→ Compiler 抽取参数 slot
→ 新 Task Goal Parsing
→ task/environment 绑定 slot
→ 缺失参数由 Observation / LLM / environment feedback 补全
→ interface validator 检查
```

例如：

```text
Action Template:
  go to {object_location}
  take {object} from {object_location}
```

运行时：

```text
object = egg 1
object_location = countertop 2
```

Tool Repository 不负责让 LLM 自己从所有 Tool 中挑选；Tool Resolver 在 Skill/Implementation 已选定后才完成具体绑定。

---

# 24. Trace 记录规范

Trace 必须记录到足以支持事后 atomicization 和 Tool evolution。

至少包含：

```text
task_id
task_type
task_goal
planning_mode
retrieved_skill_refs
selected_composite
planned_atomic_nodes
realized_atomic_nodes
implementation_refs
tool_refs
tool_parameters
observations
actions
attempts
candidate_code
state snapshots / state summaries
node validators
benchmark verifier
success/failure
failure type
token cost
latency
retries
source provenance
```

对于 cold start，`planned_atomic_nodes` 可以为空。

任务成功后由 Trace Atomicizer 事后重建：

```text
realized causal atomic chain
```

---

# 25. Trace Atomicizer：从成功轨迹发现原子能力

## 25.1 原则

FlowEvo 已经能够将完整成功 Trace 编译为可复用 Skill/Template；v2.0 在其基础上进一步增加 **State-Effect Atomicization**。

原子化的权威标准是：

```text
stable Effect + stable I/O + independent validation + reusable boundary
```

而不是：

```text
helper function name / task_type / action count
```

## 25.2 候选边界来源

根据 Benchmark 类型使用不同观测：

### Interactive Environment

- 成功 Action 边界；
- Inventory/状态变化；
- 目标谓词变化；
- 环境接受/拒绝动作；
- 可独立重放的 action subsequence；
- 已知 task blueprint。

### Code/Math

- 顶层函数；
- reachable helper；
- AST call graph；
- 输入输出变量；
- 单元测试；
- 可独立执行 helper；
- FlowEvo PrimitiveCompiler 输出。

## 25.3 原子化流程

```text
Successful Trace
    ↓
Causal Trace Normalization
    ↓
Candidate Boundary Detection
    ↓
Effect Extraction
    ↓
I/O and Precondition Inference
    ↓
Independent Validator Construction
    ↓
Atomicity Check / SplitScore
    ↓
Candidate Alignment with Existing SkillGraph
    ↓
Add / Reuse / Merge / Split
```

## 25.4 FlowEvo Primitive 不是 Atomic Skill 的最终定义

FlowEvo 当前 `PrimitiveCompiler` 可以从成功代码中提取 helper-level executable primitive。

v2.0 将它作为：

```text
Code Tool Candidate Miner
```

而不是：

```text
Atomic Skill Oracle
```

原因：

- helper function 可能只是实现细节；
- 一个 Atomic Skill 可能由多个 helper 共同实现；
- 一个 helper 也可能被多个 Atomic Skill 复用；
- Atomic Skill 必须有明确业务/环境 Effect，而 helper 只具有代码结构边界。

---

# 26. Skill 初始生成

## 26.1 Cold Start 成功后

```text
Successful Dynamic Trace
→ FlowEvo Compiler 生成原始 Layer-1/Layer-2
→ Trace Atomicizer 生成 Atomic candidates
→ 对齐已有 SkillGraph
→ 生成/复用 Abstract Atomic Skill
→ 生成 Implementation Atom
→ Tool Candidate Miner 生成 Tool skeleton
→ Tool admission
→ 构建 Composite Skill
```

## 26.2 Warm Start 成功后

如果本次复用了已有 Skill：

```text
更新 success evidence / utility
检查是否出现新 Tool parameter binding
检查是否出现更通用实现
检查 Composite 是否产生新稳定路径
```

## 26.3 失败轨迹

失败轨迹不能直接创建 active executable Tool。

可以产生：

- Failure Mode；
- guideline 更新候选；
- validator 更新；
- split candidate；
- tool update candidate；
- tool specialize candidate；
- add_tool_test；
- implementation fallback 调整。

任何新的 executable body 必须重新成功 replay/admission。

---

# 27. Tool 发现与 Skeleton 生成

## 27.1 一次成功即可提取骨架

首次成功 Trace 可以产生：

```text
Tool Skeleton
```

但 Skeleton 不等于可直接调用 Candidate。

状态流程：

```text
successful trace
→ skeleton / draft
→ admission_pending
→ admission passed
→ candidate
→ repeated successful evidence
→ active
→ best-in-class / preferred
```

Admission 失败：

```text
→ shadow / quarantined
```

## 27.2 Tool Skeleton 内容

至少包含：

- executable artifact；
- signature；
- parameter slots；
- source trace；
- source Atomic Skill candidate；
- replay case；
- safety metadata；
- extraction method。

---

# 28. Tool Admission

优先直接复用 FlowEvo：

```text
src/compiler/admission.py
src/compiler/gatekeeper.py
src/env/sandbox.py
```

## 28.1 Code Tool Admission

包括：

1. syntax/interface check；
2. static safety scan；
3. 禁止危险 imports/calls；
4. trivial solution check；
5. unit/replay tests；
6. perturbation replay；
7. benchmark-specific consistency；
8. dedup/hash check。

## 28.2 Interactive Tool Admission

Action Template 类 Tool 不使用 Python AST safety scan，而使用 Benchmark Adapter：

- action syntax legal；
- parameters complete；
- action sequence admissible；
- source trace replay；
- fresh-world / perturbation replay（若环境允许）；
- terminal Effect 验证。

## 28.3 Candidate 的定义

只有：

```text
成功 Trace 提取
+
Admission Passed
```

之后才能进入：

```text
status = candidate
```

Candidate 可以被 Runtime 使用，但优先级低于 active/preferred，并受到更严格 direct-execution gate。

---

# 29. Tool 生命周期状态

```text
draft / skeleton
    ↓
admission_pending
    ├── fail → shadow
    ↓ pass
candidate
    ↓ additional successful evidence
active
    ↓ utility / reuse / reliability 最优
preferred
    ↓ harmful evidence
suppressed
    ↓ obsolete / unsafe / superseded
retired
```

说明：

- `shadow`：保留用于审计和后续修复，不直接调用；
- `candidate`：已经通过 admission，可探索性使用；
- `active`：有额外成功证据；
- `preferred`：同等功能候选中的默认选择；
- `suppressed`：暂时禁止默认使用，可被重新验证恢复；
- `retired`：不再进入新任务候选。

---

# 30. Tool Evolution 操作全集

v2.0 正式定义以下操作：

```text
discover_tool
parameterize_tool
add_tool
update_tool
specialize_tool
generalize_tool
merge_tools
split_tool
add_tool_test
add_tool_adapter
retire_tool
rollback_tool
```

## 30.1 discover_tool

触发：成功 Trace 中出现可独立执行、可复用片段。

输出：`Tool Skeleton`。

## 30.2 parameterize_tool

将 task-specific 常量转成参数：

```text
OrderDate / InvoiceDate / ShipDate
→ target_column
```

参数化主要沿用 FlowEvo：

```text
trace-driven
+ goal-driven
+ interface-driven
+ verification-driven
```

## 30.3 add_tool

Skeleton 通过 admission 后进入 Candidate Tool Repository。

## 30.4 update_tool

语义目标不变，但 executable body、参数处理、异常处理、测试或性能需要更新。

必须生成新版本。

## 30.5 specialize_tool

当通用 Tool 在某一稳定输入/环境子类表现明显差时，生成 specialised child。

例如：

```text
normalize_date_column
  └── normalize_excel_serial_date
```

## 30.6 generalize_tool

从多个 specialised Tool / Trace 中抽象 parameterized parent。

例如：

```text
normalize_order_date
normalize_invoice_date
normalize_ship_date
        ↓
normalize_date_column(table, target_column)
```

## 30.7 merge_tools

两个 Tool 行为等价，只是重复实现时合并。

区别：

```text
merge = duplicate equivalence
generalize = abstraction over variants
```

## 30.8 split_tool

当一个 Tool 实际包含多个稳定可复用效果、失败簇或接口时拆分。

只有在 Tool 边界本身已经妨碍复用/归因时才做，禁止为代码洁癖过度拆分。

## 30.9 add_tool_test

失败 Trace、边界 case 或新输入分布可增加测试。

## 30.10 add_tool_adapter

v2.0 仅保留 Schema 与接口；不做跨 Harness 实验。

状态建议：

```text
feature_flag: disabled_by_default
```

正式跨 Harness Tool Adapter 放入 v2.1。

## 30.11 retire_tool

长期无效、危险、完全被替代时退役。

## 30.12 rollback_tool

将默认引用恢复到历史有效版本；旧版本不被物理覆盖。

---

# 31. Tool Parameterization 与跨轨迹泛化

## 31.1 不限制同 task_type

Layer-3 insight 的默认聚合要求同 `task_type` 至少 3 条 Trace。

但 Tool generalization **全局进行**。

例如：

```text
检查对象是否存在
```

可以同时用于：

```text
clean
merge
format
validate
```

不同 task type。

## 31.2 候选召回

候选 Tool 对齐可使用：

- signature；
- parameter schema；
- normalized executable structure；
- code hash；
- action template shape；
- Semantic summary；
- Atomic Skill Effect；
- source behavior evidence。

Embedding 只用于候选召回，不作为最终 merge 判据。

## 31.3 Generalization 判据

至少满足：

\[
InterfaceCompatible
\land EffectCompatible
\land Parameterizable
\land ReplayConsistent
\]

新 generalized Tool 必须在来源实例 replay。

## 31.4 Generalize 后 specialised 对象的处理

- 原始成功 Trace：保留；
- example bindings：保留；
- 重复 one-off executable：可以 shadow；
- 有独立适用域的 specialised Tool：保留；
- generalized Tool：成为主要复用候选；
- harmful Tool：suppressed；
- 长期无价值：retired/pruned。

不采用：

```text
generalize → 立即物理删除全部 specialised tool
```

---

# 32. Tool 与 Skill 的联合进化

系统不是两套完全独立的进化器。

一个成功 Trace 可同时产生：

```text
1. Abstract Atomic Skill semantic update
2. Layer-2 guideline
3. Implementation Atom
4. Tool Skeleton / Tool update
5. Composite Skill path
6. Composite Layer-3 statistics
```

但这些对象具有不同生命周期。

## 32.1 Tool 成功不会自动改写 Layer-2 文本

Tool 成功主要更新：

- success count；
- utility；
- direct reuse evidence；
- preferred status。

如果新成功 Trace 包含新的可泛化规则，Compiler/Evolution Analyzer 才生成新的 guideline 版本。

## 32.2 Tool 失败不会直接改写所有语义

Tool 失败可：

- 增加 failure evidence；
- 生成 repair candidate；
- 增加 Tool test；
- 触发 specialized Tool；
- 触发 Implementation fallback；
- 更新相关 Failure Mode。

只有证据支持时才修改 Abstract Skill Contract/Effect。

---

# 33. 失败轨迹的处理规范

## 33.1 失败可做什么

```text
update_tool candidate
specialize_tool candidate
split_tool candidate
add_tool_test
add_failure_mode
add_validator
update_guideline candidate
adjust_execution_policy
suppress harmful tool
```

## 33.2 失败不能做什么

失败轨迹不能：

```text
直接生成 recommended executable tool
直接把新代码设为 active/preferred
绕过 admission
绕过成功 replay
```

固定原则：

> **Failure proposes; successful replay admits.**

---

# 34. 节点级错误归因

## 34.1 错误分类

```text
precondition_violation
input_schema_mismatch
implementation_selection_error
tool_binding_error
tool_execution_error
tool_interface_error
tool_safety_rejection
output_schema_mismatch
effect_violation
validator_error
control_flow_error
data_flow_error
composite_validation_error
benchmark_failure
unknown
```

## 34.2 定位流程

```text
1. 找到第一个失败 Atomic Node / validator
2. 检查上游输入是否已经污染
3. 检查 Implementation Atom 是否选错
4. 检查 tool_ref / parameter binding
5. 检查 Tool-level execution/test
6. 检查 Atomic Effect 是否满足
7. 检查 control/data edge
8. 检查 Composite 高层目标
9. 输出 responsibility + confidence
```

## 34.3 原子 Tool 内部错误不默认继续细分

如果 `AcquireObject` Tool 内部执行两三个动作失败，默认责任对象是：

```text
AcquireObject Implementation / Tool
```

而不是立即继续定位到：

```text
内部第二条 action
```

只有长期证据表明其内部形成独立复用/失败簇，才触发 `split_tool` 或 `split_skill`。

---

# 35. 三级验证体系

## 35.1 Tool-level Test

回答：

> 该 executable artifact 本身是否满足接口与局部行为？

例如：

- action template 参数能否实例化；
- Python function 单测是否通过；
- replay 是否通过；
- safety 是否通过。

## 35.2 Atomic Skill Node Validator

回答：

> 该 Tool/Implementation 执行后，Atomic Skill 的核心状态 Effect 是否真的发生？

例如：

```text
AcquireObject
→ inventory contains object
```

## 35.3 Composite / Benchmark Validator

Composite Validator：

> 高层 Skill 的组合目标是否满足？

Benchmark Verifier：

> 整个任务最终是否成功？

因此：

```text
Tool Test
   ↓
Atomic Node Validator
   ↓
Composite Validator
   ↓
Benchmark Verifier
```

四层不能互相替代。

---

# 36. Composite Skill 形成与进化

## 36.1 从成功 Atomic Chain 构建

例如成功 Trace 被原子化为：

```text
AcquireObject
→ HeatObject
→ PlaceObject
```

即可形成：

```text
Composite: PickHeatThenPlace
```

## 36.2 Composite 不与 task_type 强制 1:1

允许：

```text
一个 task_type → 多个 Composite 方法
一个 Composite → 服务多个 task_type
```

## 36.3 Composite Layer-3 insight 更新

同 `task_type` Trace 默认达到：

```text
_INSIGHT_MIN_SAMPLES = 3
```

后更新：

- common locations；
- common pitfalls；
- environment facts；
- search priority；
- failure distribution；
- efficient ordering。

## 36.4 Composite Tool 明确禁止

v2.0 不将：

```text
Acquire → Heat → Place
```

整体再编译成一个 mega-tool。

Composite 的价值在于：

- 高层结构复用；
- Atomic Tool 可独立更新；
- 局部失败可定位；
- 可跨 Composite 共享 Atomic Skill/Tool。

---

# 37. Skill 节点的合并、拆分与去重

## 37.1 Abstract Skill merge

不能只用文本相似度。

至少要求：

\[
SemanticMatch
\land IOCompatible
\land CoreEffectEquivalent
\land ValidatorConsistent
\]

必要时增加跨实例行为验证。

## 37.2 Implementation merge

若：

- implements 同一 Abstract；
- Tool binding 等价；
- 参数映射等价；
- compatibility 一致；

则可以合并重复 Implementation。

## 37.3 Skill split

遵循原子性判据和 SplitScore。

## 37.4 Tool merge 与 Skill merge 独立

可能出现：

```text
两个 Skill 不同，但共享同一个 Tool
```

因此 Tool merge 不意味着 Skill merge。

也可能：

```text
两个 Skill 合并，但保留两个 specialised Tool
```

---

# 38. 版本机制

v2.0 删除 Patch 协议，但保留不可变版本与 lineage。

## 38.1 Skill 版本

```text
logical_id + semantic_version + content_hash
```

语义/Contract/Effect/Graph 变化 → 新版本。

## 38.2 Tool 版本

```text
tool_id + semantic_version + artifact_hash
```

code/template/signature/tests/safety contract 的实质变化 → 新版本。

## 38.3 稳定成功不产生新版本

只更新：

```text
statistics / evidence / utility
```

## 38.4 回滚

回滚本质是：

```text
推荐指针恢复历史版本
```

不覆盖历史 artifact。

## 38.5 不再存在 Patch

系统没有：

```text
Patch Schema
Patch upload
Patch approval
Patch aggregation
```

集中式 Evolution Engine 直接在验证通过后注册新版本与 lineage。

---

# 39. Tool Governance 与优胜劣汰

首版尽量复用 FlowEvo：

- `use_count`；
- `success_count`；
- `failure_count`；
- `utility`；
- direct/seeded usage；
- positive/negative transfer；
- contrastive evaluation；
- suppress/prune lifecycle。

## 39.1 Tool Score 不另起复杂模型

v2.0 首版不训练新 Router。

沿用规则、阈值和历史统计。

## 39.2 Tool 优先级

原则：

```text
1. Safety / Interface / Preconditions
2. Negative-transfer risk
3. Reliability / Utility
4. Specific input fit
5. Cost
```

## 39.3 Tool 与 Skill 分开记统计

Skill 统计反映能力层表现。

Tool 统计反映 executable implementation 表现。

否则无法判断：

```text
是 Skill 概念错了
还是某个 Tool 实现错了
```

---

# 40. 存储设计

推荐逻辑结构：

```text
data/
├── skill_graph/
│   ├── graph.json
│   ├── abstract_atomic/
│   │   └── <logical_id>/<version>.json
│   ├── implementation_atomic/
│   │   └── <logical_id>/<version>.json
│   └── composite/
│       └── <logical_id>/<version>.json
│
├── tools/
│   ├── registry.json
│   └── <tool_id>/
│       └── <version>/
│           ├── tool.json
│           ├── artifact.*
│           └── tests.json
│
├── traces/
│   └── ...
│
├── runtime_runs/
│   └── ...
│
└── metrics/
    └── ...
```

第一版允许继续 JSON 文件存储以最大化 FlowEvo 兼容性。

后续可以迁移 SQLite/PostgreSQL，不改变 IR。

---

# 41. 核心逻辑表

如果后续使用数据库，建议：

## `skill_nodes`

- logical_id
- kind
- current_version
- recommended_version
- status

## `skill_versions`

- logical_id
- version
- content_hash
- semantic_json
- contract_json
- validator_json
- metadata_json

## `skill_edges`

- edge_id
- source_ref
- target_ref
- type
- subtype
- metadata

## `implementation_bindings`

- implementation_ref
- abstract_ref
- execution_policy_json
- compatibility_json

## `implementation_tool_bindings`

- implementation_ref
- tool_ref
- role
- parameter_mapping_json
- priority

## `tool_assets`

- tool_id
- version
- artifact_kind
- artifact_hash
- status
- signature_json
- interface_json
- safety_json
- provenance_json

## `tool_stats`

- tool_ref
- support_count
- call_count
- success_count
- failure_count
- utility
- direct_use_count
- negative_transfer_count

## `tool_lineage`

- source_tool_ref
- target_tool_ref
- relation

## `execution_runs`

- execution_id
- task_id
- task_type
- start_mode
- runtime_graph_hash
- benchmark_result
- trace_ref

## `execution_node_results`

- execution_id
- node_ref
- implementation_ref
- tool_refs
- status
- validator_result
- error_type

---

# 42. Python 核心接口

```python
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SkillRef:
    logical_id: str
    version: str


@dataclass(frozen=True)
class ToolRef:
    tool_id: str
    version: str


@dataclass
class AbstractAtomicSkill:
    ref: SkillRef
    summary: str
    inputs: list[dict]
    outputs: list[dict]
    preconditions: list[dict]
    effects: list[dict]
    validator: dict
    failure_modes: list[dict]
    guideline: dict
    metadata: dict


@dataclass
class ToolBinding:
    tool_ref: ToolRef
    role: str
    parameter_mapping: dict


@dataclass
class ImplementationAtom:
    ref: SkillRef
    abstract_ref: SkillRef
    tool_bindings: list[ToolBinding]
    execution_policy: dict
    compatibility: dict
    quality: dict


@dataclass
class CompositeSkill:
    ref: SkillRef
    summary: str
    graph: dict
    guideline: dict
    insight: dict
    validator: dict
    metadata: dict


@dataclass
class ToolAsset:
    ref: ToolRef
    artifact_kind: str
    signature: dict
    interface: dict
    artifact_ref: str
    tests: list[dict]
    safety: dict
    provenance: dict
    statistics: dict
    lifecycle: dict


class SkillGraphRegistry(Protocol):
    def get_skill(self, ref: SkillRef): ...
    def retrieve(self, query: dict) -> Sequence[object]: ...
    def register_new_version(self, obj: object) -> SkillRef: ...


class ToolRegistry(Protocol):
    def get_tool(self, ref: ToolRef) -> ToolAsset: ...
    def retrieve_candidates(self, query: dict) -> Sequence[ToolAsset]: ...
    def register_candidate(self, tool: ToolAsset) -> ToolRef: ...
    def record_feedback(self, ref: ToolRef, result: dict) -> None: ...


class AtomicPlanner(Protocol):
    def compile_runtime_graph(self, task: dict) -> dict: ...


class ImplementationSelector(Protocol):
    def select(self, atomic_ref: SkillRef, context: dict) -> SkillRef | None: ...


class ToolResolver(Protocol):
    def resolve(self, implementation_ref: SkillRef, context: dict) -> list[dict]: ...


class ToolExecutor(Protocol):
    def execute(self, resolved_tool: dict, inputs: dict, context: dict) -> dict: ...


class ValidatorEngine(Protocol):
    def validate_tool(self, tool_ref: ToolRef, result: dict) -> dict: ...
    def validate_atomic(self, atomic_ref: SkillRef, before: dict, after: dict) -> dict: ...
    def validate_composite(self, composite_ref: SkillRef, run: dict) -> dict: ...


class TraceAtomicizer(Protocol):
    def atomicize_success(self, trace: dict) -> dict: ...


class ToolCompilerAdapter(Protocol):
    def compile_from_atomic_segment(self, segment: dict, trace: dict) -> list[ToolAsset]: ...


class AdmissionEngine(Protocol):
    def admit(self, tool: ToolAsset, context: dict) -> dict: ...


class EvolutionEngine(Protocol):
    def process_success(self, trace: dict) -> dict: ...
    def process_failure(self, trace: dict) -> dict: ...


class CompositeInsightUpdater(Protocol):
    def update_if_ready(self, composite_ref: SkillRef, task_type: str) -> dict | None: ...
```

---

# 43. Runtime 执行伪代码

```python
def run_task(task, system):
    candidates = system.skill_graph.retrieve(task)

    if not candidates:
        # cold start / unknown capability
        result = system.original_flowevo.dynamic_run(task)
        system.trace_store.save(result.trace)
        if result.success:
            system.evolution.process_success(result.trace)
        else:
            system.evolution.process_failure(result.trace)
        return result

    runtime_graph = system.atomic_planner.compile_runtime_graph(task)

    for atomic in runtime_graph.atomic_nodes:
        impl = system.implementation_selector.select(atomic.ref, task.context)

        if impl is None:
            runtime_graph.mark_dynamic_fallback(atomic.ref)
            continue

        resolved_tools = system.tool_resolver.resolve(impl, task.context)

        if not direct_execution_is_reliable(atomic, impl, resolved_tools):
            runtime_graph.mark_seeded_generation(atomic.ref)
            continue

        result = execute_atomic_with_tools(
            atomic=atomic,
            implementation=impl,
            tools=resolved_tools,
        )

        validation = system.validator.validate_atomic(
            atomic.ref,
            result.before,
            result.after,
        )

        if not validation["passed"]:
            trigger_fallback(runtime_graph, atomic.ref)

    result = finish_with_flowevo_routes(task, runtime_graph)
    system.trace_store.save(result.trace)

    if result.success:
        system.evolution.process_success(result.trace)
    else:
        system.evolution.process_failure(result.trace)

    return result
```

---

# 44. 成功轨迹进化伪代码

```python
def process_success(trace):
    # 1. FlowEvo 原始编译产物
    base_artifacts = flowevo_compiler.compile(trace)

    # 2. 原子化真实成功因果链
    atomic_result = trace_atomicizer.atomicize_success(trace)

    # 3. 对齐/创建 Abstract Atomic Skill
    atomic_refs = []
    for candidate in atomic_result.atomic_candidates:
        ref = align_or_register_atomic(candidate)
        atomic_refs.append(ref)

    # 4. 生成 Implementation Atom 与 Tool candidates
    for segment, atomic_ref in zip(atomic_result.segments, atomic_refs):
        tool_skeletons = tool_compiler.compile_from_atomic_segment(segment, trace)

        admitted_refs = []
        for skeleton in tool_skeletons:
            result = admission_engine.admit(skeleton, trace.context)
            if result.passed:
                admitted_refs.append(tool_registry.register_candidate(result.tool))
            else:
                tool_registry.store_shadow(result.tool)

        bind_or_update_implementation(atomic_ref, admitted_refs, trace)

    # 5. 构造或更新 Composite Skill
    composite_ref = build_or_align_composite(atomic_refs, trace)

    # 6. 更新 Layer-2 guideline / evidence
    update_semantic_evidence(trace, atomic_refs, composite_ref, base_artifacts)

    # 7. 达到多轨迹门槛时更新 Layer-3 insight
    composite_insight_updater.update_if_ready(
        composite_ref,
        task_type=trace.task_type,
    )

    # 8. 全局 Tool generalization / merge / specialization maintenance
    maintenance.observe_success(trace)
```

---

# 45. 失败轨迹进化伪代码

```python
def process_failure(trace):
    attribution = failure_localizer.localize(trace)

    for failure in attribution.failures:
        if failure.kind == "tool_execution_error":
            proposal = propose_tool_update_or_specialization(failure)
            store_shadow_repair_candidate(proposal)

        elif failure.kind == "missing_tool_test":
            add_tool_test_candidate(failure)

        elif failure.kind == "oversized_tool":
            propose_split_tool(failure)

        elif failure.kind == "atomic_contract_gap":
            propose_atomic_contract_revision(failure)

        elif failure.kind == "missing_atomic_boundary":
            propose_skill_split(failure)

        elif failure.kind == "composite_control_error":
            propose_composite_revision(failure)

    # 任何 executable repair 都必须：
    # successful replay -> admission -> candidate
```

---

# 46. Global Tool Generalizer

## 46.1 输入

- 全部 active/candidate Tool metadata；
- source traces；
- Atomic Skill bindings；
- executable signatures；
- replay tests；
- utility；
- failure patterns。

## 46.2 候选生成

```text
Canonical signature / AST / template-shape retrieval
+ semantic/effect recall
→ candidate pair/group
```

## 46.3 判断动作

```text
完全行为等价         → merge_tools
仅实例常量不同       → generalize_tool
特殊输入需要单独逻辑 → specialize_tool
一个 Tool 多个核心效果→ split_tool
没有足够证据         → keep separate
```

## 46.4 运行时不依赖 task_type

Global Tool Generalizer 检索全库，不按 task_type 分仓。

`task_type` 只能作为 provenance feature。

---

# 47. FlowEvo 代码复用方案

## 47.1 为什么以 FlowEvo 为首选底座

FlowEvo 已公开并按模块拆分：

```text
src/agent/
src/compiler/
src/memory/
src/governance/
src/maintenance/
src/runtime/
src/core/
src/env/
src/eval/
src/alfworld_/
src/code_math/
```

它已经实现：

- success trace compilation；
- executable skill code/template；
- admission；
- GateKeeper；
- Sandbox；
- direct/seeded/dynamic routing；
- utility；
- negative transfer suppression；
- ALFWorld 与 Code/Math runner；
- Trace schemas；
- primitive extraction。

因此 v2.0 不应重新实现完整 Agent Harness。

## 47.2 推荐直接复用/尽量保持不变

### `src/alfworld_/env.py`

保留 ALFWorld 环境交互。

### `src/alfworld_/executor.py`

保留原执行器与环境动作调用。

### `src/alfworld_/param_extractor.py`

复用 goal/task 参数解析。

### `src/alfworld_/run_20task_validation.py`

作为第一阶段 ALFWorld 对照实验入口之一。

### `src/code_math/runner.py`

作为 HumanEval/MBPP/GSM8K/MATH 原始实验入口。

### `src/eval/`

尽量保持原 verifier / benchmark runner 行为。

### `src/env/sandbox.py`

直接复用代码安全执行基础。

## 47.3 强复用但需要外层适配

### `src/compiler/compiler.py`

继续生成完整 executable skill artifact，作为：

- cold-start base artifact；
- Atomic Tool extraction source；
- baseline 对照。

### `src/alfworld_/compiler.py`

直接复用：

- Layer-1 parameterized action template；
- Layer-2 guideline extraction。

Atomicizer 在其结果和原始 Trace 上进一步拆原子节点。

### `src/compiler/primitive_compiler.py`

直接作为 Code 类：

```text
helper-level Tool Candidate Miner
```

但不能直接把 PrimitiveCard 当 Abstract Atomic Skill。

### `src/compiler/admission.py`

复用：

- compiled artifact admission；
- dedup；
- code hash；
- replay result；
- shadow/active handling 思路。

### `src/compiler/gatekeeper.py`

复用：

- static safety scan；
- unit replay；
- perturbation replay；
- trivial solution checks。

### `src/memory/trace_store.py`

复用 Trace 存储底座。

### `src/memory/skill_registry.py`

复用：

- utility stats；
- usage feedback；
- suppression；
- distillation score；
- status rules。

但 v2.0 语义层需要新增 SkillGraphRegistry。

### `src/memory/primitive_store.py`

可演化成 Tool Repository 的首个原型底座，尤其是：

- executable artifact storage；
- support_count；
- success/failure；
- utility；
- suppress；
- code path。

## 47.4 复用治理

### `src/governance/`

保留 contrastive evaluation / utility。

### `src/maintenance/governance.py`

保留 lifecycle coordination 思路，并增加：

```text
generalize_tool
specialize_tool
merge_tools
split_tool
Composite insight update
SkillGraph dedup
```

## 47.5 FlowEvo 已存在 Primitive 对本研究的意义

FlowEvo 已经实现：

```text
successful code trace
→ reachable helper extraction
→ PrimitiveCard
→ PrimitiveStore
```

因此本文不能将“从代码里抽 helper 函数”本身作为主要创新。

v2.0 的新增点必须明确是：

```text
helper/code primitive
      ↓
与稳定 State Effect 对齐
      ↓
Abstract Atomic Skill
      ↓
N:M Tool Binding
      ↓
跨 task type Tool reuse/generalization
      ↓
Composite SkillGraph
      ↓
节点级 validation & structural evolution
```

## 47.6 License

当前 FlowEvo 公开仓库 LICENSE 为 Apache License 2.0。

实现时保留原项目版权、许可与修改声明。

---

# 48. SkillOps 代码复用方案

SkillOps 提供：

```text
SkillContract(P, O, A, V, F)
Hierarchical Skill Ecosystem Graph
skill_graph.py
maintenance.py
planner.py
```

## 48.1 建议复用

重点参考/选择性复用：

```text
skillops/skill_graph.py
```

其中已有：

- SkillContract；
- Action；
- Skill；
- dependency；
- compatibility；
- redundancy；
- alternative；
- lineage；
- Graph-of-Graphs。

这与 v1.0/v2.0 的 Contract + SkillGraph 很接近。

## 48.2 不建议整体替换 FlowEvo Runtime

不建议：

```text
FlowEvo Runtime + SkillOps Planner + Trace2Skill Agent + SkillOpt Optimizer
```

全部混成同一运行路径。

原因：

- Harness 不同；
- 数据结构不同；
- baseline 难公平；
- 很难定位增益来自哪个系统。

建议：

> **FlowEvo 为 Runtime 主底座；SkillOps 仅提供 Contract/Graph/Maintenance 的数据结构和算法参考。**

SkillOps 当前公开仓库采用 MIT License。

---

# 49. Trace2Skill 代码复用方案

Trace2Skill 最有价值的是：

```text
analysis/
  run_error_analysis.py
  run_success_analysis_llm.py
skill_evolver/
spreadsheet_agent/
```

## 49.1 推荐复用

可选择性借用：

- success trajectory analysis prompt；
- error trajectory analysis；
- trajectory-local lesson extraction；
- 多 Trace 并行分析思路。

这些可以加强：

```text
Failure Localizer
Semantic Guideline Updater
Atomic Boundary Analyzer
```

## 49.2 不推荐直接复用

`skill_evolver` 的最终目标主要是文件/Markdown Skill patch/consolidation。

v2.0 已经删除：

```text
SKILL.md as trainable state
Patch protocol
```

因此不应将其完整 skill-evolution backend 作为主实现。

---

# 50. SkillOpt 代码/方法复用方案

SkillOpt 的核心价值是：

```text
rollout
→ bounded update
→ held-out validation gate
→ accept / reject
→ best version
```

## 50.1 可借鉴

- validation-gated evolution；
- rejected-candidate buffer；
- rollback；
- held-out selection；
- accuracy/token/latency 多目标评价思路。

## 50.2 不建议作为主 Skill 表示

SkillOpt 的核心 trainable state 是 Markdown Skill Document。

v2.0 的核心对象已经变成：

```text
SkillGraph + Tool Repository
```

因此 SkillOpt 更适合作为：

- evolution gate 的算法参考；
- baseline；
- offline maintenance 参考。

---

# 51. 代码复用最终决策

## Primary Substrate

```text
FlowEvo
```

## Selective Structural Reuse

```text
SkillOps.skill_graph / maintenance concepts
```

## Optional Trace Analysis Reuse

```text
Trace2Skill.analysis
```

## Validation/Evolution Reference

```text
SkillOpt
```

不构建“四个项目拼盘 Runtime”。

---

# 52. Benchmark Adapter 设计

v2.0 设计文档不锁死最终 Benchmark。

原则是：

> **第一阶段选择 FlowEvo 已有 Benchmark，使用相同 Harness、Evaluator、模型与尽可能相同的实验设置，对比原 FlowEvo 和 AtomicSkillGraph。**

FlowEvo 当前公开代码支持：

```text
ALFWorld
HumanEval
MBPP
GSM8K
MATH
```

Benchmark Adapter 必须暴露：

```text
load_tasks()
parse_task_type(task)
run_environment_action(...)
get_observation()
verify_task(...)
extract_state_summary(...)
validate_atomic_effect(...)
compile_tool_artifact(...)
replay_tool(...)
```

不要求所有 Benchmark 实现完全相同。

---

# 53. ALFWorld 原子化完整示例

## 53.1 原任务

```text
Heat egg and place it in fridge.
```

原 FlowEvo 成功 Template 可能为：

```text
go to {object_location}
take {object} from {object_location}
go to {heating_station}
heat {object} with {heating_station}
go to {target_location}
move {object} to {target_location}
```

## 53.2 v2.0 原子化

```text
Composite Skill: PickHeatThenPlace

AcquireObject
   ↓
HeatObject
   ↓
PlaceObject
```

## 53.3 AcquireObject Effect

```text
before: not agent.holds(object)
after:  agent.holds(object)
```

Tool：

```text
go to {object_location}
take {object} from {object_location}
```

## 53.4 HeatObject Effect

```text
before: object not heated
after:  object heated
```

Tool：

```text
go to {heating_station}
heat {object} with {heating_station}
```

## 53.5 PlaceObject Effect

```text
before: agent.holds(object)
after:  object at target_location
```

Tool：

```text
go to {target_location}
move {object} to {target_location}
```

## 53.6 跨 task_type 复用

`AcquireObject` 可直接被：

```text
PickAndPlace
PickCleanThenPlace
PickHeatThenPlace
PickCoolThenPlace
LookAtObjectInLight
```

多个高层 Skill 共享。

这正是原本 task-type template 不容易表达的跨流程原子复用。

---

# 54. ALFWorld Layer-2 / Layer-3 示例

## Atomic Layer-2

```text
AcquireObject guideline:
先定位目标对象，再执行获取操作，避免反复检查已排除的位置。
```

## Composite Layer-3

当 `pick_heat_then_place_in_recep` 至少有 3 条 Trace 后：

```yaml
insight:
  sample_count: 5
  search_priority:
    - countertop
    - cabinet
    - microwave
  common_pitfalls:
    - 重复访问已经搜索过的位置会增加 timeout 风险。
  environment_facts:
    - heating station 通常属于 microwave 类实体。
```

这些 insight 属于 Composite Skill，而不是 Tool。

---

# 55. Tool failure → repair 示例

若：

```text
HeatObject Tool
```

在某个环境里因为 station 参数无法绑定而失败：

```text
failure
→ tool_binding_error
→ add_tool_test / update parameter extraction candidate
→ 新 Tool 版本存入 shadow
→ 在成功 episode/replay 中验证
→ admission
→ candidate
```

禁止：

```text
失败一次
→ LLM 改代码
→ 立即 active
```

---

# 56. 测试体系

## 56.1 IR 单元测试

- Abstract Skill Schema；
- Implementation Atom Schema；
- Composite Schema；
- Tool Asset Schema；
- fixed version refs；
- content hash；
- Effect predicate；
- parameter mapping；
- task_type 非硬作用域约束。

## 56.2 Graph 测试

- 所有 Skill refs 存在；
- contains / implements 合法；
- data edge I/O 可连接；
- Composite 有入口出口；
- 除 loop 外无非法控制环；
- retired 节点不进入新默认路径；
- N:M Tool binding 合法。

## 56.3 Tool Repository 测试

- artifact hash；
- interface；
- admission status；
- tests 可执行；
- unsafe code 阻断；
- shadow 不可 direct；
- retired 不可新绑定；
- rollback 恢复历史版本。

## 56.4 Atomic Execution 测试

- direct success；
- precondition fail；
- tool binding fail；
- tool execution fail；
- Effect fail；
- node validator fail；
- fallback to seeded；
- fallback to dynamic。

## 56.5 Evolution 测试

- single-success skeleton；
- admission pass/fail；
- candidate → active；
- duplicate merge；
- cross-task-type reuse；
- generalize；
- specialize；
- split；
- suppress；
- retire；
- successful rollback。

---

# 57. 实验设计

## 57.1 最重要原则

第一阶段不要同时换：

```text
Benchmark
Model
Harness
Agent Runtime
Skill representation
Evaluation
```

而应：

> **只改变 Skill 表示和长期进化机制。**

因此优先：

```text
FlowEvo official setting
vs
AtomicSkillGraph on the same FlowEvo setting
```

## 57.2 核心实验条件

建议至少：

1. **Original Baseline / Dynamic**
2. **Original FlowEvo**
3. **FlowEvo + Atomic SkillGraph, no independent Tool evolution**
4. **FlowEvo + Tool Repository, no Composite Graph**
5. **AtomicSkillGraph Full**

## 57.3 关键消融

- Full - Node Validator；
- Full - Layer-3 Composite Insight；
- Full - Tool Generalization；
- Full - Tool Specialization；
- Full - Global Cross-Task-Type Reuse；
- Full - N:M Binding（强制 Skill:Tool=1:1）；
- Full - Governance；
- Full - Primitive Compiler Reuse；
- Full - Composite Skill；
- task_type hard-restricted vs capability-based reuse。

---

# 58. 评价指标

## 58.1 Task-level

- Success Rate / Solve Rate / pass@1；
- Late-run Success；
- First-attempt Success；
- Retry Count；
- Negative Transfer Rate。

## 58.2 Cost

- Tokens per task；
- LLM calls；
- Latency；
- Dynamic planning frequency；
- Direct execution rate；
- Seeded generation rate；
- Cost per solved task。

## 58.3 Atomic SkillGraph

- Atomic Reuse Rate；
- Cross-Task-Type Atomic Reuse Rate；
- Average Atomic Nodes per Composite；
- Duplicate Atomic Skill Rate；
- Error Localization Accuracy；
- Node Validator Coverage；
- Skill Split/Merge Precision；
- Composite Reuse Rate。

## 58.4 Tool Repository

- Tool Reuse Rate；
- Cross-Task-Type Tool Reuse；
- Tool Fan-in：一个 Tool 被多少 Skill 引用；
- Tool Fan-out：一个 Skill 平均绑定多少 Tool；
- Tool Admission Pass Rate；
- Candidate → Active Promotion Rate；
- Tool Generalization Success Rate；
- Tool Duplicate Rate；
- Tool Suppression Rate；
- Harmful Tool Reuse Rate；
- Tool direct execution success；
- Utility improvement over lifetime。

## 58.5 Knowledge Growth

- SkillGraph nodes over episodes；
- Tool count over episodes；
- Active/Shadow/Suppressed ratio；
- Graph redundancy；
- Tool redundancy；
- Layer-3 insight coverage。

---

# 59. 关键论文实验假设

### H1

Atomic SkillGraph 能比 task-type / flat skill bank 提高跨任务复用率。

### H2

独立 Tool Repository + N:M binding 能降低重复 executable artifact 数量。

### H3

节点级 validator 能降低错误定位范围并提高 repair 成功率。

### H4

Tool generalization 能降低 token 和 dynamic generation 成本，同时不显著降低成功率。

### H5

capability-based global reuse 优于 task_type hard restriction。

### H6

Composite Layer-3 insight 能改善长任务中的搜索顺序和失败规避。

---

# 60. 与现有工作的区别边界

## FlowEvo

已有：

- executable skill compile；
- Layer-1 / Layer-2；
- Layer-3 insight；
- direct/seeded/dynamic routing；
- primitive extraction；
- governance。

本方案新增重点：

- State-Effect Atomic Skill；
- SkillGraph；
- Abstract/Implementation split；
- independent Tool Repository；
- N:M Skill–Tool binding；
- node-level validation；
- global cross-task-type atomic/tool reuse；
- tool generalize/specialize/merge/split；
- Composite built from reusable atomic nodes。

## SkillOps

已有：

- typed Skill Contract；
- Graph-of-Graphs；
- maintenance。

本方案重点不同：

- 从执行 Trace 在线持续生成/进化；
- executable Tool 与 Skill N:M 分离；
- 复用 FlowEvo runtime；
- Atomic State Effect 作为最小能力边界。

## Trace2Skill

已有：

- trajectory-local lesson；
- success/error analysis；
- Markdown/skill-directory consolidation。

本方案不再以完整 Skill 文档 patch 为优化对象。

## SkillOpt

已有：

- validation-gated text skill optimization。

本方案优化对象是：

```text
SkillGraph + Tool Repository + lifecycle
```

而不是单一文本文件。

---

# 61. 风险与可能失败点

## 61.1 原子化过细

后果：

- Graph 爆炸；
- 规划成本高；
- Tool 数量膨胀；
- 节点缺少独立复用意义。

缓解：SplitScore、复用证据、Effect 唯一性。

## 61.2 原子化过粗

后果：

- 失败仍无法局部定位；
- 跨 Composite 复用差；
- Tool 过拟合完整 task type。

缓解：partial success、failure cluster、跨任务 reuse 分析。

## 61.3 Tool Generalization 过度

后果：

- generic Tool 在特殊输入上 negative transfer。

缓解：

```text
generic + specialised 共存
+ replay
+ utility
+ suppression
```

## 61.4 Tool Repository 膨胀

缓解：

- canonical signature；
- hash dedup；
- merge；
- generalize；
- shadow；
- retire。

## 61.5 Layer-3 insight 过拟合 task_type

缓解：

- task_type 仅作为 source grouping；
- insight 使用仍通过当前 Skill/Composite 检索；
- 不自动注入所有同 type insight。

## 61.6 FlowEvo 原有 Primitive 已经很强

这是必须正面处理的实验风险。

因此必须包含：

```text
Original FlowEvo primitives
vs
Atomic state-effect graph + Tool Repository
```

消融，证明提升不是简单因为“多存几个 helper”。

---

# 62. MVP 实施路线

## Stage 0：完整复现 FlowEvo Baseline

目标：

- 原论文代码正常运行；
- 保存原始结果；
- 固定模型/配置/seed；
- 确认 Trace/Compiler/GateKeeper/Governance 行为。

任何新模块开发前必须完成。

## Stage 1：Atomic Skill IR + SkillGraph

实现：

- Abstract Atomic；
- Implementation Atom；
- Composite；
- Graph Registry；
- Runtime Graph；
- Node Validator；
- cold/warm planner。

暂不改变 FlowEvo Tool 生成。

目的：单独证明原子图价值。

## Stage 2：独立 Tool Repository

将：

```text
FlowEvo executable skill / primitive
```

改为长期 Tool Asset，并建立：

```text
Implementation Atom → tool_ref
```

实现：

- ToolRegistry；
- N:M binding；
- ToolResolver；
- admission status；
- tool statistics。

## Stage 3：Tool Evolution

实现：

```text
discover
parameterize
update
generalize
specialize
merge
split
add_test
suppress
retire
rollback
```

## Stage 4：Composite Skill + Layer-3 Insight

实现：

- success atomic chain → Composite；
- `_INSIGHT_MIN_SAMPLES = 3`；
- multi-trace insight update；
- Composite retrieval / reuse。

## Stage 5：完整实验

运行：

```text
Original FlowEvo
Atomic Graph only
Tool Repository only
Full AtomicSkillGraph
关键 ablations
```

## Stage 6：v2.1 前置但不实施

仅准备：

```text
Tool Adapter Schema
Harness Compatibility metadata
```

不进行跨 Harness 实验。

---

# 63. 推荐模块划分

```text
atomic_skillgraph/
├── core/
│   ├── skill_ir.py
│   ├── tool_ir.py
│   ├── refs.py
│   └── predicates.py
│
├── graph/
│   ├── registry.py
│   ├── graph.py
│   ├── aligner.py
│   └── validator.py
│
├── atomicizer/
│   ├── trace_atomicizer.py
│   ├── effect_extractor.py
│   ├── boundary_detector.py
│   └── split_score.py
│
├── tools/
│   ├── registry.py
│   ├── resolver.py
│   ├── compiler_adapter.py
│   ├── admission_adapter.py
│   ├── generalizer.py
│   └── lifecycle.py
│
├── runtime/
│   ├── atomic_planner.py
│   ├── implementation_selector.py
│   ├── runtime_graph.py
│   └── execution_bridge.py
│
├── validation/
│   ├── tool_validator.py
│   ├── node_validator.py
│   ├── composite_validator.py
│   └── failure_localizer.py
│
├── evolution/
│   ├── success_processor.py
│   ├── failure_processor.py
│   ├── composite_builder.py
│   └── insight_updater.py
│
└── adapters/
    ├── flowevo.py
    └── benchmark.py
```

这只是模块边界规范，不要求现在立即创建该目录。

---

# 64. 3D / Graph 可视化更新

v2.0 可视化建议：

## Abstract View

显示：Atomic/Composite 关系。

## Implementation View

```text
Abstract → Implementation → Tool Asset
```

Tool 作为外部虚拟资产节点显示。

## Composite Drill-down

点击 Composite：

```text
放大进入 Atomic 子图
```

## Runtime Trace View

显示：

- planned node；
- realized node；
- Tool binding；
- direct/seeded/dynamic；
- validator；
- failure。

## Evolution View

显示：

- generalize；
- specialize；
- merge；
- split；
- supersedes；
- suppress。

---

# 65. 核心设计冻结结论

1. v2.0 只研究集中式 Skill/Tool 自进化，不研究联邦。
2. 全系统只有一份长期 Global SkillGraph。
3. Tool Repository 也是集中式全局资产库。
4. 持久化 SkillGraph 有三类节点：Abstract Atomic、Implementation Atom、Composite Skill。
5. Runtime Task Instance 不进入长期 SkillGraph。
6. Tool 不是 SkillGraph 核心节点，但 Implementation View 可以显示 Tool 资产。
7. Abstract Atomic Skill 必须拥有一个稳定核心 Effect。
8. Atomic Tool 可以封装多个机械步骤，只要整体服务于一个稳定 Skill Effect 且可整体归因。
9. Skill 与 Tool 是 N:M，不强制 1:1。
10. Implementation Atom 不保存大段 executable code，只保存 `tool_ref + bindings + execution_policy`。
11. 不做 Composite Tool；Composite 只存在于 SkillGraph。
12. 不再存在持久化 `SKILL.md`。
13. Layer-2 guideline 成为 Atomic/Composite 的语义字段。
14. Layer-3 insight 属于 Composite Skill。
15. Layer-3 默认由同 task_type 至少 3 条 Trace 聚合产生。
16. task_type 不是能力作用域，Atomic Skill/Tool 可以跨 task type 复用。
17. Cold Start 由原始 Agent 先完成任务，成功后事后学习。
18. Warm Start 对已有能力进行前置 Atomic Planning。
19. LLM 不直接看到不断增长的 Tool Repository。
20. Planner 选择 Atomic Skill，Implementation Atom 再解析 Tool。
21. Tool 的具体 executable 形式跟随 FlowEvo 当前 Benchmark/Harness，不强制 Python。
22. 一次成功即可产生 Tool Skeleton。
23. Skeleton 必须通过 admission 才能成为 candidate。
24. Failure Trace 可以提出 Tool 修复，但 executable repair 必须成功 replay 后才能进入 candidate。
25. Tool 生命周期完整支持 discover/parameterize/add/update/specialize/generalize/merge/split/add_test/add_adapter/retire/rollback。
26. `add_tool_adapter` v2.0 只保留 Schema，跨 Harness 正式放到 v2.1。
27. Tool merge 与 Skill merge 是两个独立问题。
28. Generalized Tool 与具有独立价值的 specialised Tool 可以共存。
29. 原始成功 Trace 永久保留作为 provenance / replay evidence。
30. 系统保留 FlowEvo direct → seeded → dynamic 路由与 cheapest-yet-reliable 原则。
31. Tool-level Test、Atomic Validator、Composite Validator、Benchmark Verifier 必须分层。
32. 删除 v1.0 Patch 协议，不再有 Patch upload/approval/aggregation。
33. 保留不可变版本、lineage、rollback 和 evidence statistics。
34. FlowEvo 是首选 Runtime/Compiler/Admission/Governance 底座。
35. FlowEvo PrimitiveCompiler 作为 Code Tool Candidate Miner，而不是 Atomic Skill 定义器。
36. SkillOps 主要复用/借鉴 Contract、SkillGraph、maintenance，不替换 FlowEvo Runtime。
37. Trace2Skill 主要可复用 success/error trajectory analysis，不复用其 MD patch 作为主进化对象。
38. SkillOpt 主要借鉴 held-out validation gate 与 evolution discipline。
39. 第一阶段 Benchmark 尽量沿用 FlowEvo 原始设置，证明在相同条件下本方案是否优于原版。
40. 新增模块必须可以做逐项 ablation，以确认提升来自 Atomic Graph、Tool Repository、Tool Generalization、Composite Insight 中的哪一部分。

---

# 附录 A：最小 Tool Asset JSON Schema 草案

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AtomicSkillGraphToolAsset",
  "type": "object",
  "required": [
    "tool_id",
    "version",
    "status",
    "artifact_kind",
    "signature",
    "interface",
    "artifact",
    "tests",
    "safety",
    "provenance",
    "statistics",
    "lineage"
  ],
  "properties": {
    "tool_id": {"type": "string", "minLength": 3},
    "version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
    },
    "status": {
      "enum": [
        "draft",
        "admission_pending",
        "shadow",
        "candidate",
        "active",
        "preferred",
        "suppressed",
        "retired"
      ]
    },
    "artifact_kind": {"type": "string"},
    "signature": {"type": "object"},
    "interface": {"type": "object"},
    "artifact": {"type": "object"},
    "tests": {"type": "array"},
    "safety": {"type": "object"},
    "provenance": {"type": "object"},
    "statistics": {"type": "object"},
    "lineage": {"type": "object"}
  },
  "additionalProperties": true
}
```

---

# 附录 B：Implementation Atom JSON Schema 核心

```json
{
  "kind": "implementation_atomic",
  "id": "impl.example.operation",
  "version": "1.0.0",
  "implements": {
    "id": "example.operation",
    "version": "1.0.0"
  },
  "tool_bindings": [
    {
      "tool_ref": "tool://example/tool@1.0.0",
      "role": "primary",
      "parameter_mapping": {}
    }
  ],
  "execution_policy": {
    "mode": "direct_if_eligible",
    "on_failure": "seeded_then_dynamic"
  },
  "compatibility": {},
  "quality": {}
}
```

---

# 附录 C：Tool Generalization 伪代码

```python
def maintain_tool_repository(tool_registry, skill_graph, traces):
    candidates = retrieve_structurally_similar_tools(tool_registry)

    for group in candidates:
        behavior = compare_behavior(group)
        interfaces = compare_interfaces(group)
        effects = collect_bound_atomic_effects(group, skill_graph)

        if behavior.equivalent and interfaces.equivalent:
            merge_tools(group)
            continue

        if can_parameterize(group, effects):
            generalized = propose_generalized_tool(group)
            if replay_on_all_sources(generalized, group, traces):
                admitted = admission_engine.admit(generalized)
                if admitted.passed:
                    tool_registry.register_candidate(admitted.tool)
            continue

        clusters = discover_stable_special_cases(group)
        for cluster in clusters:
            specialized = propose_specialized_tool(cluster)
            if successful_replay(specialized, cluster):
                admitted = admission_engine.admit(specialized)
                if admitted.passed:
                    tool_registry.register_candidate(admitted.tool)
```

---

# 附录 D：Capability-based Retrieval 约束

禁止：

```python
if skill.task_type != task.task_type:
    reject(skill)
```

推荐：

```python
score = semantic_score(skill, task)
score += effect_score(skill, task)
score += io_fit(skill, task)
score += precondition_fit(skill, task)
score += historical_utility(skill)

if skill.task_type in task.related_types:
    score += TASK_TYPE_SOFT_BONUS
```

`task_type` 只能影响召回分数，不得推翻 Contract/Effect 层面的真实能力匹配。

---

# 附录 E：直接参考的公开代码与论文

## FlowEvo

- Paper: `FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills`
- arXiv: https://arxiv.org/abs/2607.21596
- Code: https://github.com/DEFENSE-SEU/FlowEvo
- License: Apache-2.0

重点源码：

```text
src/compiler/compiler.py
src/compiler/admission.py
src/compiler/gatekeeper.py
src/compiler/primitive_compiler.py
src/memory/skill_registry.py
src/memory/primitive_store.py
src/memory/trace_store.py
src/governance/
src/maintenance/governance.py
src/alfworld_/compiler.py
src/alfworld_/executor.py
src/alfworld_/param_extractor.py
src/alfworld_/run_20task_validation.py
src/code_math/runner.py
src/core/schemas.py
```

## SkillOps

- Paper: `SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems`
- arXiv: https://arxiv.org/abs/2605.13716
- Code: https://github.com/Hik289/SkillOps
- License: MIT

重点源码：

```text
skillops/skill_graph.py
skillops/maintenance.py
skillops/planner.py
```

## Trace2Skill

- Paper: `Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills`
- arXiv: https://arxiv.org/abs/2603.25158
- Code: https://github.com/Qwen-Applications/Trace2Skill
- License: Apache-2.0

重点可参考：

```text
analysis/
skill_evolver/
spreadsheet_agent/
```

v2.0 主要借鉴 `analysis/`，不以其 Markdown consolidation 为主状态。

## SkillOpt

- Paper: `SkillOpt: Executive Strategy for Self-Evolving Agent Skills`
- arXiv: https://arxiv.org/abs/2605.23904
- Code: https://github.com/microsoft/SkillOpt

主要借鉴：

```text
held-out validation gate
candidate accept/reject
rollback
multi-objective accuracy/cost evaluation
```

---

# 附录 F：最终系统一句话定义

> **AtomicSkillGraph 是一个建立在现有 Self-Evolving Agent Runtime 之上的结构化能力演化层：它从成功执行轨迹中学习以稳定状态 Effect 为边界的 Atomic Skill，将可执行行为沉淀到独立 Tool Repository，通过 Implementation Atom 建立 N:M Skill–Tool 绑定，再将 Atomic Skill 组合成具有多轨迹 Insight 的 Composite Skill，并使用 admission、节点验证、全局泛化/特化和长期 utility 治理，使 Agent 从“反复生成完整工作流”逐步转变为“规划可复用能力并执行经过验证的可执行部件”。**

---

# 附录 G：v2.0 实现对齐说明（2026-08-23）

本附录只说明当前仓库对设计的落地边界，不改变正文的目标定义。

1. Runtime Graph 已使用稳定 `step_id` 区分重复调用实例，并保存类型化边、节点 Effect、执行模式、fallback 原因和分阶段 attempts。ALFWorld 等支持原地续跑的状态环境按节点执行 Direct、Seeded、Dynamic；Code/Math 仍以完整入口程序为一个原子节点。

2. 六类边共用 `GraphEdge` IR。Structural、Control、Data、Dependency、Semantic、Evolution 均有 Schema 和验证；控制边具有声明式 condition 与有界 policy，数据边具有 output-to-input mapping。普通顺序 Trace 默认只生成有证据的 `next` 和可推断 `data_flow`，不会猜测复杂分支。

3. 多正向核心 Effect 的 Skill 候选会按 Effect 实际拆分。Tool generalize、specialize 和 split candidate 共用 Admission；泛化 Tool 通过后会生成可达的 Implementation 版本。Action-template 跨结构语义泛化、自动 adapter 和无分区证据的自动 Tool split 不在当前实现范围内。

4. Tool、Atomic、Composite、Benchmark 四层验证分别持久化。进化后运行 SkillGraph Validator 并记录错误数；校验失败会被报告，但当前 JSON 存储不提供跨文件事务回滚。

5. 失败只能产生 proposal 和证据。后续成功轨迹覆盖相同 Skill、Tool 或 Composite 范围时，proposal consumer 才将其标记为 replayed。涉及 executable body 的改变仍须来自可执行候选并重新通过 Admission，系统不会根据失败文本直接激活 LLM 生成的补丁。

6. 当前自动化测试和无 API smoke 用于证明闭环及数据结构正确性，不替代真实模型、真实 ALFWorld 大样本和正式消融实验。论文结果必须另外报告每类实际生成边数量、逐节点模式分布、图验证通过率和 proposal replay 数量。
