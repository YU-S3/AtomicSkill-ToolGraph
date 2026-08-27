# AtomicSkillGraph Workflow 说明

本文依据两类材料校准：一是 `AtomicSkillGraph_集中式原子SkillGraph与Tool联合进化_完整设计文档_v2.0.md`，二是仓库中的实际代码、测试和现有运行产物。凡是设计文档与代码不一致的地方，以“当前实现”和“设计目标”分开说明，不能把尚未接入的能力写成已经完成。

## 一、先给出结论

当前系统不是 Multi-Agent 系统，也不存在多个 Agent 互相发消息、讨论、投票或交接任务。它的真实形态是：一个 `AtomicSkillGraphSystem` 负责顺序编排，一个 LLM 实例在需要开放式推理时被调用，其余工作由一组普通代码模块完成。

当前实现的核心闭环是：先执行任务并记录 Trace；任务成功后，代码框架从成功 Trace 中切分能力、建立 Skill 节点、挖掘 Tool、做 Admission，再建立 Implementation 和可能的 Composite；下一次任务先检索这些能力，再决定 Direct、Seeded 或 Dynamic。SkillGraph 不是由 LLM 在任务开始前凭空设计出来的，而是从成功执行证据中逐步长出来的。

LLM 当前主要负责两类事情：生成或修复完整代码，以及在交互环境中逐步选择动作；在少数规划场景下，它还会从代码框架已经召回的 Atomic Skill 中选择子集和顺序。原子边界检测、Effect 提取、I/O 和 Validator 构造、Skill 对齐、Tool 挖掘、Admission、Tool 泛化、失败归因和 insight 聚合，目前都不是 LLM 做的，而是确定性的规则、AST、状态差分和统计代码做的。

经过本轮实现后，状态环境已经支持逐节点混合执行，六类边有统一的类型化 IR，四层验证结果会分别落盘，原子 split 会真实执行，Tool 具备 generalize、specialize、split 的候选生成与统一 Admission 入口，失败也会进入隔离分支。仍需准确区分“框架已经支持”与“每次实验都会自动触发”：例如 branch、parallel、loop 的执行语义已经存在，但当前成功链构造器通常只自动生成 `next` 和可推断的 `data_flow`；只有同一任务中出现成功补救轨迹时，失败分支才能从正证据生成并严格验证新的 executable。完全失败、没有正轨迹时，框架不会凭失败文本自行发明可合并 executable。

## 二、系统中谁负责什么

### 1. 集中式编排器

`src/atomic_skillgraph/system.py` 中的 `AtomicSkillGraphSystem` 是总编排器。它创建并调用 Planner、Implementation Selector、Tool Resolver、Execution Bridge、Trace Store、Atomicizer、Success Processor、Failure Processor、Skill Registry 和 Tool Registry。

这些对象都是同一进程中的代码模块，不是自治 Agent。它们之间通过函数调用和 Python 对象传递数据，没有 Agent 间通信协议。

### 2. Runtime Agent 和 LLM

这里所说的 Runtime Agent，是“LLM 在当前任务中进行开放式推理和生成”这一逻辑角色，不是一个与其他 Agent 对话的独立服务。

在 Code/Math 任务中，LLM 接收题目和可选的 Skill/Tool 上下文，生成完整代码；验证失败时，代码适配器把 verifier 反馈交给同一个 LLM，让它修复代码。

在 ALFWorld 一类交互任务中，LLM 根据当前 Observation、历史动作和可选 Skill 上下文选择下一条动作。环境执行动作并返回新的 Observation，然后同一个循环继续。

在规划阶段，只有任务没有提供 `target_effects` 时，Planner 才可能让 LLM 从已经检索出的 Skill 候选中选择一个有序子集。LLM 不能浏览整个 Tool Repository，也不直接决定具体使用哪个 Tool。即使 LLM 返回参数，当前 Planner 仍会用代码从 `task.context.params` 或任务文本中重新绑定参数。

### 3. 代码框架

代码框架负责以下工作：

- 初始化和持久化 SkillGraph、Tool Repository、Trace、运行记录和指标；
- 从任务目标构造检索请求并给 Skill 候选打分；
- 检查 Implementation 的 Harness 兼容性和历史质量；
- 按 `tool_bindings` 解析 Tool 和参数；
- 判断 Direct 是否满足可用生命周期状态、可靠性、前置条件和参数门槛；
- 调用环境、Sandbox、测试和最终 verifier；
- 记录原始 Observation、Action、状态快照、验证结果和成本；
- 对成功 Trace 做边界检测、Effect 差分、Skill 对齐、Tool 挖掘和注册；
- 对失败 Trace 做规则归因并保存修复提案；
- 管理版本、状态、统计和部分 Tool 生命周期。

原始 Trace 中的事实必须由代码记录。以后即使增加 LLM 生成的语义摘要，摘要也只能作为附加信息，不能替代原始 Observation、Action、参数、状态和验证结果。

### 4. Environment、Harness 和 Verifier

Environment 或 Benchmark Harness 才真正执行动作或代码。LLM 只提出动作或候选代码，不直接改变环境状态，也不能自行宣布成功。

Code/Math 任务的最终成功由测试和 verifier 决定。ALFWorld 的最终成功由环境的 `won` 结果决定。这个 Benchmark 级结果是当前主流程中始终实际生效的最终判断。

### 5. 当前到底复用了多少 FlowEvo

实验入口把 FlowEvo baseline 和 AtomicSkillGraph ours 分成两条独立路径。baseline 条件在子进程中调用 vendored FlowEvo 的特定入口，ours 条件运行 `AtomicSkillGraphSystem`，不是在原 FlowEvo Agent 外再套一层图。子进程会重新加载自己的任务，也不与 ours 共享同一批内存 Task 对象。

baseline 本身也不是一条统一的“完整 FlowEvo”链。`baseline_dynamic` 在 Code/Math 中映射到简化 runner 的 `cot_baseline`，在 ALFWorld 中映射到 `pure_dynamic`；`flowevo` 在 Code/Math 中映射到简化 runner 的 `ours`，在 ALFWorld 中映射到 `full_library`。Code/Math runner 主要做整题 LLM 生成、验证、重试和 solution 缓存；ALFWorld 才使用其 Generator、Executor、Compiler 和 SkillLibrary。仓库中更完整的 `flowevo.eval.runner.BenchmarkRunner` 没有被当前实验入口调用，所以两个 Benchmark 下同名的 `flowevo` 条件并不代表完全相同的内部能力。

ours 当前复用的是 FlowEvo 的部分底层设施。HumanEval 使用其 loader、verifier 和 Sandbox；GSM8K 使用其 loader 和 Sandbox，但答案归一化比较由 Atomic Adapter 自己完成；ALFWorld 使用其任务加载、环境和参数提取；LLM 通过 FlowEvo 的 LLM Client 包装。Atomic Tool Admission 使用的是本项目自己的 `atomic_skillgraph.tools.sandbox.Sandbox`，不是 FlowEvo Admission/GateKeeper。Runtime 循环、Atomicizer、Skill/Tool Registry、Tool Miner、Admission、路由和 Governance 主要由本项目重新实现。

`SuccessProcessor` 虽然保留了可选的 `base_compiler` 接口，`adapters/flowevo.py` 也提供了 Trace 和 Primitive 的转换函数，但 `AtomicSkillGraphSystem` 没有给 `base_compiler` 传入原 FlowEvo Compiler，这些转换函数也没有接入主流程。因此，当前不能说“成功 Trace 会同时经过原 FlowEvo Compiler 和新的 Atomicizer”。

## 三、当前代码实际执行的完整 Workflow

本节以下步骤只描述 AtomicSkillGraph 的 ours 路径。FlowEvo baseline 不经过 `AtomicSkillGraphSystem`、Atomic Trace、Atomicizer、Tool Repository 或后面的 Success/Failure Processor。

### 第一步：初始化持久化仓库和执行模块

系统启动时创建 Skill Registry、Tool Registry、Trace Store、Runtime Run Store、Proposal Store 和 Metrics Store。新实验目录中没有历史数据时，Skill 节点、图边和 Tool 都是空的。

随后系统装配 Planner、Selector、Resolver、Execution Bridge、节点 Validator、Atomicizer、Tool Admission、Success Processor 和 Failure Processor。这个阶段只建立框架，不会预先让 LLM 生成 SkillGraph。

### 第二步：接收任务

任务至少包含任务编号、Benchmark、`task_type`、自然语言目标和上下文。任务还可以提供 `target_effects`、参数、初始状态、环境编号或入口函数等信息。

在默认完整配置中，`task_type` 只是检索加分信号，不是独立分仓，也不是硬过滤条件，因而不同 `task_type` 可以复用同一 Atomic Skill 或 Tool。消融配置可以启用 `task_type_hard_restricted` 或关闭跨类型复用，此时它会变成硬限制。

### 第三步：从 Skill Registry 检索已有能力

Planner 先构造查询文本和目标 Effect，然后让 Registry 通过确定性规则打分。分数主要来自文本词重合、Effect 匹配、I/O、前置条件、utility 和 `task_type` 加分，没有 embedding 检索，也没有 LLM 语义重排。

Planner 优先考虑可用的 Composite，再考虑 Abstract Atomic Skill。如果 Registry 没有相关节点，规划结果中的节点列表为空，任务被标记为 Cold Start。

### 第四步：形成当前的 Runtime Plan

如果任务明确给出 `target_effects`，Planner 用贪心 Effect 覆盖选择已有 Atomic Skill，整个过程由代码完成。

如果任务没有 `target_effects`，Planner 才可能让 LLM 从已召回候选中选择子集和顺序。这个 LLM 只能选择已经存在的 Skill，不能在 Runtime Plan 中创造一个尚不存在的“未知 Atomic Skill 节点”。

`RuntimeGraph` 现在保存稳定的 `step_id`、节点目标 Effect、逐节点尝试记录和类型化边。框架提供了 `next`、`branch`、`parallel`、`retry`、`fallback`、`loop` 与 `data_flow` 的确定性调度语义，条件采用声明式字段比较，不执行任意代码。当前 Planner 从普通成功链自动生成的仍主要是 `next` 和能够由相邻 I/O 推断出的 `data_flow`；复杂控制边只有在 Composite 明确声明时才进入 Runtime Graph。

如果任务提供 `target_effects`，已有节点只能覆盖其中一部分，Planner 会为未覆盖 Effect 加入 `dynamic_gap` 运行时节点。它只存在于本次 Runtime Graph，不进入长期 SkillGraph，也不伪装成已经学会的 Abstract Skill。

### 第五步：判断 Cold Start 还是 Warm Start

如果没有计划节点，系统直接进入 Dynamic。此时不会先生成 Skill，也不会为未来能力预造 Tool。

如果检索到了计划节点，系统进入 Warm Start。对每个已有 Atomic Skill，Implementation Selector 先按 active 状态、Harness 兼容性、质量和可解析 Tool 选择 Implementation；Tool Resolver 再按 `tool_bindings` 找到 Tool 并绑定参数；最后才执行 Direct Gate。

Direct Gate 检查 Implementation 是否 active、全部 Tool 是否成功解析且处于 candidate/active/preferred、历史成功证据和 utility 是否达标，以及 Atomic Skill 的前置条件是否满足。Harness 兼容性和参数绑定由 Selector/Resolver 先检查；Runtime Tool Validator 再检查当前参数、模板 slot 和 `safety.direct_execution_allowed`，任一 Tool 验证失败都会关闭 Direct。suppressed、retired、shadow 或 draft Tool 不属于 usable 状态。Atomic Skill 自身状态主要由检索阶段过滤。

### 第六步：Cold Start 的实际执行

Code/Math 冷启动时，LLM 生成完整候选代码。代码由 benchmark verifier 运行测试；失败时，同一个 LLM 可以根据测试反馈进行若干次修复。通过测试才算任务成功。

ALFWorld 冷启动时，LLM 根据 Observation 逐步给出动作，环境执行并返回下一条 Observation。达到环境目标才算成功。

这里的 Dynamic 是“没有可用 Skill 上下文的完整任务生成或交互”，不是 LLM 在后台先构造一张图。

### 第七步：Warm Start 的实际执行

状态环境现在按 Runtime Node 逐个执行。每个节点独立准备 Direct、Seeded 和 Dynamic 候选：Direct 模板执行后立即检查该节点 Effect；未通过只把当前节点降级为 Seeded，仍未通过再降级为 Dynamic。节点 Effect 一旦满足，框架保留同一环境对象和当前 Observation，从该状态继续下一个节点，不会重新 reset episode。`dynamic_gap` 节点直接由 Dynamic Agent 完成。全部节点通过但环境尚未给出 `won` 时，框架再做一次无 seed 的任务级收尾，最终成功仍由环境决定。

Planner 选择 Composite 时先计算其子 Atomic Effect 的闭包，再用任务目标 Effect 做充分性硬过滤。完整覆盖目标的 Composite 优先，summary 文本相似度和历史 utility 只能在同样充分的候选之间排序。没有完整 Composite 时才允许补 dynamic gap；gap 从任务规范参数中绑定对象、目标位置和处理设备，仍有占位符无法绑定时放弃该 partial Composite，回到 Atomic greedy plan。

完整 Composite 中经过验证的因果节点不会因为输入暂时未绑定而被删除。某节点的 Effect 如果是后续节点的 Precondition，说明它是必须保留的生产者。Runtime 依次尝试从任务语义参数、当前状态、DATA_FLOW 和受控环境发现中绑定缺失参数；仍无法执行时让该节点失败并阻断依赖它的后续节点，不允许后续节点越过 Atomic 边界把前置能力吞并到自己名下。冻结与非冻结、Atomic-only 与 Full 都遵守同一条因果规则。

Seeded 或 Dynamic 执行某个节点时，框架给 LLM 的目标是该节点绑定后的 Effect，例如“只完成拿取 apple 这一步”，而不是再次给它整题目标。因此 LLM 不会在 Acquire 节点中顺手完成 Heat 和 Place。框架在动作后只用当前节点的 Effect 判定该节点是否完成；环境整题 `won` 只负责最终任务成功，不能反过来把尚未验证的节点标成成功。

Contract 提取也按动作因果范围收紧：Acquire 只保留目标对象及其来源位置，Heat、Clean、Cool 和 Place 只保留手中目标对象这类必要条件，不把同一房间里偶然出现的碗、杯子等物体写成前置条件。`apple` 与环境实例名 `apple_1` 按同一对象类匹配；`object.in_receptacle` 在规划覆盖判断中归一为 `object.at_location`。Validator 的前置条件用于判断节点执行前是否可进入，节点执行后的成功只由目标 Effect 和后置条件决定，不能因为 Effect 已完成但旧 Observation 没显式重述前置事实而误判失败。

Tool 的跨工具泛化仍按维护周期运行，但 candidate→active 的轻量生命周期审查在每次成功后运行。这样 Admission replay 已通过且 support 达到阈值的 Tool 会及时进入 active，后续任务可以积累真实 Direct 证据；不会因为小规模实验未刚好达到第 5 次成功而一直停留在 candidate。

因此，在支持原地续跑的 ALFWorld 路径中，“Acquire 用 Direct、Clean 用 Dynamic、Place 再用 Direct”已经是可执行模式。每次动作仍由同一个 Runtime Agent 或 Tool 模板产生，不存在多个 Agent 交接环境。

Code/Math 仍以整个入口程序作为一个 Atomic Skill，无法把一个函数内部的若干语义步骤安全地原地续跑，所以它保留整题 Direct、Seeded、Dynamic 路由。Seeded 失败后 Dynamic 的 attempts 现在会追加保存并带阶段前缀，不再覆盖前一阶段；没有 seed context 时会直接记为 Dynamic，不再误记为 Seeded。

### 第八步：Agent、框架和环境如何在一次任务中交互

Code/Math 的交互顺序是：框架组织 prompt，LLM 生成代码，框架调用 verifier，verifier 返回测试结果；如果允许修复，框架把失败反馈再次交给同一个 LLM。这里没有第二个“验证 Agent”。

交互环境的顺序是：框架把 Observation 和上下文交给 LLM，LLM 返回一个动作字符串，框架检查并提交给环境，环境返回新的 Observation 和状态，框架继续下一步。Direct 时，动作主要来自已经注册的 Tool 模板，不需要 LLM 决定 Tool。

任务结束后，`AtomicSkillGraphSystem` 顺序调用 Success Processor 或 Failure Processor。Atomicizer、Tool Miner、Admission 和 Generalizer 彼此也只是函数调用，不是几个 Agent 在协商。

### 第九步：记录 Trace 并做最终验证

`TraceRecord` 的 Schema 可以保存任务、检索结果、计划节点、实现引用、Tool 引用、参数、代码尝试或环境动作、Observation、状态快照、节点验证、重试、成本和最终结果。当前没有一个独立的 Trace Recorder；这些字段由 `system.py` 和 Adapter 在执行后分段填充。

Code/Math 的状态快照目前是简化表示：开始时没有成功事实，测试通过后加入 `callable_returns_expected(entry)`。ALFWorld 的状态由适配器从自然语言 Observation 中用规则解析成累计事实。

Warm 执行时，Tool 和 Implementation 反馈会持久化。系统先保存原始 Trace，再执行成功或失败进化；进化完成后运行 SkillGraph Validator，把边 Schema、固定版本端点、Composite 顺序和 Tool Binding 的检查结果写入本轮指标，并再次保存带验证结果的 Trace。最后保存 Runtime Execution Instance 和 episode 指标。

### 第十步：成功 Trace 的原子化

当前 Atomicizer 完全由规则代码完成，没有 LLM 调用。

对 ALFWorld，Boundary Detector 比较连续状态快照中的事实变化。纯移动、查看、盘点、检查和开门等机械动作通常不单独形成原子能力，而是附着到下一个产生核心状态变化的片段。`location_checked`、`agent_at` 等导航事实会被过滤，主要保留拿取、放置、清洁、加热、冷却、点亮等任务 Effect。

Effect Extractor 用 `after facts - before facts` 得到新增效果，再用动作参数把具体实例值替换为输入槽位。输入来自动作参数，输出来自 Effect 的参数，前置条件从切分前状态中的部分已知事实推断，Validator 则由 Effect predicate 名构造。

对 Code/Math，当前不会按内部语义或 helper 调用链切出多个 Atomic Skill，而是把整个入口函数视为一个原子片段。它的 Effect 是简化的“该入口函数返回测试期望结果”。helper 只可能在后面的 Tool Miner 中成为 Tool 候选，不会因此成为 Atomic Skill 节点。

Atomicizer 还计算规则型 SplitScore。当一个片段包含多个正向核心 Effect 且判定为 `split` 时，代码会按 Effect 生成多个子候选，每个子候选有自己的 Effect、Validator、片段和引用；后续 Tool 挖掘按这些子片段对齐，不再用原整体候选代替拆分结果。

### 第十一步：构造并对齐 Abstract Atomic Skill

代码根据片段生成 `summary`、I/O、preconditions、effects、validator 和固定模板式 guideline，然后与 Registry 中的已有 Atomic Skill 对齐。

当前对齐也是确定性规则：比较 summary/guideline 的词重合、I/O、Effect 和 Validator；不同 logical id 的候选只有 Effect 严格等价且总分达到阈值才复用。同 logical id 会直接视为匹配。

没有匹配时注册新的 Abstract Atomic Skill；有匹配时尝试复用并合并证据。Abstract Skill 可以先独立存在，因为 Tool Admission 可能尚未通过。此时它只是“系统认识了这个能力合同”，不代表已经有可 Direct 的实现。

### 第十二步：从成功产物中挖掘 Tool Skeleton

Tool 不是由另一个 Tool Agent 或 LLM 重新发明的。它来自刚刚成功的代码或动作轨迹，再由代码规则转换为可执行候选。

Code/Math 的 Tool Miner 使用 AST。它会产生一个以成功完整代码为基础的主 Tool 候选，并最多提取两个可从入口函数到达的 helper 候选。函数签名被转成 Tool interface，代码正文保存在 Tool artifact 中。不过 helper 候选当前固定没有 replay tests，而 Code Admission 强制要求测试，所以首次挖出的 helper 通常进入 shadow，尚未形成稳定的 helper 复用闭环。

ALFWorld 的 Tool Miner 把成功片段的动作序列转成 action template，并用动作参数把具体对象替换成 `{slot}`。当前 artifact 实际仍保存完整片段步骤，不是只保留已经证明必要的最小核心动作。

刚挖出的 Tool 初始是 draft，版本从 `0.1.0` 开始。成功 Trace 只是 Tool 候选的来源证据，不能替代 Admission。

### 第十三步：Tool Admission

Code Tool 的 Admission 检查 AST 语法、安全黑名单、入口函数、明显无效的常量返回、Sandbox 测试、重复执行确定性和结构哈希去重。

环境 Tool 的 Admission 检查 interface、slot、步骤非空和长度，并在来源环境上 replay，比较预期 Effect 是否成立。当前所谓“动作合法性”主要只是非空和长度检查，还没有完整的命令语法验证；replay 也主要针对来源实例，没有设计文档中更强的 fresh-world 扰动测试。

Admission 通过后，Tool 才进入 candidate；失败则进入 shadow。shadow 可以保存用于分析，但不能正常执行。不能把“来自成功 Trace”直接等同为“已成为可用 Tool”。

### 第十四步：建立 Implementation Atom

至少一个 Tool 通过 Admission 后，Success Processor 才建立或更新 Implementation Atom。Implementation 保存它实现的 Abstract Skill 固定版本、兼容 Harness、执行策略、Tool Binding、参数映射和质量统计，不保存 executable body。

Abstract Skill 与 Tool 的真实关系由 Implementation 中介：Abstract Skill 表达“做什么”，Implementation 表达“在某种环境中如何实现”，Tool 保存真正可执行的代码或动作模板。

Schema 允许一个 Implementation 绑定多个 Tool，也允许不同 Implementation 共享同一个 Tool。不过当前 Code/Math 的 Direct 执行只使用解析结果中的第一个 Tool；环境 Direct 会展开同一 Implementation 中的全部动作 Tool。因而 N:M 数据结构已经存在，但各种执行路径并未完全支持组合语义。

### 第十五步：建立 Composite Skill

一次成功 Trace 只有在得到至少两个 Atomic Skill 引用时才建立 Composite。Composite 为每次调用保存独立 `step_id`，所以同一个 Atomic Skill 可以在一条流程中重复出现。普通成功链生成相邻 `next`；若前一节点 output 与后一节点 input 名称或语义类型一致，还会生成带 source output、target input、schema 和 identity transform 的 `data_flow`。

当前 Composite guideline 是规则模板生成的简短说明，不是 LLM 从 FlowEvo Layer-2 artifact 中抽象出来的。相同 logical-id 链会被视为同一个 Composite 候选。

同一 `task_type` 默认累积到至少三条成功 Trace 后，Insight Updater 可以根据位置、工作站和重复动作等词频规则生成 Layer-3 insight。这个过程当前也没有调用 LLM，而且它读取该 `task_type` 的全部成功 Trace，没有验证每条 Trace 是否形成或使用当前 Composite，所以 insight 可能混入同类型其他 Atomic 链的经验。

### 第十六步：周期性 Tool 维护

Tool 生命周期代码在 `support_count` 达到配置门槛，而且存在真实 Direct 成功或 Admission 严格 replay 成功时，把 candidate 晋升为 active；在相同 `artifact_kind:entry_point` 分组内，active Tool 的 utility 明显领先时可设为 preferred。Admission replay 单独记为 `admission_replay_success_count`，不计入 Runtime 的 `call_count`、`success_count` 或 `direct_success_count`。`record_usage` 只记录真正执行该 Tool 的调用、成功、失败和连续失败计数；达到高失败率或连续失败门槛时可 suppressed，utility 长期很低时再 retired。统计与状态都写回版本文件，检索不会继续读取旧状态。

Generalizer 能做结构重复合并和 Python Tool 的 AST 常量参数化，并为特殊输入约束生成 specialized candidate，也能按已确认的步骤/Effect 分区生成 split children。generalize、specialize 和 split 都走同一个 Admission 入口，失败只能进入 shadow。成功的泛化产物会建立新的 Implementation 版本和 `generalized_from` 关系，因此 Resolver 可以到达它。当前仍没有自动 action-template 语义泛化和 `add_adapter` 生成；specialize/split 需要上游提供有证据的约束或分区，框架不会猜测分区。

Tool 的 executable、signature、tests 或安全契约发生实质变化时会产生新版本；普通成功只更新统计。Repository 保存 recommended version 指针，回滚改变推荐指针而不是覆盖旧 artifact。Tool 修复成功时还会产生新的 Implementation 版本，并以 `supersedes` 连接新旧 Implementation；Tool 自身的父版本和来源 Trace 保存在 Tool lineage/provenance 中。

### 第十七步：失败后的处理

每个失败 attempt 都先独立记账。Direct Tool 失败时，失败立即记到实际执行的旧 Tool；随后 Seeded 或 Dynamic 即使把任务救回来，也不能给旧 Tool 记成功。同一任务对同一 Atomic 的节点执行失败和整题失败证据分开保存：`failure_count` 表示实际节点失败次数，`task_failure_count` 表示使用该节点的失败任务数，因此不会把一次任务重复算成多次 Atomic 执行失败。

失败处理随后为每个建议建立独立目录 `data/evolution/branches/<branch_id>/`，把当时的 SkillGraph 和 Tool Repository 复制到分支 bank。修改先发生在分支，main 的推荐版本不变。若 Direct 失败后 Seeded/Dynamic 成功，框架从“失败 Direct 动作 + 成功补救动作”生成新的 Tool patch，比较新旧步骤，给新 Tool 建新的 Implementation patch，并在来源失败任务上执行严格 Direct replay。这里不允许 Seeded 或 Dynamic fallback。只有严格 Direct 成功、Tool Admission 通过且分支图校验通过，才把候选版本和 `supersedes` 关系合并到 main；任何一项失败都保留分支审计，不切换 main 推荐指针。

若 Seeded 失败而 Dynamic 成功，框架把已验证的补救动作顺序写入新的 Abstract Atomic Skill guideline，仅在分支中重放一次强制 Seeded 路径。只有来源任务成功且图校验通过才合并新 Atomic 版本。Composite 校验失败也会建立独立分支，但没有成功顺序证据时不会猜测控制边。完全失败、没有成功补救轨迹时，分支中仍会实际生成 shadow/draft 候选副本，写入失败观察、回归案例和合并门槛；因为没有正样本，它只能等待后续成功证据，不能合并。

Code/Math 同一 episode 内仍可把 verifier 反馈交给 LLM 修复本题代码。若修复代码成功，框架会用成功代码形成 Tool patch，并在原任务测试上严格验证后才考虑合并；LLM 的“本题修复成功”本身不会直接修改长期仓库。

## 四、SkillGraph 如何从零开始长出来

第一轮任务开始时，`graph.json`、节点版本文件和 Tool Registry 可以都是空的。Planner 检索不到 Skill，于是走 Dynamic。

如果任务失败，系统保存失败 Trace、节点级失败反馈和隔离修复分支。完全失败且没有成功补救轨迹时，只生成不能合并的 shadow 候选；若同一任务内较低层路径成功补救，则可以从成功后缀形成 executable 或 guideline patch，但必须在原失败任务上严格 replay 成功后才注册为 main 的新推荐版本。

如果任务成功，框架先从 Trace 中找出一个或多个状态转移片段，并为每个片段构造 Abstract Atomic Skill 候选。没有匹配就新增，有匹配就复用。

接着框架从同一成功片段中挖掘 Tool Skeleton。Tool 通过 Admission 后成为 candidate，然后才以 Tool Binding 为基础建立 Implementation Atom。

如果本次成功包含两个或更多 Atomic Skill，框架再把它们的顺序链注册为 Composite Skill。

Registry 会自动产生 Implementation 指向 Abstract 的 `implements`、Composite 指向各 step 所用 Skill 的 `contains`，并持久化 Composite 的类型化 control、data、dependency、semantic 和 evolution 边。Implementation 注册时还会记录 Harness 的 `requires_environment`；相似但未合并的 Atomic 候选可产生带置信度和 Trace 证据的 `similar`；Insight 形成新版本时会产生 `supersedes`。Runtime 边只进入执行记录，不写回长期 SkillGraph。

后续任务检索到这些节点后，才可能进入 Warm Start。随着 Tool 证据增加，Tool 可能从 candidate 晋升为 active 或 preferred，Direct 的机会才逐渐增加。这个增长过程是“成功执行—事后提取—验证入库—下次复用”，不是先由 LLM 设计完图再执行。

## 五、节点的设计标准与当前代码门槛

### 1. Abstract Atomic Skill

设计标准要求一个 Atomic Skill 具有稳定的输入输出、明确前置条件、一个主要 Effect、可在任务结束前独立验证、失败可局部归因，并且有跨任务复用价值。原子性由 Contract 和 Effect 决定，不由动作数量、函数数量或代码行数决定。一个动作可能太细，多个动作也可能共同完成一个原子 Effect。

当前代码保存并校验这些字段；SplitScore 判定多个核心 Effect 需要拆分时会真正生成多个子节点。尽管如此，规则提取仍不能从数学上证明“长期可复用”或“所有环境下都可独立验证”。因此“原子节点不能混合”的准确含义是：一个 Abstract Atomic 节点只能声明一个稳定的主要状态转变，不能把“拿取并清洗并放置”塞进同一节点；它并不表示一个节点只能有一个动作，也不表示 Runtime Graph 不能混合 Direct、Seeded 和 Dynamic 模式。

Code/Math 当前尤其粗：logical id 基本是 `<benchmark>.<entrypoint>`，整个入口函数只有一个“测试通过”Effect。不同算法可能被压到同一 logical id；入口参数也没有稳定进入 Atomic Skill inputs，带参数 Tool 因而常常无法满足 Direct 参数绑定。

### 2. Implementation Atom

设计标准要求 Implementation 固定引用一个 Abstract Skill 版本，声明兼容 Harness，至少有一个 Tool Binding，并保存参数映射、执行策略、质量和验证证据。它可以组合多个 Tool，但整体只能实现所引用 Atomic Skill 的主要 Effect。

当前 IR 确实强制至少一个 Tool Binding，Selector 和 Resolver 也按兼容性、状态、utility 和参数绑定工作。不过“多个 Tool 合起来是否只实现一个 Effect”没有严格验证，Code Direct 也只执行第一个解析 Tool。

### 3. Composite Skill

设计标准要求 Composite 由 Skill 节点组成，保存控制关系、数据映射、复合 Effect、Composite Validator、Layer-2 guideline 和可选 Layer-3 insight。Composite 是可复用的工作流结构，不应编译成一个不可解释的 mega-tool。

当前 Composite 已经是可成图的 step-instance 子图：它保存固定版本节点、类型化控制边、可推断数据边和高层 Validator。普通 Trace Builder 只从顺序成功链自动生成 `next`，不会无证据创造 branch、parallel、retry、fallback 或 loop；但这些边的 Schema、校验和 Runtime 调度语义已经实现，手工或后续编译器提供相应声明即可执行。Composite Validator 已接入任务结束后的独立验证记录。

### 4. Runtime 节点和 Tool

Task Instance、Runtime Atomic Instance 和 Validator Instance 只进入 Trace 或 Runtime Run，不是长期 SkillGraph 节点。

Tool Asset 也不是核心 SkillGraph 节点。Tool 独立存放在 Tool Repository；SkillGraph 通过 Implementation 的 `tool_bindings` 间接引用 Tool。界面可以把这种关系展示成虚拟 `uses_tool`，但当前持久化图中没有正式 `uses_tool` 边。

## 六、边的设计标准与当前落地情况

设计文档定义了六类边，含义如下。

1. Structural：`contains` 表示 Composite 包含 Skill，`implements` 表示 Implementation 实现 Abstract Skill。

2. Control：`next`、`branch`、`parallel`、`retry`、`fallback`、`loop` 表示执行顺序和控制流。除显式 loop 外，一次 Runtime Graph 原则上应是 DAG。

3. Data：描述某节点输出如何绑定到另一节点输入，应包含 source output、target input、schema、参数或 artifact mapping。

4. Dependency：`requires_skill`、`requires_permission`、`requires_environment`、`requires_schema` 表示执行依赖。

5. Semantic：`equivalent`、`similar`、`alternative`、`conflict` 表示能力语义关系。

6. Evolution：`derived_from`、`supersedes`、`split_from`、`merged_from`、`generalized_from`、`specialized_from` 表示 Skill 版本或结构演化关系。Tool 自己的 lineage 应保存在 Tool Repository，不能用 SkillGraph evolution edge 代替。

六类边现在共用 `GraphEdge` IR。每条边都有确定性 `edge_id`、固定端点、category、scope、可选 step-instance、condition、mapping、policy、evidence 和 metadata，并按边类型校验必填字段。Structural、Dependency、Semantic、Evolution 已有生产路径自动生成实例；Control 和 Data 会从 Composite 进入 Runtime Graph。框架还实现了复杂 Control 和 Data 的执行语义。

设计上边的端点应引用固定版本，不能只靠会漂移的名字。当前 `implements` 的实际持久化方向是 Implementation 指向 Abstract，`contains` 是 Composite 指向成员。设计文档某些界面示例把 `implements` 画成反方向，属于表示口径不一致，应以 IR 和代码中的方向为准，或者在后续规范中统一。

重复调用通过 `step_id` 区分；branch 使用安全的声明式 condition；retry、fallback、loop 和 parallel 使用有界 policy；data mapping 至少声明 `source_output` 和 `target_input`，并可附 schema/transform；scope 明确区分 global、composite 和 runtime。这里“六类边已实现”不等于每次训练都会出现六类边：系统只根据可验证证据生成相应边，尤其不会从普通顺序 Trace 猜测复杂分支。

## 七、Tool 是怎样产生、使用和演化的

Tool 的来源必须是可追溯的成功产物。当前 Code Tool 来自成功代码的 AST，环境 Tool 来自成功动作片段的模板化。LLM 在执行任务时产生了原始代码或动作，但“哪一段成为 Tool、接口是什么、怎样参数化、是否入库”由代码框架决定。

Tool 先以 draft 形式产生，经过 Admission 后才成为 candidate；失败则进入 shadow。candidate 是否能用于 Direct 还要经过更严格的 Direct Gate。达到支持数和至少一次成功证据后可以 active，再按当前 `artifact_kind:entry_point` 分组中的 utility 成为 preferred。达到高失败率门槛时可以 suppressed；suppressed 后 utility 继续低于退役门槛才会 retired。

Runtime Planner 选择的是 Atomic Skill，不直接面对不断增长的 Tool Repository。Implementation Selector 先选择兼容实现，Tool Resolver 再根据固定 binding 找到 Tool 并绑定参数。这个分层避免让 LLM 每次都在所有 Tool 中做开放式搜索。

当前已有挖掘、参数化、Admission、注册、重复合并、Python 泛化、specialize/split candidate、生命周期晋升/抑制/退役、推荐指针和回滚。失败 proposal 可被后续成功证据消费；同一任务的成功补救轨迹也能产生 Tool patch，并绑定到新 Implementation。尚未自动化的是：仅根据完全失败的自然语言描述发明新的 executable body、action-template 的跨轨迹语义泛化、自动 add_adapter，以及在没有可靠分区证据时自动 split。论文应把这些列为受控限制，而不是已完成实验操作。

证据不足时，多个相似 Tool 应保持分开，不能只因表面相似就合并。新 generalized Tool 也必须在来源实例上 replay 并重新 Admission，不能仅凭 LLM 或规则提出的抽象直接注册为 active。

## 八、四层验证在设计中是什么，当前接到了哪里

设计文档实际定义的是四层验证，虽然其中一个章节标题误写成“三级验证体系”。四层分别是 Tool、Atomic、Composite 和 Benchmark。

Tool 层回答“这个 executable 自身是否安全、接口正确并能 replay”。入库时由 Admission 做完整安全与 replay；Runtime 准备 Direct 时还会调用 Tool Validator 检查当前参数、模板实例化和安全标志，结果写入 `validation_layers.tool`。

Atomic 层回答“这个节点执行后是否实现了自己的主要 Effect”。状态环境的 Direct、Seeded 和 Dynamic 节点都在各自状态边界验证，Effect 未满足会触发当前节点 fallback；动态 gap 也按自己的目标 Effect 验证。Code/Math 因整个入口就是一个原子节点，测试通过事实充当该节点的简化 Effect。

Composite 层回答“整条子图是否满足复合目标和结构约束”。任务结束后，如果本次选择了 Composite，主流程会检查成员 Atomic 结果、声明的高层 checks 和控制覆盖，并独立保存结果。

Benchmark 层回答“最终任务是否通过官方测试或环境目标”。这一层当前实际生效，也是最终成功标准。

四层结果现在分别保存，不能互相替代。Benchmark 仍是论文任务成功率的最终口径；中间层用于 Direct 门禁、局部 fallback、错误归因和审计。普通成功学习后会运行 Graph Validator 并写入结构错误数。失败驱动的语义修复采用更严格的事务式门禁：先校验分支，合并后再校验 main；若 main 校验异常，推荐指针回滚到父版本。正式实验仍把任何 `skill_graph_valid=0` 视为硬失败。

## 九、设计文档与当前实现仍有的误差和限制

### 1. FlowEvo 联合执行范围被写大了

设计希望最大限度复用 FlowEvo 的 Agent、Compiler、Admission、Gatekeeper、Sandbox 和 Governance。当前 ours 只按 Benchmark 复用了部分底层环境、加载、验证、执行 Sandbox 和 LLM Client；原 FlowEvo Compiler 没有接入成功学习主链，Tool Admission 使用本项目自己的 Sandbox，Admission 和 Governance 也是本项目自己的实现。

### 2. 混合路由只适用于可观察、可原地续跑的状态环境

ALFWorld 已实现逐节点 mixed execution 和动态缺口节点。Code/Math 仍把完整入口程序视为一个原子能力，无法在函数内部安全暂停并保持可验证状态，所以仍是整题路由。论文可以报告环境任务的节点级混合模式，不能把同样结论直接推广到 Code/Math 内部步骤。

### 3. Atomicity 的实际保证不足

设计以稳定 Contract、单一主要 Effect、独立验证、复用价值和局部归因为标准。当前 Env 使用规则事实差分，Code 把整个入口函数视作一项能力；最低 IR 校验和 SplitScore 不能充分保证这些性质。

### 4. Split 已执行，但边界质量仍依赖可观察 Effect

Atomicizer 会执行多 Effect 拆分，Tool 也有 specialize/split candidate 与 Admission 通道。Env 的 Effect 来自规则状态事实，Code 只有整题测试事实；没有可观察状态时，框架不能可靠发现更细边界。Tool split 还需要由失败簇或维护器提供明确分区，不能仅凭一次失败自动拆。

### 5. 六类边的 IR 完整，但自动生成范围不同

六类边均可表示、校验和持久化，Runtime 也能解释控制与数据边。不过普通顺序成功轨迹不会自然提供 branch、parallel、retry 或 loop 证据，因此 Builder 默认只生成 `next` 和可推断 `data_flow`。论文应分别报告“支持的边类型”和“实验实际产生的边类型及数量”。

### 6. 四层验证已接线，失败修复使用分支和回滚门禁

Tool、Atomic、Composite、Benchmark 四层会独立落盘，Graph Validator 也在每轮进化后运行。失败修复先在隔离 bank 中验证，再合并 main，并在异常时恢复推荐指针。历史版本文件按不可变审计保留，不做物理删除；正式实验仍必须检查图错误数和 bank 哈希。

### 7. 失败分支已能自动修复有成功补救证据的节点

Direct 失败且 Seeded/Dynamic 成功时，环境 Tool 会从补救轨迹自动产生新 executable template；Code/Math 会从 verifier 修复成功的代码产生新 Python Tool。Seeded 失败且 Dynamic 成功时会产生 Atomic guideline patch。它们都必须通过来源任务的严格目标路径 replay 才能合并。仍未实现的是：在完全失败、没有任何成功补救轨迹时，让另一个 LLM 凭失败文本独立发明 executable；此时只创建隔离 shadow 候选并等待正证据。

### 8. Tool 演化仍缺 action-template 语义泛化和 adapter 生成

Python 常量泛化、重复合并、specialize/split candidate、生命周期与 Runtime 绑定已经存在。环境动作模板目前只做实例槽位参数化，尚不能跨多个不同动作结构自动归纳一个新模板；`add_adapter` 也没有自动生成。

### 9. 统计原地更新与语义版本必须区分

状态、quality、utility、support 和 evidence 会写回版本文件；跨 id 对齐使用实际匹配引用，segment 与 atomic candidate 也按同一列表保持对齐。普通统计更新不创建新语义版本；Contract、artifact、guideline 或 insight 的内容变化才 bump version 并产生 evolution lineage。Runtime Direct、Admission replay、节点级执行结果和任务级使用结果使用不同字段，避免补救成功污染旧 Tool 成功数，也避免一次任务对 Atomic 重复计失败。

### 10. Direct 的参数和多 Tool 支持有限

Code Atomic Skill 当前常没有提取入口参数，而 Python Tool interface 又可能需要参数，Resolver 因而会拒绝 Direct。Schema 虽允许一个 Implementation 绑定多个 Tool，Code Direct 只执行第一个。Seeded 只把 Tool 当提示上下文，不执行 Tool artifact，因此 Seeded 成功不再给任何 seed Tool 记 Runtime 成功；这使统计更严格，但也意味着当前系统不估计 LLM 在文本层面采用了哪个 seed Tool。

### 11. Trace 阶段边界已保留，但环境状态解析仍是规则近似

Code attempts 会按 direct、seeded、dynamic 追加；ALFWorld fallback 在同一 episode 原地续跑，动作 mode、node_ref、状态和节点 attempts 都会保存，最终再同步 Runtime Graph。剩余误差主要来自自然语言 Observation 到事实集合的规则解析：环境已经发生但解析器未识别的 Effect 仍可能导致保守 fallback。

### 12. 当前测试更多是 smoke test

现有 23 项自动测试覆盖 IR、toy full-chain、冻结模式、ALFWorld 原地续跑、六类边 Schema、Runtime 控制/数据语义、proposal replay，以及 Direct 补救统计、Atomic 去重计数、Tool/Atomic 隔离分支、严格 replay、无正样本 shadow 副本和沙箱解释器可移植性。12 任务在线与冻结 replay smoke 均为 12/12，三个 ours 条件的 frozen bank 前后哈希一致且图错误数为零。仍缺真实 ALFWorld 大样本下的节点级 mixed 路由统计；toy 使用 `ToyAdapter + MockLLM`，不能据此宣称真实模型下的 Direct 收益已经稳定。

## 十、怎样准确概括当前系统

当前实现可以准确概括为：一个集中式、单 LLM 的任务执行与事后学习系统。它先用 Dynamic 或带经验的 Seeded/Direct 路径完成任务，再用确定性代码从成功 Trace 中建立三类 Skill 节点和独立 Tool Repository。Planner 选择 Skill，Implementation 连接 Skill 与 Tool，环境或 verifier 决定成功。失败 attempt 独立记账；有成功补救轨迹时，在隔离分支中形成 Tool 或 Atomic 新版本，严格 replay 成功后才合并；无正证据时只保留不能合并的 shadow 候选。

它已经实现了从空仓库积累 Abstract Skill、Implementation、Composite 和 Tool 的闭环；状态环境支持逐节点 mixed Runtime Graph；六类边具有统一 IR；四层验证、图验证、Tool Admission、泛化绑定、生命周期和 proposal replay 证据均进入主流程。

准确的保留项是：Code/Math 仍是整入口原子粒度；复杂控制边不会从普通顺序 Trace 自动猜出；环境 Effect 解析是规则近似；action-template 的跨任务语义泛化、自动 adapter，以及“完全失败且无正轨迹”条件下的 LLM executable 发明尚未实现。这些限制不影响当前有证据修复闭环的可运行性，但必须在论文实验范围和威胁有效性中明确说明。
