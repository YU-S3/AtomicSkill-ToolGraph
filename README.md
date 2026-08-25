# AtomicSkillGraph v2.0

集中式原子 SkillGraph 与 Tool 联合进化的论文实验代码。研究主线：在单一集中式长期运行
Agent 中，把成功执行轨迹持续蒸馏为**抽象原子技能**（Abstract Atomic Skill）、**可执行
Tool** 与**复合技能**（Composite Skill），通过 SkillGraph、Tool Repository、节点级验证
与生命周期治理实现能力的细粒度复用、泛化与优胜劣汰。

设计依据：`AtomicSkillGraph_集中式原子SkillGraph与Tool联合进化_完整设计文档_v2.0.md`
（本目录）。实现底座：vendored FlowEvo（Apache-2.0，见 `src/flowevo/LICENSE`），复用其
Harness / Verifier / Sandbox / 数据加载，并在此基础上新增 v2.0 结构化层（`src/atomic_skillgraph/`）。

---

## 1. 目录结构

```
AtomicSkill-ToolGraph/
├── src/
│   ├── atomic_skillgraph/        # v2.0 新增层（核心贡献）
│   │   ├── core/                 # SkillRef/ToolRef、Skill IR、Tool IR、Trace IR、谓词、配置
│   │   ├── graph/                # SkillGraph Registry（JSON 存储）、图结构、对齐器、图验证
│   │   ├── atomicizer/           # Trace Atomicizer、Effect 提取、边界检测、SplitScore
│   │   ├── tools/                # Tool Repository、Resolver、Miner、Admission、Generalizer、Lifecycle
│   │   ├── runtime/              # Atomic Planner、Implementation Selector、Runtime Graph、Execution Bridge
│   │   ├── validation/           # 三级验证（Tool/Atomic/Composite）+ 失败定位
│   │   ├── evolution/            # 成功/失败进化管线、Composite Builder、Layer-3 Insight
│   │   ├── adapters/             # code_math(HumanEval/GSM8K) / alfworld / toy / flowevo / mock_llm
│   │   └── system.py             # AtomicSkillGraphSystem（§43 运行时总装）
│   └── flowevo/                  # vendored FlowEvo（baseline 条件与 Harness 复用）
├── experiments/                  # 分阶段实验入口 + 指标报告
│   ├── run_smoke_ir.py           # Stage 1：无 API smoke（IR/Graph/Tools/Atomicizer/Evolution）
│   ├── run_smoke_fullchain.py    # Stage 2：无 API 完整实验链路 smoke（Mock LLM + 合成任务）
│   ├── run_small.py              # Stage 3：小规模数据（ALFWorld 10 / HumanEval 10 / GSM8K 50）
│   ├── run_full.py               # Stage 4：完整实验 + 消融
│   ├── report.py                 # §58 指标汇总与 Markdown 报告
│   └── common.py                 # 条件定义 / 工厂 / baseline 子进程分发
├── tests/                        # pytest（无 API）
├── configs/
│   ├── default.yaml              # v2.0 系统配置（API/阈值/消融开关）
│   ├── local.example.yaml        # 本地 API key 覆盖模板
│   ├── flowevo_default.yaml      # baseline 用 vendored FlowEvo 配置
│   └── flowevo_local.example.yaml
├── environment.yml / requirements.txt / pyproject.toml
└── data/  runs/                  # 运行时生成
```

## 2. 本地环境（conda）

```bash
cd D:\T3S_exp\AtomicSkill-ToolGraph

# 方式 A：一键创建（含可选 ALFWorld 依赖，先编辑 environment.yml 取消注释）
conda env create -f environment.yml -n asg
conda activate asg

# 方式 B：手动创建
conda create -n asg python=3.11 -y
conda activate asg
pip install -r requirements.txt

# 验证安装（全部应通过）
python -m experiments.run_smoke_ir
python -m pytest tests -q
```

依赖说明：
- 基础依赖：`pydantic>=2.0`、`datasets>=2.19.0`、`pyyaml`、`requests`、`pytest`。
- **HumanEval / GSM8K**：五个条件共用同一个 local-first loader。若存在
  `data/hf/openai_humaneval/test-00000-of-00001.parquet` 和
  `data/hf/gsm8k/main/test-00000-of-00001.parquet`，会直接读取本地 Parquet，
  不向 Hugging Face 发请求；文件不存在时才调用在线 `load_dataset`。
- **ALFWorld（真实环境）**：ALFWorld 0.4.x 采用 PDDL 后端，**运行时需要
  fast-downward 规划器（C++）**，Windows 下无法纯 pip 安装（需 MSVC 编译链）。
  本机已配置 **WSL2 Ubuntu** 方案（见下节）；不装 alfworld 也能跑 Stage 1/2
  与 toy 小规模联调。

### WSL2 运行 ALFWorld（Windows 本机方案）

WSL 内已配置：`~/asg_alfworld_venv`（alfworld + textworld + fast-downward）、
数据位于 `~/.cache/alfworld`（json_2.1.1 标注 + 预生成 game.tw-pddl）。
若需重装/重下数据：

```powershell
wsl -e bash /mnt/d/alfworld_deps/wsl_setup_install.sh   # 依赖安装（约 10 分钟）
wsl -e bash /mnt/d/alfworld_deps/wsl_setup_data.sh      # 数据下载（约 2-4GB）
```

运行实验（三种方式等价）：

```powershell
# 环境自检（枚举任务 + 打印第一个任务，约几分钟）
wsl -e bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph/scripts/run_alfworld_wsl.sh check

# 小规模：同一类任务 10 个（5 个核心条件）
wsl -e bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph/scripts/run_alfworld_wsl.sh small

# 完整实验
wsl -e bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph/scripts/run_alfworld_wsl.sh full
```

参数说明（`run_small.py` / `run_full.py` 均适用）：
- `--task-type pick_heat_then_place_in_recep`：**环境级任务类型过滤**（baseline 与
  ours 使用完全相同的任务集合，已消除旧版"baseline 跑混合类型前 10 个、ours 跑
  heat 前 10 个"的不公平缺口）。valid_unseen 中 heat 任务共 23 个。
- `--limit N`：任务数（heat 类型最多 23；步数/任务数杠杆实验可直接调）。
- `--max-steps N`：步数预算，**ours 与 baseline 对称生效**（默认 50）。
- `--resume`：断点续跑（复用 output-dir 下最近目录，跳过已完成条件）。

### 三个 Benchmark 的完整小规模实验

下面这一条命令依次运行 ALFWorld 10、HumanEval 10、GSM8K 50。每个
Benchmark 都包含 `baseline_dynamic`、`flowevo`、`atomic_graph_only`、
`tool_repo_only`、`atomic_skillgraph_full` 五个在线条件；随后对三个 ours 条件分别
冻结进化后的 bank，在同一 train 切分上 replay。任一 baseline 子进程失败、条件
结果缺失、冻结前后 bank 哈希变化或图校验失败，整条命令都会以非零状态退出：

```powershell
wsl -e bash -lc 'cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph && exec "$HOME/asg_alfworld_venv/bin/python" -m experiments.run_all_small --config-path configs/default.yaml --alfworld-data "$HOME/.cache/alfworld" --task-type pick_heat_then_place_in_recep --alfworld-limit 10 --humaneval-limit 10 --gsm8k-limit 50 --max-steps 100 --online-output runs/all_small_post_planner_fix --eval-output runs/all_small_frozen_post_planner_fix'
```

总清单写入 `runs/all_small_post_planner_fix/all_small_manifest.json`。已有旧目录不会被
当成本次产物；runner 会定位本次新建的时间戳目录。若只想先跑五条件在线实验而
不做冻结 replay，可临时加 `--skip-frozen`。

### 冻结 Skill 进化效果评估（Train-Evolve-Test 第二阶段）

正式的小规模 ALFWorld 实验建议使用下面的单入口命令。它会先从空输出目录完成
在线进化，再自动定位本次产物并执行严格冻结的 train replay；任一条件缺失、图
校验失败、或 replay 前后 SkillGraph/Tool 文件哈希变化，命令都会以非零状态退出：

```powershell
wsl -e bash -lc 'cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph && exec "$HOME/asg_alfworld_venv/bin/python" -m experiments.run_evolve_replay_pipeline --benchmark alfworld --limit 10 --task-type pick_heat_then_place_in_recep --conditions atomic_graph_only tool_repo_only atomic_skillgraph_full --max-steps 100 --alfworld-data "$HOME/.cache/alfworld" --config-path configs/default.yaml --online-output runs/small_post_atomic_fix --eval-output runs/evolve_eval_post_atomic_fix'
```

产物中的 `pipeline_manifest.json` 记录在线目录、冻结回放目录、每个条件的 bank
SHA-256、回放前后一致性和图校验错误数。不要把修复前生成的 bank 当作本次正式
replay 输入；本次修复改变了 Direct/Seeded 统计口径和失败分支合并行为，需要重新
执行在线阶段，才能得到同一口径的结果。

在线进化跑完后（`run_small`/`run_full` 产物），用独立模块冻结各条件进化出的
SkillGraph/Tool 库（只读、禁止一切进化与统计写入），在 train 或 hold-out test
任务上重放，单独测量"进化出的技能本身的质量"（复用覆盖率 / 成功率 / tokens /
与在线 Late-run 对照）。在线管线完全不受影响：

```powershell
# train replay：在同一批 10 个 heat 任务上重放冻结技能（把 <时间戳> 换成实际目录名）
wsl -e bash -c "cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph && ~/asg_alfworld_venv/bin/python -m experiments.run_evolve_eval --run-dir runs/small/alfworld_<时间戳> --benchmark alfworld --task-type pick_heat_then_place_in_recep --max-steps 100 --alfworld-data ~/.cache/alfworld --config-path configs/default.yaml"

# test 泛化：跳过前 10 个训练任务，在剩余 13 个 hold-out 任务上评估
wsl -e bash -c "cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph && ~/asg_alfworld_venv/bin/python -m experiments.run_evolve_eval --run-dir runs/small/alfworld_<时间戳> --benchmark alfworld --task-type pick_heat_then_place_in_recep --max-steps 100 --split test --train-limit 10 --alfworld-data ~/.cache/alfworld --config-path configs/default.yaml"

# 同时评估 FlowEvo 完整库的冻结版（加载在线 checkpoint，关闭 compile）
wsl -e bash -c "cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph && ~/asg_alfworld_venv/bin/python -m experiments.run_evolve_eval --run-dir runs/small/alfworld_<时间戳> --benchmark alfworld --task-type pick_heat_then_place_in_recep --max-steps 100 --eval-flowevo --alfworld-data ~/.cache/alfworld --config-path configs/default.yaml"
```

产物：`runs/evolve_eval/<run>_<split>_<ts>/{report.md, aggregated.json, <条件>/data/}`，
报告含 frozen vs 在线 Late-run 对照表；`bank_unchanged_after_eval` 字段校验评估
期间技能库未被写入（冻结快照为在线产物的副本，原库零改动）。

或在 WSL 终端内 `cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph` 后
`source ~/asg_alfworld_venv/bin/activate`，直接运行 `experiments/run_small.py` /
`run_full.py`（加 `--alfworld-data ~/.cache/alfworld`）。API 配置沿用
`configs/default.yaml` + `configs/local.yaml`（WSL 内同样生效）。

> Windows 原生运行 ALFWorld 需先安装 VS Build Tools 2022（C++ 工作负载）与
> CMake，成本高且易碎，不推荐；HumanEval/GSM8K 实验不受影响，继续用
> Windows 下的 `asg` conda 环境即可。

## 3. API 填写

**v2.0 runtime 与 baseline 共用同一 OpenAI 兼容端点。**

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | v2.0 系统配置（ours 条件） |
| `configs/local.yaml` | 本地 API key 覆盖（复制 `configs/local.example.yaml`，git 忽略） |
| `configs/flowevo_default.yaml` | baseline 条件（vendored FlowEvo 原版 runner） |
| `configs/flowevo_local.yaml` | baseline 本地覆盖（复制 `flowevo_local.example.yaml`） |

最小填写（以 OpenRouter 为例）：

```bash
# 1. 复制并编辑本地覆盖文件
cp configs/local.example.yaml configs/local.yaml
cp configs/flowevo_local.example.yaml configs/flowevo_local.yaml

# 2. 两个 local.yaml 中填写（或用环境变量）：
#    llm:
#      api_key: "sk-or-..."          # 你的 API key
#      base_url: https://openrouter.ai/api/v1
#      model: openai/gpt-4o-mini     # 论文 backbone，可换其他 OpenAI 兼容模型
```

> 也可不改文件，直接设置环境变量 `OPENROUTER_API_KEY`（v2.0 用
> `llm.api_key_env` 指定的变量名，默认 `OPENROUTER_API_KEY`；baseline 读取
> vendored FlowEvo 的 `configs/flowevo_local.yaml` 或同名环境变量）。
> 换成 DeepSeek 等兼容端点：修改 `base_url` / `model` 即可（例如
> `base_url: https://api.deepseek.com/v1`、`model: deepseek-chat`）。

## 4. 分阶段实验（务必按顺序）

### Stage 1 — 无 API smoke（不动 API、不联网）

验证 IR / SkillGraph / Tool Repository / Atomicizer / Admission / Evolution
全部规则逻辑（12 项检查，合成数据）：

```bash
python -m experiments.run_smoke_ir
python -m experiments.run_smoke_ir --json     # JSON 输出
python -m pytest tests -q                     # 等价 pytest（3 项：smoke + 完整链路 + 世界协议）
```

### Stage 2 — 无 API、但按有 API 方式走完整实验链路 smoke

Mock LLM + 合成 benchmark（toy_code/toy_math/toy_env），跑完整闭环：
任务 → 检索/规划 → direct/seeded/dynamic 路由 → 验证 → Trace → 原子化 →
Tool admission → SkillGraph/ToolRepo 更新 → Composite → Layer-3 insight →
全局泛化 → 生命周期治理 → 指标 → 报告。期望输出：12/12 成功，且最后
一个 env 任务 `mode=DIRECT`（零 token 直接模板执行），SkillGraph 含
abstract_atomic / implementation_atomic / composite 三类节点。

```bash
python -m experiments.run_smoke_fullchain
# 产物：runs/smoke_fullchain/run_<时间戳>/{report.md, aggregated.json, data/}
```

### Stage 3 — 小规模数据结果测试（先看有没有效果）

| benchmark | 规模 | 说明 |
|---|---|---|
| ALFWorld | 同一类任务 10 个 | 默认 `pick_heat_then_place_in_recep`，可用 `--task-type` 换 |
| HumanEval | 10 个任务 | HF `openai/openai_humaneval` test 前 10 |
| GSM8K | 50 个问题 | HF `openai/gsm8k` main/test 前 50 |

**3a. 无 API、按有 API 方式走完整实验链路**（管线验证，不看效果）：

```bash
python -m experiments.run_small --benchmark humaneval --limit 10 --mock
python -m experiments.run_small --benchmark gsm8k --limit 50 --mock
python -m experiments.run_small --benchmark alfworld --limit 10 --mock   # 需要 alfworld 数据
python -m experiments.run_small --benchmark toy --mock                    # 纯合成联调
```

**3b. 真实 API 小规模结果测试**（5 个核心条件，§57.2）：

```bash
python -m experiments.run_small --benchmark humaneval --limit 10 --config-path configs/default.yaml
python -m experiments.run_small --benchmark gsm8k --limit 50 --config-path configs/default.yaml
python -m experiments.run_small --benchmark alfworld --limit 10 \
    --task-type pick_heat_then_place_in_recep --config-path configs/default.yaml
```

条件含义：

| 条件 | 含义 |
|---|---|
| `baseline_dynamic` | 原版 FlowEvo 的 cot_baseline / pure_dynamic（子进程，零改动） |
| `flowevo` | 原版 FlowEvo 的 ours / full_library |
| `atomic_graph_only` | FlowEvo + Atomic SkillGraph，无独立 Tool 进化 |
| `tool_repo_only` | FlowEvo + Tool Repository，无 Composite Graph |
| `atomic_skillgraph_full` | AtomicSkillGraph Full |

每个条件使用**独立 data 目录**（条件间知识互不污染）；baseline 与 ours 使用
相同模型 / 相同任务顺序 / 相同验证器（Humaneval/GSM8K 复用 vendored FlowEvo
verifier 与 sandbox；ALFWorld 复用其 env 协议，won 为唯一成功信号）。

### Stage 4 — 完整实验 + 消融

```bash
python -m experiments.run_full --benchmark humaneval --config-path configs/default.yaml
python -m experiments.run_full --benchmark gsm8k --config-path configs/default.yaml
python -m experiments.run_full --benchmark alfworld --config-path configs/default.yaml
python -m experiments.run_full --all --config-path configs/default.yaml          # 全部 benchmark
python -m experiments.run_full --benchmark humaneval --ablations ...            # 追加 §57.3 消融
python -m experiments.run_full --benchmark humaneval --resume ...               # 断点续跑
```

消融开关（`--ablations` 一键追加，或 `--conditions full-no_validator ...` 单项）：
`full-no_validator`、`full-no_insight`、`full-no_generalization`、
`full-no_specialization`、`full-no_cross_task_type_reuse`、`full-1to1_binding`、
`full-no_governance`、`full-no_primitive_reuse`、`full-no_composite`、
`task_type_hard_restricted`。

## 5. 结果与指标

每次运行产出（`runs/<stage>/.../`）：

- `report.md` / `aggregated.json`：§58 指标——成功率、Late-run Success、
  首试成功率、平均 tokens/task、direct/seeded/dynamic 频率、原子复用率、
  跨 task type 复用率、admission 通过率、维护动作数、知识增长曲线、
  最终 SkillGraph / Tool Repository 状态。
- `results.json`：逐条件原始 episode 记录。
- `data/<condition>/data/`：`skill_graph/`、`tools/`、`traces/`、
  `runtime_runs/`、`metrics/`（完整审计轨迹，符合设计文档 §40 布局）。

## 6. 常见问题

- **baseline 报 API key 错误**：检查 `configs/flowevo_local.yaml` 或
  `OPENROUTER_API_KEY`；v2.0 条件报错则检查 `configs/local.yaml` / 环境变量。
- **HumanEval/GSM8K 首次运行下载数据**：需要网络；`HF_DATASETS_CACHE` 可改缓存目录。
- **ALFWorld 报缺少 alfred.pddl**：安装 `pip install alfworld` 并准备数据目录，
  用 `--alfworld-data` 指定；只想验证链路可用 `--mock` 或 toy。
- **断点续跑**：Stage 3/4 支持 `--resume`（Stage 4），基于已写出的 results.json 跳过完成条件。
- **Windows 下 long path**：如遇路径过长，把 `runs/` 移到短路径（`--output-dir`）。

## 7. 许可与致谢

- vendored FlowEvo（`src/flowevo/`）来自 [DEFENSE-SEU/FlowEvo](https://github.com/DEFENSE-SEU/FlowEvo)，
  Apache-2.0（保留 `src/flowevo/LICENSE` 与引用信息），仅作 baseline 与 Harness 复用。
- v2.0 新增层（`src/atomic_skillgraph/`、`experiments/`、`tests/`）为本项目实现。
- 若使用本代码，请同时引用 FlowEvo 论文与本工作。
