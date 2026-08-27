"""实验公共设施：条件定义 / 系统与适配器工厂 / baseline 分发 / 汇总。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from atomic_skillgraph.adapters.alfworld import AlfWorldAdapter  # noqa: E402
from atomic_skillgraph.adapters.code_math import CodeMathAdapter  # noqa: E402
from atomic_skillgraph.adapters.flowevo import FlowEvoLLM, run_flowevo_baseline  # noqa: E402
from atomic_skillgraph.adapters.mock_llm import MockLLM  # noqa: E402
from atomic_skillgraph.adapters.toy_benchmarks import ToyAdapter, build_mock_script  # noqa: E402
from atomic_skillgraph.core.config import FeatureFlags, SystemConfig, load_config  # noqa: E402
from atomic_skillgraph.system import AtomicSkillGraphSystem  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 条件（设计文档 §57.2 核心条件 + §57.3 消融）
# ---------------------------------------------------------------------------

BASELINE_FLOWEVO_CONDITIONS = {
    # condition_name -> flowevo 原版 condition（按 benchmark 映射）
    "baseline_dynamic": {"code_math": "cot_baseline", "alfworld": "pure_dynamic"},
    "flowevo": {"code_math": "ours", "alfworld": "full_library"},
}

OUR_CONDITIONS: dict[str, dict[str, Any]] = {
    "atomic_graph_only": {
        "description": "FlowEvo + Atomic SkillGraph，无独立 Tool 进化",
        "features": {"enable_tool_evolution": False},
    },
    "tool_repo_only": {
        "description": "FlowEvo + Tool Repository，无 Composite Graph",
        "features": {"enable_composite": False, "enable_layer3_insight": False},
    },
    "atomic_skillgraph_full": {
        "description": "AtomicSkillGraph Full（§57.2 条件 5）",
        "features": {},
    },
    # ---- 消融（§57.3） ----
    "full-no_validator": {"features": {"enable_node_validator": False}},
    "full-no_insight": {"features": {"enable_layer3_insight": False}},
    "full-no_generalization": {"features": {"enable_generalization": False}},
    "full-no_specialization": {"features": {"enable_specialization": False}},
    "full-no_cross_task_type_reuse": {"features": {"enable_cross_task_type_reuse": False}},
    "full-1to1_binding": {"features": {"enable_nm_binding": False}},
    "full-no_governance": {"features": {"enable_governance": False}},
    "full-no_primitive_reuse": {"features": {"enable_primitive_reuse": False}},
    "full-no_composite": {"features": {"enable_composite": False}},
    "task_type_hard_restricted": {"features": {"task_type_hard_restricted": True}},
}

ALL_OUR_CONDITIONS = list(OUR_CONDITIONS.keys())
ALL_CONDITIONS = list(BASELINE_FLOWEVO_CONDITIONS.keys()) + ALL_OUR_CONDITIONS


def apply_condition(config: SystemConfig, condition: str) -> SystemConfig:
    """把条件/消融的 feature 覆盖应用到配置（返回拷贝，不污染原配置）。"""
    import dataclasses
    spec = OUR_CONDITIONS.get(condition)
    if spec is None:
        raise ValueError(f"未知条件：{condition}")
    overrides = dict(spec.get("features") or {})
    fields = FeatureFlags.__dataclass_fields__
    features = dataclasses.replace(config.features)
    for key, value in overrides.items():
        if key in fields:
            setattr(features, key, value)
    return dataclasses.replace(config, features=features)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def make_llm(config: SystemConfig, mock_script: dict[str, Any] | None = None):
    """按配置构造 LLM（mock 或真实 API）。"""
    if config.llm.mock:
        return MockLLM(script=mock_script or {})
    api_key = config.llm.resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "未找到 API key。请在 configs/local.yaml 填写 llm.api_key，"
            "或设置环境变量 %s（见 README「API 填写」）" % config.llm.api_key_env)
    return FlowEvoLLM(config.llm)


def make_adapter(benchmark: str, config: SystemConfig, **kwargs):
    if benchmark == "toy":
        return ToyAdapter(kinds=kwargs.get("kinds", ("code", "math", "env")))
    if benchmark in ("humaneval", "gsm8k"):
        return CodeMathAdapter(benchmark, limit=int(kwargs.get("limit", 0) or 0),
                               timeout_seconds=kwargs.get("timeout_seconds",
                                                          config.thresholds.admission_timeout_seconds))
    if benchmark == "alfworld":
        return AlfWorldAdapter(
            split=kwargs.get("split", "eval_out_of_distribution"),
            max_steps=kwargs.get("max_steps", config.max_steps),
            task_type=kwargs.get("task_type"),
            alfworld_data=kwargs.get("alfworld_data"),
        )
    raise ValueError(f"未知 benchmark：{benchmark}")


def load_tasks_for(benchmark: str, adapter, limit: int, task_type: str | None) -> list:
    tasks = adapter.load_tasks(limit=limit, task_type=task_type)
    return tasks


def balanced_task_subset(tasks: list, task_types: list[str],
                         per_type_limit: int) -> list:
    """Deterministically interleave an equal number of tasks from each label."""
    labels = [str(label) for label in task_types if str(label).strip()]
    if not labels or per_type_limit <= 0:
        return []
    buckets = {
        label: [task for task in tasks
                if str(getattr(task, "task_type", "")) == label][:per_type_limit]
        for label in labels
    }
    missing = {label: len(bucket) for label, bucket in buckets.items()
               if len(bucket) < per_type_limit}
    if missing:
        raise ValueError(
            f"均衡任务不足：要求每类 {per_type_limit} 个，实际不足 {missing}")
    return [buckets[label][index]
            for index in range(per_type_limit) for label in labels]


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

def run_our_condition(condition: str, adapter, config: SystemConfig,
                      tasks: list, mock_script: dict[str, Any] | None = None,
                      output_dir: str | Path | None = None,
                      monitor=None, allow_task_extension: bool = False,
                      ) -> dict[str, Any]:
    """运行我方条件（v2.0 runtime 条件/消融）。

    每个条件使用独立 data 目录（条件间知识不互相污染）。
    """
    import dataclasses
    config = apply_condition(config, condition)
    progress_path: Path | None = None
    task_signature = [{
        "task_id": str(task.task_id),
        "task_type": str(task.task_type),
        "game_file": str(task.context.get("game_file") or ""),
    } for task in tasks]
    episodes: list[dict[str, Any]] = []
    if output_dir is not None:
        condition_dir = Path(output_dir) / condition
        config = dataclasses.replace(config, data_dir=condition_dir / "data")
        progress_path = condition_dir / "online_progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            prior_signature = list(progress.get("task_signature") or [])
            same_signature = prior_signature == task_signature
            valid_extension = (
                allow_task_extension
                and len(prior_signature) < len(task_signature)
                and prior_signature == task_signature[:len(prior_signature)]
            )
            if not (same_signature or valid_extension):
                raise RuntimeError(
                    f"{condition} 已有进度的任务清单与本次不一致；"
                    "请使用新 run-dir、--fresh-conditions，或仅以前缀追加方式"
                    "使用 --extend-online")
            episodes = list(progress.get("episodes") or [])
            if len(episodes) > len(prior_signature):
                raise RuntimeError(
                    f"{condition} online_progress 损坏：episode 数超过任务签名")
    llm = make_llm(config, mock_script=mock_script)
    system = AtomicSkillGraphSystem(config, adapter, llm)
    completed = len(episodes)
    if monitor is not None and completed:
        monitor.task_update(completed, note=f"resume {completed}/{len(tasks)}")
    for index, task in enumerate(tasks[completed:], start=completed + 1):
        episodes.append(system.run_task(task))
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "condition": condition,
                "task_signature": task_signature,
                "completed": len(episodes),
                "episodes": episodes,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(progress_path)
        if monitor is not None:
            monitor.task_update(index, note=str(task.task_id))
    return {
        "condition": condition,
        "kind": "ours",
        "episodes": episodes,
        "final_skill_graph": system.registry.stats(),
        "final_tool_repo": system.tool_registry.stats(),
    }


def run_baseline_condition(condition: str, benchmark: str, *, config_path: str,
                           output_dir: str | Path, limit: int = 0,
                           task_type: str | None = None,
                           alfworld_split: str = "eval_out_of_distribution",
                           alfworld_data: str | None = None,
                           max_steps: int | None = None,
                           monitor=None) -> dict[str, Any]:
    """运行 vendored FlowEvo 原版 baseline 条件（子进程）。"""
    flowevo_condition = BASELINE_FLOWEVO_CONDITIONS[condition].get(
        "code_math" if benchmark in ("humaneval", "gsm8k", "mbpp", "math") else "alfworld")
    on_progress = None
    if monitor is not None:
        def _on_progress(done: int, total: int, note: str) -> None:
            monitor.task_update(done, note=note.strip()[:80])
        on_progress = _on_progress
    result = run_flowevo_baseline(
        project_root=PROJECT_ROOT,
        benchmark=benchmark,
        output_dir=Path(output_dir),
        conditions=[flowevo_condition],
        config_path=config_path,
        limit=limit,
        task_type=task_type,
        alfworld_split=alfworld_split,
        alfworld_data=alfworld_data,
        max_steps=max_steps,
        on_progress=on_progress,
    )
    if int(result.get("returncode", 1)) != 0:
        raise RuntimeError(
            f"FlowEvo baseline 子进程失败：condition={condition}, "
            f"benchmark={benchmark}, returncode={result.get('returncode')}\n"
            f"{result.get('stdout_tail', '')[-2000:]}")
    episodes = parse_flowevo_results(Path(output_dir), benchmark, flowevo_condition)
    if not episodes:
        raise RuntimeError(
            f"FlowEvo baseline 成功退出但没有可解析结果：{output_dir}")
    return {
        "condition": condition,
        "kind": "flowevo_baseline",
        "flowevo_condition": flowevo_condition,
        "benchmark": benchmark,
        "subprocess": result,
        "episodes": episodes,
    }


def run_balanced_baseline_condition(
        condition: str, benchmark: str, *, config_path: str,
        output_dir: str | Path, task_types: list[str], per_type_limit: int,
        alfworld_split: str = "eval_out_of_distribution",
        alfworld_data: str | None = None, max_steps: int | None = None,
        monitor=None) -> dict[str, Any]:
    """按 task type 分别运行原版 baseline，再汇总为一个均衡结果。

    FlowEvo 的 ALFWorld 入口一次只接受一个 ``task_type``。如果直接给它
    ``limit = 类型数 * 每类数量``，它会取 split 的前 N 题，不能保证与 ours
    的均衡清单具有相同的任务类型分布。因此这里为每种类型建立独立输出目录，
    各取前 ``per_type_limit`` 题，最后按真实任务数加权汇总。
    """
    if benchmark != "alfworld":
        raise ValueError("均衡 task-type baseline 目前只适用于 ALFWorld")
    labels = [str(label) for label in task_types if str(label).strip()]
    if not labels or per_type_limit <= 0:
        raise ValueError("task_types 不能为空且 per_type_limit 必须为正整数")

    output_root = Path(output_dir)
    by_type: dict[str, dict[str, Any]] = {}
    subprocesses: dict[str, Any] = {}
    total = passed = 0
    weighted_tokens = 0.0
    completed = 0
    flowevo_condition = BASELINE_FLOWEVO_CONDITIONS[condition]["alfworld"]
    for label in labels:
        result = run_baseline_condition(
            condition, benchmark, config_path=config_path,
            output_dir=output_root / "by_task_type" / label,
            limit=per_type_limit, task_type=label,
            alfworld_split=alfworld_split, alfworld_data=alfworld_data,
            max_steps=max_steps, monitor=None)
        episode = (result.get("episodes") or [{}])[0]
        count = int(episode.get("num_tasks", 0) or 0)
        if count != per_type_limit:
            raise RuntimeError(
                f"{condition}/{label} 应评估 {per_type_limit} 题，实际为 {count}")
        success_count = int(episode.get("num_passed", 0) or 0)
        avg_tokens = float(episode.get("avg_tokens", 0.0) or 0.0)
        total += count
        passed += success_count
        weighted_tokens += avg_tokens * count
        completed += count
        by_type[label] = {
            "num_tasks": count,
            "num_passed": success_count,
            "success_rate": float(episode.get("success_rate", 0.0) or 0.0),
            "avg_tokens": avg_tokens,
            "flowevo_summary": episode.get("flowevo_summary") or {},
        }
        subprocesses[label] = result.get("subprocess") or {}
        if monitor is not None:
            monitor.task_update(
                completed, note=f"{label}: {success_count}/{count}")

    summary = {
        "total": total,
        "success": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_tokens": weighted_tokens / total if total else 0.0,
        "sampling": "balanced_first_n_per_task_type",
        "per_type_limit": per_type_limit,
        "per_task_type": by_type,
    }
    return {
        "condition": condition,
        "kind": "flowevo_baseline",
        "flowevo_condition": flowevo_condition,
        "benchmark": benchmark,
        "subprocess_by_task_type": subprocesses,
        "episodes": [_flowevo_summary_episode(
            summary, benchmark, flowevo_condition)],
        "balanced_task_types": labels,
        "per_type_limit": per_type_limit,
    }


# ---------------------------------------------------------------------------
# FlowEvo 输出解析
# ---------------------------------------------------------------------------

def parse_flowevo_results(output_dir: Path, benchmark: str,
                          flowevo_condition: str) -> list[dict[str, Any]]:
    """从 FlowEvo 输出目录解析可比较的摘要。

    code/math: summary.json = {benchmark: {condition: {"pass_rate", "avg_tokens", ...}}}
    alfworld:  validation_results*.json = {condition: {... "summary": {...}}}
    """
    if benchmark in ("humaneval", "gsm8k", "mbpp", "math"):
        path = output_dir / "summary.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        by_benchmark = data.get(benchmark, data)
        summary = by_benchmark.get(flowevo_condition, {})
        return [_flowevo_summary_episode(summary, benchmark, flowevo_condition)] if summary else []
    candidates = [
        output_dir / "validation_results.json",
        output_dir / f"validation_results_{flowevo_condition}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(flowevo_condition) or data
        summary = entry.get("summary", {}) if isinstance(entry, dict) else {}
        if summary:
            return [_flowevo_summary_episode(summary, benchmark, flowevo_condition)]
    return []


def _flowevo_summary_episode(summary: dict[str, Any], benchmark: str,
                             condition: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "condition": condition,
        "kind": "flowevo_baseline",
        "success": True,
        "flowevo_summary": summary,
        "success_rate": float(summary.get("pass_rate", 0.0) or 0.0),
        "avg_tokens": float(summary.get("avg_tokens", 0.0) or 0.0),
        "num_tasks": int(summary.get("total", summary.get("n", 0)) or 0),
        "num_passed": int(summary.get("success", 0) or 0),
    }


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def run_conditions(*, conditions: list[str], benchmark: str, config: SystemConfig,
                   adapter=None, tasks: list | None = None,
                   output_dir: str | Path, limit: int = 0,
                   config_path: str, task_type: str | None = None,
                   alfworld_split: str = "eval_out_of_distribution",
                   alfworld_data: str | None = None,
                   max_steps: int | None = None,
                   mock_script: dict[str, Any] | None = None,
                   initial_results: dict[str, Any] | None = None,
                   allow_task_extension: bool = False) -> dict[str, Any]:
    """按条件列表运行实验（baseline 走 FlowEvo 子进程，ours 走 v2.0 runtime）。"""
    from experiments.progress import ProgressMonitor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = ProgressMonitor()
    monitor.set_total_conditions(len(conditions))
    # 精准重跑时保留未选中条件的已有结果；每完成一个条件
    # 仍增量落盘，中断也不会把 baseline 从 results.json 中删掉。
    results: dict[str, Any] = dict(initial_results or {})
    for condition in conditions:
        print(f"[experiment] benchmark={benchmark} condition={condition} ...")
        if condition in BASELINE_FLOWEVO_CONDITIONS:
            monitor.condition_start(condition, task_total=limit or 0)
            result = run_baseline_condition(
                condition, benchmark, config_path=config_path,
                output_dir=output_dir / f"{condition}_flowevo", limit=limit,
                task_type=task_type, alfworld_split=alfworld_split,
                alfworld_data=alfworld_data,
                max_steps=max_steps, monitor=monitor)
        else:
            if tasks is None or adapter is None:
                raise ValueError("ours 条件需要 adapter 与 tasks")
            monitor.condition_start(condition, task_total=len(tasks))
            result = run_our_condition(condition, adapter, config, tasks,
                                       mock_script=mock_script,
                                       output_dir=output_dir, monitor=monitor,
                                       allow_task_extension=allow_task_extension)
        monitor.condition_finish()
        results[condition] = result
        # 每次条件写一次汇总（支持中断后查看已完成条件）
        (output_dir / "results.json").write_text(
            json.dumps({k: _serializable(v) for k, v in results.items()},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    monitor.finish()
    return results


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_conda_config(config_path: str | None) -> SystemConfig:
    if config_path:
        return load_config(config_path)
    return SystemConfig(data_dir=PROJECT_ROOT / "data")
