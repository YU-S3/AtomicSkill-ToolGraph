"""20-task validation: verify the three-layer skill architecture.

Five conditions exercise different combinations of Layer 1 (executable
template), Layer 2 (trajectory exemplar) and Layer 3 (structural insight)
on the first 20 ALFWorld eval_out_of_distribution tasks.

Usage::

    cd src
    conda run -n alfworld_eval --no-capture-output python -u \\
        -m alfworld_.run_20task_validation \\
        --limit 20 --output-dir /tmp/20task_validation \\
        --config-path configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ContrastiveEval / Governance constants
# ---------------------------------------------------------------------------

HOLDOUT_RATE: float = 0.20  # fraction of episodes that skip injection
AUDIT_INTERVAL: int = 20    # episodes between active periodic_audit calls
HOLDOUT_RNG_SEED: int = 1337  # deterministic per-condition holdout schedule

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from alfworld_.compiler import AlfWorldCompiler
from alfworld_.env import AlfWorldEnv
from alfworld_.executor import AlfWorldExecutor
from alfworld_.generator import AlfWorldGenerator
from alfworld_.param_extractor import ParamExtractor
from alfworld_.skill_library import SkillLibrary
from runtime.config import load_runtime_config
from runtime.llm_client import LLMClient


_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
)


CONDITIONS: dict[str, dict[str, Any]] = {
    "pure_dynamic": {
        "display": "Pure Dynamic (baseline)",
        "use_template": False,
        "use_exemplar": False,
        "use_insight": False,
        "compile": False,
    },
    "compile_only": {
        "display": "Compile Only (build library, no reuse)",
        "use_template": False,
        "use_exemplar": False,
        "use_insight": False,
        "compile": True,
    },
    "layer1_only": {
        "display": "Layer 1: Direct Reuse",
        "use_template": True,
        "use_exemplar": False,
        "use_insight": False,
        "compile": True,
    },
    "layer1_2": {
        "display": "Layer 1+2: Direct Reuse + Exemplar",
        "use_template": True,
        "use_exemplar": True,
        "use_insight": False,
        "compile": True,
    },
    "layer1_3": {
        "display": "Layer 1+3: Direct Reuse + Insight (skip exemplar)",
        "use_template": True,
        "use_exemplar": False,
        "use_insight": True,
        "compile": True,
    },
    "full_library": {
        "display": "Full Library (Layer 1+2+3)",
        "use_template": True,
        "use_exemplar": True,
        "use_insight": True,
        "compile": True,
    },
    "expel": {
        "display": "ExpeL (exemplar + insight, no template)",
        "use_template": False,
        "use_exemplar": True,
        "use_insight": True,
        "compile": True,
    },
    "no_governance": {
        "display": "No Governance (full library, no quality control)",
        "use_template": True,
        "use_exemplar": True,
        "use_insight": True,
        "compile": True,
        "no_governance": True,
    },
}


# ---------------------------------------------------------------------------
# Checkpoint helpers (per-condition resumable state)
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 3


def _checkpoint_path(output_dir: Path, condition_name: str) -> Path:
    return output_dir / ("_checkpoint_%s.json" % condition_name)


def _save_checkpoint(
    output_dir: Path,
    condition_name: str,
    episodes: list[dict[str, Any]],
    library: SkillLibrary,
) -> None:
    """Atomic per-condition checkpoint write.

    Writes to ``_checkpoint_<cond>.tmp`` then renames over the real path so a
    crash mid-write cannot leave a half-baked JSON file behind.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_path(output_dir, condition_name)
    tmp = output_dir / ("_checkpoint_%s.tmp" % condition_name)
    data = {
        "version": _CHECKPOINT_VERSION,
        "condition": condition_name,
        "completed": len(episodes),
        "episodes": episodes,
        "library": library.to_dict(),
    }
    tmp.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8",
    )
    tmp.replace(ckpt)  # atomic on POSIX


def _load_checkpoint(
    output_dir: Path,
    condition_name: str,
) -> tuple[list[dict[str, Any]], SkillLibrary, int] | None:
    """Try to load a previous checkpoint.

    Returns ``(episodes, library, start_idx)`` or ``None`` if no usable
    checkpoint exists (missing file, version mismatch, or parse error).
    """
    ckpt = _checkpoint_path(output_dir, condition_name)
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if data.get("version") != _CHECKPOINT_VERSION:
            print(
                "  [resume] checkpoint version %s != %d, ignoring."
                % (data.get("version"), _CHECKPOINT_VERSION)
            )
            return None
        episodes = list(data.get("episodes", []))
        library = SkillLibrary.from_dict(data.get("library", {}))
        return episodes, library, len(episodes)
    except Exception as exc:  # noqa: BLE001
        print(
            "  [resume] checkpoint parse error: %s -- starting fresh" % exc
        )
        return None


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    cfg: dict[str, Any],
    llm: LLMClient,
    limit: int,
    max_steps: int,
    output_dir: Path | None = None,
    task_type: str | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:

    print("\n" + "=" * 72)
    print("  Condition: %s — %s" % (condition_name, cfg["display"]))
    print("=" * 72)

    env = AlfWorldEnv(split="eval_out_of_distribution", max_steps=max_steps,
                      task_type=task_type)
    total = env.initialize()
    # limit 表示从 start_index 起要运行的任务数（hold-out 评估时 start_index>0）
    n_tasks = min(total, start_index + limit)

    generator = AlfWorldGenerator(llm_client=llm)
    executor = AlfWorldExecutor(env=env, generator=generator, max_steps=max_steps)
    # Pass the LLM client to the compiler so Layer 2 can extract
    # procedural guidelines from successful traces (v4 redesign).
    compiler = AlfWorldCompiler(llm_client=llm)
    param_extractor = ParamExtractor()

    # Per-condition deterministic RNG so holdout decisions are reproducible
    # and don't leak entropy from / into other conditions.
    rng = random.Random(HOLDOUT_RNG_SEED)

    # ---- Resume from checkpoint if available ----
    resumed = (
        _load_checkpoint(output_dir, condition_name)
        if output_dir is not None else None
    )
    if resumed is not None:
        episodes, library, start_idx = resumed
        if start_idx >= n_tasks:
            print(
                "  [resume] Condition already complete (%d/%d). Skipping."
                % (start_idx, n_tasks)
            )
            return episodes
        print(
            "  [resume] Loaded %d episodes; continuing from task %d. "
            "Library: T=%d E=%d I=%d ep=%d"
            % (
                start_idx, start_idx,
                library.snapshot()["templates_active"],
                library.snapshot()["exemplars_total"],
                library.snapshot()["insights_total"],
                library.snapshot().get("current_episode", 0),
            )
        )
        # Restore governance setting from condition config.
        library._governance_enabled = not cfg.get("no_governance", False)
        # Fast-forward env: ALFWorld eval split is deterministic, so calling
        # reset() start_idx times advances to the same position the original
        # run had reached.  Discard each (task, obs, admissible) tuple.
        for _ in range(start_idx):
            env.reset()
        # Replay holdout RNG decisions: for every resumed episode where
        # is_holdout could have been rolled, we need to consume one rng
        # tick so the post-resume schedule matches the no-resume schedule.
        # We don't actually know which past episodes rolled the dice, so we
        # accept that the post-resume holdout schedule may diverge from a
        # cold run -- the bookkeeping in CE stats stays correct because
        # is_holdout is recorded per-episode in the checkpoint.
    else:
        episodes = []
        gov = not cfg.get("no_governance", False)
        library = SkillLibrary(governance_enabled=gov)
        start_idx = 0

    # 冻结库评估：加载指定目录下同名条件的 checkpoint 库，
    # 并禁用 compile/governance（Train-Evolve-Test 第二阶段，不改原行为）
    freeze_library_dir = os.environ.get("ALFWORLD_FREEZE_LIBRARY_DIR", "").strip()
    if freeze_library_dir:
        loaded = _load_checkpoint(Path(freeze_library_dir), condition_name)
        if loaded is not None:
            library = loaded[1]
            cfg = {**cfg, "compile": False, "no_governance": True}
            library._governance_enabled = False
            start_idx = 0
            print("  [frozen] 加载冻结技能库：%s（compile 已禁用）"
                  % _checkpoint_path(Path(freeze_library_dir), condition_name))
        else:
            print("  [frozen] 警告：%s 下没有 %s 的 checkpoint，回退空库"
                  % (freeze_library_dir, condition_name))

    # hold-out 评估（--start-index）：快进环境到第 start_index 个任务再开始
    if start_index > start_idx:
        for _ in range(start_idx, start_index):
            env.reset()
        start_idx = start_index

    cum_success = sum(1 for e in episodes if e.get("success"))

    for idx in range(start_idx, n_tasks):
        library.advance_episode()
        task, obs, admissible = env.reset()
        t0 = time.time()

        try:
            # Layered retrieval, masked by condition flags.
            full = library.retrieve(task.task_type)
            template = full["template"] if cfg["use_template"] else None
            exemplar = full["exemplar"] if cfg["use_exemplar"] else None
            insight = full["insight"] if cfg["use_insight"] else None

            # ContrastiveEval holdout: in conditions that *can* inject Layer 2/3,
            # 20% of episodes drop the injection while keeping the template
            # (Layer 1 is the zero-LLM path and not affected by injection
            # quality).  These holdouts feed library.record_unguided() so the
            # contrastive delta has a real baseline to compare against.
            is_holdout = False
            if cfg["use_exemplar"] or cfg["use_insight"]:
                if exemplar is not None or insight is not None:
                    is_holdout = rng.random() < HOLDOUT_RATE
                    if is_holdout:
                        exemplar = None
                        insight = None

            if template is not None:
                best_layer = 1
            elif exemplar is not None:
                best_layer = 2
            elif insight is not None:
                best_layer = 3
            else:
                best_layer = None

            skill_library_result = {
                "template": template,
                "exemplar": exemplar,
                "insight": insight,
                "best_layer": best_layer,
            }

            # Layer 1 needs concrete bindings extracted from the goal/obs.
            bindings: dict[str, str] | None = None
            if template is not None and task.task_type != "pick_two_obj_and_place":
                bindings = param_extractor.extract(
                    task.task_type, task.goal, obs, admissible,
                )

            # Optional static policy directives from the condition config
            static_dirs = cfg.get("static_directives")

            infrastructure_attempts: list[dict[str, Any]] = []
            trace = None
            # 与 ours 相同：最多 2 次 task-level 基础设施重试，并从该任务
            # 初始状态 reset；不能从 API 出错时的中间环境继续。
            for infrastructure_try in range(3):
                trace = executor.run(
                    task=task,
                    initial_observation=obs,
                    admissible_commands=admissible,
                    skill_library_result=skill_library_result,
                    skill_bindings=bindings,
                    policy_directives=static_dirs,
                )
                if trace.failure_type != "llm_error":
                    break
                infrastructure_attempts.append({
                    "attempt": infrastructure_try + 1,
                    "failure_type": trace.failure_type,
                    "errors": list(trace.infrastructure_errors),
                })
                if infrastructure_try < 2:
                    print("  [infra-retry] task=%s reset from initial state (%d/2)" % (
                        task.task_id, infrastructure_try + 1), flush=True)
                    retry_task, obs, admissible = env.reset_to_task(idx)
                    # reset_to_task 应返回同一过滤后任务；显式沿用返回对象，避免
                    # backend 重载时 task/observation 不一致。
                    task = retry_task
                    if template is not None and task.task_type != "pick_two_obj_and_place":
                        bindings = param_extractor.extract(
                            task.task_type, task.goal, obs, admissible)
            assert trace is not None
            infrastructure_failure = trace.failure_type == "llm_error"

            wall = time.time() - t0

            # Track Layer 1 utility
            if not infrastructure_failure and template is not None and best_layer == 1:
                library.record_template_usage(template.template_id, trace.success)

            # ContrastiveEval bookkeeping (only meaningful in conditions that
            # can actually inject Layer 2/3).  Use trace.planning_mode (what
            # actually fired) rather than best_layer: when a template exists,
            # best_layer is always 1 even though the trajectory may end up
            # driven by an exemplar after a direct-skill fallback.
            if not infrastructure_failure and (cfg["use_exemplar"] or cfg["use_insight"]):
                if is_holdout:
                    library.record_unguided(task.task_type, trace.success)
                elif trace.planning_mode in (
                    "exemplar_guided", "insight_guided",
                    "direct_then_exemplar", "direct_then_insight",
                ):
                    library.record_guided(task.task_type, trace.success)

            # Compile + ingest into the library.  Failure traces also feed
            # record_trace_stats so the Layer 3 insight picks up failure
            # patterns (timeouts, action loops) into common_pitfalls.
            if cfg["compile"] and not infrastructure_failure:
                if trace.success:
                    compile_result = compiler.compile(trace)
                    if compile_result.template is not None:
                        library.add_template(compile_result.template)
                    if compile_result.exemplar is not None:
                        library.add_exemplar(compile_result.exemplar)
                library.record_trace_stats(trace)

            cum_success += int(trace.success)
            snap = library.snapshot()

            ep: dict[str, Any] = {
                "condition": condition_name,
                "task_index": idx,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "goal": task.goal,
                "success": trace.success,
                "total_steps": trace.total_steps,
                "llm_call_count": trace.llm_call_count,
                "llm_total_tokens": trace.llm_total_tokens_total,
                "planning_mode": trace.planning_mode,
                "best_layer": best_layer,
                "is_holdout": is_holdout,
                "contrastive_delta": library.get_contrastive_delta(task.task_type),
                "templates_active": snap["templates_active"],
                "exemplars_total": snap["exemplars_total"],
                "insights_total": snap["insights_total"],
                "injection_suppressed_types": snap.get(
                    "injection_suppressed_types", 0,
                ),
                "failure_type": trace.failure_type or "",
                "infrastructure_failure": infrastructure_failure,
                "task_failure": bool(not trace.success and not infrastructure_failure),
                "infrastructure_episode_retries": max(
                    0, len(infrastructure_attempts)
                    - int(trace.failure_type == "llm_error")),
                "infrastructure_attempts": infrastructure_attempts,
                "infrastructure_errors": list(trace.infrastructure_errors),
                "wall_time_s": round(wall, 2),
            }

            status = "PASS" if trace.success else "FAIL"
            holdout_tag = " HO" if is_holdout else "   "
            print(
                "  [%2d/%d] %s%s mode=%-20s %-25s steps=%2d tk=%6d "
                "T=%d E=%d I=%d cum=%.0f%%" % (
                    idx + 1, n_tasks, status, holdout_tag,
                    trace.planning_mode[:20],
                    task.task_type[:25],
                    trace.total_steps,
                    trace.llm_total_tokens_total,
                    snap["templates_active"],
                    snap["exemplars_total"],
                    snap["insights_total"],
                    100.0 * cum_success / (idx + 1),
                )
            )

            # Periodic audit: every AUDIT_INTERVAL episodes, ask the library
            # to suppress low-utility / inactive templates and detect negative
            # transfer.  Only meaningful in conditions that compile.
            if cfg["compile"] and (idx + 1) % AUDIT_INTERVAL == 0:
                audit_result = library.periodic_audit()
                if any(audit_result.values()):
                    summary_line = ", ".join(
                        "%s=%d" % (k.replace("suppressed_", ""), len(v))
                        for k, v in audit_result.items() if v
                    )
                    print("    [audit @%d] %s" % (idx + 1, summary_line))

        except Exception as exc:  # noqa: BLE001
            wall = time.time() - t0
            ep = {
                "condition": condition_name,
                "task_index": idx,
                "task_id": getattr(task, "task_id", ""),
                "task_type": getattr(task, "task_type", ""),
                "goal": getattr(task, "goal", ""),
                "success": False,
                "total_steps": 0,
                "llm_call_count": 0,
                "llm_total_tokens": 0,
                "planning_mode": "error",
                "best_layer": None,
                "is_holdout": False,
                "contrastive_delta": None,
                "templates_active": library.active_template_count,
                "exemplars_total": sum(
                    len(v) for v in library._exemplars.values()  # noqa: SLF001
                ),
                "insights_total": len(library._insights),  # noqa: SLF001
                "injection_suppressed_types": len(
                    library._injection_suppressed  # noqa: SLF001
                ),
                "failure_type": str(exc)[:200],
                "wall_time_s": round(wall, 2),
            }
            print(
                "  [%2d/%d] ERROR %-25s :: %s"
                % (idx + 1, n_tasks, ep["task_type"][:25], str(exc)[:120])
            )

        episodes.append(ep)

        if output_dir is not None:
            _save_checkpoint(output_dir, condition_name, episodes, library)

    return episodes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary(eps: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(eps)
    if n == 0:
        return {}
    succ = sum(1 for e in eps if e["success"])
    tokens = sum(e["llm_total_tokens"] for e in eps)
    steps = sum(e["total_steps"] for e in eps)
    walls = sum(e.get("wall_time_s", 0.0) for e in eps)

    holdout_eps = [e for e in eps if e.get("is_holdout", False)]
    # Guided = LLM was actually steered by Layer 2 / Layer 3 content.
    # Use planning_mode (what actually fired) rather than best_layer (which
    # is 1 whenever a template existed, even when the trajectory was driven
    # by an exemplar after a direct-skill fallback).
    _GUIDED_MODES = (
        "exemplar_guided", "insight_guided",
        "direct_then_exemplar", "direct_then_insight",
    )
    guided_eps = [
        e for e in eps
        if not e.get("is_holdout", False)
        and e.get("planning_mode", "") in _GUIDED_MODES
    ]
    holdout_succ = sum(1 for e in holdout_eps if e["success"])
    guided_succ = sum(1 for e in guided_eps if e["success"])

    # Final delta observed at end of run (per-task contrastive_delta is the
    # most recent observation in the episode trail).
    final_deltas: list[float] = []
    for e in eps:
        d = e.get("contrastive_delta")
        if d is not None:
            final_deltas.append(d)
    avg_delta = sum(final_deltas) / len(final_deltas) if final_deltas else None

    return {
        "total": n,
        "success": succ,
        "pass_rate": succ / n,
        "avg_tokens": tokens / n,
        "avg_steps": steps / n,
        "wall_time_s": walls,
        "holdout_count": len(holdout_eps),
        "guided_count": len(guided_eps),
        "holdout_success": holdout_succ,
        "guided_success": guided_succ,
        "holdout_pass_rate": holdout_succ / len(holdout_eps) if holdout_eps else None,
        "guided_pass_rate": guided_succ / len(guided_eps) if guided_eps else None,
        "avg_contrastive_delta": avg_delta,
    }


def _planning_mode_breakdown(eps: list[dict[str, Any]]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for e in eps:
        m = e.get("planning_mode") or "unknown"
        counter[m] = counter.get(m, 0) + 1
    return counter


def generate_report(
    all_eps: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Write JSON, CSV and Markdown reports.

    When ``filename_suffix`` is non-empty (e.g. ``"_pure_dynamic"``) the
    output files become ``validation_results_pure_dynamic.json``, etc.
    This is what single-condition runs use to avoid clobbering each other
    when multiple processes share an output directory.  An empty suffix
    produces the unified ``validation_results.json`` etc., which is what
    a multi-condition run or ``--merge`` produces.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_name = "validation_results%s.json" % filename_suffix
    csv_name = "validation_episodes%s.csv" % filename_suffix
    md_name = "validation_report%s.md" % filename_suffix

    # JSON
    payload: dict[str, Any] = {}
    for cond, eps in all_eps.items():
        payload[cond] = {
            "episodes": eps,
            "summary": _summary(eps),
            "planning_mode_breakdown": _planning_mode_breakdown(eps),
        }
    (output_dir / json_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # CSV
    flat: list[dict[str, Any]] = []
    for eps in all_eps.values():
        flat.extend(eps)
    if flat:
        # Union the keys across all rows so that error-path episodes (which
        # may have a smaller dict) don't break csv.DictWriter.
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in flat:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with (output_dir / csv_name).open(
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in flat:
                writer.writerow(row)

    # Markdown
    lines: list[str] = ["# 20-task Three-Layer Validation\n"]
    lines.append("## Summary\n")
    lines.append(
        "| Condition | Tasks | Pass Rate | Avg Steps | Avg Tokens | Wall (s) "
        "| Holdout | Guided | Δ |"
    )
    lines.append(
        "|-----------|-------|-----------|-----------|------------|----------"
        "|---------|--------|---|"
    )
    for cond, cfg in CONDITIONS.items():
        eps = all_eps.get(cond, [])
        if not eps:
            continue
        s = _summary(eps)
        holdout_cell = (
            "%d/%d (%.0f%%)" % (
                s["holdout_success"], s["holdout_count"],
                s["holdout_pass_rate"] * 100,
            )
            if s.get("holdout_pass_rate") is not None else "—"
        )
        guided_cell = (
            "%d/%d (%.0f%%)" % (
                s["guided_success"], s["guided_count"],
                s["guided_pass_rate"] * 100,
            )
            if s.get("guided_pass_rate") is not None else "—"
        )
        delta_cell = (
            "%+.2f" % s["avg_contrastive_delta"]
            if s.get("avg_contrastive_delta") is not None else "—"
        )
        lines.append(
            "| %s | %d | %.0f%% | %.1f | %.0f | %.0f | %s | %s | %s |" % (
                cond, s["total"], s["pass_rate"] * 100,
                s["avg_steps"], s["avg_tokens"], s["wall_time_s"],
                holdout_cell, guided_cell, delta_cell,
            )
        )

    lines.append("\n## Planning Mode Breakdown\n")
    for cond in CONDITIONS:
        eps = all_eps.get(cond, [])
        if not eps:
            continue
        lines.append("### %s" % cond)
        bd = _planning_mode_breakdown(eps)
        for mode in sorted(bd, key=lambda m: -bd[m]):
            lines.append("- %s: %d" % (mode, bd[mode]))
        lines.append("")

    # Layer effect comparisons
    def _delta(a: str, b: str, key: str) -> float:
        if a not in all_eps or b not in all_eps:
            return 0.0
        sa = _summary(all_eps[a])
        sb = _summary(all_eps[b])
        return sa.get(key, 0.0) - sb.get(key, 0.0)

    lines.append("\n## Layer Effects\n")
    if "layer1_only" in all_eps and "compile_only" in all_eps:
        lines.append(
            "- **Layer 1 effect** (layer1_only - compile_only): "
            "pass_rate %+.0f%%, avg_tokens %+.0f, avg_steps %+.1f"
            % (
                _delta("layer1_only", "compile_only", "pass_rate") * 100,
                _delta("layer1_only", "compile_only", "avg_tokens"),
                _delta("layer1_only", "compile_only", "avg_steps"),
            )
        )
    if "layer1_2" in all_eps and "layer1_only" in all_eps:
        lines.append(
            "- **Layer 2 effect** (layer1_2 - layer1_only): "
            "pass_rate %+.0f%%, avg_tokens %+.0f, avg_steps %+.1f"
            % (
                _delta("layer1_2", "layer1_only", "pass_rate") * 100,
                _delta("layer1_2", "layer1_only", "avg_tokens"),
                _delta("layer1_2", "layer1_only", "avg_steps"),
            )
        )
    if "full_library" in all_eps and "layer1_2" in all_eps:
        lines.append(
            "- **Layer 3 effect** (full_library - layer1_2): "
            "pass_rate %+.0f%%, avg_tokens %+.0f, avg_steps %+.1f"
            % (
                _delta("full_library", "layer1_2", "pass_rate") * 100,
                _delta("full_library", "layer1_2", "avg_tokens"),
                _delta("full_library", "layer1_2", "avg_steps"),
            )
        )

    (output_dir / md_name).write_text(
        "\n".join(lines), encoding="utf-8",
    )
    print(
        "\nReports written to %s (suffix=%r)"
        % (output_dir, filename_suffix or "<unified>")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _merge_from_checkpoints(output_dir: Path) -> None:
    """Reconstruct a unified report from per-condition checkpoints.

    Loads ``_checkpoint_<cond>.json`` for every condition that exists on
    disk and writes the unified ``validation_results.json``,
    ``validation_episodes.csv`` and ``validation_report.md``.  Used after
    parallel single-condition runs.
    """
    print("Merging checkpoints from %s" % output_dir)
    all_eps: dict[str, list[dict[str, Any]]] = {}
    for cond in CONDITIONS:
        loaded = _load_checkpoint(output_dir, cond)
        if loaded is None:
            print("  [merge] %s: no checkpoint, skipping" % cond)
            continue
        eps, _lib, _idx = loaded
        all_eps[cond] = eps
        print("  [merge] %s: %d episodes" % (cond, len(eps)))
    if not all_eps:
        print("  [merge] No checkpoints found; nothing to write.")
        return
    generate_report(all_eps, output_dir, filename_suffix="")


def main() -> None:
    parser = argparse.ArgumentParser(description="20-task three-layer validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--config-path", default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model in the loaded config (frozen dataclass replace).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS.keys()),
        help="Subset of conditions to run.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Skip running; rebuild unified report from per-condition checkpoints.",
    )
    parser.add_argument(
        "--task-type",
        default=None,
        help="只运行指定类型任务（如 pick_heat_then_place_in_recep），环境级过滤",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="从第 N 个（类型过滤后的）任务开始（hold-out 冻结评估用）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --merge: pure offline reconstruction, no LLM client needed
    if args.merge:
        _merge_from_checkpoints(output_dir)
        return

    base_config = load_runtime_config(config_path=args.config_path)
    if args.model:
        config = dataclasses.replace(base_config, model=args.model)
    else:
        config = base_config
    llm = LLMClient(config=config)

    # Single-condition mode = parallel-safe per-condition output files
    is_single = len(args.conditions) == 1
    suffix = ("_" + args.conditions[0]) if is_single else ""

    print("20-task Three-Layer Validation")
    print("  Limit:    %d" % args.limit)
    print("  Max step: %d" % args.max_steps)
    print("  Output:   %s" % output_dir)
    print("  Model:    %s" % config.model)
    print("  Provider: %s" % config.provider)
    print("  Conditions: %s" % ", ".join(args.conditions))
    print("  Mode:     %s" % ("single (per-cond files)" if is_single else "unified"))

    all_eps: dict[str, list[dict[str, Any]]] = {}
    grand_t0 = time.time()

    for cond in args.conditions:
        if cond not in CONDITIONS:
            print("  [warn] Unknown condition '%s' — skipping." % cond)
            continue
        eps = run_condition(
            cond, CONDITIONS[cond], llm, args.limit, args.max_steps,
            output_dir=output_dir, task_type=args.task_type,
            start_index=args.start_index,
        )
        all_eps[cond] = eps
        s = _summary(eps)
        print(
            "\n  >> %s: %d/%d pass (%.0f%%), avg_tokens=%.0f, "
            "avg_steps=%.1f, wall=%.0fs"
            % (
                cond, s["success"], s["total"], s["pass_rate"] * 100,
                s["avg_tokens"], s["avg_steps"], s["wall_time_s"],
            )
        )
        if s["holdout_count"] > 0 or s["guided_count"] > 0:
            holdout_str = (
                "%d/%d (%.0f%%)" % (
                    s["holdout_success"], s["holdout_count"],
                    s["holdout_pass_rate"] * 100,
                )
                if s["holdout_pass_rate"] is not None else "n/a"
            )
            guided_str = (
                "%d/%d (%.0f%%)" % (
                    s["guided_success"], s["guided_count"],
                    s["guided_pass_rate"] * 100,
                )
                if s["guided_pass_rate"] is not None else "n/a"
            )
            delta_str = (
                "%+.2f" % s["avg_contrastive_delta"]
                if s["avg_contrastive_delta"] is not None else "n/a"
            )
            print(
                "     holdout=%s  guided=%s  avg_delta=%s"
                % (holdout_str, guided_str, delta_str)
            )

    grand_wall = time.time() - grand_t0
    print(
        "\nTotal wall: %.0fs (%.1f min)"
        % (grand_wall, grand_wall / 60.0)
    )
    generate_report(all_eps, output_dir, filename_suffix=suffix)


if __name__ == "__main__":
    main()
