"""FlowEvo 桥接（设计文档 v2.0 §47）。

- FlowEvoLLM：把 vendored FlowEvo LLMClient 包装成 v2.0 LLM 协议
- run_flowevo_baseline：以子进程运行原版 FlowEvo（baseline 条件，零改动）
- flowevo_trace_to_record：FlowEvo ExecutionTrace → v2.0 TraceRecord
- flowevo_primitives_to_tools：PrimitiveCard → ToolAsset（PrimitiveCompiler 作为
  Code Tool Candidate Miner，§25.4/§47.5）
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable

from ..core.llm import LLM, LLMResponse, LLMUsage
from ..core.refs import ToolRef
from ..core.status import ArtifactKind, ToolLifecycle
from ..core.tool_ir import ToolAsset
from ..core.trace_ir import AttemptRecord, TraceRecord
from .code_math import ensure_flowevo_path


class FlowEvoLLM:
    """把 vendored FlowEvo 的 LLMClient 包装为 v2.0 LLM 协议（真实 API）。"""

    def __init__(self, llm_settings, *, temperature: float | None = None,
                 max_output_tokens: int | None = None,
                 request_timeout_seconds: float | None = None,
                 fallback_disable_thinking_on_length: bool = False,
                 reasoning_effort: str | None = None,
                 stream_response: bool = False,
                 app_name: str = "atomic-skillgraph") -> None:
        ensure_flowevo_path()
        from runtime.config import (
            GenerationSettings,
            RuntimeLLMConfig,
            SkillContextBudgets,
        )
        from runtime.llm_client import LLMClient
        api_key = llm_settings.resolve_api_key()
        runtime_config = RuntimeLLMConfig(
            provider=llm_settings.provider,
            api_key=api_key,
            base_url=llm_settings.base_url,
            model=llm_settings.model,
            app_name=app_name,
            skill_top_k=3,
            skill_context_budgets=SkillContextBudgets(
                draft_total_tokens=4096, repair_total_tokens=4096,
                draft_per_skill_tokens=1500, repair_per_skill_tokens=1500,
            ),
            draft=GenerationSettings(
                temperature=(llm_settings.temperature if temperature is None else temperature),
                max_output_tokens=(llm_settings.max_output_tokens if max_output_tokens is None
                                   else max_output_tokens)),
            repair=GenerationSettings(
                temperature=max(0.0, (llm_settings.temperature if temperature is None
                                      else temperature) - 0.05),
                max_output_tokens=(llm_settings.max_output_tokens if max_output_tokens is None
                                   else max_output_tokens)),
            config_path="", local_override_path="",
        )
        self.client = LLMClient(runtime_config)
        self._usage = LLMUsage()
        from runtime.config import GenerationSettings
        self._settings = GenerationSettings(
            temperature=(llm_settings.temperature if temperature is None else temperature),
            max_output_tokens=(llm_settings.max_output_tokens if max_output_tokens is None
                               else max_output_tokens),
            request_timeout_seconds=request_timeout_seconds,
            fallback_disable_thinking_on_length=fallback_disable_thinking_on_length,
            reasoning_effort=reasoning_effort,
            stream_response=stream_response)
        self._llm_settings = llm_settings

    def fork(self, *, temperature: float | None = None,
             max_output_tokens: int | None = None,
             request_timeout_seconds: float | None = None,
             fallback_disable_thinking_on_length: bool = False,
             reasoning_effort: str | None = None,
             stream_response: bool = False,
             app_name: str = "atomic-skillgraph-extractor") -> "FlowEvoLLM":
        """Create a stateless independent client using the same provider/model credentials."""
        return FlowEvoLLM(
            self._llm_settings, temperature=temperature,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
            fallback_disable_thinking_on_length=fallback_disable_thinking_on_length,
            reasoning_effort=reasoning_effort,
            stream_response=stream_response,
            app_name=app_name)

    @property
    def usage(self) -> LLMUsage:
        return self._usage

    # 调用级硬超时覆盖 client 的 3 次请求（初次 + 2 次重试）及退避。
    # 若外层先超时，内部 daemon 仍会继续请求，episode retry 会制造重复并发。
    # 180s 只兜底 requests 自身超时机制失效的病理情况。
    HARD_TIMEOUT_SECONDS = 180.0

    def generate(self, instructions: str, input_text: str,
                 *, temperature: float | None = None,
                 thinking: str | None = None,
                 structured_output: bool = False) -> LLMResponse:
        from runtime.config import GenerationSettings
        effective_temp = self._settings.temperature if temperature is None else temperature

        holder: dict[str, Any] = {}

        def _call() -> None:
            try:
                holder["result"] = self.client.generate(
                    instructions=instructions,
                    input_text=input_text,
                    settings=GenerationSettings(temperature=effective_temp,
                                                max_output_tokens=self._settings.max_output_tokens,
                                                thinking=thinking,
                                                reasoning_effort=self._settings.reasoning_effort,
                                                request_timeout_seconds=self._settings.request_timeout_seconds,
                                                fallback_disable_thinking_on_length=(
                                                    self._settings.fallback_disable_thinking_on_length
                                                    and structured_output),
                                                fallback_response_format=(
                                                    "json_object" if structured_output else None),
                                                stream_response=self._settings.stream_response),
                )
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        request_timeout = self._settings.request_timeout_seconds
        hard_timeout = (3600.0 if self._settings.stream_response else
                        self.HARD_TIMEOUT_SECONDS if request_timeout is None else
                        max(self.HARD_TIMEOUT_SECONDS,
                            (self.client.MAX_RETRIES + 1) * request_timeout + 45.0))
        thread.join(hard_timeout)
        if thread.is_alive():
            raise RuntimeError(
                "LLM call stalled: no response within %ss (hard timeout)"
                % hard_timeout)
        if holder.get("error") is not None:
            raise holder["error"]
        result = holder["result"]
        response = LLMResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            provider=result.provider,
            model=result.model,
            reasoning_text=result.reasoning_text,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
        )
        self._usage.add(response)
        return response


# ---------------------------------------------------------------------------
# Baseline 子进程
# ---------------------------------------------------------------------------

def run_flowevo_baseline(*, project_root: str | Path, benchmark: str,
                         output_dir: str | Path, conditions: list[str],
                         config_path: str | Path, limit: int = 0,
                         task_type: str | None = None,
                         max_steps: int | None = None,
                         start_index: int | None = None,
                         on_progress: Callable[[int, int, str], None] | None = None,
                         extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """以子进程运行 vendored FlowEvo 原版 runner（baseline 条件）。

    code/math:  flowevo.code_math.runner
    alfworld:   flowevo.alfworld_.run_20task_validation

    ``on_progress(done, total, line)``：流式解析子进程 stdout 中 ``[i/n]``
    进度行（ALFWorld 与 code/math runner 均按此格式打印），供全局进度条使用。
    """
    project_root = Path(project_root)
    flowevo_src = str(project_root / "src" / "flowevo")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(project_root / "src"), flowevo_src, env.get("PYTHONPATH", "")]))
    # 关键：stdout 走管道时 Python 默认块缓冲（8KB），子进程 print 的
    # [i/n] 进度行不会实时到达父进程 → 全局进度条"冻结"。强制无缓冲。
    env["PYTHONUNBUFFERED"] = "1"
    for key, value in (extra_env or {}).items():
        env[key] = value

    cmd = [sys.executable, "-m"]
    if benchmark in ("humaneval", "mbpp", "gsm8k", "math"):
        cmd += ["flowevo.code_math.runner",
                "--benchmark", benchmark,
                "--output-dir", str(Path(output_dir)),
                "--config-path", str(config_path),
                "--conditions", *conditions]
        if limit and limit > 0:
            cmd += ["--limit", str(limit)]
    elif benchmark == "alfworld":
        cmd += ["flowevo.alfworld_.run_20task_validation",
                "--output-dir", str(Path(output_dir)),
                "--config-path", str(config_path),
                "--conditions", *conditions]
        if limit and limit > 0:
            cmd += ["--limit", str(limit)]
        if task_type:
            cmd += ["--task-type", task_type]
        if max_steps and max_steps > 0:
            cmd += ["--max-steps", str(max_steps)]
        if start_index and start_index > 0:
            cmd += ["--start-index", str(start_index)]
    else:
        raise ValueError(f"FlowEvo baseline 不支持 benchmark={benchmark}")

    # 流式读取 stdout（stderr 合并），实时解析 [i/n] 进度行并保留尾部日志
    proc = subprocess.Popen(
        cmd, cwd=str(project_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    progress_re = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\]")
    tail: deque[str] = deque(maxlen=200)
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line)
        if on_progress is not None:
            match = progress_re.search(line)
            if match:
                on_progress(int(match.group(1)), int(match.group(2)), line.strip())
    returncode = proc.wait()
    merged = "".join(tail)[-6000:]
    return {
        "returncode": returncode,
        "stdout_tail": merged[-3000:],
        "stderr_tail": merged[-3000:],
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# FlowEvo ExecutionTrace → v2.0 TraceRecord
# ---------------------------------------------------------------------------

def flowevo_trace_to_record(trace, benchmark: str) -> TraceRecord:
    """把 FlowEvo ExecutionTrace 转成 v2.0 规范 Trace（供事后原子化）。"""
    record = TraceRecord(
        trace_id=getattr(trace, "trace_id", "") or f"flowevo_{_short_id()}",
        task_id=str(getattr(trace, "task_id", "")),
        task_type=str(getattr(trace, "task_pattern", "") or ""),
        task_goal=str(getattr(trace, "query", "") or ""),
        benchmark=benchmark,
        start_mode="cold",
        planning_mode=str(getattr(trace, "planning_mode", "pure_dynamic")),
        success=bool(getattr(trace, "success", False)),
        failure_type=str(getattr(trace, "failure_type", "") or ""),
        retries=int(getattr(trace, "retry_count", 0) or 0),
        token_cost=float(getattr(trace, "llm_total_tokens_total", 0) or 0),
        latency_ms=float(getattr(trace, "llm_latency_ms_total", 0) or 0),
        provenance={"source": "flowevo_execution_trace"},
        benchmark_result={"passed": bool(getattr(trace, "success", False)),
                          "flowevo_utility": float(getattr(trace, "utility_score", 0) or 0)},
    )
    for attempt_index, attempt in enumerate(getattr(trace, "attempts", []) or []):
        feedback = getattr(attempt, "verifier_feedback", None)
        record.attempts.append(AttemptRecord(
            index=attempt_index,
            stage=str(getattr(attempt, "stage", "draft")),
            candidate=str(getattr(attempt, "candidate_code", "") or ""),
            passed=bool(getattr(feedback, "passed", False)) if feedback else False,
            feedback={"stderr": str(getattr(feedback, "stderr", "") or "")} if feedback else {},
            failure_type=str(getattr(attempt, "failure_type", "") or ""),
            repair_source=str(getattr(attempt, "repair_source", "") or ""),
        ))
    # 通过候选代码
    for attempt in reversed(record.attempts):
        if attempt.passed and attempt.candidate:
            record.candidate_code = attempt.candidate
            break
    if not record.candidate_code:
        history = getattr(trace, "code_history", []) or []
        if history:
            record.candidate_code = str(history[-1])
    # state snapshots（以验证结果为状态信号）
    passed = record.success
    record.state_snapshots = [
        {"step": 0, "state": {"facts": [], "text": "start"}},
        {"step": len(record.attempts), "state": {
            "facts": ["callable_returns_expected(solve)"] if passed else [],
            "text": "passed" if passed else "failed"}},
    ]
    return record


def _short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# PrimitiveCard → ToolAsset（PrimitiveCompiler 输出作为 Tool Skeleton 来源）
# ---------------------------------------------------------------------------

def flowevo_primitives_to_tools(trace, primitive_store, benchmark: str) -> list[ToolAsset]:
    """把 vendored FlowEvo 提取的 PrimitiveCard 映射为 ToolAsset Skeleton。"""
    ensure_flowevo_path()
    from memory.primitive_store import PrimitiveStore
    tools: list[ToolAsset] = []
    store = primitive_store if isinstance(primitive_store, PrimitiveStore) else PrimitiveStore(str(primitive_store))
    cards = store.list_all()
    trace_id = getattr(trace, "trace_id", "")
    for card in cards:
        if trace_id and trace_id not in (card.source_trace_ids or []):
            continue
        code_path = Path(card.code_path)
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        if not code.strip():
            continue
        tool = ToolAsset(
            ref=ToolRef(tool_id=f"{benchmark}.{card.helper_name}"[:120], version="0.1.0"),
            artifact_kind=ArtifactKind.PYTHON_CALLABLE,
            summary=str(card.description or f"FlowEvo primitive helper: {card.helper_name}"),
            signature={"entry_point": str(card.helper_name),
                       "parameters": [{"name": str(a), "required": True}
                                      for a in (card.signature or {}).get("args", [])]},
            interface={"inputs": list((card.signature or {}).get("args", [])),
                       "outputs": [{"name": "result"}]},
            artifact={"code": code},
            tests=[],
            safety={"direct_execution_allowed": False, "checks_passed": []},
            provenance={"source_trace_ids": list(card.source_trace_ids or []),
                        "source_skill_ids": list(card.source_skill_ids or []),
                        "extraction_method": "flowevo_primitive_compiler"},
            statistics={"support_count": int(card.support_count or 1),
                        "call_count": 0, "success_count": 0, "failure_count": 0,
                        "utility": float(card.utility or 0.5)},
            lineage={"generalized_from": [], "specialized_from": [], "supersedes": None},
            status=ToolLifecycle.DRAFT,
        )
        tools.append(tool)
    return tools
