"""Regression tests for all-in FlowEvo online episode token accounting."""

from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.adapters.code_math import ensure_flowevo_path


ensure_flowevo_path()

from alfworld_.compiler import AlfWorldCompiler  # noqa: E402
from alfworld_.generator import AlfWorldGenerator  # noqa: E402
from alfworld_.schemas import AlfWorldAction, AlfWorldTrace  # noqa: E402
from compiler.strategy_compiler import StrategyCompiler  # noqa: E402


class _FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, **_kwargs):
        return self.response


def _response(*, text: str, prompt: int, completion: int, total: int,
              latency_ms: float = 0.0):
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        latency_ms=latency_ms,
    )


def _successful_alfworld_trace() -> AlfWorldTrace:
    return AlfWorldTrace(
        trace_id="trace-1",
        task_id="task-1",
        task_type="pick_and_place_simple",
        goal="put an object in a receptacle",
        success=True,
        actions=[AlfWorldAction(
            step=0,
            action="go to countertop 1",
            observation="You arrive at countertop 1.",
            score=1.0,
            done=True,
            admissible_commands=[],
        )],
        llm_prompt_tokens_total=100,
        llm_completion_tokens_total=40,
        llm_total_tokens_total=150,
        llm_call_count=2,
        llm_latency_ms_total=10.0,
    )


def test_alfworld_compiler_charges_provider_total_to_online_trace():
    response = _response(
        text=(
            "When searching, inspect likely containers, because the object location is unknown.\n"
            "When carrying the object, visit the destination, because delivery is required.\n"
            "When an action fails, avoid repeating it, because loops waste the budget."
        ),
        prompt=11,
        completion=7,
        # Deliberately differs from prompt+completion: the provider total is
        # authoritative and may include additional token classes.
        total=23,
        latency_ms=4.5,
    )
    trace = _successful_alfworld_trace()

    result = AlfWorldCompiler(llm_client=_FakeLLM(response)).compile(trace)

    assert result.exemplar is not None
    assert result.exemplar.extraction_tokens == 23
    assert trace.llm_prompt_tokens_total == 111
    assert trace.llm_completion_tokens_total == 47
    assert trace.llm_total_tokens_total == 173
    assert trace.llm_call_count == 3
    assert trace.llm_latency_ms_total == 14.5


def test_alfworld_compiler_charges_call_even_when_semantic_parse_rejects_output():
    trace = _successful_alfworld_trace()
    response = _response(text="short", prompt=5, completion=2, total=9)

    result = AlfWorldCompiler(llm_client=_FakeLLM(response)).compile(trace)

    assert result.exemplar is None
    assert trace.llm_total_tokens_total == 159
    assert trace.llm_call_count == 3


def test_alfworld_step_preserves_authoritative_total_and_latency():
    response = _response(
        text="Think: inspect the room\nAct: look",
        prompt=6,
        completion=4,
        total=15,
        latency_ms=2.25,
    )
    output = AlfWorldGenerator(_FakeLLM(response)).step(
        task_goal="find an object",
        observation="You are in a room.",
        admissible_commands=["look", "inventory"],
    )

    assert output.action == "look"
    assert output.total_tokens == 15
    assert output.latency_ms == 2.25


def test_strategy_compiler_charges_post_runtime_llm_usage():
    response = _response(
        text=(
            "APPROACH: Use a frequency map\n"
            "STRATEGY: Count values once and query the map. This avoids repeated scans.\n"
            "APPLICABILITY: Collection counting tasks"
        ),
        prompt=13,
        completion=8,
        total=25,
        latency_ms=3.0,
    )
    trace = SimpleNamespace(
        success=True,
        task_pattern="count_items",
        query="count repeated items",
        benchmark="humaneval",
        task_id="HumanEval/1",
        llm_prompt_tokens_total=20,
        llm_completion_tokens_total=10,
        llm_total_tokens_total=32,
        llm_call_count=1,
        llm_latency_ms_total=5.0,
    )
    skill = SimpleNamespace(
        skill_id="skill-1",
        signature={"args": ["items"], "entry_point": "count_items"},
    )

    card = StrategyCompiler(
        llm_client=_FakeLLM(response),
        llm_settings=SimpleNamespace(),
    ).compile_strategy(trace, skill, "def count_items(items):\n    return {}")

    assert card is not None
    assert trace.llm_prompt_tokens_total == 33
    assert trace.llm_completion_tokens_total == 18
    assert trace.llm_total_tokens_total == 57
    assert trace.llm_call_count == 2
    assert trace.llm_latency_ms_total == 8.0
