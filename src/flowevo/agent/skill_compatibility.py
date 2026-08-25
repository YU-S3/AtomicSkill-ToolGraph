"""Static skill metadata inference and task-skill compatibility heuristics."""

from __future__ import annotations

import re
from typing import Any, Mapping


_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE_IO_PATTERN = re.compile(r"\b(file|path|directory|folder|stdin|stdout|json|csv|os_path|filepath|filename)\b")
_STRING_PATTERN = re.compile(r"\b(string|strings|text|substring|character|characters|letter|letters|word|words)\b")
_NUMERIC_PATTERN = re.compile(r"\b(number|numbers|integer|integers|digit|digits|count|counts|sum|product|prime)\b")
_LIST_PATTERN = re.compile(r"\b(list|lists|array|arrays|sequence|sequences|tuple|tuples)\b")
_GRID_PATTERN = re.compile(r"\b(grid|matrix|board|maze|row|rows|column|columns)\b")
_CODE_PATTERN = re.compile(r"\b(code|python|function|refactor|transform|rewrite|syntax|ast)\b")
_GUARD_PATTERN = re.compile(r"\b(check|guard|validate|verify|assert|sanity)\b")
_REPAIR_PATTERN = re.compile(r"\b(repair|fix|patch|debug|correct)\b")
_FORMATTER_PATTERN = re.compile(r"\b(format|formatter|serialize|markdown|json|csv|pretty)\b")


def _text_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text or "")}


def _normalize_text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts if str(part or "").strip()).strip().lower()


def _negative_transfer_label(score: float, *, default_high: bool = False) -> str:
    if default_high or score >= 1.25:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _environment_affordances(text: str) -> list[str]:
    affordances: list[str] = []
    if _FILE_IO_PATTERN.search(text):
        affordances.extend(["file_path", "io"])
    if _STRING_PATTERN.search(text):
        affordances.append("string")
    if _NUMERIC_PATTERN.search(text):
        affordances.append("numeric")
    if _LIST_PATTERN.search(text):
        affordances.append("list")
    if _GRID_PATTERN.search(text):
        affordances.append("grid")
    if _CODE_PATTERN.search(text):
        affordances.append("code_transform")
    seen: set[str] = set()
    ordered: list[str] = []
    for affordance in affordances:
        if affordance in seen:
            continue
        seen.add(affordance)
        ordered.append(affordance)
    return ordered


def _manual_override(text: str) -> dict[str, Any]:
    if "humaneval_141" in text or "file_name_check" in text:
        return {
            "role": "guard",
            "task_family": "file_path_guard",
            "environment_affordance": ["file_path", "io", "string"],
            "insertion_mode": "local_guard",
            "negative_transfer_risk_label": "high",
            "skill_family": "file_path_guard",
            "manual_override": "humaneval_141_file_name_check",
        }
    return {}


def infer_skill_compatibility_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Infer a lightweight compatibility schema for a skill card or retrieval row."""
    signature = payload.get("signature") or {}
    preconditions = payload.get("preconditions") or {}
    tests = payload.get("tests") or []
    compile_metadata = payload.get("compile_metadata") or {}
    entry_point = str(payload.get("entry_point") or signature.get("entry_point") or preconditions.get("entry_point") or "")
    task_pattern = str(payload.get("task_pattern") or preconditions.get("task_pattern") or payload.get("family") or "")
    role_text = _normalize_text(
        payload.get("skill_id"),
        payload.get("name"),
        payload.get("description"),
        entry_point,
        task_pattern,
        preconditions,
        compile_metadata,
    )
    evidence_text = _normalize_text(
        role_text,
        *(test.get("code") if isinstance(test, dict) else test for test in tests),
    )
    override = _manual_override(evidence_text)
    if override:
        return override

    affordances = _environment_affordances(evidence_text)
    role = "solver"
    if _REPAIR_PATTERN.search(role_text):
        role = "repair"
    elif _FORMATTER_PATTERN.search(role_text):
        role = "formatter"
    elif _GUARD_PATTERN.search(role_text):
        role = "guard"
    elif "verifier" in role_text or "unit test" in role_text:
        role = "verifier"

    if role in {"verifier", "formatter"}:
        insertion_mode = "verifier_only" if role == "verifier" else "soft_hint"
    elif role == "guard":
        insertion_mode = "local_guard"
    elif bool(payload.get("exact_entry_match", False)) or bool(payload.get("direct_eligible", False)):
        insertion_mode = "hard_skeleton"
    else:
        insertion_mode = "soft_hint"

    negative_transfer_score = float(payload.get("negative_transfer_risk", 0.0) or 0.0)
    label = _negative_transfer_label(
        negative_transfer_score,
        default_high=role == "guard" and "file_path" in affordances,
    )
    task_family = task_pattern or str(preconditions.get("benchmark") or payload.get("family") or "")
    if role == "guard" and "file_path" in affordances:
        skill_family = "file_path_guard"
    elif role == "guard":
        skill_family = f"{task_family or 'generic'}_guard"
    elif role == "verifier":
        skill_family = f"{task_family or 'generic'}_verifier"
    else:
        skill_family = task_family or "generic_solver"
    return {
        "role": role,
        "task_family": task_family,
        "environment_affordance": affordances,
        "insertion_mode": insertion_mode,
        "negative_transfer_risk_label": label,
        "skill_family": skill_family,
        "manual_override": "",
    }


def task_compatibility_context(task: Any, *, runtime_error_text: str = "") -> dict[str, Any]:
    query_text = _normalize_text(
        getattr(task, "benchmark", ""),
        getattr(task, "prompt", ""),
        getattr(task, "text", ""),
        runtime_error_text,
    )
    evidence_text = _normalize_text(
        query_text,
        getattr(task, "test", ""),
        " ".join(getattr(task, "test_list", []) or []),
    )
    tokens = _text_tokens(query_text)
    affordances = _environment_affordances(evidence_text)
    evidence = {
        "text": evidence_text,
        "tokens": tokens,
        "environment_affordance": affordances,
        "has_file_path_evidence": bool({"file", "path", "directory", "folder", "stdin", "stdout", "json", "csv", "filepath", "filename"} & tokens),
        "has_guard_evidence": bool({"check", "guard", "validate", "verify", "assert", "sanity"} & tokens),
        "task_family": str(getattr(task, "metadata", {}).get("task_family") or getattr(task, "benchmark", "") or ""),
    }
    return evidence


def compatibility_assessment(
    task: Any,
    skill_payload: Mapping[str, Any],
    *,
    runtime_error_text: str = "",
    negative_memory_hits: int = 0,
) -> dict[str, Any]:
    """Assess task-skill compatibility and emit veto reasons."""
    metadata = infer_skill_compatibility_metadata(skill_payload)
    context = task_compatibility_context(task, runtime_error_text=runtime_error_text)
    score = 1.0
    reasons: list[str] = []
    soft_only_reasons: list[str] = []
    tokens = context["tokens"]
    affordances = set(metadata.get("environment_affordance") or [])
    entry_point = str(skill_payload.get("entry_point") or (skill_payload.get("signature") or {}).get("entry_point") or "")
    task_entry_point = str(getattr(task, "entry_point", "") or "")

    if metadata.get("role") == "guard":
        score -= 0.15
        if not context["has_guard_evidence"] and "file_path" not in affordances:
            reasons.append("guard_without_evidence")
    if "file_path" in affordances and not context["has_file_path_evidence"]:
        reasons.append("file_path_without_task_evidence")
    if metadata.get("negative_transfer_risk_label") == "high":
        score -= 0.25
        soft_only_reasons.append("high_negative_transfer_risk")
    if negative_memory_hits > 0:
        score -= min(0.45, 0.25 * float(negative_memory_hits))
        soft_only_reasons.append("planner_negative_memory")
    if entry_point and task_entry_point and entry_point != task_entry_point:
        score -= 0.15
    if affordances and not affordances.intersection(set(context["environment_affordance"])):
        score -= 0.2
        soft_only_reasons.append("environment_affordance_mismatch")
    score = max(0.0, min(score, 1.2))
    return {
        "score": round(score, 4),
        "hard_veto": reasons,
        "soft_only": soft_only_reasons,
        "metadata": metadata,
        "context": {
            "task_family": context["task_family"],
            "environment_affordance": context["environment_affordance"],
        },
    }
