"""Admission helpers for compiled skills."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compiler.compiler import Compiler
from compiler.gatekeeper import GateKeeper
from core.schemas import ExecutionTrace
from env.sandbox import Sandbox
from memory.skill_registry import SkillRegistry

@dataclass(frozen=True)
class CompileAdmissionResult:
    attempted: bool
    admitted: bool
    skill_id: str = ""
    provenance: str = ""
    rejection_stage: str = ""
    rejection_reason: str = ""
    status: str = ""
    benchmark: str = ""
    entry_point: str = ""
    code_path: str = ""
    normalized_code_hash: str = ""
    admission_bank: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(lines) + ("\n" if lines else "")


def normalized_code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()[:16]


def existing_dedup_keys(registry: SkillRegistry) -> set[tuple[str, str, str]]:
    keys = set()
    for card in registry.list_all():
        code_path = Path(card.code_path)
        if not code_path.exists():
            continue
        code_hash = normalized_code_hash(code_path.read_text(encoding="utf-8"))
        keys.add((card.preconditions.get("benchmark", card.family), card.signature.get("entry_point", ""), code_hash))
    return keys


def find_existing_card_for_dedup_key(
    registry: SkillRegistry,
    dedup_key: tuple[str, str, str],
):
    benchmark, entry_point, code_hash = dedup_key
    for card in registry.list_all():
        code_path = Path(card.code_path)
        if not code_path.exists():
            continue
        if (card.preconditions.get("benchmark", card.family), card.signature.get("entry_point", "")) != (benchmark, entry_point):
            continue
        if normalized_code_hash(code_path.read_text(encoding="utf-8")) == code_hash:
            return card
    return None


def _is_humaneval_seed_only_card(card) -> bool:
    benchmark = str((card.preconditions or {}).get("benchmark", card.family) or "")
    task_pattern = str((card.preconditions or {}).get("task_pattern", "") or "")
    return benchmark == "humaneval" or task_pattern in {
        "function_synthesis",
        "generic_function_synthesis",
        "humaneval_function_synthesis",
    }


def _admission_bank(card) -> str:
    if str(card.status) != "active":
        return "shadow"
    if bool(card.direct_eligible or (card.preconditions or {}).get("direct_eligible", False)):
        return "active_direct"
    return "active_seed_only"


def admit_compiled_trace(
    trace: ExecutionTrace,
    *,
    provenance: str,
    registry: SkillRegistry,
    timeout_seconds: float,
    dedup_keys: set[tuple[str, str, str]] | None = None,
    force_shadow: bool = False,
    allow_unsupported_family_direct_admission: bool = False,
) -> CompileAdmissionResult:
    compiler = Compiler(skills_root=os.fspath(registry.skills_root))
    card, code = compiler.compile(trace=trace, provenance=provenance)
    dedup_key = (
        card.preconditions.get("benchmark", card.family),
        card.signature.get("entry_point", ""),
        normalized_code_hash(code),
    )
    existing = find_existing_card_for_dedup_key(registry, dedup_key) if dedup_keys is not None and dedup_key in dedup_keys else None
    if existing is not None:
        if existing.status == "shadow" and provenance == "pure_dynamic_success":
            result = GateKeeper(sandbox=Sandbox(timeout_seconds=timeout_seconds)).validate(code=code, card=card)
            if result.allowed:
                existing.status = "active"
                if trace.trace_id and trace.trace_id not in existing.lineage:
                    existing.lineage.append(trace.trace_id)
                registry.upsert(existing)
                checks = dict(result.checks)
                checks["shadow_promotion"] = True
                return CompileAdmissionResult(
                    attempted=True,
                    admitted=True,
                    skill_id=existing.skill_id,
                    provenance=existing.provenance,
                    status="active",
                    benchmark=existing.family,
                    entry_point=existing.signature.get("entry_point", ""),
                    code_path=existing.code_path,
                    normalized_code_hash=dedup_key[2],
                    admission_bank=_admission_bank(existing),
                    checks=checks,
                )
        return CompileAdmissionResult(
            attempted=True,
            admitted=False,
            skill_id=card.skill_id,
            provenance=card.provenance,
            rejection_stage="dedup",
            rejection_reason="deduped_by_benchmark_entry_point_code_hash",
            status="deduped",
            benchmark=card.family,
            entry_point=card.signature.get("entry_point", ""),
            code_path=card.code_path,
            normalized_code_hash=dedup_key[2],
        )

    if force_shadow:
        code_path = registry.save_code(card.skill_id, code)
        card.code_path = os.fspath(code_path)
        card.status = "shadow"
        registry.upsert(card)
        if dedup_keys is not None:
            dedup_keys.add(dedup_key)
        return CompileAdmissionResult(
            attempted=True,
            admitted=False,
            skill_id=card.skill_id,
            provenance=card.provenance,
            rejection_stage="shadow_compile",
            rejection_reason="seeded_shadow_compile",
            status="shadow",
            benchmark=card.family,
            entry_point=card.signature.get("entry_point", ""),
            code_path=card.code_path,
            normalized_code_hash=dedup_key[2],
            admission_bank="shadow",
            checks={"shadow_compile": True},
        )

    result = GateKeeper(sandbox=Sandbox(timeout_seconds=timeout_seconds)).validate(code=code, card=card)
    card.compile_metadata.update(dict(result.metrics or {}))
    card.replay_validation_count += 1
    if not bool(result.checks.get("fresh_world_source_replay", True)) or not bool(result.checks.get("same_family_perturbation_replay", True)):
        card.replay_failure_count += 1
    if _is_humaneval_seed_only_card(card):
        card.direct_eligible = False
        card.preconditions["direct_eligible"] = False
        card.direct_reuse_evidence_count = 0
        card.compile_metadata["admission_bank"] = "active_seed_only"
        card.compile_metadata["same_family_perturbation_replay_supported"] = bool(
            result.checks.get("same_family_perturbation_replay_supported", False)
        )
    if card.direct_eligible and bool(result.checks.get("unsupported_family_replay", False)):
        if allow_unsupported_family_direct_admission:
            code_path = registry.save_code(card.skill_id, code)
            card.code_path = os.fspath(code_path)
            card.status = "active"
            card.compile_metadata["admission_bank"] = _admission_bank(card)
            card.compile_metadata["unsupported_family_direct_admission"] = True
            registry.upsert(card)
            if dedup_keys is not None:
                dedup_keys.add(dedup_key)
            return CompileAdmissionResult(
                attempted=True,
                admitted=True,
                skill_id=card.skill_id,
                provenance=card.provenance,
                status=card.status,
                benchmark=card.family,
                entry_point=card.signature.get("entry_point", ""),
                code_path=card.code_path,
                normalized_code_hash=dedup_key[2],
                admission_bank=_admission_bank(card),
                checks=result.checks,
                metrics=result.metrics,
            )
        code_path = registry.save_code(card.skill_id, code)
        card.code_path = os.fspath(code_path)
        card.status = "shadow"
        card.compile_metadata["admission_bank"] = "shadow"
        registry.upsert(card)
        return CompileAdmissionResult(
            attempted=True,
            admitted=False,
            skill_id=card.skill_id,
            provenance=card.provenance,
            rejection_stage="gatekeeper",
            rejection_reason="unsupported_family_direct_skill_shadowed",
            status="shadow",
            benchmark=card.family,
            entry_point=card.signature.get("entry_point", ""),
            code_path=card.code_path,
            normalized_code_hash=dedup_key[2],
            admission_bank="shadow",
            checks=result.checks,
            metrics=result.metrics,
        )
    if not result.allowed:
        code_path = registry.save_code(card.skill_id, code)
        card.code_path = os.fspath(code_path)
        card.status = "shadow"
        card.compile_metadata["admission_bank"] = "shadow"
        registry.upsert(card)
        return CompileAdmissionResult(
            attempted=True,
            admitted=False,
            skill_id=card.skill_id,
            provenance=card.provenance,
            rejection_stage="gatekeeper",
            rejection_reason="; ".join(result.reasons)[:500],
            status="shadow",
            benchmark=card.family,
            entry_point=card.signature.get("entry_point", ""),
            code_path=card.code_path,
            normalized_code_hash=dedup_key[2],
            admission_bank="shadow",
            checks=result.checks,
            metrics=result.metrics,
        )

    code_path = registry.save_code(card.skill_id, code)
    card.code_path = os.fspath(code_path)
    card.status = "active"
    card.compile_metadata["admission_bank"] = _admission_bank(card)
    registry.upsert(card)
    if dedup_keys is not None:
        dedup_keys.add(dedup_key)
    return CompileAdmissionResult(
        attempted=True,
        admitted=True,
        skill_id=card.skill_id,
        provenance=card.provenance,
        status=card.status,
        benchmark=card.family,
        entry_point=card.signature.get("entry_point", ""),
        code_path=card.code_path,
        normalized_code_hash=dedup_key[2],
        admission_bank=_admission_bank(card),
        checks=result.checks,
        metrics=result.metrics,
    )
