"""Skill retrieval for executable-skill experiments."""

from __future__ import annotations

import re
from typing import Any

from core.schemas import BenchmarkTaskInstance, SkillCard
from core.utils import infer_task_arg_count, infer_task_entry_point, infer_task_pattern
from agent.skill_compatibility import infer_skill_compatibility_metadata
from memory.negative_memory_store import NegativeMemoryStore
from memory.skill_registry import SkillRegistry


class Retriever:
    """Retrieve top-k candidate skills from the local registry."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        negative_memory_store: NegativeMemoryStore | None = None,
        top_k: int = 3,
    ) -> None:
        self.registry = registry or SkillRegistry(file_path="data/skills/registry.json")
        self.negative_memory_store = negative_memory_store or NegativeMemoryStore(file_path="data/governance/negative_memory.json")
        self.top_k = top_k

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower()))

    def _visible_cards(self, *, include_shadow: bool = False) -> list[SkillCard]:
        cards = list(self.registry.list_active())
        if include_shadow:
            cards.extend(self.registry.list_shadow())
        return cards

    def _task_retrieval_text(self, task: BenchmarkTaskInstance) -> str:
        if task.benchmark == "gsm8k":
            return " ".join(part for part in [task.text, task.prompt] if part)
        return " ".join(part for part in [task.prompt, task.text, task.test, " ".join(task.test_list)] if part)

    def active_bank_snapshot(self, *, include_shadow: bool = False) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for card in self._visible_cards(include_shadow=include_shadow):
            direct_eligible = bool(
                card.status == "active" and (card.direct_eligible or card.preconditions.get("direct_eligible", False))
            )
            snapshot.append(
                {
                    "skill_id": card.skill_id,
                    "callable_type": str(card.preconditions.get("callable_type", "") or ""),
                    "direct_eligible": direct_eligible,
                    "preferred_usage_subtypes": list(card.preferred_usage_subtypes or card.preconditions.get("preferred_usage_subtypes") or []),
                    "entry_point": str(card.signature.get("entry_point", "") or ""),
                    "task_pattern": str(card.preconditions.get("task_pattern", "") or ""),
                    "utility": float(card.utility),
                    "positive_transfer_count": int(card.positive_transfer_count or 0),
                    "negative_transfer_count": int(card.negative_transfer_count or 0),
                    "last_used_routing_mode": str(card.last_used_routing_mode or ""),
                    "governance_seed_mode_override": str((card.compile_metadata or {}).get("governance_seed_mode_override", "") or ""),
                    "status": card.status,
                }
            )
        snapshot.sort(key=lambda item: str(item["skill_id"]))
        return snapshot

    def _historical_positive_transfer(self, card: SkillCard) -> float:
        seeded_uses = max(int(card.seeded_use_count or 0), 1)
        seeded_success_rate = float(card.seeded_success_count or 0) / float(seeded_uses)
        score = (
            0.35 * float(card.positive_transfer_count or 0)
            + 0.75 * seeded_success_rate
            + 0.40 * max(float(card.avg_delta_first_attempt or 0.0), 0.0)
            + 0.25 * max(-float(card.avg_delta_retry_count or 0.0), 0.0)
        )
        return round(min(score, 2.5), 4)

    def _negative_transfer_risk(self, card: SkillCard) -> float:
        toxicity = 0.0
        if card.skill_id:
            toxicity = float(self.negative_memory_store.toxicity_score_for_artifact(card.skill_id))
        score = (
            0.45 * float(card.negative_transfer_count or 0)
            + 0.20 * float(card.consecutive_failures or 0)
            + 0.60 * toxicity
        )
        return round(min(score, 3.0), 4)

    def _estimated_seed_token_cost(self, card: SkillCard) -> float:
        compile_metadata = dict(card.compile_metadata or {})
        helper_count = int(compile_metadata.get("helper_function_count", 0) or 0)
        replay_assertion_count = int(compile_metadata.get("replay_assertion_count", 0) or 0)
        source_line_count = int(compile_metadata.get("source_line_count", 0) or 0)
        score = 0.25
        score += 0.03 * min(helper_count, 5)
        score += 0.04 * min(replay_assertion_count, 2)
        score += 0.015 * min(source_line_count, 15)
        return round(min(score, 1.25), 4)

    def _lexical_overlap_score(self, task: BenchmarkTaskInstance, card: SkillCard) -> float:
        query_text = self._task_retrieval_text(task)
        task_tokens = self._tokenize(query_text)
        if task.benchmark == "gsm8k":
            task_tokens.discard("solve")
        if not task_tokens:
            return 0.0
        desc_tokens = self._tokenize(card.description)
        pattern_tokens = self._tokenize(str(card.preconditions.get("task_pattern", "")))
        entry_tokens = self._tokenize(str(card.signature.get("entry_point", "")))
        verifier_tokens = self._tokenize(" ".join(str(test.get("code", "")) for test in (card.tests or [])))
        overlap = 0.0
        overlap += 0.2 * len(task_tokens & desc_tokens)
        overlap += 0.4 * len(task_tokens & pattern_tokens)
        overlap += 0.6 * len(task_tokens & entry_tokens)
        # Verifier snippets are the most reliable semantic hint we have for
        # seeded-only retrieval when many skills share the same coarse signature.
        overlap += 0.6 * len(task_tokens & verifier_tokens)
        return overlap

    def score_skill(self, task: BenchmarkTaskInstance, card: SkillCard, a: float = 1.0, b: float = 1.0, c: float = 1.0) -> dict[str, Any]:
        task_entry_point = infer_task_entry_point(task)
        task_arg_count = infer_task_arg_count(task)
        task_pattern = infer_task_pattern(task)
        callable_type = str(card.preconditions.get("callable_type", "") or "")
        entry_point = str(card.signature.get("entry_point", "") or card.preconditions.get("entry_point", "") or "")
        direct_eligible = bool(
            card.status == "active" and (card.direct_eligible or card.preconditions.get("direct_eligible", False))
        )
        exact_entry_match = bool(task_entry_point and entry_point == task_entry_point)
        if task.benchmark == "gsm8k":
            exact_entry_match = False
        signature_args = list(card.signature.get("args") or [])
        skill_arg_count = len(signature_args)
        signature_compatible = task_arg_count is None or task_arg_count == skill_arg_count
        task_pattern_match = card.preconditions.get("task_pattern") == task_pattern

        sem = 0.0
        if card.family == task.benchmark:
            sem += 1.0
        if str(card.preconditions.get("benchmark", "")) == task.benchmark:
            sem += 1.0
        if exact_entry_match:
            sem += 3.0
        if signature_compatible:
            sem += 0.1 if task.benchmark == "gsm8k" else 0.75
        if task_pattern_match:
            sem += 1.2 if task.benchmark == "gsm8k" else 1.0
        sem += self._lexical_overlap_score(task, card)

        compat = 0.0
        if card.preconditions.get("benchmark") == task.benchmark:
            compat += 1.0
        if card.preconditions.get("language") in (None, "", "python"):
            compat += 0.5
        if callable_type in (None, "", "python_function"):
            compat += 0.5
        if exact_entry_match:
            compat += 1.0
        elif signature_compatible:
            compat += 0.1 if task.benchmark == "gsm8k" else 0.5
        else:
            compat -= 0.75
        if task_pattern_match:
            compat += 0.6 if task.benchmark == "gsm8k" else 0.5

        utility = float(card.utility)
        historical_positive_transfer = self._historical_positive_transfer(card)
        negative_transfer_risk = self._negative_transfer_risk(card)
        estimated_seed_token_cost = self._estimated_seed_token_cost(card)
        compile_metadata = dict(card.compile_metadata or {})
        compatibility_metadata = infer_skill_compatibility_metadata(
            {
                "skill_id": card.skill_id,
                "name": card.name,
                "description": card.description,
                "family": card.family,
                "signature": dict(card.signature or {}),
                "preconditions": dict(card.preconditions or {}),
                "direct_eligible": direct_eligible,
                "tests": list(card.tests or []),
                "compile_metadata": compile_metadata,
                "negative_transfer_risk": negative_transfer_risk,
            }
        )
        seed_replay_pass_rate = float(compile_metadata.get("seed_replay_pass_rate", 0.0) or 0.0)
        seed_replay_attempt_count = int(compile_metadata.get("seed_replay_attempt_count", 0) or 0)
        seed_replay_pass_count = int(compile_metadata.get("seed_replay_pass_count", 0) or 0)
        recall_score = a * sem
        hard_filter_reasons: list[str] = []
        if card.preconditions.get("benchmark") != task.benchmark:
            hard_filter_reasons.append("benchmark_mismatch")
        if callable_type not in {"python_function"}:
            hard_filter_reasons.append("unsupported_callable_type")
        hard_filter_passed = not hard_filter_reasons

        rerank_score = b * compat + c * utility
        if hard_filter_passed and direct_eligible and task.benchmark != "gsm8k":
            rerank_score += 0.25
        total = recall_score + rerank_score
        return {
            "skill_id": card.skill_id,
            "score": round(total, 4),
            "recall_score": round(recall_score, 4),
            "rerank_score": round(rerank_score, 4),
            "sem": round(sem, 4),
            "compat": round(compat, 4),
            "utility": round(utility, 4),
            "entry_point": entry_point,
            "task_pattern": card.preconditions.get("task_pattern"),
            "provenance": card.provenance,
            "status": card.status,
            "code_path": card.code_path,
            "signature": card.signature,
            "tests": card.tests,
            "call_count": card.call_count,
            "success_count": card.success_count,
            "consecutive_failures": card.consecutive_failures,
            "exact_entry_match": exact_entry_match,
            "signature_compatible": signature_compatible,
            "task_arg_count": task_arg_count,
            "skill_arg_count": skill_arg_count,
            "callable_type": callable_type,
            "direct_eligible": direct_eligible,
            "preferred_usage_subtypes": list(card.preferred_usage_subtypes or card.preconditions.get("preferred_usage_subtypes") or []),
            "direct_reuse_evidence_count": int(card.direct_reuse_evidence_count),
            "positive_transfer_count": int(card.positive_transfer_count or 0),
            "negative_transfer_count": int(card.negative_transfer_count or 0),
            "historical_positive_transfer": historical_positive_transfer,
            "negative_transfer_risk": negative_transfer_risk,
            "estimated_seed_token_cost": estimated_seed_token_cost,
            "last_used_routing_mode": str(card.last_used_routing_mode or ""),
            "seed_replay_pass_rate": seed_replay_pass_rate,
            "seed_replay_attempt_count": seed_replay_attempt_count,
            "seed_replay_pass_count": seed_replay_pass_count,
            "governance_seed_mode_override": str(compile_metadata.get("governance_seed_mode_override", "") or ""),
            "task_family_match_kind": "exact" if hard_filter_passed else "mismatch",
            "app_overlap_count": 0,
            "api_overlap_count": 0,
            "hard_filter_passed": hard_filter_passed,
            "hard_filter_reasons": hard_filter_reasons,
            "negative_precondition_conflict_kind": "",
            "role": str(compatibility_metadata.get("role") or ""),
            "task_family": str(compatibility_metadata.get("task_family") or ""),
            "environment_affordance": list(compatibility_metadata.get("environment_affordance") or []),
            "insertion_mode": str(compatibility_metadata.get("insertion_mode") or ""),
            "negative_transfer_risk_label": str(compatibility_metadata.get("negative_transfer_risk_label") or ""),
            "skill_family": str(compatibility_metadata.get("skill_family") or ""),
        }

    def retrieve(self, task: BenchmarkTaskInstance, top_k: int | None = None, *, include_shadow: bool = False) -> list[dict[str, Any]]:
        results, _ = self.retrieve_with_diagnostics(task=task, top_k=top_k, include_shadow=include_shadow)
        return results

    def retrieve_with_diagnostics(
        self,
        task: BenchmarkTaskInstance,
        top_k: int | None = None,
        *,
        include_shadow: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        k = top_k or self.top_k
        cards = self._visible_cards(include_shadow=include_shadow)
        scored = [self.score_skill(task, card) for card in cards]
        scored.sort(
            key=lambda item: (
                -float(item["recall_score"]),
                -float(item["sem"]),
                -float(item["utility"]),
                str(item["skill_id"]),
            )
        )
        recall_pool = scored[: max(k * 4, 10)]
        hard_filter_drop_reasons = {
            str(item.get("skill_id", "")): list(item.get("hard_filter_reasons") or [])
            for item in recall_pool
            if not bool(item.get("hard_filter_passed", True))
        }
        filtered = [item for item in recall_pool if bool(item.get("hard_filter_passed", True))]
        filtered.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                -float(item["score"]),
                -int(bool(item["exact_entry_match"])),
                -float(item["utility"]),
                -int(item["call_count"]),
                str(item["skill_id"]),
            )
        )
        selected = filtered[:k]
        diagnostics = {
            "active_skill_bank_at_episode_start": self.active_bank_snapshot(include_shadow=include_shadow),
            "recall_candidates": recall_pool,
            "hard_filter_survivors": selected,
            "hard_filter_drop_reasons": hard_filter_drop_reasons,
            "rerank_score_decomposition": {
                str(item.get("skill_id", "")): {
                    "recall_score": float(item.get("recall_score", 0.0)),
                    "rerank_score": float(item.get("rerank_score", 0.0)),
                    "sem": float(item.get("sem", 0.0)),
                    "compat": float(item.get("compat", 0.0)),
                    "utility": float(item.get("utility", 0.0)),
                }
                for item in selected
            },
        }
        return selected, diagnostics
