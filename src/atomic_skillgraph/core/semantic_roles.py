"""Semantic-role families used by persistent parameter bindings.

Grounded equality is evidence that two occurrences mentioned the same value;
it is not evidence that a source, a destination, and an execution resource are
interchangeable.  This module keeps that distinction explicit for builders and
validators without depending on benchmark task-type names.
"""

from __future__ import annotations

import re


SOURCE_LOCATION = "source_location"
TARGET_LOCATION = "target_location"
EXECUTION_RESOURCE = "execution_resource"
GENERIC_LOCATION = "generic_location"
ENTITY = "entity"

LOCATION_FAMILIES = frozenset({
    SOURCE_LOCATION,
    TARGET_LOCATION,
    EXECUTION_RESOURCE,
    GENERIC_LOCATION,
})


def normalize_semantic_role(role: str) -> str:
    """Return a stable snake-case spelling for a semantic role name."""

    return re.sub(
        r"_+", "_",
        re.sub(r"[^a-z0-9]+", "_", str(role or "").strip().lower()),
    ).strip("_")


def semantic_role_family(role: str) -> str:
    """Classify a role by its data-flow meaning, not its grounded value."""

    normalized = normalize_semantic_role(role)
    tokens = set(normalized.split("_")) if normalized else set()
    location_tokens = {
        "location", "container", "receptacle", "position", "place",
    }
    directional_location = bool(tokens.intersection(location_tokens))
    if (normalized == "object_location"
            or (directional_location
                and tokens.intersection({"source", "origin"}))):
        return SOURCE_LOCATION
    if (normalized == "destination"
            or (directional_location
                and tokens.intersection({"target", "destination", "goal"}))):
        return TARGET_LOCATION
    if tokens.intersection({
            "station", "resource", "fixture", "appliance"}):
        return EXECUTION_RESOURCE
    if tokens.intersection({
            "location", "container", "receptacle", "position", "place"}):
        return GENERIC_LOCATION
    return ENTITY


def task_role_binding_compatible(occurrence_role: str,
                                 task_role: str) -> bool:
    """Whether a task role can safely parameterize an occurrence role.

    Explicit source and destination families accept aliases within the same
    family.  Execution resources are deliberately stricter: two station roles
    are not interchangeable merely because one trace used the same fixture.
    A neutral location role may be refined by grounded evidence to any explicit
    location family.
    """

    occurrence = normalize_semantic_role(occurrence_role)
    task = normalize_semantic_role(task_role)
    if not occurrence or not task:
        return False
    if occurrence == task:
        return True
    occurrence_family = semantic_role_family(occurrence)
    task_family = semantic_role_family(task)
    if occurrence_family in {SOURCE_LOCATION, TARGET_LOCATION}:
        return occurrence_family == task_family
    if occurrence_family == EXECUTION_RESOURCE:
        # ``cleaning_station`` and ``heating_station`` remain distinct even if
        # a malformed trace grounds both to one entity.
        return False
    if occurrence_family == GENERIC_LOCATION:
        return task_family in LOCATION_FAMILIES
    return task_family == ENTITY


def unsafe_persistent_task_binding(slot_role: str, task_role: str) -> bool:
    """Detect task bindings that must never enter a reusable Composite.

    Source and station slots have authoritative directional semantics.  A
    target-shaped Atomic slot can be an operational navigation/opening slot,
    so its reusable meaning may legitimately be supplied by an explicit
    ``$task.object_location`` or ``$flow.*`` role and is not rejected here.
    """

    family = semantic_role_family(slot_role)
    if family == SOURCE_LOCATION:
        return semantic_role_family(task_role) != SOURCE_LOCATION
    if family == EXECUTION_RESOURCE:
        return not task_role_binding_compatible(slot_role, task_role)
    return False


def unsafe_composite_task_role_binding(slot_role: str, binding: object) -> bool:
    """Return whether a serialized Composite task binding is role-unsafe.

    ``binding`` is intentionally accepted in its persisted form so planning,
    lifecycle validation, and migration code can share one guard without each
    reimplementing placeholder parsing.
    """

    if not isinstance(binding, str) or not binding.startswith("$task."):
        return False
    task_role = binding[len("$task."):]
    return unsafe_persistent_task_binding(slot_role, task_role)
