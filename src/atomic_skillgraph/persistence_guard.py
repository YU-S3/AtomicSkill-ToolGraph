"""Fail-closed checks for data written into portable long-term assets."""

from __future__ import annotations

import re
from typing import Any


_WINDOWS_PATH = re.compile(
    r"(?i)(?:\b[a-z]:[\\/](?:[^\s'\"<>|]+[\\/]?)+|"
    r"\\\\[a-z0-9_.-]+[\\/][^\s'\"<>|]+)")
_POSIX_PATH = re.compile(
    r"(?<![a-z0-9_])/(?:home|users|mnt|tmp|var|etc|opt|workspace|workspaces|"
    r"private|srv)/(?:[^\s'\"<>]+)", re.IGNORECASE)
_URL = re.compile(r"(?i)\bhttps?://[^\s'\"<>]+")
_EMAIL = re.compile(r"(?i)(?<![\w.+-])[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
_INSTANCE = re.compile(
    r"(?i)(?<![a-z0-9])([a-z][a-z0-9-]*)(?:_|\s+)(\d+)(?![a-z0-9])")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer\s+)[a-z0-9._~+/=-]{12,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|"
        r"client[_-]?secret|private[_-]?key)\s*[:=]\s*['\"]?"
        r"(?!\$|\{|<redacted>)[^\s,'\"}]{6,}"),
)
_SECRET_KEYS = {
    "api_key", "apikey", "access_key", "access_token", "refresh_token",
    "auth_token", "password", "passwd", "secret", "client_secret",
    "private_key", "credential", "credentials",
}
_SAFE_SECRET_DECLARATIONS = {
    "", "string", "str", "secret", "secret_ref", "credential_ref",
    "api_key_ref", "token_ref", "required", "optional", "<redacted>",
}
_INSTANCE_PREFIX_ALLOWLIST = {
    "arg", "attempt", "edge", "episode", "event", "from", "input",
    "output", "param", "parameter", "phase", "step", "task", "to",
    "tool", "trace", "var", "variable", "version",
}
_EVIDENCE_ONLY_KEYS = {
    "grounded_params", "source_value", "target_value", "before", "after",
    "bindings", "prefix", "source_graph_revision", "summary_raw",
    "implicit_dependencies_raw", "llm_reason_raw", "task_id", "env_index",
}
_STRUCTURAL_ID_KEYS = {
    "edge_id", "event_id", "occurrence_id", "origin_step_id", "phase_id",
    "source_step", "step_id", "target_step",
}
_SAFE_OPAQUE_ID = re.compile(r"^[a-zA-Z0-9_.:-]{1,240}$")
_EVIDENCE_REF = re.compile(
    r"^evidence://[a-z0-9_.-]+/[0-9a-f]{64}$")
_BRANCH_REF = re.compile(r"^branch:[0-9a-f]{64}$")
_VERSIONED_REF = re.compile(
    r"^(?:(?:skill|tool)://)?[a-zA-Z0-9_.:-]+@\d+\.\d+\.\d+$")
_HASH_VALUE = re.compile(r"^[0-9a-f]{16,128}$", re.IGNORECASE)


def validate_long_term_asset(payload: Any, *, asset_kind: str = "asset") -> list[str]:
    """Return deterministic findings for literals forbidden in portable assets.

    Evidence references and hashes are allowed.  Concrete replay payloads must
    be moved to :class:`EvidenceStore` before this function is called.
    """
    findings: list[str] = []
    _walk(payload, path="$", asset_kind=str(asset_kind or "asset"),
          findings=findings)
    return sorted(set(findings))


def _walk(value: Any, *, path: str, asset_kind: str,
          findings: list[str]) -> None:
    if isinstance(value, dict):
        # Evidence pointers are a two-field integrity object, not a naming
        # convention.  A well-formed URI paired with a different digest would
        # otherwise pass leaf-level regex checks and leave an unauditable
        # long-term asset behind.
        ref_value = value.get("evidence_ref")
        hash_value = value.get("evidence_hash")
        if isinstance(ref_value, str) or hash_value is not None:
            match = (_EVIDENCE_REF.fullmatch(ref_value)
                     if isinstance(ref_value, str) else None)
            if (match is None or not isinstance(hash_value, str)
                    or ref_value.rsplit("/", 1)[-1] != hash_value):
                findings.append(f"invalid_evidence_pointer:{path}")
            elif set(value) != {"evidence_ref", "evidence_hash"}:
                findings.append(f"noncanonical_evidence_pointer:{path}")
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            lowered = key_text.strip().lower()
            if lowered in _EVIDENCE_ONLY_KEYS and item not in (None, "", [], {}):
                findings.append(f"inline_evidence_field:{item_path}")
            if lowered in _SECRET_KEYS and _is_concrete_secret(item):
                findings.append(f"secret_field:{item_path}")
            _walk(item, path=item_path, asset_kind=asset_kind,
                  findings=findings)
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _walk(item, path=f"{path}[{index}]", asset_kind=asset_kind,
                  findings=findings)
        return
    if not isinstance(value, str) or not value:
        return

    lowered_path = path.lower()
    # A field name ending in ``_ref`` is not sufficient to make arbitrary
    # content safe.  Skip scanning only after both the field role and the
    # value's opaque-reference grammar have been verified.
    if _validated_opaque_reference(lowered_path, value):
        return

    for label, pattern in (
        ("absolute_windows_path", _WINDOWS_PATH),
        ("absolute_posix_path", _POSIX_PATH),
        ("url", _URL),
        ("email", _EMAIL),
    ):
        if pattern.search(value):
            findings.append(f"{label}:{path}")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        findings.append(f"secret_literal:{path}")

    # Python source legitimately contains numbered identifiers and examples.
    # Inspect only its quoted constants for instance residue; action templates,
    # summaries and structured fields are scanned in full.
    instance_texts = ([match.group(2) for match in re.finditer(
        r"(['\"])(.*?)(?<!\\)\1", value, flags=re.DOTALL)]
        if lowered_path.endswith(".artifact.code") else [value])
    for text in instance_texts:
        for match in _INSTANCE.finditer(_without_symbols(text)):
            if match.group(1).lower() not in _INSTANCE_PREFIX_ALLOWLIST:
                findings.append(f"concrete_instance:{path}:{match.group(0)}")


def _reference_leaf(path: str) -> str:
    # Lists are walked as ``$.source_trace_ids[0]``.  Strip all trailing
    # indexes before classifying the owning field; otherwise an opaque trace
    # reference is accidentally scanned as ordinary prose and values such as
    # ``trace_12`` are rejected as grounded entity instances.
    normalized = re.sub(r"(?:\[\d+\])+$", "", path)
    return normalized.rsplit(".", 1)[-1]


def _validated_opaque_reference(path: str, value: str) -> bool:
    leaf = _reference_leaf(path)
    if leaf == "evidence_ref":
        return bool(_EVIDENCE_REF.fullmatch(value))
    if leaf == "evidence_hash" or leaf.endswith("_hash"):
        return bool(_HASH_VALUE.fullmatch(value))
    if leaf in {"source_trace_id", "source_trace_ids", "trace_id"} \
            or leaf.endswith(("_trace_id", "_trace_ids",
                               "_occurrence_id", "_occurrence_ids",
                               "_gap_id", "_gap_ids")) \
            or leaf in _STRUCTURAL_ID_KEYS:
        return bool(_SAFE_OPAQUE_ID.fullmatch(value))
    if leaf == "evidence":
        return bool(_SAFE_OPAQUE_ID.fullmatch(value)
                    or _EVIDENCE_REF.fullmatch(value)
                    or _BRANCH_REF.fullmatch(value)
                    or _VERSIONED_REF.fullmatch(value))
    if leaf.endswith(("_ref", "_refs")):
        if _BRANCH_REF.fullmatch(value) or _EVIDENCE_REF.fullmatch(value):
            return True
        return bool(_VERSIONED_REF.fullmatch(value)
                    and not _contains_forbidden_instance(value))
    return False


def _contains_forbidden_instance(value: str) -> bool:
    return any(match.group(1).lower() not in _INSTANCE_PREFIX_ALLOWLIST
               for match in _INSTANCE.finditer(_without_symbols(value)))


def _is_concrete_secret(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(
            text.lower() not in _SAFE_SECRET_DECLARATIONS
            and not text.startswith(("$", "{", "evidence://"))
        )
    # A schema object such as {"type": "string"} declares a parameter; an
    # arbitrary object/list under a secret key is a concrete credential.
    if isinstance(value, dict):
        return set(value) - {"type", "semantic_type", "required", "description"} != set()
    return True


def _without_symbols(text: str) -> str:
    text = re.sub(r"\{[a-z_][a-z0-9_]*\}", "", text, flags=re.IGNORECASE)
    return re.sub(r"\$(?:inputs|task|flow)\.[a-z_][a-z0-9_]*", "", text,
                  flags=re.IGNORECASE)
