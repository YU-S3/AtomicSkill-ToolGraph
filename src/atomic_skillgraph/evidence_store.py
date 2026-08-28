"""Content-addressed local evidence storage.

Long-lived Skill/Tool assets may refer to execution evidence, but must not
embed task instances, state snapshots, paths, credentials, or replay inputs.
This store keeps that evidence in a separate, local-only namespace.  A bank
export can therefore copy ``skill_graph`` and ``tools`` without copying the
private payloads under ``evidence``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .persistence import atomic_write_json


_KIND_RE = re.compile(r"[^a-z0-9_.-]+")
_REF_RE = re.compile(
    r"^evidence://(?P<kind>[a-z0-9_.-]+)/(?P<digest>[0-9a-f]{64})$")


def evidence_hash(payload: Any) -> str:
    """Return a stable full SHA-256 for a JSON-serializable payload."""
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_ref_parts(ref: str) -> tuple[str, str] | None:
    """Return ``(kind, digest)`` for one canonical evidence reference."""
    match = _REF_RE.fullmatch(str(ref or ""))
    if match is None:
        return None
    return match.group("kind"), match.group("digest")


def evidence_pointer_valid(pointer: Any, *, kind: str | None = None) -> bool:
    """Validate that an evidence pointer's URI and explicit hash agree."""
    if not isinstance(pointer, dict):
        return False
    parts = evidence_ref_parts(str(pointer.get("evidence_ref") or ""))
    if parts is None:
        return False
    ref_kind, digest = parts
    return bool(
        (kind is None or ref_kind == _safe_kind(kind))
        and str(pointer.get("evidence_hash") or "") == digest
    )


class EvidenceStore:
    """Small content-addressed JSON store for Trace/Replay evidence."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, kind: str, payload: Any, trace_id: str = "",
            event_start: int | None = None,
            event_end: int | None = None) -> dict[str, str]:
        """Persist private evidence and return only its opaque ref and hash."""
        safe_kind = _safe_kind(kind)
        digest = evidence_hash(payload)
        ref = f"evidence://{safe_kind}/{digest}"
        path = self._path(safe_kind, digest)
        association = {
            "trace_id": str(trace_id or ""),
            "event_start": event_start,
            "event_end": event_end,
        }
        has_association = bool(
            association["trace_id"]
            or association["event_start"] is not None
            or association["event_end"] is not None
        )
        envelope = {
            "schema_version": 2,
            "kind": safe_kind,
            "evidence_hash": digest,
            "trace_id": str(trace_id or ""),
            "event_start": event_start,
            "event_end": event_end,
            "associations": [association] if has_association else [],
            "payload": payload,
        }
        if path.exists():
            existing = self._read(path)
            if (not isinstance(existing, dict)
                    or existing.get("evidence_hash") != digest
                    or evidence_hash(existing.get("payload")) != digest):
                raise ValueError(
                    f"evidence_hash_collision_or_corruption:{ref}")
            # The content address deliberately ignores occurrence provenance:
            # identical private evidence is stored once.  Its envelope must,
            # however, retain *every* trace/span association or the second
            # supporting occurrence becomes unauditable.
            associations = _normalized_associations(existing)
            if has_association and association not in associations:
                associations.append(association)
            if (associations != list(existing.get("associations") or [])
                    or int(existing.get("schema_version") or 1) < 2):
                existing["schema_version"] = 2
                existing["associations"] = associations
                atomic_write_json(path, existing)
        else:
            atomic_write_json(path, envelope)
        return {"evidence_ref": ref, "evidence_hash": digest}

    def get(self, ref: str | dict[str, Any]) -> Any | None:
        """Resolve evidence, returning ``None`` when absent or corrupted."""
        if isinstance(ref, dict) and not evidence_pointer_valid(ref):
            return None
        ref_text = str(ref.get("evidence_ref") or "") \
            if isinstance(ref, dict) else str(ref or "")
        parts = evidence_ref_parts(ref_text)
        if parts is None:
            return None
        kind, digest = parts
        envelope = self._read(self._path(kind, digest))
        if not isinstance(envelope, dict):
            return None
        payload = envelope.get("payload")
        if (str(envelope.get("kind") or "") != kind
                or str(envelope.get("evidence_hash") or "") != digest
                or evidence_hash(payload) != digest):
            return None
        return payload

    def get_associations(self, ref: str | dict[str, Any]) -> list[dict[str, Any]]:
        """Return all verified trace/span associations for one evidence item."""
        if isinstance(ref, dict) and not evidence_pointer_valid(ref):
            return []
        ref_text = str(ref.get("evidence_ref") or "") \
            if isinstance(ref, dict) else str(ref or "")
        parts = evidence_ref_parts(ref_text)
        if parts is None:
            return []
        kind, digest = parts
        envelope = self._read(self._path(kind, digest))
        if not isinstance(envelope, dict):
            return []
        if (str(envelope.get("kind") or "") != kind
                or str(envelope.get("evidence_hash") or "") != digest
                or evidence_hash(envelope.get("payload")) != digest):
            return []
        return _normalized_associations(envelope)

    def _path(self, kind: str, digest: str) -> Path:
        return self.root / kind / digest[:2] / f"{digest}.json"

    @staticmethod
    def _read(path: Path) -> Any:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _safe_kind(kind: str) -> str:
    value = _KIND_RE.sub("-", str(kind or "evidence").strip().lower()).strip("-.")
    return value or "evidence"


def _normalized_associations(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize v1 provenance fields and v2 associations without duplicates."""
    normalized: list[dict[str, Any]] = []
    raw = list(envelope.get("associations") or [])
    legacy = {
        "trace_id": str(envelope.get("trace_id") or ""),
        "event_start": envelope.get("event_start"),
        "event_end": envelope.get("event_end"),
    }
    if (legacy["trace_id"] or legacy["event_start"] is not None
            or legacy["event_end"] is not None):
        raw.insert(0, legacy)
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = {
            "trace_id": str(item.get("trace_id") or ""),
            "event_start": item.get("event_start"),
            "event_end": item.get("event_end"),
        }
        if (value["trace_id"] or value["event_start"] is not None
                or value["event_end"] is not None):
            if value not in normalized:
                normalized.append(value)
    return normalized
