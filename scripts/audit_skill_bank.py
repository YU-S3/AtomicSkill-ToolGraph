"""Audit a persisted SkillGraph before frozen evaluation.

The command is read-only by default.  It emits a machine-readable report and
returns exit code 2 when a normal-planner safety invariant is violated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.skill_ir import CompositeSkill
from atomic_skillgraph.core.status import EdgeType, SkillStatus
from atomic_skillgraph.graph.graph import composite_step_order
from atomic_skillgraph.graph.registry import SkillGraphRegistry
from atomic_skillgraph.runtime.plan_validator import validate_composite_binding_closure


def _symbolic(value: Any, path: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, str) and value.startswith("$"):
        found.append({"path": path, "value": value})
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_symbolic(item, f"{path}.{key}".strip(".")))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_symbolic(item, f"{path}[{index}]"))
    return found


def audit(root: Path) -> dict[str, Any]:
    registry = SkillGraphRegistry(root)
    graph = registry._read_graph()  # audit intentionally inspects the index
    findings: list[dict[str, Any]] = []
    for logical_id, entry in graph.get("nodes", {}).items():
        recommended = str(entry.get("recommended_version") or "")
        if recommended:
            obj = registry.get(SkillRef(logical_id, recommended))
            if obj is None:
                findings.append({"severity": "error", "kind": "recommended_missing",
                                 "ref": f"{logical_id}@{recommended}"})
            elif obj.status != SkillStatus.ACTIVE:
                findings.append({"severity": "error", "kind": "non_active_recommended",
                                 "ref": str(obj.ref), "status": obj.status.value})

    for obj in registry.list_all_versions():
        if not isinstance(obj, CompositeSkill):
            continue
        ref = str(obj.ref)
        steps, graph_report = composite_step_order(obj, registry)
        closure = validate_composite_binding_closure(obj, registry)
        unresolved = _symbolic(obj.graph, "graph")
        # Resolved task/data-flow specs legitimately serialize a symbol.  Only
        # raw unresolved placeholders are reported by the closure validator;
        # keep the full list as audit context, not automatically as an error.
        if obj.status == SkillStatus.ACTIVE and not graph_report.passed:
            findings.append({"severity": "error", "kind": "active_graph_invalid",
                             "ref": ref, "errors": graph_report.errors})
        if obj.status == SkillStatus.ACTIVE and not closure.passed:
            findings.append({"severity": "error", "kind": "active_binding_invalid",
                             "ref": ref, "report": closure.to_dict()})
        unsupported = [edge.type.value for edge in obj.edge_objects()
                       if edge.type in {EdgeType.BRANCH, EdgeType.PARALLEL,
                                        EdgeType.LOOP, EdgeType.RETRY,
                                        EdgeType.FALLBACK}]
        if obj.status == SkillStatus.ACTIVE and unsupported:
            findings.append({"severity": "error", "kind": "unsupported_control",
                             "ref": ref, "edge_types": sorted(set(unsupported))})
        findings.append({"severity": "info", "kind": "composite_audit",
                         "ref": ref, "status": obj.status.value,
                         "step_count": len(steps),
                         "symbolic_bindings": unresolved,
                         "graph_passed": graph_report.passed,
                         "closure_passed": closure.passed})
    errors = [item for item in findings if item["severity"] == "error"]
    return {"passed": not errors, "error_count": len(errors),
            "finding_count": len(findings), "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-graph", required=True,
                        help="Path to the persisted skill_graph directory")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = audit(Path(args.skill_graph))
    output = (Path(args.output) if args.output else
              Path(args.skill_graph).parent / "audit" / "bank_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "error_count": report["error_count"],
                      "output": str(output)}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
