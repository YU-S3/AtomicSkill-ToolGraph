"""Stage-1 无 API smoke 的 pytest 版本（复用 experiments.smoke_checks）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.smoke_checks import (  # noqa: E402
    ALL_CHECKS,
    check_admission,
    check_aligner,
    check_atomicizer_code,
    check_atomicizer_env,
    check_evolution,
    check_generalizer,
    check_refs_and_semver,
    check_registry_and_graph,
    check_skill_ir_roundtrip,
    check_tool_ir_lifecycle,
    check_tool_registry_and_resolver,
    check_validators,
)

NO_ARG_CHECKS = [check_refs_and_semver, check_skill_ir_roundtrip, check_tool_ir_lifecycle]
ARG_CHECKS = [check_registry_and_graph, check_aligner, check_tool_registry_and_resolver,
              check_admission, check_atomicizer_env, check_atomicizer_code,
              check_generalizer, check_evolution, check_validators]


def test_all_smoke_checks(workspace_tmp):
    from experiments.smoke_checks import run_all_checks
    results = run_all_checks(tmp_root=workspace_tmp)
    failed = {name: message for name, message in results.items() if message != "PASS"}
    assert not failed, f"smoke 检查失败：{failed}"
    assert len(results) == len(ALL_CHECKS)
