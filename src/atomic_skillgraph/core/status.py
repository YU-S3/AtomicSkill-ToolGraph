"""v2.0 生命周期与图状态枚举。

设计文档 v2.0 §29（Tool 生命周期）、§15（边类型）、§34（错误分类）、§22（路由）。
"""

from __future__ import annotations

from enum import Enum


class SkillNodeKind(str, Enum):
    """SkillGraph 长期持久化节点类型（§7：只允许这三类）。"""

    ABSTRACT_ATOMIC = "abstract_atomic"
    IMPLEMENTATION_ATOMIC = "implementation_atomic"
    COMPOSITE = "composite"


class SkillStatus(str, Enum):
    """Skill 版本状态。"""

    DRAFT = "draft"
    ACTIVE = "active"
    SHADOW = "shadow"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"


class ToolLifecycle(str, Enum):
    """Tool Asset 生命周期（§29）。

    draft/skeleton -> admission_pending -> (fail) shadow | (pass) candidate
    -> active -> preferred -> suppressed | retired
    """

    DRAFT = "draft"
    ADMISSION_PENDING = "admission_pending"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PREFERRED = "preferred"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"


# 生命周期偏序：status 只能沿合法方向迁移
_TOOL_TRANSITIONS: dict[ToolLifecycle, set[ToolLifecycle]] = {
    ToolLifecycle.DRAFT: {ToolLifecycle.ADMISSION_PENDING, ToolLifecycle.RETIRED},
    ToolLifecycle.ADMISSION_PENDING: {ToolLifecycle.CANDIDATE, ToolLifecycle.SHADOW, ToolLifecycle.RETIRED},
    ToolLifecycle.SHADOW: {ToolLifecycle.ADMISSION_PENDING, ToolLifecycle.RETIRED},
    ToolLifecycle.CANDIDATE: {
        ToolLifecycle.ACTIVE,
        ToolLifecycle.SHADOW,
        ToolLifecycle.SUPPRESSED,
        ToolLifecycle.RETIRED,
    },
    ToolLifecycle.ACTIVE: {
        ToolLifecycle.PREFERRED,
        ToolLifecycle.CANDIDATE,  # 降级
        ToolLifecycle.SUPPRESSED,
        ToolLifecycle.RETIRED,
        ToolLifecycle.SHADOW,
    },
    ToolLifecycle.PREFERRED: {
        ToolLifecycle.ACTIVE,
        ToolLifecycle.SUPPRESSED,
        ToolLifecycle.RETIRED,
        ToolLifecycle.SHADOW,
    },
    ToolLifecycle.SUPPRESSED: {ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE, ToolLifecycle.RETIRED},
    ToolLifecycle.RETIRED: set(),
}


def tool_transition_allowed(frm: ToolLifecycle, to: ToolLifecycle) -> bool:
    return to in _TOOL_TRANSITIONS.get(frm, set())


# 可被 Runtime 默认选用的状态（§29：candidate 可用但优先级低于 active/preferred）
USABLE_TOOL_STATUSES = {ToolLifecycle.CANDIDATE, ToolLifecycle.ACTIVE, ToolLifecycle.PREFERRED}


class ArtifactKind(str, Enum):
    """Tool 可执行形态（§18：跟随当前 Benchmark/Harness，不强制 Python）。"""

    PYTHON_CALLABLE = "python_callable"
    ACTION_TEMPLATE = "action_template"
    EXECUTABLE_SCRIPT = "executable_script"
    HARNESS_MACRO = "harness_macro"


class EdgeType(str, Enum):
    """SkillGraph 边类型（§15）。"""

    # structural
    CONTAINS = "contains"
    IMPLEMENTS = "implements"
    # control
    NEXT = "next"
    BRANCH = "branch"
    PARALLEL = "parallel"
    RETRY = "retry"
    FALLBACK = "fallback"
    LOOP = "loop"
    # data
    DATA_FLOW = "data_flow"
    # dependency
    REQUIRES_SKILL = "requires_skill"
    REQUIRES_PERMISSION = "requires_permission"
    REQUIRES_ENVIRONMENT = "requires_environment"
    REQUIRES_SCHEMA = "requires_schema"
    # semantic
    EQUIVALENT = "equivalent"
    SIMILAR = "similar"
    ALTERNATIVE = "alternative"
    CONFLICT = "conflict"
    # evolution
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    SPLIT_FROM = "split_from"
    MERGED_FROM = "merged_from"
    GENERALIZED_FROM = "generalized_from"
    SPECIALIZED_FROM = "specialized_from"


class ErrorKind(str, Enum):
    """节点级错误分类（§34.1）。"""

    PRECONDITION_VIOLATION = "precondition_violation"
    INPUT_SCHEMA_MISMATCH = "input_schema_mismatch"
    IMPLEMENTATION_SELECTION_ERROR = "implementation_selection_error"
    TOOL_BINDING_ERROR = "tool_binding_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TOOL_INTERFACE_ERROR = "tool_interface_error"
    TOOL_SAFETY_REJECTION = "tool_safety_rejection"
    OUTPUT_SCHEMA_MISMATCH = "output_schema_mismatch"
    EFFECT_VIOLATION = "effect_violation"
    VALIDATOR_ERROR = "validator_error"
    CONTROL_FLOW_ERROR = "control_flow_error"
    DATA_FLOW_ERROR = "data_flow_error"
    COMPOSITE_VALIDATION_ERROR = "composite_validation_error"
    BENCHMARK_FAILURE = "benchmark_failure"
    UNKNOWN = "unknown"


class ExecutionMode(str, Enum):
    """单节点执行路由（§22：direct -> seeded -> dynamic）。"""

    DIRECT = "direct"
    SEEDED = "seeded"
    DYNAMIC = "dynamic"


class ValidationLevel(str, Enum):
    """三级验证体系（§35）：tool / atomic / composite / benchmark。"""

    TOOL = "tool"
    ATOMIC = "atomic"
    COMPOSITE = "composite"
    BENCHMARK = "benchmark"
