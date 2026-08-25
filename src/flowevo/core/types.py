from typing import Literal


# 通用工作流步骤类型
StepType = Literal[
    "inspect_schema",
    "call_skill",
    "write_code",
    "run_code",
    "finalize",
    "draft_code",
    "run_tests",
    "inspect_failure",
    "repair_code",
    "draft_from_skill",
    "repair_with_skill",
]


# 技能状态
SkillStatus = Literal[
    "draft",
    "active",
    "shadow",
    "suppressed",
    "pruned",
]
