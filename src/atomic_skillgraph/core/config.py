"""系统配置与消融开关（设计文档 v2.0 §57 实验条件与消融）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# 阈值（§6.4 SplitScore、§36.3 insight、§22 路由 —— 全部可配置）
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    split_score_force: float = 0.70          # >= 强制 split candidate
    split_score_consider: float = 0.45       # 0.45-0.70 由 replay/复用证据决定
    insight_min_samples: int = 3             # Layer-3 insight 最少轨迹数
    direct_min_utility: float = 0.5          # direct 执行最小历史 utility
    direct_min_success: int = 3              # direct 执行最小历史成功数
    direct_max_consecutive_failures: int = 2 # 连续失败降级阈值
    candidate_min_support: int = 3           # candidate -> active 额外成功证据
    preferred_margin: float = 0.1            # preferred 选择的 utility 领先幅度
    preferred_min_direct_success_rate: float = 0.95 # preferred 必须有稳定 Direct 证据
    suppress_failure_threshold: int = 2      # 失败证据达到阈值 → 抑制审查
    retirement_utility: float = 0.2          # 长期 utility 低于阈值 → 退役候选
    llm_generalize_min_group: int = 2        # generalize 最少同构组大小
    max_tool_versions: int = 20              # 每个 tool_id 保留的历史版本上限
    admission_timeout_seconds: float = 10.0  # 工具 admission 沙箱超时
    acquire_discovery_max_locations: int = 30 # Acquire 有界位置发现上限
    llm_max_consecutive_errors: int = 1       # provider 已重试；禁止决策层再放大
    infrastructure_episode_retries: int = 2  # 从任务初始状态重跑次数
    composite_min_support: int = 2            # 单轨迹 Composite 仅为 draft；多轨迹支持后晋升
    env_node_max_steps: int = 30              # 单个 Atomic（含 fallback）的环境步数上限
    env_attempt_max_steps: int = 15           # 单种执行模式的环境步数上限
    env_dynamic_node_max_steps: int = 20      # Dynamic gap 单节点动作上限


@dataclass
class FeatureFlags:
    """消融开关（§57.3）。

    条件映射（§57.2）：
      - baseline_dynamic / flowevo：走 vendored FlowEvo 原版 runner（子进程）
      - atomic_graph_only: enable_tool_evolution=False
      - tool_repo_only:     enable_composite=False
      - atomic_skillgraph_full: 全部开启
    """

    enable_composite: bool = True            # Composite Skill 构建/检索
    enable_tool_evolution: bool = True       # 独立 Tool 进化（generalize/merge/specialize/split）
    enable_node_validator: bool = True       # 节点级验证
    enable_layer3_insight: bool = True       # Composite Layer-3 insight
    enable_generalization: bool = True
    enable_specialization: bool = True
    enable_cross_task_type_reuse: bool = True  # False = task_type 硬过滤（对照实验）
    enable_nm_binding: bool = True           # False = 强制 Skill:Tool = 1:1
    enable_governance: bool = True           # utility/negative-transfer 治理
    enable_primitive_reuse: bool = True      # PrimitiveCompiler 作为 Tool 挖掘器
    task_type_hard_restricted: bool = False  # task_type 是否作为硬过滤
    # Optional Tool-only discovery. Declared Atomic source-location contracts
    # are resolved by bounded discovery regardless of this ablation flag.
    enable_framework_discovery: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "enable_composite": self.enable_composite,
            "enable_tool_evolution": self.enable_tool_evolution,
            "enable_node_validator": self.enable_node_validator,
            "enable_layer3_insight": self.enable_layer3_insight,
            "enable_generalization": self.enable_generalization,
            "enable_specialization": self.enable_specialization,
            "enable_cross_task_type_reuse": self.enable_cross_task_type_reuse,
            "enable_nm_binding": self.enable_nm_binding,
            "enable_governance": self.enable_governance,
            "enable_primitive_reuse": self.enable_primitive_reuse,
            "task_type_hard_restricted": self.task_type_hard_restricted,
            "enable_framework_discovery": self.enable_framework_discovery,
        }


@dataclass
class LLMSettings:
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    api_key: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = 900
    mock: bool = False                       # 使用 MockLLM（smoke 阶段）
    extractor_temperature: float = 0.1       # 独立 Extractor Agent 使用同一模型的低温会话
    extractor_max_output_tokens: int = 384000  # DeepSeek V4 官方最大输出上限
    extractor_thinking: str = "enabled"      # enabled | disabled；Runtime 可继续按调用关闭
    extractor_reasoning_effort: str = "low"
    extractor_read_timeout_seconds: float = 600.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMSettings":
        data = data or {}
        return cls(
            provider=str(data.get("provider", "openrouter")),
            base_url=str(data.get("base_url", "https://openrouter.ai/api/v1")),
            model=str(data.get("model", "openai/gpt-4o-mini")),
            api_key=str(data.get("api_key", "")),
            api_key_env=str(data.get("api_key_env", "OPENROUTER_API_KEY")),
            temperature=float(data.get("temperature", 0.2)),
            max_output_tokens=int(data.get("max_output_tokens", 900)),
            mock=bool(data.get("mock", False)),
            extractor_temperature=float(data.get("extractor_temperature", 0.1)),
            extractor_max_output_tokens=int(data.get("extractor_max_output_tokens", 384000)),
            extractor_thinking=str(data.get("extractor_thinking", "enabled")),
            extractor_reasoning_effort=str(
                data.get("extractor_reasoning_effort", "low")),
            extractor_read_timeout_seconds=float(
                data.get("extractor_read_timeout_seconds", 600.0)),
        )

    def resolve_api_key(self) -> str:
        """api_key 字段优先，其次环境变量。"""
        key = self.api_key.strip()
        if key and key not in ("YOUR_OPENROUTER_API_KEY", "YOUR_API_KEY"):
            return key
        return os.getenv(self.api_key_env, "")


@dataclass
class SystemConfig:
    data_dir: Path = Path("data")
    llm: LLMSettings = field(default_factory=LLMSettings)
    thresholds: Thresholds = field(default_factory=Thresholds)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    seed: int = 42
    max_repairs: int = 2
    retrieval_top_k: int = 5
    task_type_soft_bonus: float = 0.3        # 附录 D：task_type 弱召回加分
    planning_min_score: float = 0.4          # 检索得分低于阈值视为无可用能力（cold）
    max_steps: int = 50                      # 交互环境最大步数
    maintenance_interval: int = 5            # 全局维护（generalize/lifecycle）周期（成功数）
    freeze_skills: bool = False              # 冻结评估模式：只读技能库，禁止一切进化/统计写入

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, base_dir: Path | None = None) -> "SystemConfig":
        data = data or {}
        base_dir = base_dir or Path.cwd()
        data_dir = Path(str(data.get("data_dir", "data")))
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir
        return cls(
            data_dir=data_dir,
            llm=LLMSettings.from_dict(data.get("llm")),
            thresholds=Thresholds(**{k: v for k, v in (data.get("thresholds") or {}).items() if k in Thresholds.__dataclass_fields__}),
            features=FeatureFlags(**{k: v for k, v in (data.get("features") or {}).items() if k in FeatureFlags.__dataclass_fields__}),
            seed=int(data.get("seed", 42)),
            max_repairs=int(data.get("max_repairs", 2)),
            retrieval_top_k=int(data.get("retrieval_top_k", 5)),
            task_type_soft_bonus=float(data.get("task_type_soft_bonus", 0.3)),
            planning_min_score=float(data.get("planning_min_score", 0.4)),
            max_steps=int(data.get("max_steps", 50)),
            maintenance_interval=int(data.get("maintenance_interval", 5)),
            freeze_skills=bool(data.get("freeze_skills", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "llm": {
                "provider": self.llm.provider,
                "base_url": self.llm.base_url,
                "model": self.llm.model,
                "api_key_env": self.llm.api_key_env,
                "temperature": self.llm.temperature,
                "max_output_tokens": self.llm.max_output_tokens,
                "mock": self.llm.mock,
                "extractor_temperature": self.llm.extractor_temperature,
                "extractor_max_output_tokens": self.llm.extractor_max_output_tokens,
                "extractor_thinking": self.llm.extractor_thinking,
                "extractor_reasoning_effort": self.llm.extractor_reasoning_effort,
                "extractor_read_timeout_seconds": self.llm.extractor_read_timeout_seconds,
            },
            "thresholds": {
                "split_score_force": self.thresholds.split_score_force,
                "split_score_consider": self.thresholds.split_score_consider,
                "insight_min_samples": self.thresholds.insight_min_samples,
                "direct_min_utility": self.thresholds.direct_min_utility,
                "direct_min_success": self.thresholds.direct_min_success,
                "candidate_min_support": self.thresholds.candidate_min_support,
                "preferred_margin": self.thresholds.preferred_margin,
                "preferred_min_direct_success_rate": self.thresholds.preferred_min_direct_success_rate,
                "admission_timeout_seconds": self.thresholds.admission_timeout_seconds,
                "acquire_discovery_max_locations": self.thresholds.acquire_discovery_max_locations,
                "llm_max_consecutive_errors": self.thresholds.llm_max_consecutive_errors,
                "infrastructure_episode_retries": self.thresholds.infrastructure_episode_retries,
                "composite_min_support": self.thresholds.composite_min_support,
                "env_node_max_steps": self.thresholds.env_node_max_steps,
                "env_attempt_max_steps": self.thresholds.env_attempt_max_steps,
                "env_dynamic_node_max_steps": self.thresholds.env_dynamic_node_max_steps,
            },
            "features": self.features.to_dict(),
            "seed": self.seed,
            "max_repairs": self.max_repairs,
            "retrieval_top_k": self.retrieval_top_k,
            "task_type_soft_bonus": self.task_type_soft_bonus,
            "planning_min_score": self.planning_min_score,
            "max_steps": self.max_steps,
            "maintenance_interval": self.maintenance_interval,
            "freeze_skills": self.freeze_skills,
        }


def load_config(path: str | Path, *, base_dir: Path | None = None) -> SystemConfig:
    """从 YAML 加载配置；不存在时返回默认。

    - 若同目录存在 local.yaml，则其值覆盖基础配置（key 不入库）
    - 相对 data_dir 解析到项目根（configs/ 的上一级）
    """
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    local_path = path.parent / "local.yaml"
    if local_path.exists() and local_path != path:
        raw_local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
        if isinstance(raw_local, dict):
            data = _merge_nested(data, raw_local)
    root = base_dir or path.resolve().parent.parent
    return SystemConfig.from_dict(data, base_dir=root)


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged
