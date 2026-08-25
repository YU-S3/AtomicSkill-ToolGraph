"""不可变引用（frozen refs）与语义版本工具。

设计文档 v2.0 §38：
  - Skill 版本 = logical_id + semantic_version + content_hash
  - Tool 版本 = tool_id + semantic_version + artifact_hash
  - 引用形式：`skill://<logical_id>@<version>` / `tool://<tool_id>@<version>`
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class RefError(ValueError):
    """引用解析/版本错误。"""


@dataclass(frozen=True, order=True)
class SkillRef:
    """指向某个逻辑 Skill 的某个不可变版本的引用。"""

    logical_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.logical_id or not self.logical_id.strip():
            raise RefError("SkillRef.logical_id 不能为空")
        _check_version(self.version, what="SkillRef")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"skill://{self.logical_id}@{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {"logical_id": self.logical_id, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillRef":
        return cls(logical_id=str(data["logical_id"]), version=str(data["version"]))

    @classmethod
    def parse(cls, text: str) -> "SkillRef":
        text = str(text).strip()
        prefix = "skill://"
        if text.startswith(prefix):
            text = text[len(prefix):]
        if "@" not in text:
            raise RefError(f"非法 SkillRef（缺少版本）：{text!r}")
        logical_id, version = text.rsplit("@", 1)
        return cls(logical_id=logical_id.strip(), version=version.strip())


@dataclass(frozen=True, order=True)
class ToolRef:
    """指向某个 Tool Asset 某个版本的引用。"""

    tool_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.tool_id or not self.tool_id.strip():
            raise RefError("ToolRef.tool_id 不能为空")
        _check_version(self.version, what="ToolRef")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"tool://{self.tool_id}@{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {"tool_id": self.tool_id, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolRef":
        return cls(tool_id=str(data["tool_id"]), version=str(data["version"]))

    @classmethod
    def parse(cls, text: str) -> "ToolRef":
        text = str(text).strip()
        prefix = "tool://"
        if text.startswith(prefix):
            text = text[len(prefix):]
        if "@" not in text:
            raise RefError(f"非法 ToolRef（缺少版本）：{text!r}")
        tool_id, version = text.rsplit("@", 1)
        return cls(tool_id=tool_id.strip(), version=version.strip())


# ---------------------------------------------------------------------------
# 语义版本
# ---------------------------------------------------------------------------

def check_version(version: str) -> bool:
    return bool(_VERSION_RE.match(version))


def _check_version(version: str, *, what: str) -> None:
    if not check_version(version):
        raise RefError(f"{what} 需要 semver 版本（x.y.z），实际为：{version!r}")


def bump_version(version: str, part: str = "patch") -> str:
    """语义版本 +1（part: major/minor/patch）。"""
    _check_version(version, what="bump_version")
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise RefError(f"未知版本段：{part!r}")


# ---------------------------------------------------------------------------
# 内容哈希
# ---------------------------------------------------------------------------

def content_hash(obj: Any, *, exclude: tuple[str, ...] = ()) -> str:
    """规范化 JSON 的 sha256 摘要（用于不可变版本与去重）。

    exclude: 需要从哈希中排除的顶层字段（如 statistics / status 等可变统计）。
    """
    if isinstance(obj, dict):
        data = {k: v for k, v in obj.items() if k not in exclude}
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    else:
        canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def artifact_hash(text: str) -> str:
    """可执行 artifact（代码/模板）的规范化哈希。"""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def normalize_code(code: str) -> str:
    """可执行代码规范化（去首尾空白、统一换行），供相似性比较。"""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())
