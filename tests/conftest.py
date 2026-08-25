"""pytest 入口：确保 src 在 sys.path；提供 workspace 临时目录 fixture。

workspace_tmp：使用 Path.mkdir 在项目 runs/.pytest_tmp 下创建（部分受限环境
对 tempfile.mkdtemp 目录的子目录写入有限制），测试结束自动清理。
"""

import shutil
import sys
import uuid
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace_tmp():
    base = PROJECT_ROOT / "runs" / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    root = base / f"test_{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)
