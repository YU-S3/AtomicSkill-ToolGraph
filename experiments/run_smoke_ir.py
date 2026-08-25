"""Stage 1：无 API smoke —— IR / Graph / Tools / Atomicizer / Evolution 检查。

用法：
    python -m experiments.run_smoke_ir
    python -m experiments.run_smoke_ir --json            # 输出 JSON
不联网、不调用 LLM、不需要任何数据文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.smoke_checks import run_all_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: no-API IR smoke")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    args = parser.parse_args()

    results = run_all_checks()
    failed = {name: message for name, message in results.items()
              if message != "PASS"}
    if args.json:
        print(json.dumps({"results": results, "failed": failed,
                          "passed": len(results) - len(failed),
                          "total": len(results)}, ensure_ascii=False, indent=2))
    else:
        print("=" * 64)
        print("Stage 1 无 API smoke：IR / Graph / Tools / Atomicizer / Evolution")
        print("=" * 64)
        for name, message in results.items():
            print(f"  [{message}] {name}")
        print("-" * 64)
        print(f"通过 {len(results) - len(failed)}/{len(results)}")
        if failed:
            print("失败项：")
            for name, message in failed.items():
                print(f"  - {name}: {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
