"""清理指定运行目录中的 baseline 结果（用于 API 故障后独立重跑 baseline）。

只删除 results.json 里的 baseline_dynamic / flowevo 条目与对应输出子目录，
ours 条件的结果完全保留；随后用 run_small --resume 重跑即可写回同一目录。

用法：
    python scripts/reset_baseline_results.py runs/small/alfworld_20260822T123602
"""

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：python scripts/reset_baseline_results.py <运行目录>")
        return 1
    run_dir = Path(sys.argv[1])
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"[错误] 找不到 {results_path}")
        return 1
    data = json.loads(results_path.read_text(encoding="utf-8"))
    removed = [k for k in ("baseline_dynamic", "flowevo") if data.pop(k, None) is not None]
    results_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    for sub in ("baseline_dynamic_flowevo", "flowevo_flowevo"):
        shutil.rmtree(run_dir / sub, ignore_errors=True)
    print(f"已清理：{removed}；保留条件：{sorted(data.keys())}")
    print(f"目录：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
