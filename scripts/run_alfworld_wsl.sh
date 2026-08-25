#!/bin/bash
# ALFWorld 实验运行入口（在 WSL Ubuntu 内使用）
# 用法（Windows PowerShell 里执行）：
#     wsl -e bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph/scripts/run_alfworld_wsl.sh
# 或在 WSL 终端内：
#     cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph
#     bash scripts/run_alfworld_wsl.sh [stage] [condition...]
set -e
cd "$(dirname "$0")/.."   # 项目根
export PYTHONUNBUFFERED=1

PY="$HOME/asg_alfworld_venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "[错误] 未找到 WSL venv：$PY。请先执行：wsl -e bash /mnt/d/alfworld_deps/wsl_setup_install.sh"
  exit 1
fi

# ALFWorld 数据目录：默认 WSL 的 ~/.cache/alfworld（download 脚本的默认位置）
DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
if [ ! -d "$DATA/json_2.1.1" ]; then
  echo "[错误] 未找到 ALFWorld 数据：$DATA。请先执行：wsl -e bash /mnt/d/alfworld_deps/wsl_setup_data.sh"
  exit 1
fi

STAGE="${1:-small}"
shift 2>/dev/null || true

echo "=== WSL ALFWorld 运行（venv=$PY, data=$DATA）==="
case "$STAGE" in
  small)
    "$PY" -m experiments.run_small --benchmark alfworld --limit 10 \
        --task-type pick_heat_then_place_in_recep \
        --alfworld-data "$DATA" \
        --config-path configs/default.yaml "$@"
    ;;
  full)
    "$PY" -m experiments.run_full --benchmark alfworld \
        --alfworld-data "$DATA" \
        --config-path configs/default.yaml "$@"
    ;;
  check)
    "$PY" -c "import alfworld, textworld, fast_downward; print('依赖 OK：textworld', textworld.__version__)"
    "$PY" -c "
import sys; sys.path.insert(0, 'src')
from flowevo.alfworld_.env import AlfWorldEnv
env = AlfWorldEnv(split='eval_out_of_distribution', alfworld_data='$DATA')
print('环境初始化（枚举任务，需要几分钟）...')
n = env.initialize()
print('任务总数:', n)
task, obs, admissible = env.reset()
print('样例任务:', task.task_id, '|', task.task_type)
print('目标:', task.goal[:120])
print('admissible 前 5:', admissible[:5])
print('ENV_OK')
"
    ;;
  *)
    echo "用法：run_alfworld_wsl.sh [small|full|check]"
    exit 1
    ;;
esac
