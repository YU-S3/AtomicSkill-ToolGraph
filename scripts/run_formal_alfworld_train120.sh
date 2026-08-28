#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

run_dir="${ASG_RUN_DIR:-runs/formal_alfworld_train120_binding_v1}"
python_bin="${ASG_PYTHON:-$HOME/asg_alfworld_venv/bin/python}"
alfworld_data="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
config_path="${ASG_CONFIG:-configs/default.yaml}"
resume_args=()

case "${1:-}" in
  "")
    if [[ -e "$run_dir" ]]; then
      echo "ERROR: $run_dir already exists; rerun with --resume or set ASG_RUN_DIR." >&2
      exit 2
    fi
    ;;
  --resume)
    resume_args=(--resume)
    ;;
  *)
    echo "Usage: $0 [--resume]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: Python environment not found or not executable: $python_bin" >&2
  exit 2
fi
if [[ ! -d "$alfworld_data" ]]; then
  echo "ERROR: ALFWorld data directory not found: $alfworld_data" >&2
  exit 2
fi

exec "$python_bin" -m experiments.run_small \
  --benchmark alfworld \
  --alfworld-split train \
  --task-types \
    pick_and_place_simple \
    pick_clean_then_place_in_recep \
    pick_heat_then_place_in_recep \
    pick_cool_then_place_in_recep \
    look_at_obj_in_light \
    pick_two_obj_and_place \
  --per-type-limit 20 \
  --conditions \
    atomic_graph_only \
    tool_repo_only \
    atomic_skillgraph_full \
  --max-steps 100 \
  --alfworld-data "$alfworld_data" \
  --config-path "$config_path" \
  --run-dir "$run_dir" \
  "${resume_args[@]}"
