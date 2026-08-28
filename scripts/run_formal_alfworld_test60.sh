#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

train_run="${ASG_TRAIN_RUN_DIR:-runs/formal_alfworld_train120_runtime_ir_v1}"
eval_dir="${ASG_EVAL_RUN_DIR:-runs/formal_alfworld_eval60_runtime_ir_v1}"
python_bin="${ASG_PYTHON:-$HOME/asg_alfworld_venv/bin/python}"
alfworld_data="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
config_path="${ASG_CONFIG:-configs/default.yaml}"
resume=false

case "${1:-}" in
  "") ;;
  --resume) resume=true ;;
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
if [[ ! -d "$train_run" ]]; then
  echo "ERROR: completed train run not found: $train_run" >&2
  exit 2
fi
if [[ "$resume" == false && -e "$eval_dir" ]]; then
  echo "ERROR: $eval_dir already exists; rerun with --resume or set ASG_EVAL_RUN_DIR." >&2
  exit 2
fi

# Frozen evaluation is allowed only after every persisted bank passes its
# structural/lifecycle audit. run_evolve_eval performs the independent
# completion, task-signature, split-isolation, source-milestone and hash gates.
for condition in atomic_graph_only tool_repo_only atomic_skillgraph_full; do
  PYTHONPATH=src "$python_bin" scripts/audit_skill_bank.py \
    --skill-graph "$train_run/$condition/data/skill_graph" \
    --output "$train_run/$condition/data/audit/bank_audit.json"
done

exec "$python_bin" -m experiments.run_evolve_eval \
  --run-dir "$train_run" \
  --benchmark alfworld \
  --conditions \
    atomic_graph_only \
    tool_repo_only \
    atomic_skillgraph_full \
  --split heldout \
  --alfworld-split eval_out_of_distribution \
  --task-types \
    pick_and_place_simple \
    pick_clean_then_place_in_recep \
    pick_heat_then_place_in_recep \
    pick_cool_then_place_in_recep \
    look_at_obj_in_light \
    pick_two_obj_and_place \
  --per-type-limit 10 \
  --limit 60 \
  --expected-train-count 120 \
  --expected-heldout-count 60 \
  --baseline-conditions baseline_dynamic flowevo \
  --max-steps 100 \
  --alfworld-data "$alfworld_data" \
  --config-path "$config_path" \
  --eval-dir "$eval_dir"
