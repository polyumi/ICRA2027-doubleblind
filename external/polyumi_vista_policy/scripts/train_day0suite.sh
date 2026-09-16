#!/usr/bin/env bash
# Train hardcoded baseline policies on a PolyUMI multimodal export.
#
# Models: see_hear_feel, sparsh_x, polytouch, qformer, VisTA, vta_diffusion
#
# Usage:
#   DAY0SUITE_DATASET=/path/to/export.zarr.zip ./scripts/train_day0suite.sh
#   ./scripts/train_day0suite.sh /path/to/export.zarr.zip
#   ./scripts/train_day0suite.sh --model see_hear_feel
#   ./scripts/train_day0suite.sh --model qformer
#   ./scripts/train_day0suite.sh --model vista
#   ./scripts/train_day0suite.sh --model vta_diffusion
#   ./scripts/train_day0suite.sh --model qformer -- ablation=vt   # Qformer sensor ablation
#   DRY_RUN=1 ./scripts/train_day0suite.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DAY0SUITE_DATASET:-}"
MODEL_FILTER=""
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_FILTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h | --help)
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$DATASET" && ( -f "$1" || "$1" == *.zarr* ) ]]; then
        DATASET="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

DATASET="${DATASET:-/data/day0suite.zarr.zip}"

declare -A CONFIG_FOR=(
  [see_hear_feel]=train_see_hear_feel
  [sparsh_x]=train_sparsh_x
  [polytouch]=train_polytouch
  [qformer]=train_qformer
  [vista]=train_vista
  [vta_diffusion]=train_vta_diffusion
)

run_train() {
  local exp_name="$1"
  local config_name="$2"
  shift 2

  local cmd=(
    python train_vista.py
    --config-name="${config_name}"
    "task.dataset_path=${DATASET}"
    "exp_name=${exp_name}"
    "$@"
  )
  if ((${#EXTRA_ARGS[@]})); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "==> ${cmd[*]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    "${cmd[@]}"
  fi
}

should_run() {
  local exp_name="$1"
  [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "$exp_name" ]]
}

train_suite() {
  for model in see_hear_feel sparsh_x polytouch qformer vista vta_diffusion; do
    if should_run "$model"; then
      run_train "$model" "${CONFIG_FOR[$model]}"
    fi
  done
}

if [[ -n "$MODEL_FILTER" ]]; then
  case "$MODEL_FILTER" in
    see_hear_feel | sparsh_x | polytouch | qformer | vista | vta_diffusion) ;;
    *)
      echo "Unknown model '${MODEL_FILTER}'. Choose from: see_hear_feel sparsh_x polytouch qformer vista vta_diffusion" >&2
      exit 1
      ;;
  esac
fi

echo "day0suite dataset: ${DATASET}"
train_suite
