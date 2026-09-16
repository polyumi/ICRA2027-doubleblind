#!/usr/bin/env bash
# Thin training entrypoint for the PolyUMI diffusion-policy image (default container command).
#
# Runs UMI's train.py with our workspace config; any extra arguments pass straight through as
# Hydra overrides, e.g.:
#   docker run ... polyumi-dp bash docker/train.sh task.dataset_path=/data/d044.zarr.zip
#   docker run ... polyumi-dp bash docker/train.sh training.num_epochs=5 logging.mode=offline
#
# CONFIG_NAME selects the Hydra workspace config (defaults to ours). The base image's
# _entrypoint.sh has already activated the 'umi' env, so `python` is the env's interpreter.
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-train_diffusion_unet_timm_polyumi_workspace}"

exec python train.py --config-name="${CONFIG_NAME}" "$@"
