#!/usr/bin/env bash
# Inference-server entrypoint for the PolyUMI diffusion-policy image.
#
# Serves the trained policy over the same HTTP contract as the ROS-side dummy_server
# (POST /predict_cartesian/). Run by overriding the container command:
#   docker run ... -p 8000:8000 -e CKPT_PATH=/data/checkpoints/latest.ckpt polyumi-dp bash docker/serve.sh
#
# The 'umi' env is already active (base image _entrypoint.sh), so `uvicorn` is the env's.
set -euo pipefail

HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8000}"

exec uvicorn serve_policy:app --host "${HOST}" --port "${PORT}"
