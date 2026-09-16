#!/usr/bin/env bash
# Container entrypoint for the day0suite Vista training suite.
#
# Bind-mount your export and point DAY0SUITE_DATASET at it, e.g.:
#   docker run ... -v /datasets/day0suite.zarr.zip:/data/day0suite.zarr.zip \
#     polyumi-vista bash docker/train_day0suite.sh
#
# Train one model:
#   docker run ... polyumi-vista bash docker/train_day0suite.sh --model qformer -- ablation=vt
set -euo pipefail

exec bash scripts/train_day0suite.sh "$@"
