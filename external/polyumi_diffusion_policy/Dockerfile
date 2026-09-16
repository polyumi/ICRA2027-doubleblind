# Training + inference image for the PolyUMI visuomotor diffusion policy.
#
# Reproduces UMI's conda environment (conda_environment.yaml) via micromamba, so no host conda
# is needed — which is the whole point (conda fights the ROS install on the laptop). One image
# serves both entrypoints: docker/train.sh (default) and docker/serve.sh (inference server),
# so the serving env is byte-identical to the training env (checkpoints are dill-pickled and
# must unpickle against the same dep tree).
#
# Build + run instructions, including the rootless-Docker flags, live in the PolyUMI repo at
# docs/training-instructions.md. Build on the GPU workstation (rootless, no registry push).

ARG MICROMAMBA_VERSION=1.5.8
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# libGL for headless OpenCV: the conda py-opencv imports libGL.so.1 at load, which a slim base
# lacks. The sim/teleop system libs (libspnav-dev, mesa/mujoco) are gone with the eval-only pip
# deps they served (see conda_environment.yaml). apt runs as build-time root even under rootless
# Docker, so no host sudo is involved.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Create the 'umi' env. conda_environment.yaml is UMI's spec minus the real-robot/sim/teleop pip
# deps (they don't build cleanly here and nothing on the training/inference path imports them —
# see the note in that file).
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_environment.yaml /tmp/conda_environment.yaml
RUN micromamba env create -y -f /tmp/conda_environment.yaml \
    && micromamba clean --all --yes

# Server deps layered on top rather than added to the yaml. fastapi/uvicorn are tiny and don't
# perturb the DP dep tree.
RUN micromamba run -n umi pip install --no-cache-dir \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32"

# Pin transitive deps that UMI's 2023-era libraries need but a present-day solve floats too new.
# conda_environment.yaml pins the top-level libs (wandb 0.15.8, diffusers 0.18.2, timm 0.9.7)
# but not these transitives, so conda-forge resolves them to current versions that broke the
# pinned libs:
#   - protobuf: newer versions leave wandb.proto.wandb_internal_pb2 without `Result`, so
#     `import wandb` (done unconditionally by the training workspace, even offline) crashes.
#   - huggingface_hub: >=0.26 removed `cached_download`, which diffusers 0.18.2 imports at load.
# Kept in a layer after env-create so that ~23-min conda solve stays cached. The
# google-api-core / proto-plus "requires protobuf>=4" pip warnings are from ray cloud features
# training does not use.
RUN micromamba run -n umi pip install --no-cache-dir \
        "protobuf==3.20.3" \
        "huggingface_hub==0.16.4"

# Project code. The dataset and output/checkpoint dirs are bind-mounted at run time (see
# .dockerignore and the run wrapper), not copied — native I/O speed, no image bloat.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app

# Activate 'umi' for subsequent RUNs and for the container's process, so `python` is the env's.
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=umi

# Default entrypoint trains; override the command (docker/serve.sh) to run the inference server.
# The base image's _entrypoint.sh activates ENV_NAME before exec'ing the command.
CMD ["bash", "docker/train.sh"]
