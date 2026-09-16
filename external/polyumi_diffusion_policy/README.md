# PolyUMI Diffusion Policy

Visuomotor diffusion policy forked from [UMI](https://umi-gripper.github.io/)'s implementation.

### PolyUMI Fork Notes:
- Deleted code & deps not needed for training or inference; pinned deps that no longer built (protobuf, huggingface_hub)
- Added `serve_policy.py`, a `PolicyBackend` for PolyUMI's `polyumi_inference` protocol (obs translation, checkpoint load, `/reset`), plus a Dockerfile/entrypoints (`docker/train.sh`, `docker/serve.sh`) so training and serving run the same image
- Retuned the PolyUMI export config (`obs_down_sample_steps` for the 30 Hz export, val_loss re-enabled) and added per-request timing to the serve path
