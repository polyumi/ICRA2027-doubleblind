# Vista models — hardcoded baselines + ours

Quick reference for paper baselines and Qformer. Each is one `BaseVistaPolicy` subclass with encoders/fusion/head wired in code; YAML only sets hyperparameters.

**Architecture deep-dive (SHF, Sparsh-X, Qformer):** [`SHF_SPARSH_VISTA.md`](SHF_SPARSH_VISTA.md).  
**Sensor rates / shared mel / history:** [`SENSOR_PROCESSING.md`](SENSOR_PROCESSING.md).  
Ops / agent layout: [`VISTA_AGENT_GUIDE.md`](VISTA_AGENT_GUIDE.md).

---

## Expected input

Vision / tactile / proprio use `n_obs_history` (default 2) with `ds=3` (~100 ms).
Audio uses a contiguous `audio_obs_horizon` (default **10**, ~0.33 s) → shared log-mel inside SHF / Sparsh / Qformer.

| Key | Batch shape | Notes |
|-----|-------------|-------|
| `camera0_rgb` | `(B, H, 3, 224, 224)` | Wrist RGB; `H = n_obs_history` |
| `finger_rgb` | `(B, H, 3, 224, 224)` | Finger camera |
| `mic_0` | `(B, 10, 536)` | Contiguous PCM rows (~0.33 s @ 16 kHz); not downsampled |
| proprio keys | `(B, H, D)` | pos / rot6d / gripper / wrt_start |
| `action` | `(B, 16, 10)` | Relative rot6d |

---

## Models

| Class | File | Fusion | Head |
|-------|------|--------|------|
| `SeeHearFeelPolicy` | [`see_hear_feel.py`](../vista/models/see_hear_feel.py) | ResNet vectors + MHA + MLP cond | Diffusion U-Net |
| `SparshXPolicy` | [`sparsh_x.py`](../vista/models/sparsh_x.py) | MBT bottleneck (reimpl.) | DiT + flow matching |
| `PolyTouchPolicy` | [`polytouch.py`](../vista/models/polytouch.py) | 6×12 CLIP↔T3 cross-attn; 3 CLS concat | Diffusion U-Net |
| `QformerPolicy` | [`qformer.py`](../vista/models/qformer.py) | CNN stems → 6-layer Q-Former | DiT + flow matching |
| `VisTAPolicy` | [`vista.py`](../vista/models/vista.py) | CNN stems → TransformerEncoder | DiT + flow matching (AdaLN-zero) |

### QformerPolicy

- **Encoders:** `VisionCNNStem` / `TactileCNNStem` (5× stride-2 → **49 tokens/frame**, **98** with H=2) + shared log-mel + `AudioCNNStem` (3× stride-2 → **96** tokens); proprio MLP → 1 token.
- **Embeddings:** per-camera spatial `(1, 49, D)` at 224²; **shared** temporal `(1, H, 1, D)` on wrist and finger.
- **Q-Former:** 128 queries, 6 layers, each `SelfAttn → CrossAttn×3 → MLP(ratio=2)`. Full `vta` context ≈ **293**.
- **Head:** DiT (`dit_layers=4`, `d_embed=384`) + flow matching (~32M params at full `vta`).
- **Sensor ablation:** `sensor_group` ∈ `{v, vt, va, vta}` ([`vista/config/ablation/`](../vista/config/ablation/)). Dropped sensors are not built/encoded; Q-Former context shrinks. Proprio always kept. DiT / Q-Former width unchanged.

| Group | Sensors (plus proprio) |
|-------|------------------------|
| `v` | wrist RGB |
| `vt` | wrist + finger |
| `va` | wrist + mic |
| `vta` | wrist + finger + mic (default) |

Train:

```bash
# Baselines (full sensor set)
python train_vista.py --config-name=train_see_hear_feel task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_sparsh_x task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_polytouch task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_qformer task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_vista task.dataset_path=/path/to.zarr.zip

# Qformer / Mitas sensor ablations (compose Hydra ablation/*.yaml → policy.sensor_group)
python train_vista.py --config-name=train_qformer ablation=v task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_qformer ablation=vt task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_qformer ablation=va task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_qformer ablation=vta task.dataset_path=/path/to.zarr.zip
python train_vista.py --config-name=train_vista ablation=vt task.dataset_path=/path/to.zarr.zip
```

Or `./scripts/train_day0suite.sh --model vista` (full `vta`); pass Hydra overrides after `--`, e.g. `./scripts/train_day0suite.sh --model qformer -- ablation=vt`.

### VisTA Policy

- **Encoders:** Same CNN stems / embeddings / shared log-mel as Qformer; full context (~294 at `vta`, H=2 with 2 proprio tokens) kept (no query bottleneck). Proprio is one token per history step (no mean-pool).
- **Fusion:** `TransformerEncoder` (`fusion_mode=joint`, default) or per-sensor encoders then concat (`fusion_mode=per_sensor`, `fusion_layers=2` for param match at `vta`).
- **Head:** DiT (AdaLN-zero) + `objective_type=flow_matching` (default) or `diffusion` (DDPM ε); conditions on fused tokens (`max_cond_tokens=320`).
- **Sensor ablation:** Same `sensor_group` as Qformer; context length shrinks.

**Train VisTA ablations** (Hydra overlays under `vista/config/ablation/`):

```bash
# Default VisTA (joint fusion + flow matching, full vta)
python train_vista.py --config-name=train_vista task.dataset_path=/path/to.zarr.zip

# Diffusion objective (same DiT; DDPM epsilon, 16 infer steps)
python train_vista.py --config-name=train_vista ablation=objective_diffusion \
  task.dataset_path=/path/to.zarr.zip

# Per-sensor fusion (no cross-modal attn; fusion_layers=2 ≈ joint param count @ vta)
python train_vista.py --config-name=train_vista ablation=fusion_per_sensor \
  task.dataset_path=/path/to.zarr.zip

# Compose both via policy overrides (ablation group is single-select)
python train_vista.py --config-name=train_vista \
  policy.objective_type=diffusion policy.n_inference_steps=16 \
  policy.num_train_timesteps=100 policy.input_perturb=0.1 \
  policy.fusion_mode=per_sensor policy.fusion_layers=2 \
  task.dataset_path=/path/to.zarr.zip

# Sensor group + fusion ablation (set sensor_group on CLI; fusion overlay)
python train_vista.py --config-name=train_vista ablation=fusion_per_sensor \
  policy.sensor_group=vt task.dataset_path=/path/to.zarr.zip
```

Or `./scripts/train_day0suite.sh --model mitas -- ablation=fusion_per_sensor`.

Touch-in-the-Wild is **not** implemented.
