# SeeHearFeel, Sparsh-X, and Qformer

Architecture notes for the three multimodal imitation policies that share the
PolyUMI observation contract. Each model is a hardcoded `BaseVistaPolicy`
subclass: encoders, fusion, and head live in `__init__`; Hydra YAML only sets
hyperparameters.

For agent/ops layout see [`VISTA_AGENT_GUIDE.md`](VISTA_AGENT_GUIDE.md).
Sensor rates / horizons / shared mel: [`SENSOR_PROCESSING.md`](SENSOR_PROCESSING.md).
PolyTouch is a fourth baseline (CLIP/T3 + diffusion U-Net); it is out of scope here.

---

## Shared observation contract

Vision / tactile / proprio use `n_obs_history` (default **2**) with `obs_down_sample_steps=3`
(~100 ms between frames on the ~30 Hz grid).
Audio uses a contiguous `audio_obs_horizon` (default **10**) of `mic_0` PCM rows
(536 samples @ 16 kHz ≈ 33.5 ms each → ~0.33 s), **not** downsampled.
All three models share one log-mel frontend (`shared_log_mel`: 128 mels, 25/10 ms).

| Key | Shape | Role |
|-----|-------|------|
| `camera0_rgb` | `(B, H, 3, 224, 224)` | Wrist / GoPro |
| `finger_rgb` | `(B, H, 3, 224, 224)` | Finger camera |
| `mic_0` | `(B, 10, 536)` | Contact mic chunks |
| proprio keys | `(B, H, D)` | pos / rot6d / gripper / wrt_start |
| `action` | `(B, 16, 10)` | Relative rot6d chunk |

Train entrypoint: `python train_vista.py --config-name=<config>`.

| Model | Class | Config |
|-------|-------|--------|
| SeeHearFeel (SHF) | `SeeHearFeelPolicy` | `train_see_hear_feel` |
| Sparsh-X | `SparshXPolicy` | `train_sparsh_x` |
| Qformer | `QformerPolicy` | `train_qformer` (+ optional `ablation=…`) |

```bash
python -m vista.scripts.param_report --model see_hear_feel|sparsh_x|qformer
```

---

## SeeHearFeel (SHF)

**Paper lineage:** JunzheJosephZhu/see_hear_feel (MIT). CoordConv ResNet encoders and
actor MHA adapted in-tree; discrete 3^k classifier replaced by an MLP that emits a
global condition for ConditionalUnet1D (DDPM action chunks).

**Code:** [`vista/models/see_hear_feel.py`](../vista/models/see_hear_feel.py),
encoders in [`vista/encoders/shf_resnet.py`](../vista/encoders/shf_resnet.py).

### Pipeline

```
camera / finger  → CoordConv ResNet (per frame) → vector (d·H)
mic_0            → shared log-mel (128 bands, 25/10 ms) → ResNet → vector (d·H)
proprio          → MLP over history → vector (d)
                 → stack 4 modality vectors → LayerNorm → MHA (residual)
                 → concat + bottleneck Linear → MLP → cond vector
                 → ConditionalUnet1D (DDPM) → action chunk (H_a × 10)
```

### Defaults (`train_see_hear_feel.yaml`)

| Knob | Default |
|------|---------|
| `d_embed` | 256 |
| `n_heads` | 8 |
| `mlp_hidden` | 2048 |
| `cond_dim` | 512 |
| `down_dims` | [256, 512, 1024] |
| Objective | DDPM epsilon (U-Net head) |

### Audio

[`shared_log_mel`](../vista/preproc/log_mel.py) with `time_frames=None` (AdaptiveAvgPool). Same 128-mel 25/10 ms STFT as Sparsh / Qformer.

### Notes

- Fusion is **late**: one vector per modality, then MHA over four tokens; MLP emits the U-Net global condition.
- Action chunking lives in the diffusion U-Net (not a flat MSE-MLP).
- Full sensor set only (no `sensor_group` ablation).

---

## Sparsh-X

**Paper lineage:** Architecture inspired by Sparsh-X (Higuera et al., CoRL 2025).
MBT fusion is a **reimplementation** of Nagrani et al. 2021 bottleneck transformers —
do **not** copy `facebookresearch/sparsh-multisensory-touch` (CC-BY-NC).

**Code:** [`vista/models/sparsh_x.py`](../vista/models/sparsh_x.py),
[`vista/fusion/mbt.py`](../vista/fusion/mbt.py),
[`vista/heads/dit.py`](../vista/heads/dit.py).

### Pipeline

```
camera / finger  → stack H=2 on channel (6ch) → patch-16 stem → 196 tokens each
mic_0            → shared log-mel (1×128×48) → patch-8 stem → 96 tokens
proprio          → MLP over (B, H, D_prop) → tokens
                 → MBT (per-modal self-attn, then shared bottleneck fusion)
                 → concat modal streams → DiT decoder (AdaLN-Zero)
                 → flow matching → action (16 × 10)
```

### Defaults (`train_sparsh_x.yaml`)

| Knob | Default |
|------|---------|
| `d_embed` | 256 |
| MBT `depth` / `fusion_layer` | 8 / 4 |
| `num_bottlenecks` | 4 |
| `dit_layers` | 4 |
| `n_inference_steps` | 10 |
| Params (full) | ~30.5M |

### Audio

[`shared_log_mel`](../vista/preproc/log_mel.py): 128 mels, 25/10 ms, pad/crop to **48** → `(B, 1, 128, 48)`. Patch size **8** (audio only) → 96 tokens. Wrist/finger stay patch **16**.

### MBT fusion

1. First `fusion_layer` blocks: **uni-modal** self-attention per stream.
2. Remaining depth: each modality attends with shared **bottleneck** tokens; bottlenecks are averaged across modalities after each block.

### Notes

- Token-level fusion (not SHF’s four vectors). Image history is channel-stacked (Sparsh Digit recipe), not unrolled as `H` separate patch grids.
- Same DiT + conditional flow matching stack Qformer reuses.
- Full sensor set only. MBT context ≈ 196+196+96+H_proprio ≈ **500** tokens.

---

## Qformer

**Ours.** Lightweight CNN stems + Q-Former bottleneck queries + DiT / flow matching,
sized near Sparsh-X (~30–35M).

**Code:** [`vista/models/qformer.py`](../vista/models/qformer.py),
[`vista/fusion/qformer.py`](../vista/fusion/qformer.py),
[`vista/encoders/cnn.py`](../vista/encoders/cnn.py).

### Pipeline

```
camera           → VisionCNNStem (5× stride-2) → 49 × H tokens (+ spatial + shared temporal)
finger           → TactileCNNStem               → 49 × H tokens (+ spatial + shared temporal)
mic_0            → shared log-mel → AudioCNNStem (3× stride-2) → 96 tokens
proprio          → mean over H → MLP → 1 token
                 → concat active streams = Q-Former context (~293 at full vta, H=2)
learnable Q (128)→ 6× [SelfAttn → CrossAttn×3 → MLP(ratio=2)]
                 → DiT (flow matching) → action (16 × 10)
```

### Defaults (`train_qformer.yaml`)

| Knob | Default |
|------|---------|
| `d_embed` | 384 |
| `n_queries` | 128 |
| `qformer_layers` | 6 |
| `n_cross_attn` | 3 |
| `mlp_ratio` (Q-Former FFN) | 2 |
| `dit_layers` | 4 |
| `n_heads` | 8 |
| Params (full `vta`) | ~32M |

### Embeddings

- **Spatial:** learnable `(1, N, D)` per camera (`N = (img/32)²` → 49 at 224).
- **Temporal (shared):** learnable `(1, H, 1, D)` added to both wrist and finger at matching timesteps.
- Audio / proprio: no spatial or temporal tables.

### Q-Former layer

Each of 6 layers (queries only; context is frozen K/V):

1. **Self-attention** among learnable queries  
2. **Cross-attention ×3** from queries to the concatenated sensor context (more context influence than a single cross block)  
3. **MLP** with hidden size `2 · d_embed`  

Output is always `(B, n_queries, D)` regardless of how many sensors are active.

### Sensor ablation

`sensor_group ∈ {v, vt, va, vta}` ([`SENSOR_GROUPS`](../vista/models/sensor_ablation.py)).

| Group | Active exogenous sensors |
|-------|--------------------------|
| `v` | `camera0_rgb` |
| `vt` | camera + finger |
| `va` | camera + mic |
| `vta` | camera + finger + mic |

- Dropped sensors are **not constructed** and **not encoded**.
- Q-Former **context length shrinks**; queries / DiT / flow matching unchanged.
- Proprio is **always** kept.
- No mask tokens or zero-fill.

```bash
# Full sensors (default sensor_group=vta)
python train_vista.py --config-name=train_qformer

# Ablations via vista/config/ablation/{v,vt,va,vta}.yaml
python train_vista.py --config-name=train_qformer ablation=v
python train_vista.py --config-name=train_qformer ablation=vt
python train_vista.py --config-name=train_qformer ablation=va
python train_vista.py --config-name=train_qformer ablation=vta
```

Approx. context lengths (+ 1 proprio), `H=2`: `v` ≈ 99, `vt` ≈ 197, `va` ≈ 195, `vta` ≈ **293**.
See also [`SENSOR_PROCESSING.md`](SENSOR_PROCESSING.md).

---

## Side-by-side

| | SHF | Sparsh-X | Qformer |
|--|-----|----------|-------|
| Vision/tactile | ResNet per frame → vector | 6ch stack → patch-16 (196) | CNN /32 → 49×H + pos |
| Audio | shared mel + ResNet → vector | shared mel + patch-8 (96) | shared mel + CNN /8 (96) |
| Fusion | 4-token MHA + bottleneck | MBT (uni then bottleneck) | Q-Former (queries ← context) |
| Policy head | MHA+MLP cond → U-Net DDPM | DiT + flow matching | DiT + flow matching |
| Width | `d=256` | `d=256` | `d=384` |
| Ablation | — | — | `sensor_group` |
| ~Params | ~35.9M | ~30.5M | ~32M |

```mermaid
flowchart TB
  subgraph shf [SeeHearFeel]
    sEnc[ResNet vectors] --> sMHA[MHA over 4 modals]
    sMHA --> sMLP[MLP cond]
    sMLP --> sUNet[U-Net DDPM]
  end
  subgraph sparsh [Sparsh-X]
    pTok[Patch tokens] --> mbt[MBT fusion]
    mbt --> pDiT[DiT FM]
  end
  subgraph qformer [Qformer]
    cTok[CNN tokens] --> qf[Q-Former queries]
    qf --> vDiT[DiT FM]
  end
```

---

## Synthetic inference smoke test

Untrained weights, identity normalizer, CPU, batch size 1, `224×224` images,
`n_obs_history=2`, `audio_obs_horizon=10`. One `compute_loss` step + timed
`predict_action` (1 warmup + 3 timed runs). Finite actions with shape `(1, 16, 10)` required.

| Model | Params | Head | Action shape | Train loss (1 step) | Infer latency (CPU, mean±std) | Infer (Hz) | Status |
|-------|--------|------|--------------|---------------------|-------------------------------|------------|--------|
| SeeHearFeel | 35,881,926 | MLP BC | `(1, 16, 10)` | 1.0590 | 232 ± 6 ms | 4.3 | ok |
| Sparsh-X | 30,488,432 | DiT FM (10 steps) | `(1, 16, 10)` | 2.2348 | 174 ± 1 ms | 5.7 | ok |
| Qformer (`vta`) | 32,021,008 | DiT FM (10 steps) | `(1, 16, 10)` | 2.9768 | 218 ± 1 ms | 4.6 | ok |
| Qformer (`vt`) | 31,060,432 | DiT FM (10 steps) | `(1, 16, 10)` | 2.2602 | 226 ± 5 ms | 4.4 | ok |

Loss values are not meaningful for quality (random init); they only confirm the train path runs. Latency / Hz are host-dependent (`Hz ≈ 1000 / mean_ms`).

---

## Licensing

- **SHF:** encoders/MHA from `JunzheJosephZhu/see_hear_feel` (MIT); headers cite sources.
- **Sparsh-X MBT:** reimplemented from Nagrani 2021; Meta Sparsh repo is CC-BY-NC — do not copy it.
- **Qformer:** original to this fork.
