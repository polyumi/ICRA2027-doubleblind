# Sensor processing

Shared observation contract for SeeHearFeel, Sparsh-X, and Qformer.
PolyTouch keeps its own AST / Kaldi path and is out of scope here.

Architecture detail: [`SHF_SPARSH_VISTA.md`](SHF_SPARSH_VISTA.md). Ops: [`VISTA_AGENT_GUIDE.md`](VISTA_AGENT_GUIDE.md).

## Rates and horizons

| Stream | Stored grid | Policy sampling | Span |
|--------|-------------|-----------------|------|
| Wrist `camera0_rgb` | ~30 Hz (GoPro stride 2) | **H=2**, `ds=3` | ~**100 ms** between frames |
| Finger `finger_rgb` | nearest-frame on 30 Hz (~10 fps source) | **H=2**, `ds=3` | same instants as wrist |
| Proprio | same 30 Hz | **H=2**, `ds=3` | same |
| Action | same 30 Hz | H=16, `ds=3` (~10 Hz) | unchanged ROS control rate |
| Contact mic `mic_0` | one PCM row / step (536 @ 16 kHz ≈ 33.5 ms) | **10 contiguous rows**, `ds=1` | ~**0.33 s** |

One task knob `obs_down_sample_steps=3` covers wrist, finger, proprio, and action.
Audio is never downsampled: rows flatten to a gapless waveform before the mel.

**Finger camera caveat:** source rate is ~10 fps. With `ds=3` the two history frames are usually unique; rare duplicates can still appear when staleness approaches ~0.14 s.

## Shared log-mel (SHF / Sparsh / Qformer)

Defined in [`vista/preproc/log_mel.py`](../vista/preproc/log_mel.py) as `shared_log_mel`:

| Knob | Value |
|------|--------|
| Sample rate | 16 kHz (native piezo) |
| Window / hop | 25 ms / 10 ms (400 / 160 samples) |
| `n_fft` | 1024 (25 ms window zero-padded; enough bins for 128 mels @ 16 kHz) |
| Mels | **128**, `fmin=20`, `fmax=8000` |
| Amplitude | `log(clamp(S, 1e-8))` |
| Time grid | pad/crop to **48** frames (~0.33 s / 10 rows; native STFT ~34) |

SHF uses the same STFT with `time_frames=None` and AdaptiveAvgPool.
PolyTouch does **not** use this frontend.

## Per-model image history

Sampler order is oldest → newest (`I_{t−1}`, `I_t`).

| Model | History handling | Image tokens (224²) | Audio tokens |
|-------|------------------|---------------------|--------------|
| **SeeHearFeel** | ResNet **per frame**, flatten to `d·H` | 1 vector / camera | 1 vector (mel → ResNet) |
| **Sparsh-X** | Channel-**stack** 2 RGB → 6ch, then patch stem | patch **16** → **14×14 = 196** / camera | patch **8** → **16×6 = 96** |
| **Qformer** | CNN **per frame**, concat on sequence | 5× stride-2 → **7×7 = 49**/frame (**98** with H=2) | 3× stride-2 → **16×6 = 96** |

Sparsh-X keeps patch 16 on wrist/finger (paper-like Digit recipe with a 6-channel stack) and uses a finer patch 8 only on the mel. Qformer deliberately downsamples RGB more and audio less so wrist, finger, and audio sit near ~96–98 tokens before the Q-Former.

## Qformer embeddings

After each RGB stem, before the Q-Former:

- **Spatial:** learnable `(1, N, D)` table per camera (`N = (img_size / 32)²` → 49 at 224).
- **Temporal (shared):** learnable `(1, H, 1, D)` added to **both** wrist and finger at the matching timestep.

Audio and proprio have no spatial/temporal tables. Dropping a camera in `sensor_group` drops its spatial table; the shared temporal table still applies to whatever RGB remains.

Full `vta` Q-Former context ≈ 98 + 98 + 96 + 1 ≈ **293** tokens (DiT still sees 128 queries).
