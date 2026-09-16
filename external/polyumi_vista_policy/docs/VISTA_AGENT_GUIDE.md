# Vista agent guide

Architecture detail for SHF / Sparsh-X / Qformer: [`SHF_SPARSH_VISTA.md`](SHF_SPARSH_VISTA.md). Quick model table: [`VISTA_MODELS.md`](VISTA_MODELS.md). Sensor processing (rates, shared mel, history): [`SENSOR_PROCESSING.md`](SENSOR_PROCESSING.md).

## Architecture (hardcoded policies)

Each model is **one `nn.Module` class** subclassing [`BaseVistaPolicy`](../vista/policy/base.py). Encoders, fusion, and policy head are constructed in `__init__`. Hydra YAML only sets hyperparameters and `_target_`.

| Policy | Config |
|--------|--------|
| `SeeHearFeelPolicy` | `vista/config/train_see_hear_feel.yaml` |
| `SparshXPolicy` | `vista/config/train_sparsh_x.yaml` |
| `PolyTouchPolicy` | `vista/config/train_polytouch.yaml` |
| `QformerPolicy` | `vista/config/train_qformer.yaml` (+ optional `ablation={v,vt,va,vta}`) |
| `VisTAPolicy` | `vista/config/train_vista.yaml` (+ optional `ablation={v,vt,va,vta}`) |

```
obs (B, N, …)
  → shared-weight encoders (per timestep / history)
  → paper / Qformer / Mitas fusion
  → paper / Qformer / Mitas head
  → action chunk (B, H, 10)
```

Observation history: `task.n_obs_history=2` with `ds=3` (~100 ms) for wrist, finger, and proprio; `task.audio_obs_horizon=10` contiguous mic rows (no downsample) → shared log-mel for SHF / Sparsh / Qformer / VisTA. See [`SENSOR_PROCESSING.md`](SENSOR_PROCESSING.md).

Qformer / VisTA sensor ablation: compose `ablation={v,vt,va,vta}` (sets `policy.sensor_group`). Dropped sensors are not encoded; context shrinks only. Proprio always kept.

## Entry points

- `python train_vista.py --config-name=train_see_hear_feel`
- `python train_vista.py --config-name=train_qformer`                         # full vta
- `python train_vista.py --config-name=train_vista`                           # CNN → TransformerEncoder → DiT
- `python train_vista.py --config-name=train_qformer ablation=v`              # wrist only
- `python train_vista.py --config-name=train_qformer ablation=vt`             # wrist + finger
- `python train_vista.py --config-name=train_qformer ablation=va`             # wrist + mic
- `python train_vista.py --config-name=train_qformer ablation=vta`            # wrist + finger + mic
- `./scripts/train_day0suite.sh --model vista`
- `./scripts/train_day0suite.sh --model qformer -- ablation=vt`
- Tests: `python -m pytest test/test_vista_shapes.py test/test_sensor_ablation.py test/test_vista_dataset.py -q`

## Layout

```
vista/
  policy/base.py          # BaseVistaPolicy
  models/see_hear_feel.py # SeeHearFeelPolicy
  models/sparsh_x.py      # SparshXPolicy
  models/polytouch.py     # PolyTouchPolicy
  models/qformer.py       # QformerPolicy (CNN → Q-Former → DiT)
  models/vista.py         # VisTAPolicy (CNN → TransformerEncoder → DiT)
  encoders/cnn.py         # Vision / tactile / audio CNN stems
  encoders/shf_resnet.py  # SHF CoordConv ResNet (vendored MIT)
  fusion/mbt.py           # MBT reimplementation (not Meta code)
  fusion/qformer.py       # Q-Former (stacked self + multi cross + MLP)
  fusion/transformer.py   # TransformerEncoder fusion (Mitas)
  fusion/polytouch_combiner.py
  heads/dit.py            # TransformerDenoiser (DiT)
  data/vista_dataset.py
```

## Licensing notes

- SeeHearFeel encoder/MHA: copied from `JunzheJosephZhu/see_hear_feel` (MIT); source cited in file headers.
- Sparsh-X MBT: **reimplemented** from Nagrani 2021; do not copy `facebookresearch/sparsh-multisensory-touch` (CC-BY-NC). Head is DiT + flow matching.
- PolyTouch combiner: from paper text `arxiv:2504.19341` §IV-A (no public policy repo).

## Not in this pass

- Touch-in-the-Wild (deleted)
