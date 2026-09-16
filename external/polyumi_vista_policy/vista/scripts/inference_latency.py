"""Dummy inference-latency benchmark for Vista / PolyUMI policies.

Builds each model with train-config defaults, feeds synthetic batch-1
observations, and reports mean ``predict_action`` wall time.

Examples:
  python -m vista.scripts.inference_latency
  python -m vista.scripts.inference_latency --iters 100 --warmup 10
  python -m vista.scripts.inference_latency --model qformer --device cuda
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from vista.scripts.param_report import (
    MODEL_CHOICES,
    _diffusion_shape_meta,
    _vista_shape_meta,
    build_model,
)


def _identity_normalizer(shape_meta: dict) -> LinearNormalizer:
    n = LinearNormalizer()
    for key in shape_meta["obs"]:
        n[key] = get_image_identity_normalizer()
    stat = array_to_stats(
        np.zeros((1, shape_meta["action"]["shape"][0]), dtype=np.float32)
    )
    n["action"] = get_identity_normalizer_from_stat(stat)
    for key in n.params_dict.keys():
        params = n.params_dict[key]
        for field in ("scale", "offset", "input_stats"):
            if field not in params:
                continue
            if isinstance(params[field], dict):
                for sk in params[field]:
                    if torch.is_tensor(params[field][sk]):
                        params[field][sk] = params[field][sk].float()
            elif torch.is_tensor(params[field]):
                params[field] = params[field].float()
    return n


def _dummy_obs(shape_meta: dict, batch: int = 1, device: torch.device = torch.device("cpu")):
    obs: Dict[str, torch.Tensor] = {}
    for key, meta in shape_meta["obs"].items():
        shape = list(meta["shape"])
        horizon = int(meta.get("horizon", 1))
        typ = meta.get("type", "low_dim")
        if typ == "rgb":
            # CHW in shape_meta; batch layout (B, T, C, H, W)
            c, h, w = shape
            obs[key] = torch.rand(batch, horizon, c, h, w, device=device)
        else:
            d = int(shape[0])
            obs[key] = torch.randn(batch, horizon, d, device=device)
    return obs


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_predict(
    model: torch.nn.Module,
    obs: Dict[str, torch.Tensor],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    # Warmup (compile caches / allocator / first-step overhead).
    for _ in range(max(0, warmup)):
        _ = model.predict_action(obs)
    _sync(device)

    times_ms: List[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        out = model.predict_action(obs)
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        # Touch output so lazy work cannot be optimized away.
        _ = out["action"].reshape(-1)[0].item()

    mean_ms = statistics.fmean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "hz": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


def _shape_for(name: str) -> dict:
    if name == "diffusion_unet":
        return _diffusion_shape_meta()
    return _vista_shape_meta()


def run_one(
    name: str,
    *,
    device: torch.device,
    iters: int,
    warmup: int,
    batch: int,
    sensor_group: str,
) -> Dict[str, object]:
    shape = _shape_for(name)
    model = build_model(name, sensor_group=sensor_group, lite=False)
    model.to(device)
    model.set_normalizer(_identity_normalizer(shape))
    # Normalizer params stay on CPU by default; move buffers used in normalize.
    if hasattr(model, "normalizer"):
        model.normalizer.to(device)
    obs = _dummy_obs(shape, batch=batch, device=device)
    stats = benchmark_predict(
        model, obs, iters=iters, warmup=warmup, device=device
    )
    return {"model": name, **stats}


def _format_table(rows: Sequence[Dict[str, object]]) -> str:
    headers = ("model", "mean_ms", "std_ms", "min_ms", "max_ms", "Hz")
    body = []
    for r in rows:
        body.append(
            (
                str(r["model"]),
                f"{r['mean_ms']:.2f}",
                f"{r['std_ms']:.2f}",
                f"{r['min_ms']:.2f}",
                f"{r['max_ms']:.2f}",
                f"{r['hz']:.2f}",
            )
        )
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        sep,
    ]
    for row in body:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Dummy predict_action latency bench.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Single model (default: all)",
    )
    parser.add_argument("--iters", type=int, default=100, help="Timed forward passes")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup passes")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sensor-group",
        default="vta",
        choices=["v", "vt", "va", "vta"],
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    names = [args.model] if args.model else list(MODEL_CHOICES)
    print(
        f"device={device}  batch={args.batch}  warmup={args.warmup}  "
        f"iters={args.iters}  models={len(names)}"
    )

    rows = []
    for name in names:
        print(f"\n[{name}] building + timing ...", flush=True)
        try:
            row = run_one(
                name,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                batch=args.batch,
                sensor_group=args.sensor_group,
            )
            rows.append(row)
            print(
                f"[{name}] mean={row['mean_ms']:.2f} ms  "
                f"({row['hz']:.2f} Hz)",
                flush=True,
            )
        except Exception as exc:  # keep going across models
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append(
                {
                    "model": f"{name} (FAILED)",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                    "hz": float("nan"),
                }
            )

    print("\n" + _format_table(rows))


if __name__ == "__main__":
    main()
