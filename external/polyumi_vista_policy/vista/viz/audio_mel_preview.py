"""
Standalone Sparsh-style audio preprocessing visualization.

Stacks contiguous PolyUMI ``mic_0`` rows (536 samples each @ 16 kHz by default)
into a waveform, computes a log-mel spectrogram, and saves a figure.

Modes:
  - window (default): ~0.55 s Sparsh crop → ``(128, 224)`` mel, waveform above mel
  - ``--episode``: full ``mic_0`` episode, native-length mel beside the waveform

Does not wire into training — for inspecting the mel transform only.

Usage (from repo root, with PYTHONPATH=.):

  python -m vista.viz.audio_mel_preview
  python -m vista.viz.audio_mel_preview --out /tmp/mel.png
  python -m vista.viz.audio_mel_preview --zarr /path/to/export.zarr.zip --index 100
  python -m vista.viz.audio_mel_preview --episode --zarr /path/to/vista_export
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchaudio is required for audio_mel_preview") from exc


# PolyUMI defaults
DEFAULT_SR = 16_000
SAMPLES_PER_ROW = 536  # ~33.5 ms @ 16 kHz

# Sparsh-X paper (single-mic adaptation): 0.55 s window, 128 mels,
# 5 ms Hamming window, 2.5 ms hop. Dual-mic concat → 256 mels is omitted.
SPARSH_WINDOW_S = 0.55
SPARSH_N_MELS = 128
SPARSH_WIN_S = 0.005
SPARSH_HOP_S = 0.0025


def rows_for_duration(
    duration_s: float,
    sample_rate: int = DEFAULT_SR,
    row_len: int = SAMPLES_PER_ROW,
) -> int:
    """How many contiguous mic rows cover ``duration_s``."""
    n_samples = int(round(duration_s * sample_rate))
    return max(1, int(np.ceil(n_samples / row_len)))


def stack_mic_rows(rows: np.ndarray) -> np.ndarray:
    """
    Flatten contiguous mic rows to a 1-D waveform.

    Args:
        rows: (T, F) or (F,) float — PolyUMI ``mic_0`` chunks
    Returns:
        waveform: (T * F,) float32
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        return rows
    if rows.ndim != 2:
        raise ValueError(f"Expected mic rows (T, F), got shape {rows.shape}")
    return rows.reshape(-1)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SR,
    n_mels: int = SPARSH_N_MELS,
    win_s: float = SPARSH_WIN_S,
    hop_s: float = SPARSH_HOP_S,
    target_frames: Optional[int] = 224,
    crop_to_window: bool = True,
    window_s: float = SPARSH_WINDOW_S,
) -> torch.Tensor:
    """
    Sparsh-like log-mel of a mono waveform.

    Args:
        crop_to_window: If True, pad/crop to ``window_s`` (Sparsh 0.55 s).
            Set False for a full-episode spectrogram.
        target_frames: If set, bilinear-resize time axis (Sparsh ViT 224).
            Pass None to keep native STFT length.

    Returns:
        mel: (n_mels, time) float tensor
    """
    wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).float()
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    if crop_to_window:
        n_target = int(round(window_s * sample_rate))
        if wav.numel() < n_target:
            wav = torch.nn.functional.pad(wav, (0, n_target - wav.numel()))
        else:
            wav = wav[-n_target:]

    win_length = max(1, int(round(win_s * sample_rate)))
    hop_length = max(1, int(round(hop_s * sample_rate)))
    # At 16 kHz, a 5 ms window is only 80 samples — too few FFT bins for 128
    # mel filters. Keep Sparsh win/hop in seconds, but zero-pad the FFT so
    # n_freqs = n_fft//2+1 can support n_mels (torchaudio warns otherwise).
    n_fft = max(1024, 2 ** int(np.ceil(np.log2(max(win_length, 4 * n_mels)))))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        window_fn=torch.hamming_window,
        center=True,
        power=2.0,
        f_min=0.0,
        f_max=float(sample_rate // 2),
    )
    mel = mel_transform(wav.unsqueeze(0)).squeeze(0)  # (n_mels, T)
    mel = torch.log(mel.clamp_min(1e-8))

    if target_frames is not None and mel.shape[-1] != target_frames:
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0).unsqueeze(0),
            size=(n_mels, target_frames),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return mel


def synthetic_mic_rows(
    n_rows: int,
    row_len: int = SAMPLES_PER_ROW,
    sample_rate: int = DEFAULT_SR,
    seed: int = 0,
) -> np.ndarray:
    """Tone + noise demo waveform chunked into PolyUMI-sized rows."""
    rng = np.random.default_rng(seed)
    n = n_rows * row_len
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Chirp-ish contact: 200–2000 Hz sweep + bursts
    phase = 2 * np.pi * (200 * t + 900 * t * t)
    signal = 0.4 * np.sin(phase)
    bursts = (rng.random(n) < 0.002).astype(np.float32)
    signal = signal + bursts * rng.normal(0, 1.0, size=n).astype(np.float32)
    signal = signal + 0.05 * rng.normal(0, 1.0, size=n).astype(np.float32)
    signal = signal.astype(np.float32)
    peak = np.max(np.abs(signal)) + 1e-8
    signal /= peak
    return signal.reshape(n_rows, row_len)


def _open_zarr_root(zarr_path: str):
    """Open a PolyUMI export (directory or .zarr.zip) read-only.

    Returns ``(root, store_or_None)``. Caller must ``store.close()`` if not None.
    """
    import zarr

    path = Path(zarr_path)
    if path.is_dir():
        return zarr.open(str(path), mode="r"), None
    if path.is_file() and str(path).endswith(".zarr.zip"):
        store = zarr.ZipStore(str(path), mode="r")
        root = zarr.open_group(store, mode="r")
        return root, store
    raise FileNotFoundError(zarr_path)


def _mic_array_from_root(root, key: str = "mic_0") -> np.ndarray:
    """Resolve ``mic_0`` from ReplayBuffer-style or ``data/mic_0`` layouts."""
    if "data" in root and key in root["data"]:
        return np.asarray(root["data"][key][:], dtype=np.float32)
    if key in root:
        return np.asarray(root[key][:], dtype=np.float32)
    try:
        from diffusion_policy.common.replay_buffer import ReplayBuffer

        buf = ReplayBuffer.create_from_group(root)
        return np.asarray(buf[key][:], dtype=np.float32)
    except Exception as exc:
        raise KeyError(f"Could not find {key!r} in zarr root") from exc


def load_all_mic_rows_from_zarr(zarr_path: str, key: str = "mic_0") -> np.ndarray:
    """Load every ``mic_0`` row in the export (one episode / concatenated buffer)."""
    root, store = _open_zarr_root(zarr_path)
    try:
        arr = _mic_array_from_root(root, key=key)
    finally:
        if store is not None:
            store.close()
    if arr.ndim == 1:
        arr = arr.reshape(-1, SAMPLES_PER_ROW)
    return arr


def load_mic_rows_from_zarr(
    zarr_path: str,
    index: int,
    n_rows: int,
    key: str = "mic_0",
) -> np.ndarray:
    """Load ``n_rows`` contiguous ``mic_0`` rows ending at ``index`` (inclusive)."""
    arr = load_all_mic_rows_from_zarr(zarr_path, key=key)
    end = min(index + 1, arr.shape[0])
    start = max(0, end - n_rows)
    rows = arr[start:end]
    if rows.shape[0] < n_rows:
        pad = np.zeros((n_rows - rows.shape[0], rows.shape[1]), dtype=np.float32)
        rows = np.concatenate([pad, rows], axis=0)
    return rows


def _magma_rgb(x: np.ndarray) -> np.ndarray:
    """Map [0, 1] → RGB uint8 with a simple magma-like LUT (no matplotlib)."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.7 * x - 0.2, 0, 1)
    g = np.clip(1.5 * x - 0.5, 0, 1)
    b = np.clip(2.0 * x, 0, 1) * (1.0 - 0.6 * x)
    # darken low end a bit
    scale = 0.15 + 0.85 * x
    rgb = np.stack([r * scale, g * scale, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _render_waveform_strip(
    waveform: np.ndarray,
    width: int,
    height: int = 120,
) -> np.ndarray:
    """Rasterize a mono waveform into an RGB strip."""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    mid = height // 2
    img[mid, :, :] = 200
    if waveform.size == 0:
        return img
    # Downsample / upsample waveform to ``width`` columns
    idx = (np.linspace(0, waveform.size - 1, width)).astype(np.int64)
    y = waveform[idx]
    peak = float(np.max(np.abs(y))) + 1e-8
    y = y / peak
    ys = (mid - (y * (height * 0.42))).astype(np.int32)
    ys = np.clip(ys, 0, height - 1)
    for x, yy in enumerate(ys):
        lo, hi = (mid, yy) if yy >= mid else (yy, mid)
        img[lo : hi + 1, x, :] = (70, 130, 180)
    return img


def visualize_mel(
    mel: torch.Tensor,
    waveform: np.ndarray,
    sample_rate: int,
    out_path: Path,
    title: str,
) -> Path:
    """
    Save waveform above log-mel as a PNG (window / Sparsh preview).

    Uses Pillow only (no matplotlib dependency) so this stays runnable in the
    training conda env.
    """
    from PIL import Image, ImageDraw

    mel_np = mel.detach().cpu().numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize mel for display
    m = mel_np - mel_np.min()
    m = m / (m.max() + 1e-8)
    mel_rgb = _magma_rgb(m[::-1])  # low freq at bottom
    # Upscale for readability
    mel_h, mel_w = mel_rgb.shape[:2]
    scale = max(2, 400 // max(mel_h, 1))
    mel_img = Image.fromarray(mel_rgb, mode="RGB").resize(
        (mel_w * scale, mel_h * scale), resample=Image.NEAREST
    )

    wave_w = mel_img.width
    wave = _render_waveform_strip(waveform, width=wave_w, height=120)
    wave_img = Image.fromarray(wave, mode="RGB")

    pad = 8
    header = 36
    canvas = Image.new(
        "RGB",
        (wave_w + 2 * pad, header + wave_img.height + mel_img.height + 3 * pad),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    label = (
        f"{title} | wav={waveform.shape[0]}@{sample_rate}Hz | "
        f"log-mel={tuple(mel_np.shape)}"
    )
    draw.text((pad, 10), label[:120], fill=(30, 30, 30))
    y = header
    canvas.paste(wave_img, (pad, y))
    y += wave_img.height + pad
    canvas.paste(mel_img, (pad, y))
    canvas.save(out_path)
    return out_path


def visualize_mel_beside_waveform(
    mel: torch.Tensor,
    waveform: np.ndarray,
    sample_rate: int,
    out_path: Path,
    title: str,
    panel_height: int = 256,
    max_width: int = 1600,
) -> Path:
    """
    Full-episode view: waveform (left) next to log-mel (right), shared time on X.
    """
    from PIL import Image, ImageDraw

    mel_np = mel.detach().cpu().numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = mel_np - mel_np.min()
    m = m / (m.max() + 1e-8)
    mel_rgb = _magma_rgb(m[::-1])
    mel_h, mel_w = mel_rgb.shape[:2]

    # Fit time axis into max_width while keeping mel bins readable.
    time_w = min(max_width // 2, max(mel_w, 64))
    mel_img = Image.fromarray(mel_rgb, mode="RGB").resize(
        (time_w, panel_height), resample=Image.BILINEAR
    )
    wave = _render_waveform_strip(waveform, width=time_w, height=panel_height)
    wave_img = Image.fromarray(wave, mode="RGB")

    pad = 8
    gap = 10
    header = 52
    footer = 22
    canvas_w = time_w * 2 + gap + 2 * pad
    canvas_h = header + panel_height + footer + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    dur_s = waveform.shape[0] / float(sample_rate)
    label = (
        f"{title} | {dur_s:.2f}s | wav={waveform.shape[0]}@{sample_rate}Hz | "
        f"log-mel={tuple(mel_np.shape)} (native)"
    )
    draw.text((pad, 8), label[:140], fill=(30, 30, 30))
    draw.text((pad, 30), "waveform", fill=(90, 90, 90))
    draw.text((pad + time_w + gap, 30), "log-mel", fill=(90, 90, 90))

    y = header
    canvas.paste(wave_img, (pad, y))
    canvas.paste(mel_img, (pad + time_w + gap, y))
    draw.text((pad, y + panel_height + 4), "time ->", fill=(90, 90, 90))
    canvas.save(out_path)
    return out_path


def run_preview(
    zarr_path: Optional[str] = None,
    index: int = 0,
    sample_rate: int = DEFAULT_SR,
    duration_s: float = SPARSH_WINDOW_S,
    out: str = "data/viz/audio_mel_preview.png",
    seed: int = 0,
) -> Tuple[Path, torch.Tensor]:
    n_rows = rows_for_duration(duration_s, sample_rate=sample_rate)
    if zarr_path:
        rows = load_mic_rows_from_zarr(zarr_path, index=index, n_rows=n_rows)
        title = f"zarr mic_0 end={index} n_rows={n_rows}"
    else:
        rows = synthetic_mic_rows(n_rows, sample_rate=sample_rate, seed=seed)
        title = f"synthetic n_rows={n_rows}"

    waveform = stack_mic_rows(rows)
    mel = log_mel_spectrogram(waveform, sample_rate=sample_rate)
    path = visualize_mel(mel, waveform, sample_rate, Path(out), title=title)
    return path, mel


def run_episode_preview(
    zarr_path: Optional[str] = None,
    sample_rate: int = DEFAULT_SR,
    out: str = "data/viz/audio_mel_episode.png",
    seed: int = 0,
    n_synth_rows: int = 119,
) -> Tuple[Path, torch.Tensor]:
    """Full-episode stacked waveform + native-length log-mel, side by side."""
    if zarr_path:
        rows = load_all_mic_rows_from_zarr(zarr_path)
        title = f"episode mic_0 n_rows={rows.shape[0]}"
    else:
        rows = synthetic_mic_rows(n_synth_rows, sample_rate=sample_rate, seed=seed)
        title = f"synthetic episode n_rows={n_synth_rows}"

    waveform = stack_mic_rows(rows)
    mel = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        target_frames=None,
        crop_to_window=False,
    )
    path = visualize_mel_beside_waveform(
        mel, waveform, sample_rate, Path(out), title=title
    )
    return path, mel


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", type=str, default=None, help="Optional .zarr.zip or zarr dir with mic_0")
    p.add_argument("--index", type=int, default=100, help="End index into mic_0 when using --zarr")
    p.add_argument("--sr", type=int, default=DEFAULT_SR, help="Sample rate (PolyUMI default 16000)")
    p.add_argument("--duration", type=float, default=SPARSH_WINDOW_S, help="Window mode: seconds")
    p.add_argument("--out", type=str, default=None, help="Output PNG path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--episode",
        action="store_true",
        help="Full-episode log-mel beside waveform (no 0.55s crop / no 224 resize)",
    )
    args = p.parse_args(argv)

    if args.episode:
        out = args.out or "data/viz/audio_mel_episode.png"
        path, mel = run_episode_preview(
            zarr_path=args.zarr,
            sample_rate=args.sr,
            out=out,
            seed=args.seed,
        )
        dur = mel.shape[-1]  # frames
        print(
            f"Wrote {path}  mel_shape={tuple(mel.shape)}  "
            f"(full episode, native STFT frames≈{dur})"
        )
    else:
        out = args.out or "data/viz/audio_mel_preview.png"
        path, mel = run_preview(
            zarr_path=args.zarr,
            index=args.index,
            sample_rate=args.sr,
            duration_s=args.duration,
            out=out,
            seed=args.seed,
        )
        print(
            f"Wrote {path}  mel_shape={tuple(mel.shape)}  "
            f"(need ~{rows_for_duration(args.duration, args.sr)} stacked rows "
            f"for {args.duration}s)"
        )


if __name__ == "__main__":
    main()
