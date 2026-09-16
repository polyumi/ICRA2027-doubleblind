"""
Shared log-mel spectrogram frontend for SHF / Sparsh-X / Qformer.

PolyUMI ``mic_0`` rows are contiguous PCM chunks (536 samples @ 16 kHz ≈ 33.5 ms).
Policies concatenate ``audio_obs_horizon`` rows into one waveform, then build log-mel.
Default window is 10 rows ≈ 0.33 s.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchaudio is required for log-mel audio. Install torchaudio."
    ) from exc


def mic_rows_to_waveform(mic: torch.Tensor) -> torch.Tensor:
    """
    Flatten ``(B, H, F)`` mic feature rows into ``(B, H*F)`` waveform.

    Also accepts ``(B, T)`` already-flat waveforms.
    """
    if mic.ndim == 2:
        return mic
    if mic.ndim != 3:
        raise ValueError(f"Expected mic (B, H, F) or (B, T), got {tuple(mic.shape)}")
    b, _h, _f = mic.shape
    return mic.reshape(b, -1)


class LogMelSpectrogram(nn.Module):
    """
    Waveform → log-mel image ``(B, 1, n_mels, time_frames)``.

    ``time_frames`` pads / truncates the time axis so patch / CNN stems see a fixed grid.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        win_ms: float = 25.0,
        hop_ms: float = 10.0,
        time_frames: Optional[int] = 48,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        n_fft: Optional[int] = 1024,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.time_frames = time_frames
        win_length = max(16, int(round(sample_rate * win_ms / 1000.0)))
        hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
        if n_fft is None:
            # 128 mels @ 16 kHz need enough FFT bins for non-empty filterbanks.
            min_fft = max(win_length, 8 * n_mels)
            n_fft = 1
            while n_fft < min_fft:
                n_fft <<= 1
        # Short windows (25 ms → 400) are zero-padded into n_fft.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else float(sample_rate // 2),
            power=2.0,
            center=True,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: ``(B, T)`` mono PCM.
        Returns:
            ``(B, 1, n_mels, time_frames)`` log-mel (or native T if time_frames is None).
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform (B, T), got {tuple(waveform.shape)}")
        spec = self.mel(waveform.float())  # (B, n_mels, T')
        log_spec = torch.log(spec.clamp_min(1e-8))
        if self.time_frames is not None:
            t = log_spec.shape[-1]
            if t < self.time_frames:
                log_spec = F.pad(log_spec, (0, self.time_frames - t))
            elif t > self.time_frames:
                log_spec = log_spec[..., : self.time_frames]
        return log_spec.unsqueeze(1)


def shared_log_mel(
    sample_rate: int = 16000, time_frames: Optional[int] = 48
) -> LogMelSpectrogram:
    """
    Shared 16 kHz contact-mic frontend for SHF / Sparsh-X / Qformer.

    128 mels, 25 ms / 10 ms, n_fft=1024, fmin=20. Pads / crops to ``time_frames``
    (48 covers ~0.33 s / 10 mic rows; native STFT length is ~34). Pass
    ``time_frames=None`` for native length (SHF ResNet adaptive-pools).
    """
    return LogMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        win_ms=25.0,
        hop_ms=10.0,
        time_frames=time_frames,
        f_min=20.0,
        n_fft=1024,
    )


# Back-compat aliases used by older call sites / tests.
def shf_log_mel(sample_rate: int = 16000) -> LogMelSpectrogram:
    """Alias for :func:`shared_log_mel` with native time length (SHF ResNet pools)."""
    return LogMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        win_ms=25.0,
        hop_ms=10.0,
        time_frames=None,
        f_min=20.0,
        n_fft=1024,
    )


def sparsh_log_mel(sample_rate: int = 16000, time_frames: int = 48) -> LogMelSpectrogram:
    """Alias for :func:`shared_log_mel` (fixed time grid for patch / CNN stems)."""
    return shared_log_mel(sample_rate=sample_rate, time_frames=time_frames)
