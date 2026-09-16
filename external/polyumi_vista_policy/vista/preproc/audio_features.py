"""Audio fbank front-end with global min-max normalization."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import torchaudio

    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


class AudioFrontEnd(nn.Module):
    """Kaldi fbank features with optional global min-max scaling."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_length: int = 200,
    ):
        super().__init__()
        if not _HAS_TORCHAUDIO:
            raise ImportError(
                "torchaudio is required for AudioFrontEnd (kaldi fbank). "
                "Install torchaudio or omit audio modalities."
            )
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.register_buffer("_min", torch.tensor(0.0))
        self.register_buffer("_max", torch.tensor(1.0))
        self._fitted = False

    def fit(self, samples: np.ndarray) -> None:
        """Fit global min/max from raw waveform samples ``(N, T)`` or fbank cache."""
        if samples.ndim == 1:
            samples = samples[None, :]
        fbanks = []
        for row in samples:
            wav = torch.from_numpy(row.astype(np.float32)).unsqueeze(0)
            fb = torchaudio.compliance.kaldi.fbank(
                wav, num_mel_bins=self.n_mels, sample_frequency=self.sample_rate
            )
            fbanks.append(fb.numpy())
        stacked = np.concatenate(fbanks, axis=0)
        self._min.fill_(float(stacked.min()))
        self._max.fill_(float(stacked.max()))
        self._fitted = True

    def _to_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: B, T or B, To, T
        orig_shape = wav.shape
        if wav.ndim == 3:
            b, to, t = wav.shape
            wav = wav.reshape(b * to, t)
        else:
            b, to = wav.shape[0], 1
        feats = []
        for i in range(wav.shape[0]):
            fbank = torchaudio.compliance.kaldi.fbank(
                wav[i : i + 1],
                num_mel_bins=self.n_mels,
                sample_frequency=self.sample_rate,
            )
            if fbank.shape[0] > self.max_length:
                fbank = fbank[: self.max_length]
            elif fbank.shape[0] < self.max_length:
                pad = torch.zeros(
                    self.max_length - fbank.shape[0],
                    fbank.shape[1],
                    device=fbank.device,
                    dtype=fbank.dtype,
                )
                fbank = torch.cat([fbank, pad], dim=0)
            feats.append(fbank)
        out = torch.stack(feats, dim=0)
        if len(orig_shape) == 3:
            out = out.reshape(b, to, self.max_length, self.n_mels)
        else:
            out = out.unsqueeze(1)
        return out

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(obs)
        for key, wav in obs.items():
            if not key.startswith("mic") and "audio" not in key:
                continue
            fbank = self._to_fbank(wav)
            if self._fitted:
                denom = (self._max - self._min).clamp_min(1e-6)
                fbank = ((fbank - self._min) / denom).clamp(0.0, 1.0)
            out[key] = fbank
        return out
