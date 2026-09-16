"""
Contact-mic encoder: raw waveform rows -> log-mel -> small CNN -> feature vector.

`mic_0` reaches the policy as the exporter wrote it: a stack of raw 16 kHz waveform rows, one row
per exported step (`ingest/polyumi_ingest/export/dp/audio.py`). The rows are contiguous in time, so
the window is reassembled by concatenating them before any spectral transform -- computing a
spectrogram per row would impose a 33.5 ms analysis boundary that has nothing to do with the
signal, and would destroy exactly the onset structure a contact event lives in.

The mel filterbank and STFT are built from torch primitives rather than torchaudio because neither
torchaudio nor librosa is in `conda_environment.yaml`, and checkpoints are dill-pickled against the
exact dependency tree -- adding a dependency invalidates every existing checkpoint for the sake of
a filterbank that is twenty lines.
"""

# The policy container's conda env is python=3.9 (see CLAUDE.md), where `X | Y` in an annotation
# raises at import. This makes every annotation lazy, so the 3.10+ syntax below is inert there.
from __future__ import annotations

import math

import torch
import torch.nn as nn


def hz_to_mel(f: torch.Tensor | float) -> torch.Tensor | float:
    """Slaney-style hz->mel, matching librosa's default and the ingest-side log-mel."""
    return 2595.0 * math.log10(1.0 + f / 700.0) if isinstance(f, float) else 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, f_min: float, f_max: float) -> torch.Tensor:
    """
    Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Built once at construction and registered as a buffer, so it moves with the module and costs
    nothing per forward.
    """
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    m_pts = torch.linspace(hz_to_mel(float(f_min)), hz_to_mel(float(f_max)), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    # Slopes from each band edge to every FFT bin; the triangle is the lower/upper slope minimum.
    slopes = f_pts.unsqueeze(0) - freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    down = -slopes[:, :-2] / (f_pts[1:-1] - f_pts[:-2])
    up = slopes[:, 2:] / (f_pts[2:] - f_pts[1:-1])
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    return fb.T.contiguous()


class MelAudioEncoder(nn.Module):
    """
    Encode ``(B, T_rows, samples_per_row)`` contact-mic waveform to ``(B, out_dim)``.

    Three stride-2 conv blocks over the log-mel image, then global average pooling. Deliberately
    small: the slip corpus is on the order of 20k steps, and a heavier stem would fit the recording
    session rather than the contact.

    ``f_min`` defaults to 20 Hz -- the full band, i.e. the raw signal. It is left there on
    purpose: the first runs establish what the unprocessed contact mic can do, and a baseline that
    was never trained is a number nobody has.

    A higher floor is a large, measured win if you want it later. Sweeping it over 62 board contact
    events and 43 slip events, contact SNR runs 9.4 dB unfiltered, peaks at 25.0 dB (board) /
    22.5 dB (slip) at **300 Hz**, and falls to 18.8 / 20.3 dB by 1 kHz -- almost all of the noise
    masking contacts is handling rumble below 300 Hz. Set ``f_min: 300.0`` in a task's ``mic_0``
    shape_meta to turn it on; nothing else has to change and no re-export is needed.

    What a higher floor does NOT buy is speech rejection, which is the tempting reason to reach for
    one: speech spans 300-3400 Hz, so raising the floor to 1 kHz removes only its lowest sixth and
    leaves the speech share of background energy essentially unchanged (33.7% unfiltered, 37.9% at
    300 Hz, 32.7% at 1 kHz) while costing several dB of signal. Speech overlaps the contact band and
    cannot be high-passed away. Separating them needs a temporal method instead -- contacts decay in
    10-60 ms, speech syllables last 100-300 ms -- and subtracting a ~0.3 s running median per mel
    band halved the background fluctuation in testing. Neither is applied here.
    """

    def __init__(
        self,
        samples_per_row: int = 536,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        out_dim: int = 128,
        log_offset: float = 1e-6,
        f_min: float = 20.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_row = samples_per_row
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_offset = log_offset
        self.out_dim = out_dim
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            'fb', mel_filterbank(n_mels, n_fft, sample_rate, f_min, sample_rate / 2), persistent=False
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Log-mel spectrogram ``(B, 1, n_mels, frames)`` from a ``(B, samples)`` waveform."""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.matmul(self.fb, power)
        return torch.log(mel + self.log_offset).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T_rows, samples_per_row)`` to ``(B, out_dim)``."""
        if x.ndim != 3:
            raise ValueError(f'MelAudioEncoder expects (B, T_rows, samples), got {tuple(x.shape)}')
        b = x.shape[0]
        # Rows are contiguous in time: concatenate before the transform, do not treat as a batch.
        feat = self.stem(self.log_mel(x.reshape(b, -1)))
        return feat.mean(dim=(2, 3))
