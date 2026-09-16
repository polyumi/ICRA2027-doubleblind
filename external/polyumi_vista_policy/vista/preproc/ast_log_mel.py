"""
AST log-mel frontend for PolyTouch only.

Matches HuggingFace ``ASTFeatureExtractor`` used by
``MIT/ast-finetuned-audioset-10-10-0.4593``:

- Kaldi fbank, Hanning window, 128 mel bins, 16 kHz
- Pad / truncate time to ``max_length`` (1024)
- AudioSet normalize: ``(x - mean) / (std * 2)``

SHF / Sparsh / Qformer keep :mod:`vista.preproc.log_mel` unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vista.preproc.log_mel import mic_rows_to_waveform

try:
    import torchaudio
    from torchaudio.compliance import kaldi as ta_kaldi

    _HAS_TORCHAUDIO = True
except ImportError:  # pragma: no cover
    _HAS_TORCHAUDIO = False


# AudioSet stats from ASTFeatureExtractor / preprocessor_config.json
AST_AUDIOSET_MEAN = -4.2677393
AST_AUDIOSET_STD = 4.5689974
AST_MAX_LENGTH = 1024
AST_NUM_MEL_BINS = 128


class ASTLogMel(nn.Module):
    """
    PolyUMI ``mic_0`` rows → ``(B, max_length, n_mels)`` for HF AST.

    Input: ``(B, H, 536)`` contiguous PCM chunks or ``(B, T)`` waveform @ 16 kHz.
    Output: AudioSet-normalized Kaldi fbank, shape ``(B, 1024, 128)`` by default.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = AST_NUM_MEL_BINS,
        max_length: int = AST_MAX_LENGTH,
        mean: float = AST_AUDIOSET_MEAN,
        std: float = AST_AUDIOSET_STD,
        do_normalize: bool = True,
    ):
        super().__init__()
        if not _HAS_TORCHAUDIO:
            raise ImportError(
                "torchaudio is required for ASTLogMel. Install torchaudio."
            )
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.max_length = int(max_length)
        self.do_normalize = bool(do_normalize)
        self.register_buffer("_mean", torch.tensor(float(mean)))
        self.register_buffer("_std", torch.tensor(float(std)))

    def _fbank_one(self, waveform_1d: torch.Tensor) -> torch.Tensor:
        """Kaldi fbank for one mono clip → ``(T', n_mels)`` on CPU float32."""
        # HF ASTFeatureExtractor: float waveform, Hanning, default frame 25/10 ms.
        wav = waveform_1d.detach().float().cpu().unsqueeze(0)
        fbank = ta_kaldi.fbank(
            wav,
            sample_frequency=self.sample_rate,
            window_type="hanning",
            num_mel_bins=self.n_mels,
        )
        n_frames = fbank.shape[0]
        if n_frames < self.max_length:
            fbank = torch.nn.functional.pad(
                fbank, (0, 0, 0, self.max_length - n_frames)
            )
        elif n_frames > self.max_length:
            fbank = fbank[: self.max_length]
        return fbank

    def forward(self, mic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mic: ``(B, H, F)`` mic rows or ``(B, T)`` waveform.
        Returns:
            ``(B, max_length, n_mels)`` ready for ``ASTModel(input_values=...)``.
        """
        wav = mic_rows_to_waveform(mic)
        if wav.ndim != 2:
            raise ValueError(f"Expected waveform (B, T), got {tuple(wav.shape)}")
        device, dtype = wav.device, wav.dtype
        frames = [self._fbank_one(wav[i]) for i in range(wav.shape[0])]
        out = torch.stack(frames, dim=0).to(device=device, dtype=dtype)
        if self.do_normalize:
            out = (out - self._mean) / (self._std * 2.0)
        return out
