"""Audio Spectrogram Transformer encoder."""

import torch
import torch.nn as nn

from vista.encoders.base import SensorEncoder

try:
    from transformers import ASTModel

    _HAS_AST = True
except ImportError:
    _HAS_AST = False


class ASTEncoder(SensorEncoder):
    """
    Wrap HuggingFace AST; raises on forward if transformers missing.

    Expects AudioSet-style log-mel ``(B, max_length, num_mel_bins)`` —
    typically ``(B, 1024, 128)`` from :class:`vista.preproc.ast_log_mel.ASTLogMel`.
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        d_embed: int = 256,
    ):
        super().__init__(d_embed=d_embed)
        if not _HAS_AST:
            self.model = None
        else:
            self.model = ASTModel.from_pretrained(model_name)
            self.proj = nn.Linear(self.model.config.hidden_size, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ImportError(
                "transformers is required for ASTEncoder. "
                "pip install transformers or choose another audio encoder."
            )
        # x: B, T_frames, n_mels or B, To, T_frames, n_mels
        if x.ndim == 4:
            b, to, tf, nm = x.shape
            x = x.reshape(b * to, tf, nm)
            out = self.model(input_values=x).last_hidden_state
            out = self.proj(out)
            return out.reshape(b, to * out.shape[1], -1)
        out = self.model(input_values=x).last_hidden_state
        return self.proj(out)
