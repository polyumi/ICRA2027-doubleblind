"""ASTLogMel frontend matches HF ASTFeatureExtractor (PolyTouch-only)."""

import numpy as np
import pytest
import torch

from vista.preproc.ast_log_mel import (
    AST_AUDIOSET_MEAN,
    AST_AUDIOSET_STD,
    AST_MAX_LENGTH,
    AST_NUM_MEL_BINS,
    ASTLogMel,
)


def test_ast_log_mel_shape_from_mic_rows():
    front = ASTLogMel()
    # 64 rows × 536 @ 16 kHz ≈ 2.1 s (PolyTouch train override).
    mic = torch.randn(2, 64, 536)
    out = front(mic)
    assert out.shape == (2, AST_MAX_LENGTH, AST_NUM_MEL_BINS)
    assert torch.isfinite(out).all()


def test_ast_log_mel_shape_from_waveform():
    front = ASTLogMel()
    wav = torch.randn(1, 16000)  # 1 s
    out = front(wav)
    assert out.shape == (1, AST_MAX_LENGTH, AST_NUM_MEL_BINS)


def test_ast_log_mel_matches_hf_feature_extractor():
    pytest.importorskip("transformers")
    from transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer import (
        ASTFeatureExtractor,
    )

    rng = np.random.RandomState(0)
    wav_np = rng.randn(16000).astype(np.float32) * 0.1

    fe = ASTFeatureExtractor(
        sampling_rate=16000,
        num_mel_bins=AST_NUM_MEL_BINS,
        max_length=AST_MAX_LENGTH,
        do_normalize=True,
        mean=AST_AUDIOSET_MEAN,
        std=AST_AUDIOSET_STD,
    )
    ref = fe(wav_np, sampling_rate=16000, return_tensors="pt")["input_values"]
    # HF may return (1, T, F) or (T, F) depending on version.
    if ref.ndim == 2:
        ref = ref.unsqueeze(0)

    ours = ASTLogMel()(torch.from_numpy(wav_np).unsqueeze(0))
    assert ours.shape == ref.shape
    # Kaldi path should match closely; allow tiny numeric drift.
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4)
