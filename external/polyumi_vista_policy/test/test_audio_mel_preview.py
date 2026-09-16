"""Tests for standalone Sparsh-style audio mel preview preprocessing."""

from vista.viz.audio_mel_preview import (
    log_mel_spectrogram,
    rows_for_duration,
    stack_mic_rows,
    synthetic_mic_rows,
)


def test_stack_and_log_mel_shapes():
    n_rows = rows_for_duration(0.55, sample_rate=16000)
    assert n_rows == 17  # ceil(8800 / 536)
    rows = synthetic_mic_rows(n_rows, seed=1)
    assert rows.shape == (n_rows, 536)
    wav = stack_mic_rows(rows)
    assert wav.ndim == 1 and wav.shape[0] == n_rows * 536
    mel = log_mel_spectrogram(wav, sample_rate=16000, target_frames=224)
    assert mel.shape == (128, 224)


def test_audio_mel_preview_writes_png(tmp_path):
    from vista.viz.audio_mel_preview import run_preview

    out = tmp_path / "mel.png"
    path, mel = run_preview(out=str(out), seed=2)
    assert path.exists()
    assert path.stat().st_size > 0
    assert mel.shape[0] == 128


def test_episode_preview_native_mel_beside_waveform(tmp_path):
    from vista.viz.audio_mel_preview import run_episode_preview

    out = tmp_path / "episode.png"
    path, mel = run_episode_preview(out=str(out), seed=3, n_synth_rows=40)
    assert path.exists()
    assert path.stat().st_size > 0
    assert mel.shape[0] == 128
    # No Sparsh 224 resize; full stack is longer than a 0.55 s window.
    assert mel.shape[1] > 224
