"""
Unit tests for serve_policy._sync_batchnorm_stats — EMA weights served with real BatchNorm stats.

Tiny torch modules; no checkpoint or GPU. Run inside the container's umi env:
    python -m pytest test/test_serve_bn_sync.py -q
"""

# ruff: noqa: D103  - test functions are self-describing via names + inline comments

import copy

import torch
import torch.nn as nn

from serve_policy import _sync_batchnorm_stats


def _net():
    return nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.BatchNorm1d(64))  # 4 ch x 4x4 from a 6x6 input


def _trained_and_ema():
    """A model whose BN stats have tracked data, and an EMA copy taken before they did."""
    torch.manual_seed(0)
    model = _net()
    ema = copy.deepcopy(model)  # what the workspace does at construction: stats still 0/1
    model.train()
    for _ in range(20):
        model(torch.randn(8, 1, 6, 6) * 3.0 + 2.0)
    return model, ema


def test_copies_running_stats_from_the_trained_model():
    model, ema = _trained_and_ema()
    assert not torch.allclose(ema[1].running_mean, model[1].running_mean)  # the bug being fixed

    n = _sync_batchnorm_stats(model, ema)

    assert n == 2
    for i in (1, 4):
        assert torch.equal(ema[i].running_mean, model[i].running_mean)
        assert torch.equal(ema[i].running_var, model[i].running_var)
        assert torch.equal(ema[i].num_batches_tracked, model[i].num_batches_tracked)


def test_leaves_the_ema_parameters_alone():
    model, ema = _trained_and_ema()
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(1.0)  # make EMA params distinguishable from the model's
    before = [p.clone() for p in ema.parameters()]

    _sync_batchnorm_stats(model, ema)

    for b, p in zip(before, ema.parameters()):
        assert torch.equal(b, p)


def test_eval_output_matches_the_model_once_synced():
    model, ema = _trained_and_ema()
    x = torch.randn(4, 1, 6, 6) * 3.0 + 2.0
    model.eval()
    ema.eval()
    assert not torch.allclose(ema(x), model(x))

    _sync_batchnorm_stats(model, ema)

    assert torch.allclose(ema(x), model(x))


def test_a_model_without_batchnorm_is_a_no_op():
    model = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    assert _sync_batchnorm_stats(model, copy.deepcopy(model)) == 0
