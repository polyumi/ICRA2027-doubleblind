"""FoTa T3-small encoder+trunk weight loading."""

import torch

from vista.encoders.t3 import T3Encoder, T3_SIZE_PRESETS


def test_t3_small_forward_random_init():
    enc = T3Encoder(
        d_embed=768,
        pretrained=False,
        t3_size="small",
        hf_repo="",
    )
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 197, 768)
    x5 = torch.randn(2, 2, 3, 224, 224)
    out5 = enc(x5)
    assert out5.shape == (2, 2 * 197, 768)


def test_t3_small_loads_local_digit_and_trunk():
    enc_path = "/tmp/t3_small_ckpts/models/t3_small/encoders/digit.pth"
    tru_path = "/tmp/t3_small_ckpts/models/t3_small/trunk.pth"
    try:
        open(enc_path, "rb").close()
        open(tru_path, "rb").close()
    except OSError:
        import pytest

        pytest.skip("local FoTa t3_small checkpoints not downloaded")

    enc = T3Encoder(
        d_embed=768,
        pretrained=False,
        t3_size="small",
        t3_sensor="digit",
        hf_repo="",
        t3_ckpt=enc_path,
        trunk_ckpt=tru_path,
    )
    # Stem/trunk embed is 384; proj maps to 768.
    assert enc.embed_dim == T3_SIZE_PRESETS["small"][0] == 384
    out = enc(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 197, 768)
    assert torch.isfinite(out).all()
