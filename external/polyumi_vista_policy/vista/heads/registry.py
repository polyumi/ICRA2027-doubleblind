"""Policy head registry."""

from typing import Any, Dict, Type

from vista.heads.dit import TransformerDenoiser
from vista.heads.mlp import MLPHead
from vista.heads.transformer_decoder import TransformerActionDecoder
from vista.heads.unet1d import Unet1DHead

HEAD_REGISTRY: Dict[str, Type] = {
    "unet1d": Unet1DHead,
    "dit": TransformerDenoiser,
    "mlp": MLPHead,
    "transformer_decoder": TransformerActionDecoder,
}


def build_head(name: str, **kwargs: Any):
    if name not in HEAD_REGISTRY:
        raise KeyError(f"Unknown head '{name}'. Available: {list(HEAD_REGISTRY)}")
    aliases = {
        "d_model": "d_embed",
        "n_layers": "n_layer",
        "n_heads": "n_head",
    }
    for src, dst in aliases.items():
        if src in kwargs and dst not in kwargs:
            kwargs[dst] = kwargs.pop(src)
        elif src in kwargs:
            kwargs.pop(src)
    return HEAD_REGISTRY[name](**kwargs)
