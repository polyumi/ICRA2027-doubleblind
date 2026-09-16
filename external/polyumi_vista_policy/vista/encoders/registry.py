"""Encoder registry."""

from typing import Any, Dict, Type

from vista.encoders.ast import ASTEncoder
from vista.encoders.clip_vit import CLIPViTEncoder
from vista.encoders.cnn import AudioCNNStem, TactileCNNStem, VisionCNNStem
from vista.encoders.patch_stem import PatchStem
from vista.encoders.proprio import ProprioMLP
from vista.encoders.resnet import ResNetEncoder
from vista.encoders.t3 import T3Encoder
from vista.encoders.vit import TimmViTEncoder

ENCODER_REGISTRY: Dict[str, Type] = {
    "proprio_mlp": ProprioMLP,
    "vit": TimmViTEncoder,
    "clip_vit": CLIPViTEncoder,
    "resnet": ResNetEncoder,
    "vision_cnn": VisionCNNStem,
    "tactile_cnn": TactileCNNStem,
    "audio_cnn": AudioCNNStem,
    "patch_stem": PatchStem,
    "ast": ASTEncoder,
    "t3": T3Encoder,
}


def build_encoder(name: str, **kwargs: Any):
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](**kwargs)
