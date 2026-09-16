"""Fusion registry."""

from typing import Any, Dict, Type

from vista.fusion.concat_proj import ConcatProjFusion
from vista.fusion.cross_attention import CrossAttentionFusion
from vista.fusion.per_sensor_transformer import PerSensorTransformerFusion
from vista.fusion.qformer import QFormerEncoder
from vista.fusion.sparsh import SparshFusionEncoder
from vista.fusion.transformer import TransformerFusion

FUSION_REGISTRY: Dict[str, Type] = {
    "concat_proj": ConcatProjFusion,
    "cross_attention": CrossAttentionFusion,
    "transformer": TransformerFusion,
    "per_sensor_transformer": PerSensorTransformerFusion,
    "sparsh": SparshFusionEncoder,
    "qformer": QFormerEncoder,
}


def build_fusion(name: str, **kwargs: Any):
    if name not in FUSION_REGISTRY:
        raise KeyError(f"Unknown fusion '{name}'. Available: {list(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](**kwargs)
