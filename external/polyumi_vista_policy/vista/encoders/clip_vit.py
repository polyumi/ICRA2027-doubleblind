"""CLIP ViT encoder via timm."""

from vista.encoders.vit import TimmViTEncoder


class CLIPViTEncoder(TimmViTEncoder):
    """OpenAI CLIP ViT (default ViT-B/16) from timm."""

    def __init__(
        self,
        d_embed: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        model_name: str = "vit_base_patch16_clip_224.openai",
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            d_embed=d_embed,
            freeze=freeze,
        )
