"""U-Net con encoder preentrenado (segmentation_models_pytorch)."""
from __future__ import annotations

import torch


def build_model(encoder: str = "resnet34", in_channels: int = 4, init: str = "imagenet"):
    """init: 'imagenet' (encoder preentrenado), 'random' o la ruta a un checkpoint propio.

    Con in_channels != 3, smp adapta la primera convolución reutilizando los pesos de ImageNet.
    """
    import segmentation_models_pytorch as smp

    weights = "imagenet" if init == "imagenet" else None
    model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=1)

    if init not in ("imagenet", "random"):
        ckpt = torch.load(init, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[models] checkpoint cargado desde {init} (faltan {len(missing)} claves, sobran {len(unexpected)})")
    return model
