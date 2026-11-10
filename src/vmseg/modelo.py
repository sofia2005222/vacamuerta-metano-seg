"""
U-Net de segmentation-models-pytorch.
"""
import segmentation_models_pytorch as smp


def crear_unet(in_channels=6, num_classes=3, encoder="resnet34", encoder_weights="imagenet"):
    """
    U-Net con encoder ResNet.

    encoder_weights="imagenet" usa el encoder preentrenado. Con in_channels distinto de 3, smp adapta sola la primera capa.
    encoder_weights=None la entrena desde cero.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model
