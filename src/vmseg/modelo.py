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

def crear_fusion_temprana(canales_por_fecha=6, num_classes=3, encoder="resnet34", encoder_weights="imagenet"):
    """
    Línea de base bitemporal (ADR-0011): una U-Net que recibe T1 y T2 apiladas,
    o sea 2 × canales_por_fecha canales de entrada.
    """
    return crear_unet(
        in_channels=2 * canales_por_fecha,
        num_classes=num_classes,
        encoder=encoder,
        encoder_weights=encoder_weights,
    )
