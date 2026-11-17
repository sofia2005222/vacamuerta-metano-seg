"""
Modelos bitemporales de ADR-0011: la red siamesa (modelo principal) y la fusión temprana
(línea de base), con la estructura del modelo de smp.

Los dos reciben T1 y T2 apiladas (como las devuelve dataset_pares.py): primero los
canales de T1 y después los de T2.
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from vmseg.dataset_pares import INDICES
from vmseg.recorte import BANDAS

# canales de los pesos de ImageNet, en orden R, G, B, y la banda de Sentinel-2 de cada uno
RGB_IMAGENET = ["B4", "B3", "B2"]

MODOS = ["cero", "congelado", "finetuning"]


def nombres_canales(config):
    """Nombres de los canales de una fecha según la configuración: bandas y, si van, índices."""
    nombres = [BANDAS[i] for i in config.bandas]
    if config.indices:
        nombres += INDICES
    return nombres

def adaptar_primera_capa(encoder, canales):
    """
    Cambia la primera capa del encoder (preentrenado con R, G, B) para que reciba `canales`.

    Cada banda RGB de Sentinel-2 recibe el peso de su color en ImageNet (B4 rojo, B3 verde,
    B2 azul); los demás canales (NIR, SWIR, índices) arrancan con el promedio de los tres.
    """
    viejo = encoder.conv1.weight.data                      # (64, 3, 7, 7), canales R, G, B
    promedio = viejo.mean(dim=1)
    nuevo = torch.stack([
        viejo[:, RGB_IMAGENET.index(c)] if c in RGB_IMAGENET else promedio
        for c in canales
    ], dim=1)

    conv = nn.Conv2d(len(canales), viejo.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
    conv.weight.data = nuevo.clone()
    encoder.conv1 = conv


def preparar_encoder(modelo, canales, modo):
    """Primera capa adaptada a los canales y, en el modo congelado, el resto del encoder fijo."""
    adaptar_primera_capa(modelo.encoder, canales)
    if modo == "congelado":
        # se congela todo el encoder salvo la primera capa, que siempre se entrena
        for nombre, parametro in modelo.encoder.named_parameters():
            parametro.requires_grad = nombre.startswith("conv1")

class SiamesaUnet(nn.Module):
    """
    Red siamesa estilo FC-Siam-diff (Daudt et al., 2018): el mismo encoder procesa T1 y T2,
    y el decoder de la U-Net recibe la diferencia absoluta de sus features en cada nivel.
    """
    def __init__(self, canales, num_classes=3, encoder="resnet34",
                 encoder_weights="imagenet", modo="finetuning"):
        super().__init__()
        assert modo in MODOS
        self.canales_por_fecha = len(canales)

        if modo == "cero":
            encoder_weights = None
        # U-Net de smp para RGB; después se adapta la primera capa
        self.unet = smp.create_model(
            arch="unet",
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        preparar_encoder(self.unet, canales, modo)

    def forward(self, x):
        # separo las dos fechas
        x1 = x[:, :self.canales_por_fecha]
        x2 = x[:, self.canales_por_fecha:]

        # el mismo encoder (mismos pesos) para las dos fechas
        features_t1 = self.unet.encoder(x1)
        features_t2 = self.unet.encoder(x2)

        # diferencia absoluta en cada nivel: lo que cambió entre T1 y T2
        diferencias = [torch.abs(f1 - f2) for f1, f2 in zip(features_t1, features_t2)]

        decoder_output = self.unet.decoder(diferencias)
        return self.unet.segmentation_head(decoder_output)


def crear_modelo(arquitectura, canales, num_classes=3, encoder="resnet34",
                 encoder_weights="imagenet", modo="finetuning"):
    """
    Crea uno de los modelos de la tabla de docs/METODO.md.

    arquitectura: "siamesa" o "fusion" (fusión temprana)
    canales: nombres de los canales de una fecha (ver nombres_canales)
    modo: "cero", "congelado" o "finetuning" (H2)
    """
    if arquitectura == "siamesa":
        return SiamesaUnet(canales, num_classes, encoder, encoder_weights, modo)

    if arquitectura == "fusion":
        # misma inicialización que la siamesa, pero con las dos fechas apiladas
        if modo == "cero":
            encoder_weights = None
        modelo = smp.create_model(
            arch="unet",
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        preparar_encoder(modelo, list(canales) + list(canales), modo)
        return modelo

    raise ValueError(f"arquitectura desconocida: {arquitectura}")
