"""
Funciones de pérdida para segmentar locaciones (fondo y locación).

Se combinan la cross entropy con pérdidas de
segmentación de `smp.losses` para el desbalance de clases:
    "ce"              cross entropy (con pesos por clase si se pasan)
    "ce_dice"         cross entropy + Dice
    "ce_lovasz"       cross entropy + Lovász (aproxima directamente el IoU)
    "ce_dice_lovasz"  las tres sumadas
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from vmseg.recorte import SIN_ETIQUETA

PERDIDAS = ["ce", "ce_dice", "ce_lovasz", "ce_dice_lovasz"]


class PerdidaSegmentacion(nn.Module):
    def __init__(self, nombre="ce", pesos_clases=None):
        super().__init__()
        assert nombre in PERDIDAS, f"pérdida desconocida: {nombre}"
        self.nombre = nombre

        pesos = None if pesos_clases is None else torch.tensor(pesos_clases, dtype=torch.float)
        # los píxeles sin etiqueta (255) no cuentan en ninguna de las pérdidas
        self.ce = nn.CrossEntropyLoss(weight=pesos, ignore_index=SIN_ETIQUETA)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True,
                                        ignore_index=SIN_ETIQUETA)
        self.lovasz = smp.losses.LovaszLoss(mode=smp.losses.MULTICLASS_MODE, ignore_index=SIN_ETIQUETA)

    def forward(self, logits, targets):
        perdida = self.ce(logits, targets)
        if "dice" in self.nombre:
            perdida = perdida + self.dice(logits, targets)
        if "lovasz" in self.nombre:
            perdida = perdida + self.lovasz(logits, targets)
        return perdida