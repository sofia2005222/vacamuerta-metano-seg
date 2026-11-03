"""Funciones de pérdida para segmentación binaria con clases muy desbalanceadas y etiquetas ruidosas."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice, promediado sobre el batch. Compensa que casi todo el parche es fondo."""
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    total = p.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (total + eps)).mean()


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def bootstrap_bce_dice(logits: torch.Tensor, target: torch.Tensor, beta: float = 0.8) -> torch.Tensor:
    """'Soft bootstrapping' (Reed et al., 2015) para etiquetas ruidosas.

    El objetivo pasa a ser beta * etiqueta + (1 - beta) * predicción actual del modelo (sin gradiente).
    Cuando la etiqueta débil es dudosa (el punto del pozo no coincide con el contorno real), el modelo
    deja de ser forzado a reproducir el error al pie de la letra.
    """
    with torch.no_grad():
        soft = torch.sigmoid(logits)
    t = beta * target + (1.0 - beta) * soft
    return F.binary_cross_entropy_with_logits(logits, t) + dice_loss(logits, t)


def build_loss(name: str, beta: float = 0.8):
    if name == "bce_dice":
        return bce_dice
    if name == "bootstrap_bce_dice":
        return lambda logits, target: bootstrap_bce_dice(logits, target, beta=beta)
    raise ValueError(f"Pérdida desconocida: {name}")
