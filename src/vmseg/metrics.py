"""Métricas de segmentación: a nivel píxel y a nivel objeto (locación)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def confusion(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Cuentas TP/FP/FN/TN entre dos máscaras binarias de cualquier forma."""
    p, g = pred.astype(bool), gt.astype(bool)
    return {
        "tp": int(np.logical_and(p, g).sum()),
        "fp": int(np.logical_and(p, ~g).sum()),
        "fn": int(np.logical_and(~p, g).sum()),
        "tn": int(np.logical_and(~p, ~g).sum()),
    }


def add_confusions(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("tp", "fp", "fn", "tn")}


def pixel_metrics(c: dict, eps: float = 1e-9) -> dict:
    """IoU, F1, precisión y recall a partir de las cuentas acumuladas sobre TODO el conjunto."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def object_counts(pred: np.ndarray, gt: np.ndarray, min_overlap: float = 0.3) -> dict:
    """Cuenta objetos (componentes conexas).

    - Una locación real está 'detectada' si la predicción cubre al menos `min_overlap` de su área.
    - Un objeto predicho es 'correcto' si al menos `min_overlap` de su área cae sobre la verdad terreno.
    """
    p, g = pred.astype(bool), gt.astype(bool)
    lg, ng = ndimage.label(g)
    lp, np_ = ndimage.label(p)

    detected = 0
    if ng:
        areas = np.bincount(lg.ravel())[1:]
        hits = np.bincount(lg.ravel(), weights=p.ravel().astype(float))[1:]
        detected = int(np.sum(hits / areas >= min_overlap))

    correct = 0
    if np_:
        areas = np.bincount(lp.ravel())[1:]
        hits = np.bincount(lp.ravel(), weights=g.ravel().astype(float))[1:]
        correct = int(np.sum(hits / areas >= min_overlap))

    return {"n_gt": int(ng), "n_pred": int(np_), "gt_detected": detected, "pred_correct": correct}


def object_metrics(counts: dict, eps: float = 1e-9) -> dict:
    """Recall y precisión a nivel objeto a partir de las cuentas acumuladas."""
    return {
        "object_recall": counts["gt_detected"] / (counts["n_gt"] + eps),
        "object_precision": counts["pred_correct"] / (counts["n_pred"] + eps),
    }
