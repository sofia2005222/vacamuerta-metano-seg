"""Etiquetas débiles: de los puntos oficiales de pozos a máscaras de segmentación.

Idea: el padrón oficial de pozos da UN punto por pozo (sin el contorno de la locación). Dibujamos un
disco de radio `radius_m` alrededor de cada punto. Los pozos de una misma locación (pad) están
cerca, así que sus discos se funden en una mancha que aproxima la locación.

OJO: el radio por defecto es un SUPUESTO. Hay que calibrarlo midiendo a mano unas 30 locaciones
en la imagen (ver docs/LABELING.md) y ajustarlo con `--radius-m`.
"""
from __future__ import annotations

import numpy as np

from .geo import rasterize_geoms

DEFAULT_RADIUS_M = 60.0


def points_to_weak_mask(points_gdf, transform, out_shape, raster_crs, radius_m: float = DEFAULT_RADIUS_M) -> np.ndarray:
    """Máscara débil: disco de `radius_m` metros alrededor de cada punto de pozo."""
    return rasterize_geoms(points_gdf, transform, out_shape, raster_crs, buffer_m=radius_m)


def polygons_to_manual_mask(polys_gdf, transform, out_shape, raster_crs) -> np.ndarray:
    """Máscara 'verdad terreno' a partir de polígonos dibujados a mano (QGIS, etc.)."""
    return rasterize_geoms(polys_gdf, transform, out_shape, raster_crs, buffer_m=None)


def weak_label_quality(weak: np.ndarray, manual: np.ndarray) -> dict:
    """IoU y fracciones entre la etiqueta débil y la manual: mide cuán ruidosa es la etiqueta débil."""
    w, m = weak.astype(bool), manual.astype(bool)
    inter = np.logical_and(w, m).sum()
    union = np.logical_or(w, m).sum()
    return {
        "iou": float(inter / union) if union else float("nan"),
        "weak_covers_manual": float(inter / m.sum()) if m.sum() else float("nan"),
        "manual_covers_weak": float(inter / w.sum()) if w.sum() else float("nan"),
    }
