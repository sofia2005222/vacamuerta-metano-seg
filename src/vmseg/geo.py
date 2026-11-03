"""Utilidades geoespaciales: AOIs, zona UTM y rasterización de geometrías sobre la grilla de una escena."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG de la zona UTM que contiene (lon, lat). Neuquén cae en la zona 19 sur (EPSG:32719)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def pixel_to_world(transform, col: float, row: float) -> tuple[float, float]:
    """(columna, fila) -> (x, y) del CRS. Escrito a mano para no depender de la versión de `affine`."""
    return (
        transform.c + transform.a * col + transform.b * row,
        transform.f + transform.d * col + transform.e * row,
    )


def load_aoi(name: str, path: str | Path = "configs/aois.yaml") -> dict:
    """Lee un AOI de configs/aois.yaml y devuelve {'bbox': (oeste, sur, este, norte), 'center': (lon, lat), ...}."""
    with open(path, "r", encoding="utf-8") as f:
        aois = yaml.safe_load(f)
    if name not in aois:
        raise KeyError(f"AOI '{name}' no está en {path}. Opciones: {sorted(aois)}")
    aoi = dict(aois[name])
    lon, lat = aoi["center"]
    h = float(aoi["half_size_deg"])
    aoi["bbox"] = (lon - h, lat - h, lon + h, lat + h)
    return aoi


def rasterize_geoms(gdf, transform, out_shape, raster_crs, buffer_m: float | None = None) -> np.ndarray:
    """Rasteriza las geometrías de `gdf` sobre una grilla (transform, out_shape) -> máscara uint8 {0,1}.

    - `gdf` se reproyecta al CRS del raster.
    - `buffer_m` agranda cada geometría esa cantidad de metros (sirve para convertir puntos en discos);
      exige un CRS proyectado (en metros), como UTM.
    """
    import rasterio.features
    from pyproj import CRS

    crs = CRS.from_user_input(raster_crs)
    if buffer_m and not crs.is_projected:
        raise ValueError("El buffer en metros requiere un CRS proyectado (p. ej. EPSG:32719).")
    if len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.uint8)

    geoms = gdf.to_crs(crs).geometry
    if buffer_m:
        geoms = geoms.buffer(buffer_m)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterio.features.rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8"
    )
