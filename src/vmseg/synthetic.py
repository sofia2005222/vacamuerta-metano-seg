"""Escena sintética para probar TODO el pipeline sin descargar nada.

Simula una escena de 4 bandas (azul, verde, rojo, NIR) con 'locaciones' claras y rectangulares sobre un
fondo ruidoso, y un padrón de 'pozos' (puntos) desplazado un poco, como pasa con los datos reales.
No es un sustituto del problema real: sirve para detectar errores de código.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geo import pixel_to_world


def make_scene(h: int = 768, w: int = 768, n_pads: int = 12, seed: int = 0, pad_px: tuple[int, int] = (10, 16)):
    """Devuelve (image[4,h,w] float32, true_mask[h,w] uint8, centros[(row, col)])."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.08, 0.02, size=(4, h, w)).astype(np.float32)
    # textura suave de fondo
    for c in range(4):
        base[c] += 0.03 * np.sin(np.linspace(0, 6 * np.pi, w))[None, :] * rng.uniform(0.5, 1.5)
    mask = np.zeros((h, w), dtype=np.uint8)
    centers = []
    tries = 0
    while len(centers) < n_pads and tries < 1000:
        tries += 1
        ph, pw = rng.integers(pad_px[0], pad_px[1] + 1, size=2)
        r = int(rng.integers(ph, h - ph))
        c = int(rng.integers(pw, w - pw))
        if any(abs(r - rr) < 40 and abs(c - cc) < 40 for rr, cc in centers):
            continue
        sl = (slice(r - ph // 2, r + ph // 2), slice(c - pw // 2, c + pw // 2))
        base[0][sl] += 0.18
        base[1][sl] += 0.16
        base[2][sl] += 0.15
        base[3][sl] += 0.05
        mask[sl] = 1
        centers.append((r, c))
    base = np.clip(base, 0.0, 1.0).astype(np.float32)
    return base, mask, centers


def write_scene(out_dir: str | Path, name: str = "synthetic", seed: int = 0, jitter_px: float = 2.0, **kwargs) -> dict:
    """Escribe <name>.tif (escena), <name>_wells.gpkg (puntos con ruido) y <name>_manual.gpkg (polígonos)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import Point, shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image, mask, centers = make_scene(seed=seed, **kwargs)
    _, h, w = image.shape
    epsg = 32719
    # esquina noroeste arbitraria pero realista (zona de Neuquén en UTM 19S), píxeles de 10 m
    transform = from_origin(380000.0, 5700000.0, 10.0, 10.0)

    tif = out_dir / f"{name}.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", height=h, width=w, count=4, dtype="float32", crs=f"EPSG:{epsg}", transform=transform
    ) as dst:
        dst.write(image)

    rng = np.random.default_rng(seed + 1)
    pts = []
    for r, c in centers:
        dr, dc = rng.normal(0, jitter_px, size=2)
        x, y = pixel_to_world(transform, c + dc + 0.5, r + dr + 0.5)
        pts.append(Point(x, y))
    wells = gpd.GeoDataFrame({"id": range(len(pts))}, geometry=pts, crs=f"EPSG:{epsg}").to_crs("EPSG:4326")
    wells_path = out_dir / f"{name}_wells.gpkg"
    wells.to_file(wells_path, driver="GPKG")

    polys = [shape(geom) for geom, v in shapes(mask, mask=mask.astype(bool), transform=transform) if v == 1]
    manual = gpd.GeoDataFrame({"id": range(len(polys))}, geometry=polys, crs=f"EPSG:{epsg}")
    manual_path = out_dir / f"{name}_manual.gpkg"
    manual.to_file(manual_path, driver="GPKG")

    # zona "etiquetada exhaustivamente": en la escena sintética, toda
    from shapely.geometry import box

    west, north = pixel_to_world(transform, 0, 0)
    east, south = pixel_to_world(transform, w, h)
    labeled = gpd.GeoDataFrame({"id": [0]}, geometry=[box(west, south, east, north)], crs=f"EPSG:{epsg}")
    labeled_path = out_dir / f"{name}_labeled.gpkg"
    labeled.to_file(labeled_path, driver="GPKG")

    return {
        "scene": tif, "wells": wells_path, "manual": manual_path, "labeled": labeled_path,
        "true_mask": mask, "transform": transform, "epsg": epsg,
    }
