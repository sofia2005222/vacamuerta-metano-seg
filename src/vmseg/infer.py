"""Inferencia sobre escenas completas: ventana deslizante -> mapa de probabilidad -> polígonos."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tiling import window_starts


def predict_raster(image: np.ndarray, predict_fn, size: int = 256, stride: int = 128, batch_size: int = 8) -> np.ndarray:
    """Aplica `predict_fn` sobre ventanas solapadas y promedia las probabilidades.

    image: [C,H,W] float32 ya normalizado y sin NaN.
    predict_fn: recibe [B,C,size,size] y devuelve [B,size,size] con probabilidades en [0,1].
    """
    _, h, w = image.shape
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    coords = [(y, x) for y in window_starts(h, size, stride) for x in window_starts(w, size, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        batch = np.stack([image[:, y : y + size, x : x + size] for y, x in chunk])
        probs = np.asarray(predict_fn(batch), dtype=np.float32)
        for (y, x), pr in zip(chunk, probs):
            acc[y : y + size, x : x + size] += pr
            cnt[y : y + size, x : x + size] += 1.0
    return acc / np.maximum(cnt, 1.0)


def polygonize(prob: np.ndarray, transform, crs, threshold: float = 0.5, min_area_m2: float = 2000.0):
    """Mapa de probabilidad -> GeoDataFrame de polígonos (con área en m²), descartando manchas diminutas."""
    import geopandas as gpd
    import rasterio.features
    from shapely.geometry import shape

    binary = (prob > threshold).astype(np.uint8)
    pix_area = abs(transform.a * transform.e)
    geoms, areas, scores = [], [], []
    for geom, val in rasterio.features.shapes(binary, mask=binary.astype(bool), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area < min_area_m2 or pix_area == 0:
            continue
        geoms.append(g)
        areas.append(g.area)
    return gpd.GeoDataFrame({"area_m2": areas}, geometry=geoms, crs=crs)


def make_predict_fn(run_dir: str | Path, device=None):
    """Carga best.pt de una corrida y devuelve una función numpy -> numpy lista para `predict_raster`."""
    import torch

    from .evaluate import load_run

    model, cfg = load_run(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    @torch.no_grad()
    def fn(batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch).to(device)
        return torch.sigmoid(model(x))[:, 0].cpu().numpy()

    return fn, cfg


def run_prediction(run_dir: str | Path, scene: str | Path, out_path: str | Path, threshold: float = 0.5, min_area_m2: float = 2000.0) -> dict:
    """Predice sobre una escena entera y guarda polígonos (.gpkg) + mapa de probabilidad (.tif)."""
    import rasterio

    from .build import read_scene
    from .tiling import normalize_reflectance

    fn, cfg = make_predict_fn(run_dir)
    image, transform, crs = read_scene(scene)
    image = np.nan_to_num(normalize_reflectance(image), nan=0.0)
    channels = cfg["data"].get("channels")
    if channels is not None:
        image = image[channels]
    prob = predict_raster(image, fn)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = polygonize(prob, transform, crs, threshold, min_area_m2)
    gdf.to_file(out_path, driver="GPKG")
    prob_path = out_path.with_suffix(".prob.tif")
    with rasterio.open(prob_path, "w", driver="GTiff", height=prob.shape[0], width=prob.shape[1], count=1, dtype="float32", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(prob, 1)
    return {"n_polygons": int(len(gdf)), "total_area_km2": float(gdf["area_m2"].sum() / 1e6) if len(gdf) else 0.0, "polygons": str(out_path), "prob_raster": str(prob_path)}
