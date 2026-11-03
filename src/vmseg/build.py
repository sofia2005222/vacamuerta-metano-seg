"""Construye el dataset de parches: escena Sentinel-2 + puntos de pozos (+ polígonos manuales opcionales)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geo import rasterize_geoms
from .labels import DEFAULT_RADIUS_M, points_to_weak_mask, polygons_to_manual_mask, weak_label_quality
from .tiling import (
    assign_splits,
    block_key,
    iter_patches,
    normalize_reflectance,
    save_patch,
    write_index,
)


def read_scene(path: str | Path):
    """Lee un GeoTIFF multibanda -> (imagen [C,H,W] float32 con NaN en sin-dato, transform, crs)."""
    import rasterio

    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            img[img == src.nodata] = np.nan
        return img, src.transform, src.crs


def build_patches(
    scenes: list[str | Path],
    wells_path: str | Path,
    out_dir: str | Path,
    size: int = 256,
    stride: int | None = None,
    radius_m: float = DEFAULT_RADIUS_M,
    manual_path: str | Path | None = None,
    labeled_path: str | Path | None = None,
    block_px: int = 512,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    min_valid: float = 0.9,
) -> dict:
    """Escribe out_dir/index.csv, out_dir/*.npz y out_dir/to_label.gpkg (parches de val/test a etiquetar a mano).

    Etiquetas manuales: `manual_path` tiene los polígonos de locaciones; `labeled_path` tiene las zonas que
    se etiquetaron de forma EXHAUSTIVA (un parche sin polígonos dentro de una zona etiquetada es un negativo
    verdadero; fuera de esa zona, 'sin polígono' solo significa 'no se miró').
    """
    import geopandas as gpd
    import rasterio.windows
    from shapely.geometry import box

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = stride or size
    wells = gpd.read_file(wells_path)
    manual = gpd.read_file(manual_path) if manual_path else None
    labeled = gpd.read_file(labeled_path) if labeled_path else None
    if manual is not None and labeled is None:
        raise ValueError("Si pasás --manual también hay que pasar --labeled (las zonas etiquetadas exhaustivamente).")

    records = []  # (scene, patch dict, has_manual, geometry)
    quality = []
    for scene_path in map(Path, scenes):
        scene = scene_path.stem
        image, transform, crs = read_scene(scene_path)
        image = normalize_reflectance(image)
        _, h, w = image.shape
        weak = points_to_weak_mask(wells, transform, (h, w), crs, radius_m)
        man = labeled_mask = None
        if manual is not None:
            man = polygons_to_manual_mask(manual, transform, (h, w), crs)
            labeled_mask = rasterize_geoms(labeled, transform, (h, w), crs)
        for p in iter_patches(image, weak, man, size=size, stride=stride, min_valid=min_valid):
            y, x = p["y"], p["x"]
            has_manual = bool(labeled_mask is not None and labeled_mask[y : y + size, x : x + size].mean() >= 0.99)
            if has_manual:
                quality.append(weak_label_quality(p["weak"], p["manual"]))
            win = rasterio.windows.Window(x, y, size, size)
            geom = box(*rasterio.windows.bounds(win, transform))
            records.append((scene, p, has_manual, geom, str(crs)))

    if not records:
        raise RuntimeError("No salió ningún parche. Revisá la escena (¿todo nubes/NaN?) y el tamaño de parche.")

    keys = [block_key(s, p["x"], p["y"], block_px) for s, p, *_ in records]
    split_of = assign_splits(keys, val_frac, test_frac, seed)

    rows, foot = [], []
    for (scene, p, has_manual, geom, crs_str), k in zip(records, keys):
        split = split_of[k]
        name = f"{scene}_{p['y']:05d}_{p['x']:05d}.npz"
        save_patch(out_dir / name, p["image"], p["weak"], p["manual"] if has_manual else None)
        rows.append(
            {
                "file": name, "scene": scene, "x": p["x"], "y": p["y"], "size": size, "split": split,
                "has_manual": int(has_manual), "weak_frac": float(p["weak"].mean()),
                "manual_frac": float(p["manual"].mean()) if (has_manual and p["manual"] is not None) else 0.0,
            }
        )
        if split in ("val", "test"):
            foot.append({"file": name, "split": split, "geometry": geom, "crs": crs_str})
    write_index(out_dir / "index.csv", rows)

    if foot:
        fp = gpd.GeoDataFrame(foot, geometry="geometry", crs=foot[0]["crs"]).drop(columns=["crs"]).to_crs("EPSG:4326")
        fp.to_file(out_dir / "to_label.gpkg", driver="GPKG")

    summary = {
        "n_patches": len(rows),
        "per_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "n_with_manual": sum(r["has_manual"] for r in rows),
        "patches_with_weak_positives": sum(r["weak_frac"] > 0 for r in rows),
        "radius_m": radius_m,
        "weak_vs_manual": (
            {k: float(np.nanmean([q[k] for q in quality])) for k in quality[0]} if quality else None
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
