"""Cortar escenas grandes en parches y repartirlos en train/val/test SIN fuga espacial.

Los parches vecinos se parecen mucho. Si uno cae en train y su vecino en test, el modelo "ve" el test
durante el entrenamiento. Por eso la partición se hace por BLOQUES espaciales, no por parche.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

S2_SCALE = 0.3  # reflectancia a partir de la cual saturamos al normalizar (la mayoría de los píxeles < 0.3)


def normalize_reflectance(image: np.ndarray) -> np.ndarray:
    """Reflectancia [0, ~1] -> [0, 1] saturando en S2_SCALE. Los NaN (nubes/sin dato) se respetan."""
    return np.clip(image / S2_SCALE, 0.0, 1.0).astype(np.float32)


def window_starts(n: int, size: int, stride: int) -> list[int]:
    """Posiciones de inicio de las ventanas a lo largo de un eje, cubriendo también el borde final."""
    if n < size:
        raise ValueError(f"La escena ({n}px) es más chica que el parche ({size}px).")
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def iter_patches(
    image: np.ndarray,
    weak: np.ndarray,
    manual: np.ndarray | None,
    size: int = 256,
    stride: int = 256,
    min_valid: float = 0.9,
) -> Iterator[dict]:
    """Recorre la escena y devuelve parches con al menos `min_valid` de píxeles válidos (sin NaN)."""
    _, h, w = image.shape
    for y in window_starts(h, size, stride):
        for x in window_starts(w, size, stride):
            img = image[:, y : y + size, x : x + size]
            valid = np.isfinite(img).all(axis=0)
            if valid.mean() < min_valid:
                continue
            yield {
                "x": x,
                "y": y,
                "image": np.nan_to_num(img, nan=0.0).astype(np.float32),
                "weak": weak[y : y + size, x : x + size].astype(np.uint8),
                "manual": None if manual is None else manual[y : y + size, x : x + size].astype(np.uint8),
            }


def assign_splits(keys, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> dict:
    """Asigna cada clave de bloque a 'train' / 'val' / 'test' de forma determinista."""
    uniq = sorted(set(keys))
    n = len(uniq)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    out = {}
    for rank, idx in enumerate(order):
        if rank < n_test:
            out[uniq[idx]] = "test"
        elif rank < n_test + n_val:
            out[uniq[idx]] = "val"
        else:
            out[uniq[idx]] = "train"
    return out


def block_key(scene: str, x: int, y: int, block_px: int = 512) -> tuple:
    return (scene, y // block_px, x // block_px)


INDEX_COLUMNS = ["file", "scene", "x", "y", "size", "split", "has_manual", "weak_frac", "manual_frac"]


def save_patch(path: Path, image: np.ndarray, weak: np.ndarray, manual: np.ndarray | None) -> None:
    """Guarda un parche comprimido. La imagen va en float16 para ocupar la mitad."""
    manual_arr = np.zeros_like(weak) if manual is None else manual
    np.savez_compressed(path, image=image.astype(np.float16), weak=weak, manual=manual_arr)


def write_index(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
