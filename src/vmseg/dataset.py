"""Dataset de PyTorch sobre los parches .npz generados por `build-patches`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Cada ítem: (imagen [C,H,W] float32 en [0,1], máscara [1,H,W] float32 en {0,1}).

    label: 'weak' (puntos oficiales con buffer) o 'manual' (polígonos dibujados a mano).
    only_manual: quedarse solo con parches que fueron etiquetados a mano de forma exhaustiva.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        label: str = "weak",
        augment: bool = False,
        only_manual: bool = False,
        channels: list[int] | None = None,
    ):
        if label not in ("weak", "manual"):
            raise ValueError("label debe ser 'weak' o 'manual'")
        self.root = Path(root)
        df = pd.read_csv(self.root / "index.csv")
        df = df[df["split"] == split]
        if only_manual:
            df = df[df["has_manual"] == 1]
        self.df = df.reset_index(drop=True)
        self.label = label
        self.augment = augment
        self.channels = list(channels) if channels is not None else None  # p. ej. [2, 1, 0] = solo RGB
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_channels(self) -> int:
        if self.channels is not None:
            return len(self.channels)
        with np.load(self.root / self.df.loc[0, "file"]) as d:
            return int(d["image"].shape[0])

    def __getitem__(self, i: int):
        with np.load(self.root / self.df.loc[i, "file"]) as d:
            img = d["image"].astype(np.float32)
            mask = d[self.label].astype(np.float32)
        if self.channels is not None:
            img = img[self.channels]
        if self.augment:
            if self.rng.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            if self.rng.random() < 0.5:
                img, mask = img[:, ::-1, :], mask[::-1, :]
            k = int(self.rng.integers(0, 4))
            if k:
                img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k, axes=(0, 1))
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask))[None]
