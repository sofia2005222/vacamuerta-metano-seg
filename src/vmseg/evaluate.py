"""Evaluación de una corrida sobre val/test: métricas por píxel y por objeto + figura cualitativa."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .metrics import add_confusions, confusion, object_counts, object_metrics, pixel_metrics
from .models import build_model
from .train import get_device, pick_eval_set


def load_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = yaml.safe_load(ckpt["cfg_yaml"])
    state = ckpt["model"]
    in_ch = next(v.shape[1] for k, v in state.items() if v.ndim == 4)  # primera conv del encoder
    model = build_model(cfg["model"]["encoder"], in_ch, "random")
    model.load_state_dict(state)
    return model, cfg


@torch.no_grad()
def run_evaluation(run_dir: str | Path, split: str = "test", threshold: float | None = None, n_figures: int = 6) -> dict:
    model, cfg = load_run(run_dir)
    device = get_device()
    model.to(device).eval()
    thr = float(threshold if threshold is not None else cfg["train"]["threshold"])
    root = Path(cfg["data"]["root"])
    label, only_manual = pick_eval_set(root, split, cfg["data"]["eval_label"])
    channels = cfg["data"].get("channels")
    ds = PatchDataset(root, split, label=label, only_manual=only_manual, channels=channels)
    if len(ds) == 0:
        raise RuntimeError(f"No hay parches en '{split}'.")
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    objs = {"n_gt": 0, "n_pred": 0, "gt_detected": 0, "pred_correct": 0}
    examples = []
    for x, y in loader:
        pred = (torch.sigmoid(model(x.to(device))) > thr).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0] > 0.5
        for xi, pi, gi in zip(x.numpy(), pred, gt):
            conf = add_confusions(conf, confusion(pi, gi))
            for k, v in object_counts(pi, gi).items():
                objs[k] += v
            if len(examples) < n_figures and gi.any():
                examples.append((xi, gi, pi))

    result = {
        "split": split, "label": label, "threshold": thr, "n_patches": len(ds),
        **pixel_metrics(conf), **object_metrics(objs), "objects": objs,
        "init": cfg["model"]["init"], "loss": cfg["train"]["loss"], "train_label": cfg["data"]["train_label"],
    }
    run_dir = Path(run_dir)
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if examples:
        save_figure(examples, run_dir / f"examples_{split}.png")
    return result


def save_figure(examples, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i, (img, gt, pred) in enumerate(examples):
        # img llega con las bandas de entrada del modelo; si son 3 canales se asume RGB invertido (B,G,R)
        rgb = np.clip(np.stack([img[2], img[1], img[0]], axis=-1) * 1.5, 0, 1)
        axes[i, 0].imshow(rgb); axes[i, 0].set_title("Sentinel-2 (RGB)")
        axes[i, 1].imshow(gt, cmap="gray"); axes[i, 1].set_title("Verdad terreno")
        axes[i, 2].imshow(pred, cmap="gray"); axes[i, 2].set_title("Predicción")
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
