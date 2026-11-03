"""Entrenamiento de la U-Net sobre los parches generados por `build-patches`."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .losses import build_loss
from .metrics import add_confusions, confusion, pixel_metrics
from .models import build_model


def load_config(path: str | Path, overrides=()) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_eval_set(root: str | Path, split: str, preferred: str):
    """Devuelve (label, only_manual). Usa etiquetas manuales si existen en ese split; si no, las débiles."""
    if preferred == "manual":
        ds = PatchDataset(root, split, label="manual", only_manual=True)  # solo se mira cuántos hay
        if len(ds) > 0:
            return "manual", True
        print(f"[train] AVISO: no hay parches con etiqueta manual en '{split}'; se usa la etiqueta débil (métricas optimistas).")
    return "weak", False


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold: float = 0.5) -> dict:
    """IoU/F1/precisión/recall píxel a píxel acumulados sobre todo el loader."""
    model.eval()
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for x, y in loader:
        logits = model(x.to(device))
        pred = (torch.sigmoid(logits) > threshold).cpu().numpy()
        total = add_confusions(total, confusion(pred, y.numpy() > 0.5))
    return pixel_metrics(total)


def run_training(cfg: dict) -> Path:
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    seed = int(tcfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = Path(dcfg["root"])
    channels = dcfg.get("channels")  # None = todas las bandas; [2,1,0] = RGB (el orden guardado es B,G,R,NIR)
    train_ds = PatchDataset(root, "train", label=dcfg["train_label"], augment=True, channels=channels)
    if len(train_ds) == 0:
        raise RuntimeError("No hay parches de entrenamiento. ¿Corriste `build-patches`?")
    val_label, only_manual = pick_eval_set(root, "val", dcfg["eval_label"])
    val_ds = PatchDataset(root, "val", label=val_label, only_manual=only_manual, channels=channels)
    if len(val_ds) == 0:
        raise RuntimeError("No hay parches de validación; usá más escenas o parches más chicos.")

    device = get_device()
    model = build_model(mcfg["encoder"], train_ds.num_channels, mcfg["init"]).to(device)
    criterion = build_loss(tcfg["loss"], beta=float(tcfg.get("bootstrap_beta", 0.8)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    nw = int(tcfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, num_workers=nw, drop_last=len(train_ds) > int(tcfg["batch_size"]))
    val_loader = DataLoader(val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False, num_workers=nw)

    run_dir = Path(tcfg["out_dir"]) / tcfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    (run_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")

    best, log_rows = -1.0, []
    print(f"[train] {len(train_ds)} parches train ({dcfg['train_label']}), {len(val_ds)} val ({val_label}), dispositivo: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        m = evaluate_loader(model, val_loader, device, float(tcfg["threshold"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}}
        log_rows.append(row)
        print(f"[train] ep {epoch:03d}  loss {row['train_loss']:.4f}  val IoU {m['iou']:.3f}  F1 {m['f1']:.3f}  ({time.time() - t0:.1f}s)")
        if m["iou"] > best:
            best = m["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": best, "cfg_yaml": cfg_yaml}, run_dir / "best.pt")

    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train] listo. Mejor val IoU = {best:.3f}. Resultados en {run_dir}")
    return run_dir
