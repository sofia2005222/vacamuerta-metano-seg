"""Pruebas de punta a punta: sintético -> parches -> entrenar -> evaluar -> predecir. Requieren PyTorch + smp.

`test_learns_on_synthetic` es lenta (~1-2 min en CPU): `pytest -m "not slow"` la saltea.
"""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from vmseg import cli  # noqa: E402
from vmseg.losses import bce_dice, bootstrap_bce_dice, dice_loss  # noqa: E402


def _prepare(tmp_path, size, n_pads, patch, stride, block, seed=5):
    syn, proc = tmp_path / "syn", tmp_path / "proc"
    cli.main(["synth", "--out", str(syn), "--seed", str(seed), "--size", str(size), "--n-pads", str(n_pads)])
    cli.main([
        "build-patches", "--scene", str(syn / "synthetic.tif"), "--wells", str(syn / "synthetic_wells.gpkg"),
        "--manual", str(syn / "synthetic_manual.gpkg"), "--labeled", str(syn / "synthetic_labeled.gpkg"),
        "--out", str(proc), "--size", str(patch), "--stride", str(stride), "--block-px", str(block),
    ])
    return syn, proc


def _train(proc, runs, name, epochs, extra=()):
    args = [
        "train", "--config", "configs/default.yaml",
        "--set", f"data.root={proc}", "--set", "model.encoder=resnet18", "--set", "model.init=random",
        "--set", f"train.epochs={epochs}", "--set", "train.num_workers=0",
        "--set", f"train.out_dir={runs}", "--set", f"train.run_name={name}",
    ]
    for e in extra:
        args += ["--set", e]
    cli.main(args)
    return runs / name


def test_losses_behave():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, :4, :4] = 1.0
    good = target * 20 - 10   # logits muy seguros y correctos
    bad = -(target * 20 - 10)  # logits muy seguros y al revés
    for fn in (bce_dice, bootstrap_bce_dice):
        assert fn(good, target) < fn(bad, target)
    assert dice_loss(good, target) < 0.05
    assert torch.isfinite(bootstrap_bce_dice(torch.randn(2, 1, 8, 8), target))


def test_pipeline_smoke(tmp_path):
    syn, proc = _prepare(tmp_path, size=768, n_pads=12, patch=64, stride=64, block=192)
    run = _train(proc, tmp_path / "runs", "smoke", 3, ["train.batch_size=8", "train.loss=bootstrap_bce_dice"])
    assert (run / "best.pt").exists() and (run / "log.csv").exists() and (run / "config.yaml").exists()

    cli.main(["evaluate", "--run", str(run), "--split", "test"])
    metrics = json.loads((run / "metrics_test.json").read_text())
    assert metrics["label"] == "manual"
    for k in ("iou", "f1", "precision", "recall", "object_recall", "object_precision"):
        assert 0.0 <= metrics[k] <= 1.0 + 1e-6

    out = tmp_path / "pred.gpkg"
    cli.main(["predict", "--run", str(run), "--scene", str(syn / "synthetic.tif"), "--out", str(out)])
    assert out.exists() and out.with_suffix(".prob.tif").exists()


def test_rgb_only_channels(tmp_path):
    syn, proc = _prepare(tmp_path, size=512, n_pads=8, patch=64, stride=64, block=128)
    run = _train(proc, tmp_path / "runs", "rgb", 1, ["train.batch_size=8", "data.channels=[2,1,0]"])
    cli.main(["evaluate", "--run", str(run), "--split", "val"])
    out = tmp_path / "pred_rgb.gpkg"
    cli.main(["predict", "--run", str(run), "--scene", str(syn / "synthetic.tif"), "--out", str(out)])
    assert out.exists()


@pytest.mark.slow
def test_learns_on_synthetic(tmp_path):
    """Con etiquetas débiles ruidosas, el modelo debe generalizar a la verdad manual (IoU alto)."""
    _, proc = _prepare(tmp_path, size=1024, n_pads=60, patch=64, stride=32, block=256, seed=7)
    run = _train(proc, tmp_path / "runs", "learn", 12, ["train.batch_size=32", "train.lr=1e-3"])
    cli.main(["evaluate", "--run", str(run), "--split", "test"])
    metrics = json.loads((run / "metrics_test.json").read_text())
    assert metrics["iou"] > 0.5, metrics
    assert metrics["object_recall"] > 0.5, metrics
