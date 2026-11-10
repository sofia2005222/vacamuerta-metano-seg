import os

import torch

from vmseg.config import DatasetConfig, TrainingConfig, deterministic
from vmseg.dataset import get_data
from vmseg.entrenar import main
from vmseg.modelo import crear_unet
from vmseg.recorte import generar_recortes_falsos


def test_dataset_devuelve_tensores(tmp_path):
    generar_recortes_falsos(tmp_path, n=8, tamano=64)
    config = DatasetConfig(data_root=tmp_path, batch_size=2, num_workers=0, augment=True)
    train_loader, valid_loader, mean, std = get_data(config)

    X, y = next(iter(train_loader))
    assert X.shape == (2, 6, 64, 64) and X.dtype == torch.float32
    assert y.shape == (2, 64, 64) and y.dtype == torch.int64
    assert mean.shape == (6,)


def test_loss_baja_con_recortes_falsos(tmp_path):
    deterministic(0)
    # en recortes falsos alcanzan train y val; cada uno con 4 recortes
    generar_recortes_falsos(tmp_path / "datos", n=16, tamano=64)
    config = DatasetConfig(data_root=tmp_path / "datos", batch_size=4, num_workers=0)
    train_loader, valid_loader, _, _ = get_data(config)

    model = crear_unet(in_channels=6, encoder_weights=None)   # sin descargar pesos
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_config = TrainingConfig(epochs=5, checkpoints_dir=tmp_path / "runs")

    history = main(model, optimizer, train_loader, valid_loader, train_config)

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert os.path.exists(tmp_path / "runs" / "best.pt")
