import os

import numpy as np
import torch

from vmseg.config import DatasetConfig, TrainingConfig, deterministic
from vmseg.dataset_pares import calcular_indices, get_data_pares
from vmseg.entrenar import main
from vmseg.modelo import crear_fusion_temprana
from vmseg.par import generar_pares_falsos
from vmseg.recorte import LOCACION


def test_indices_con_valores_conocidos():
    # un píxel con vegetación: el NIR (B8) mucho más alto que el rojo (B4)
    img = np.zeros((6, 1, 1), dtype=np.float32)
    img[:, 0, 0] = [0.05, 0.08, 0.10, 0.40, 0.20, 0.15]   # B2, B3, B4, B8, B11, B12
    ndvi, ndbi, bsi = calcular_indices(img)[:, 0, 0]

    assert np.isclose(ndvi, (0.40 - 0.10) / (0.40 + 0.10), atol=1e-4)    # 0,6
    assert np.isclose(ndbi, (0.20 - 0.40) / (0.20 + 0.40), atol=1e-4)    # negativo
    assert np.isclose(bsi, (0.30 - 0.45) / (0.30 + 0.45), atol=1e-4)     # negativo


def test_dataset_apila_las_dos_fechas(tmp_path):
    generar_pares_falsos(tmp_path, n=8, tamano=64)

    # 6 bandas por fecha -> 12 canales
    config = DatasetConfig(data_root=tmp_path, batch_size=2, num_workers=0)
    train_loader, _, mean, _ = get_data_pares(config)
    X, y = next(iter(train_loader))
    assert X.shape == (2, 12, 64, 64) and X.dtype == torch.float32
    assert y.shape == (2, 64, 64) and y.dtype == torch.int64
    assert mean.shape == (6,)

    # RGB + índices -> (3 + 3) por fecha = 12 canales
    config = DatasetConfig(data_root=tmp_path, bandas=(2, 1, 0), indices=True,
                           batch_size=2, num_workers=0)
    train_loader, _, mean, _ = get_data_pares(config)
    X, _ = next(iter(train_loader))
    assert X.shape[1] == 12 and mean.shape == (6,)


def test_aumento_de_datos_mueve_fechas_y_mascara_juntas(tmp_path):
    generar_pares_falsos(tmp_path, n=8, tamano=64, semilla=3)
    config = DatasetConfig(data_root=tmp_path, batch_size=1, num_workers=0, augment=True)
    train_loader, _, _, _ = get_data_pares(config)
    rojo_t1, rojo_t2 = 2, 6 + 2     # canal B4 de T1 y de T2

    for _ in range(3):              # varias pasadas, con giros distintos
        for X, y in train_loader:
            nueva = y[0] == LOCACION
            if nueva.any():
                # donde la máscara dice "locación nueva", T2 es más clara que T1
                assert X[0, rojo_t2][nueva].mean() > X[0, rojo_t1][nueva].mean()


def test_fusion_temprana_entrena_con_pares_falsos(tmp_path):
    deterministic(0)
    generar_pares_falsos(tmp_path / "datos", n=16, tamano=64)
    config = DatasetConfig(data_root=tmp_path / "datos", batch_size=4, num_workers=0)
    train_loader, valid_loader, _, _ = get_data_pares(config)

    model = crear_fusion_temprana(canales_por_fecha=6, encoder_weights=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_config = TrainingConfig(epochs=5, checkpoints_dir=tmp_path / "runs")

    history = main(model, optimizer, train_loader, valid_loader, train_config)

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert os.path.exists(tmp_path / "runs" / "best.pt")
