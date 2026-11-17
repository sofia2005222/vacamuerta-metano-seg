import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import pytest

from vmseg.config import DatasetConfig, TrainingConfig, deterministic
from vmseg.dataset_pares import get_data_pares
from vmseg.entrenar import guardar_curvas, main
from vmseg.par import generar_pares_falsos
from vmseg.siamesa import adaptar_primera_capa, crear_modelo, nombres_canales


def test_nombres_canales():
    assert nombres_canales(DatasetConfig()) == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert nombres_canales(DatasetConfig(bandas=(2, 1, 0), indices=True)) == \
        ["B4", "B3", "B2", "NDVI", "NDBI", "BSI"]


def test_primera_capa_usa_el_color_de_cada_banda():
    unet = smp.create_model(arch="unet", encoder_name="resnet34", encoder_weights=None,
                            in_channels=3, classes=3)
    rgb = unet.encoder.conv1.weight.data.clone()       # canales R, G, B

    adaptar_primera_capa(unet.encoder, ["B2", "B3", "B4", "B8"])
    nuevo = unet.encoder.conv1.weight.data

    assert nuevo.shape[1] == 4
    assert torch.equal(nuevo[:, 0], rgb[:, 2])          # B2 (azul) <- B de ImageNet
    assert torch.equal(nuevo[:, 1], rgb[:, 1])          # B3 (verde) <- G
    assert torch.equal(nuevo[:, 2], rgb[:, 0])          # B4 (rojo) <- R
    assert torch.allclose(nuevo[:, 3], rgb.mean(dim=1))  # B8 (NIR) <- promedio


def test_modo_congelado_solo_entrena_la_primera_capa_del_encoder():
    canales = nombres_canales(DatasetConfig())
    modelo = crear_modelo("siamesa", canales, encoder_weights=None, modo="congelado")

    entrenables = [n for n, p in modelo.unet.encoder.named_parameters() if p.requires_grad]
    assert entrenables == ["conv1.weight"]
    assert all(p.requires_grad for p in modelo.unet.decoder.parameters())


def test_siamesa_y_fusion_devuelven_una_mascara():
    canales = nombres_canales(DatasetConfig())
    x = torch.randn(2, 12, 64, 64)
    for arquitectura in ("siamesa", "fusion"):
        modelo = crear_modelo(arquitectura, canales, encoder_weights=None)
        assert modelo(x).shape == (2, 3, 64, 64)


def test_siamesa_aprende_el_cambio_con_pares_falsos(tmp_path):
    deterministic(0)
    generar_pares_falsos(tmp_path / "datos", n=24, tamano=64)
    config = DatasetConfig(data_root=tmp_path / "datos", batch_size=4, num_workers=0)
    train_loader, valid_loader, _, _ = get_data_pares(config)

    modelo = crear_modelo("siamesa", nombres_canales(config), encoder_weights=None)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
    train_config = TrainingConfig(epochs=8, checkpoints_dir=tmp_path / "runs")

    history = main(modelo, optimizer, train_loader, valid_loader, train_config)
    assert history["train_loss"][-1] < history["train_loss"][0]

    # si T1 y T2 son iguales no hay cambio: la predicción tiene que ser casi todo fondo
    X, _ = next(iter(valid_loader))
    sin_cambio = torch.cat([X[:, :6], X[:, :6]], dim=1)
    modelo.eval()
    with torch.no_grad():
        pred = modelo(sin_cambio).argmax(dim=1)
    assert (pred != 0).float().mean() < 0.05

    guardar_curvas(history, tmp_path / "curvas.png")
    assert os.path.exists(tmp_path / "curvas.png")

# los 6 entrenamientos de la tabla de docs/METODO.md: (arquitectura, bandas, índices, modo)
SEIS_ENTRENAMIENTOS = [
    ("siamesa", (0, 1, 2, 3, 4, 5), False, "finetuning"),   # 1. modelo principal
    ("fusion",  (0, 1, 2, 3, 4, 5), False, "finetuning"),   # 2. línea de base (H4)
    ("siamesa", (2, 1, 0),          False, "finetuning"),   # 3. RGB (H1)
    ("siamesa", (0, 1, 2, 3, 4, 5), True,  "finetuning"),   # 4. 6 bandas + índices (H1)
    ("siamesa", (0, 1, 2, 3, 4, 5), False, "cero"),         # 5. desde cero (H2)
    ("siamesa", (0, 1, 2, 3, 4, 5), False, "congelado"),    # 6. encoder congelado (H2)
]


@pytest.mark.parametrize("arquitectura, bandas, indices, modo", SEIS_ENTRENAMIENTOS)
def test_corren_los_seis_entrenamientos(tmp_path, arquitectura, bandas, indices, modo):
    deterministic(0)
    generar_pares_falsos(tmp_path / "datos", n=8, tamano=64)
    config = DatasetConfig(data_root=tmp_path / "datos", bandas=bandas, indices=indices,
                           batch_size=2, num_workers=0)
    train_loader, valid_loader, _, _ = get_data_pares(config)

    modelo = crear_modelo(arquitectura, nombres_canales(config), encoder_weights=None, modo=modo)
    optimizer = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-3)
    train_config = TrainingConfig(epochs=2, checkpoints_dir=tmp_path / "runs")

    history = main(modelo, optimizer, train_loader, valid_loader, train_config)
    guardar_curvas(history, tmp_path / "curvas.png")

    assert len(history["train_loss"]) == 2
    assert os.path.exists(tmp_path / "curvas.png")
