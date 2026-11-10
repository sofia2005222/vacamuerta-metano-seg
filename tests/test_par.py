import numpy as np

from vmseg.par import (
    COLUMNAS_INDICE_PARES, generar_pares_falsos, leer_indice_pares, leer_par,
)
from vmseg.recorte import BANDAS, LOCACION, PILETA, SIN_ETIQUETA


def test_genera_y_lee_pares_falsos(tmp_path):
    indice = generar_pares_falsos(tmp_path, n=4, tamano=64)

    leido = leer_indice_pares(tmp_path)
    assert list(leido.columns) == COLUMNAS_INDICE_PARES
    assert len(leido) == 4

    for id_par in indice["id"]:
        img_t1, img_t2, mask = leer_par(tmp_path, id_par)

        for img in (img_t1, img_t2):
            assert img.shape == (len(BANDAS), 64, 64)
            assert img.dtype == np.float32
            assert img.min() >= 0 and img.max() <= 1

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1, 2, SIN_ETIQUETA}


def test_positivos_tienen_cambio_y_negativos_no(tmp_path):
    indice = generar_pares_falsos(tmp_path, n=12, tamano=64, semilla=1)

    for _, fila in indice.iterrows():
        _, _, mask = leer_par(tmp_path, fila["id"])
        hay_cambio = np.isin(mask, [LOCACION, PILETA]).any()
        assert hay_cambio == (fila["locacion_id"] >= 0)

    # tiene que haber de los dos tipos
    assert (indice["locacion_id"] >= 0).any() and (indice["locacion_id"] < 0).any()


def test_lo_viejo_es_fondo_y_lo_nuevo_no_estaba_en_t1(tmp_path):
    indice = generar_pares_falsos(tmp_path, n=6, tamano=64, semilla=2)
    rojo = BANDAS.index("B4")

    for id_par in indice["id"]:
        img_t1, img_t2, mask = leer_par(tmp_path, id_par)

        # locaciones viejas: en T1 hay píxeles claros (locación) y la máscara ahí es fondo
        vieja = img_t1[rojo] > 0.22
        assert vieja.any()
        assert (mask[vieja] == 0).all()

        # locaciones nuevas: en T1 ahí había suelo (oscuro) y en T2 hay locación (clara)
        nueva = mask == LOCACION
        if nueva.any():
            assert img_t1[rojo][nueva].mean() < 0.18
            assert img_t2[rojo][nueva].mean() > 0.2
