import numpy as np

from vmseg.recorte import (
    BANDAS, COLUMNAS_INDICE, SIN_ETIQUETA, SPLITS,
    generar_recortes_falsos, leer_indice, leer_recorte,
)


def test_genera_y_lee_recortes_falsos(tmp_path):
    indice = generar_recortes_falsos(tmp_path, n=4, tamano=64)

    # el índice tiene las columnas del formato y un split válido
    leido = leer_indice(tmp_path)
    assert list(leido.columns) == COLUMNAS_INDICE
    assert len(leido) == 4
    assert set(leido["split"]) <= set(SPLITS)

    for id_recorte in indice["id"]:
        img, mask = leer_recorte(tmp_path, id_recorte)

        assert img.shape == (len(BANDAS), 64, 64)
        assert img.dtype == np.float32
        assert img.min() >= 0 and img.max() <= 1

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1, 2, SIN_ETIQUETA}


def test_recorte_falso_tiene_locaciones(tmp_path):
    indice = generar_recortes_falsos(tmp_path, n=1, tamano=64, semilla=0)
    _, mask = leer_recorte(tmp_path, indice["id"][0])
    assert (mask == 1).any()
