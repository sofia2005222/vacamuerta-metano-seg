import numpy as np
import pytest

from vmseg.metricas import (
    conteo_objetos, emparejar, evaluar, f1_por_clase, iou_por_clase, matriz_confusion, objetos,
    precision_recall,
)
from vmseg.par import generar_pares_falsos, leer_indice_pares, leer_par
from vmseg.recorte import FONDO, LOCACION, PILETA, SIN_ETIQUETA


def _vacia(n=20):
    return np.zeros((n, n), dtype=np.uint8)


# ---------------------------------------------------------------- por píxel

def test_prediccion_perfecta():
    real = _vacia()
    real[2:8, 2:8] = LOCACION
    real[4:6, 4:6] = PILETA
    m = matriz_confusion(real, real)
    assert m.sum() == 400 and np.array_equal(m, np.diag(np.diag(m)))
    assert np.allclose(iou_por_clase(m), 1) and np.allclose(f1_por_clase(m), 1)


def test_matriz_cuenta_cada_confusion():
    real = np.array([[0, 1, 1, 2]])
    pred = np.array([[0, 1, 0, 1]])
    m = matriz_confusion(real, pred)
    assert m[0, 0] == 1          # fondo bien
    assert m[1, 1] == 1          # locación bien
    assert m[1, 0] == 1          # locación predicha como fondo
    assert m[2, 1] == 1          # pileta predicha como locación


def test_sin_etiqueta_no_cuenta():
    real = np.array([[1, 1, SIN_ETIQUETA, SIN_ETIQUETA]])
    pred = np.array([[1, 1, 0, 2]])
    m = matriz_confusion(real, pred)
    assert m.sum() == 2 and m[1, 1] == 2


def test_iou_y_f1_calculados_a_mano():
    # locación real: 10 px; predicha: 10 px corridos 5 → se pisan 5
    real, pred = _vacia(), _vacia()
    real[0, 0:10] = LOCACION
    pred[0, 5:15] = LOCACION
    m = matriz_confusion(real, pred)
    # VP = 5, FP = 5, FN = 5 → IoU = 5/15, F1 = 10/20
    assert iou_por_clase(m)[LOCACION] == pytest.approx(1 / 3)
    assert f1_por_clase(m)[LOCACION] == pytest.approx(0.5)


def test_clase_ausente_da_nan():
    real = pred = _vacia()
    assert np.isnan(iou_por_clase(matriz_confusion(real, pred))[PILETA])


def test_clase_fuera_de_rango_es_error():
    with pytest.raises(ValueError):
        matriz_confusion(np.array([[0]]), np.array([[7]]))


# ---------------------------------------------------------------- por objeto

def test_objetos_con_vecindad_de_8():
    m = _vacia()
    m[0, 0] = m[1, 1] = LOCACION        # se tocan por la esquina: un solo objeto
    m[10:12, 10:12] = LOCACION          # otro objeto
    _, n = objetos(m, LOCACION)
    assert n == 2


def test_emparejamiento_greedy_usa_cada_objeto_una_vez():
    # un objeto predicho grande pisa dos reales: solo se empareja con el de mayor IoU
    real, pred = _vacia(), _vacia()
    real[0:4, 0:4] = LOCACION           # 16 px
    real[0:4, 6:8] = LOCACION           # 8 px
    pred[0:4, 0:6] = LOCACION           # 24 px: pisa todo el primero
    et_r, n_r = objetos(real, LOCACION)
    et_p, n_p = objetos(pred, LOCACION)
    pares = emparejar(et_r, n_r, et_p, n_p)
    assert len(pares) == 1 and pares[0][2] == pytest.approx(16 / 24)


def test_umbral_de_iou():
    real, pred = _vacia(), _vacia()
    real[0, 0:10] = LOCACION
    pred[0, 7:17] = LOCACION            # IoU = 3 / 17 < 0,3
    assert conteo_objetos(real, pred, LOCACION) == {"vp": 0, "fp": 1, "fn": 1}
    pred[:] = 0
    pred[0, 4:14] = LOCACION            # IoU = 6 / 14 ≥ 0,3
    assert conteo_objetos(real, pred, LOCACION) == {"vp": 1, "fp": 0, "fn": 0}


def test_conteo_vp_fp_fn():
    real, pred = _vacia(), _vacia()
    real[0:4, 0:4] = LOCACION           # encontrada
    real[10:14, 10:14] = LOCACION       # no encontrada → FN
    pred[0:4, 0:4] = LOCACION
    pred[15:18, 0:3] = LOCACION         # inventada → FP
    c = conteo_objetos(real, pred, LOCACION)
    assert c == {"vp": 1, "fp": 1, "fn": 1}
    assert precision_recall(c) == (0.5, 0.5)


def test_lo_predicho_sobre_sin_etiqueta_no_es_falso_positivo():
    real, pred = _vacia(), _vacia()
    real[0:2, :] = SIN_ETIQUETA
    pred[0:2, 0:4] = LOCACION
    assert conteo_objetos(real, pred, LOCACION) == {"vp": 0, "fp": 0, "fn": 0}


def test_negativo_sin_nada_da_nan():
    assert all(np.isnan(precision_recall({"vp": 0, "fp": 0, "fn": 0})))


# ---------------------------------------------------------------- evaluar

def test_evaluar_suma_pares():
    real1, pred1 = _vacia(), _vacia()
    real1[0:4, 0:4] = LOCACION
    pred1[0:4, 0:4] = LOCACION
    real2, pred2 = _vacia(), _vacia()
    real2[0:4, 0:4] = LOCACION          # el modelo no la encuentra
    tabla, matriz = evaluar([(real1, pred1), (real2, pred2)])
    assert matriz[LOCACION, LOCACION] == 16 and matriz[LOCACION, FONDO] == 16
    fila = tabla.loc["locacion"]
    assert (fila.vp, fila.fp, fila.fn) == (1, 0, 1)
    assert fila.recall == 0.5 and fila.precision == 1.0
    assert fila.iou == pytest.approx(0.5)
    assert np.isnan(tabla.loc["fondo", "precision"])


def test_evaluar_con_pares_falsos(tmp_path):
    """Con pares falsos: la máscara contra sí misma da todo perfecto; contra todo fondo, recall 0."""
    indice = generar_pares_falsos(tmp_path, n=6, tamano=64, semilla=0)
    mascaras = [leer_par(tmp_path, i)[2] for i in leer_indice_pares(tmp_path)["id"]]

    perfecta, _ = evaluar([(m, np.where(m == SIN_ETIQUETA, 0, m)) for m in mascaras])
    assert perfecta.loc["locacion", "iou"] == pytest.approx(1)
    assert perfecta.loc["locacion", "recall"] == pytest.approx(1)

    nada, _ = evaluar([(m, np.zeros_like(m)) for m in mascaras])
    assert nada.loc["locacion", "recall"] == 0
    assert nada.loc["locacion", "iou"] == 0
    assert len(indice) == 6
