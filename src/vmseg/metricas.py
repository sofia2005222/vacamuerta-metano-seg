"""
Métricas de evaluación del modelo contra las máscaras de los pares (issue #17, docs/EVALUACION.md).

La máscara de un par marca lo que **apareció entre T1 y T2** (ADR-0011): 0 fondo, 1 locación nueva,
2 pileta nueva, 255 sin etiqueta. Las métricas comparan esa máscara (la "real") con la predicción
del modelo, a dos niveles:

- **Por píxel:** matriz de confusión, IoU y F1 por clase. Cuentan cuántos píxeles coinciden.
- **Por objeto** (Ramachandran et al., 2024): cada componente conexo de una clase es un objeto.
  Un objeto predicho y uno real coinciden si su IoU es ≥ 0,3, con emparejamiento greedy
  (primero los pares con más IoU; cada objeto se usa una sola vez). Los pares que coinciden son
  verdaderos positivos (VP), los predichos sin pareja falsos positivos (FP) y los reales sin
  pareja falsos negativos (FN). Con eso: precisión = VP / (VP + FP), recall = VP / (VP + FN).
  Responden "¿encontró las locaciones?" sin castigar diferencias de un píxel en el borde.

Los píxeles sin etiqueta (255) no cuentan en ninguna métrica. Todo se puede acumular sobre muchos
pares: se suman matrices y conteos, y las métricas se calculan al final (`evaluar`).
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from vmseg.recorte import CLASES, LOCACION, PILETA, SIN_ETIQUETA

UMBRAL_IOU = 0.3
CLASES_OBJETO = (LOCACION, PILETA)
# vecindad de 8 (incluye diagonales): dos píxeles que se tocan por una esquina son el mismo objeto
CONECTIVIDAD = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------- por píxel

def matriz_confusion(real, pred, n_clases=len(CLASES), ignorar=SIN_ETIQUETA):
    """
    Matriz de confusión de n_clases × n_clases: fila = clase real, columna = clase predicha.
    m[1, 0] es cuántos píxeles de locación se predijeron como fondo. Los píxeles con
    `ignorar` en la máscara real no cuentan.
    """
    real = np.asarray(real).ravel()
    pred = np.asarray(pred).ravel()
    validos = real != ignorar
    real, pred = real[validos].astype(np.int64), pred[validos].astype(np.int64)
    if real.size and (pred.max() >= n_clases or real.max() >= n_clases):
        raise ValueError("hay clases fuera de rango en la máscara o en la predicción")
    return np.bincount(real * n_clases + pred, minlength=n_clases ** 2).reshape(n_clases, n_clases)


def _dividir(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def iou_por_clase(matriz):
    """IoU de cada clase = VP / (VP + FP + FN). NaN si la clase no aparece ni en la real ni en la predicha."""
    vp = np.diag(matriz).astype(float)
    fp = matriz.sum(axis=0) - vp
    fn = matriz.sum(axis=1) - vp
    return _dividir(vp, vp + fp + fn)


def f1_por_clase(matriz):
    """F1 de cada clase = 2·VP / (2·VP + FP + FN) (equivale al coeficiente de Dice)."""
    vp = np.diag(matriz).astype(float)
    fp = matriz.sum(axis=0) - vp
    fn = matriz.sum(axis=1) - vp
    return _dividir(2 * vp, 2 * vp + fp + fn)


# ---------------------------------------------------------------- por objeto

def objetos(mask, clase):
    """Componentes conexos de `clase` en `mask`. Devuelve (etiquetas, n): 0 = fondo, 1..n = objetos."""
    return ndimage.label(np.asarray(mask) == clase, structure=CONECTIVIDAD)


def emparejar(etiquetas_real, n_real, etiquetas_pred, n_pred, umbral=UMBRAL_IOU):
    """
    Emparejamiento greedy entre objetos reales y predichos por IoU.

    Devuelve una lista de (objeto_real, objeto_pred, iou) con los pares que coinciden (IoU ≥ umbral).
    """
    if n_real == 0 or n_pred == 0:
        return []
    # intersección de cada par (real, pred) en una sola pasada
    ambos = (etiquetas_real > 0) & (etiquetas_pred > 0)
    inter = np.zeros((n_real + 1, n_pred + 1), dtype=np.int64)
    np.add.at(inter, (etiquetas_real[ambos], etiquetas_pred[ambos]), 1)
    area_real = np.bincount(etiquetas_real.ravel(), minlength=n_real + 1)
    area_pred = np.bincount(etiquetas_pred.ravel(), minlength=n_pred + 1)
    union = area_real[:, None] + area_pred[None, :] - inter
    iou = _dividir(inter.astype(float), union.astype(float))
    iou[0, :] = iou[:, 0] = 0

    candidatos = np.argwhere(np.nan_to_num(iou) >= umbral)
    candidatos = sorted(candidatos, key=lambda rp: -iou[rp[0], rp[1]])
    usados_real, usados_pred, pares = set(), set(), []
    for r, p in candidatos:
        if r in usados_real or p in usados_pred:
            continue
        usados_real.add(int(r)); usados_pred.add(int(p))
        pares.append((int(r), int(p), float(iou[r, p])))
    return pares


def conteo_objetos(real, pred, clase, umbral=UMBRAL_IOU, ignorar=SIN_ETIQUETA):
    """
    VP, FP y FN por objeto de una clase en un par. Lo predicho sobre píxeles sin etiqueta se descarta
    (no hay con qué compararlo).
    Devuelve un dict {"vp", "fp", "fn"}.
    """
    real = np.asarray(real)
    pred = np.where(real == ignorar, 0, np.asarray(pred))
    et_real, n_real = objetos(real, clase)
    et_pred, n_pred = objetos(pred, clase)
    vp = len(emparejar(et_real, n_real, et_pred, n_pred, umbral))
    return {"vp": vp, "fp": n_pred - vp, "fn": n_real - vp}


def precision_recall(conteo):
    """Precisión y recall a partir de un conteo {"vp", "fp", "fn"} (NaN si no hay con qué calcular)."""
    vp, fp, fn = conteo["vp"], conteo["fp"], conteo["fn"]
    precision = vp / (vp + fp) if vp + fp else float("nan")
    recall = vp / (vp + fn) if vp + fn else float("nan")
    return precision, recall


# ---------------------------------------------------------------- todo junto

def evaluar(pares, umbral=UMBRAL_IOU):
    """
    Evalúa una colección de pares.

    pares: iterable de (real, pred), máscaras (H, W) con valores 0/1/2 (la real puede tener 255).
    Devuelve (tabla, matriz):
        tabla: DataFrame con una fila por clase (fondo, locacion, pileta) y columnas
               iou, f1 (por píxel) y vp, fp, fn, precision, recall (por objeto; NaN para fondo).
        matriz: matriz de confusión por píxel sumada sobre todos los pares.
    """
    matriz = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    conteos = {c: {"vp": 0, "fp": 0, "fn": 0} for c in CLASES_OBJETO}
    for real, pred in pares:
        matriz += matriz_confusion(real, pred)
        for c in CLASES_OBJETO:
            for k, v in conteo_objetos(real, pred, c, umbral).items():
                conteos[c][k] += v

    filas = []
    for i, nombre in enumerate(CLASES):
        fila = {"clase": nombre, "iou": iou_por_clase(matriz)[i], "f1": f1_por_clase(matriz)[i],
                "vp": np.nan, "fp": np.nan, "fn": np.nan, "precision": np.nan, "recall": np.nan}
        if i in conteos:
            precision, recall = precision_recall(conteos[i])
            fila.update(conteos[i], precision=precision, recall=recall)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("clase"), matriz
