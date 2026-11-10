"""
Formato de un par (ver docs/FORMATO_RECORTE.md y ADR-0011).

Cada par son tres GeoTIFF en la misma grilla:
    <id>_t1.tif    recorte antes (T1): 6 bandas float32 con reflectancia en [0, 1]
    <id>_t2.tif    recorte después (T2): igual que T1
    <id>_mask.tif  máscara de lo que apareció entre T1 y T2:
                   0 fondo, 1 locación, 2 pileta, 255 sin etiqueta
y una fila en indice.csv con sus metadatos.
"""
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from vmseg.recorte import (
    BANDAS, CRS, FONDO, LOCACION, PILETA, RESOLUCION, SIN_ETIQUETA, SPLITS, TAMANO,
)

COLUMNAS_INDICE_PARES = [
    "id", "encuadre", "locacion_id", "zona", "bloque", "split",
    "mes_t1", "mes_t2", "x_min", "y_max", "crs", "validada", "nota",
]

def guardar_tif(ruta, datos, x_min, y_max, crs, dtype):
    """Guarda un array (bandas, H, W) como GeoTIFF con la grilla del par."""
    bandas, alto, ancho = datos.shape
    transform = from_origin(x_min, y_max, RESOLUCION, RESOLUCION)
    with rasterio.open(ruta, "w", driver="GTiff", height=alto, width=ancho, count=bandas,
                       dtype=dtype, crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(datos.astype(dtype))


def guardar_par(carpeta, id_par, img_t1, img_t2, mask, x_min, y_max, crs=CRS):
    """
    Guarda los dos recortes y la máscara de un par como GeoTIFF.

    img_t1, img_t2: arrays (6, H, W) float32 con reflectancia en [0, 1]
    mask: array (H, W) uint8 con lo que apareció entre T1 y T2
    x_min, y_max: esquina superior izquierda, en metros del CRS
    """
    assert img_t1.shape == img_t2.shape and img_t1.shape[0] == len(BANDAS)
    assert mask.shape == img_t1.shape[1:]

    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, id_par)
    guardar_tif(f"{base}_t1.tif", img_t1, x_min, y_max, crs, "float32")
    guardar_tif(f"{base}_t2.tif", img_t2, x_min, y_max, crs, "float32")
    guardar_tif(f"{base}_mask.tif", mask[None], x_min, y_max, crs, "uint8")


def leer_par(carpeta, id_par):
    """
    Lee un par. Devuelve (img_t1, img_t2, mask):
        img_t1, img_t2: arrays (6, H, W) float32
        mask: array (H, W) uint8
    """
    base = os.path.join(carpeta, id_par)
    with rasterio.open(f"{base}_t1.tif") as src:
        img_t1 = src.read().astype(np.float32)
    with rasterio.open(f"{base}_t2.tif") as src:
        img_t2 = src.read().astype(np.float32)
    with rasterio.open(f"{base}_mask.tif") as src:
        mask = src.read(1)
    return img_t1, img_t2, mask


def leer_indice_pares(carpeta):
    """Lee indice.csv de una carpeta de pares."""
    return pd.read_csv(os.path.join(carpeta, "indice.csv"), dtype={"nota": str})

# reflectancia típica por banda (B2, B3, B4, B8, B11, B12) para los pares falsos
SUELO = np.array([0.08, 0.11, 0.14, 0.20, 0.28, 0.22])
LOCACION_REFL = np.array([0.18, 0.22, 0.26, 0.30, 0.38, 0.33])
AGUA = np.array([0.06, 0.07, 0.05, 0.03, 0.01, 0.01])


def rectangulo_libre(rng, ocupado, tamano):
    """Elige un rectángulo de 5 a 25 píxeles de lado que no pise lo ya dibujado."""
    for _ in range(50):
        alto, ancho = rng.integers(5, 26, size=2)
        y0 = rng.integers(2, tamano - alto)
        x0 = rng.integers(0, tamano - ancho)
        if not ocupado[y0:y0 + alto, x0:x0 + ancho].any():
            return y0, x0, alto, ancho
    return None


def generar_pares_falsos(carpeta, n=8, tamano=TAMANO, semilla=42, frac_negativos=0.25):
    """
    Genera n pares falsos con el formato real, para probar sin imágenes.

    En los dos recortes hay suelo con ruido y alguna locación vieja (está en T1 y en T2,
    así que en la máscara es fondo). En los pares positivos, T2 además tiene locaciones
    nuevas (rectángulos claros), a veces con una pileta (círculo oscuro): eso es lo que
    marca la máscara. Los negativos no tienen nada nuevo. T2 tiene un brillo un poco
    distinto, como si fuera otra estación.
    Devuelve el índice como DataFrame y lo guarda en carpeta/indice.csv.
    """
    rng = np.random.default_rng(semilla)
    filas_y, columnas_x = np.mgrid[0:tamano, 0:tamano]
    filas = []

    for i in range(n):
        # el mismo terreno en las dos fechas, con un poco de ruido distinto en cada una
        terreno = SUELO[:, None, None] + rng.normal(0, 0.02, (len(BANDAS), tamano, tamano))
        img_t1 = terreno + rng.normal(0, 0.005, terreno.shape)
        img_t2 = terreno + rng.normal(0, 0.005, terreno.shape)
        mask = np.full((tamano, tamano), FONDO, dtype=np.uint8)
        ocupado = np.zeros((tamano, tamano), dtype=bool)

        # locaciones viejas: están en T1 y en T2, no son cambio
        for _ in range(rng.integers(1, 3)):
            rect = rectangulo_libre(rng, ocupado, tamano)
            if rect is None:
                continue
            y0, x0, alto, ancho = rect
            img_t1[:, y0:y0 + alto, x0:x0 + ancho] = LOCACION_REFL[:, None, None]
            img_t2[:, y0:y0 + alto, x0:x0 + ancho] = LOCACION_REFL[:, None, None]
            ocupado[y0:y0 + alto, x0:x0 + ancho] = True

        # locaciones nuevas: solo en T2 (los negativos no tienen)
        negativo = rng.random() < frac_negativos
        nuevas = 0 if negativo else rng.integers(1, 3)
        for _ in range(nuevas):
            rect = rectangulo_libre(rng, ocupado, tamano)
            if rect is None:
                continue
            y0, x0, alto, ancho = rect
            img_t2[:, y0:y0 + alto, x0:x0 + ancho] = LOCACION_REFL[:, None, None]
            mask[y0:y0 + alto, x0:x0 + ancho] = LOCACION
            ocupado[y0:y0 + alto, x0:x0 + ancho] = True

            # a veces, una pileta nueva adentro de la locación nueva
            if rng.random() < 0.5:
                radio = max(2, min(alto, ancho) // 4)
                cy, cx = y0 + alto // 2, x0 + ancho // 2
                circulo = (filas_y - cy) ** 2 + (columnas_x - cx) ** 2 <= radio ** 2
                img_t2[:, circulo] = AGUA[:, None]
                mask[circulo] = PILETA

        # T2 de otra estación: un poco más claro u oscuro en todas las bandas
        img_t2 = img_t2 * rng.uniform(0.9, 1.1)

        # algunos píxeles sin etiqueta, para probar el ignore_index
        mask[:2, :] = SIN_ETIQUETA

        img_t1 = np.clip(img_t1, 0, 1).astype(np.float32)
        img_t2 = np.clip(img_t2, 0, 1).astype(np.float32)

        split = SPLITS[i % len(SPLITS)]
        zona = "externa" if split == "test_externo" else "anelo"
        bloque = -1 if zona == "externa" else int(rng.integers(0, 16))
        encuadre = f"{zona}_b{bloque:02d}_e{i % 9}"
        locacion_id = -1 if (negativo or not (mask == LOCACION).any()) else 1000 + i
        id_par = f"falso_{i:04d}"
        x_min, y_max = 330000 + i * tamano * RESOLUCION, 5770000

        guardar_par(carpeta, id_par, img_t1, img_t2, mask, x_min, y_max)
        filas.append(dict(
            id=id_par, encuadre=encuadre, locacion_id=locacion_id, zona=zona,
            bloque=bloque, split=split, mes_t1="2021-01", mes_t2="2021-08",
            x_min=x_min, y_max=y_max, crs=CRS, validada=False, nota="falso",
        ))

    indice = pd.DataFrame(filas, columns=COLUMNAS_INDICE_PARES)
    indice.to_csv(os.path.join(carpeta, "indice.csv"), index=False)
    return indice
