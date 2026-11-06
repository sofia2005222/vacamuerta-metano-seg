"""
Formato de un recorte (ver docs/FORMATO_RECORTE.md).

Cada recorte son dos GeoTIFF en la misma grilla:
    <id>.tif       imagen: 6 bandas float32 con reflectancia en [0, 1]
    <id>_mask.tif  máscara: 1 banda uint8 (0 fondo, 1 locación, 2 pileta, 255 sin etiqueta)
y una fila en indice.csv con sus metadatos.
"""
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# bandas de Sentinel-2, en este orden
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# clases de la máscara
FONDO = 0
LOCACION = 1
PILETA = 2
SIN_ETIQUETA = 255
CLASES = ["fondo", "locacion", "pileta"]

TAMANO = 256          # píxeles por lado
RESOLUCION = 10       # metros por píxel
CRS = "EPSG:32719"    # UTM 19S

SPLITS = ["train", "val", "test_anelo", "test_externo"]

COLUMNAS_INDICE = [
    "id", "zona", "bloque", "split", "mes",
    "x_min", "y_max", "crs", "validada", "nota",
]


def guardar_recorte(carpeta, id_recorte, img, mask, x_min, y_max, crs=CRS):
    """
    Guarda la imagen y la máscara de un recorte como GeoTIFF.

    img:  array (6, H, W) float32 con reflectancia en [0, 1]
    mask: array (H, W) uint8
    x_min, y_max: esquina superior izquierda, en metros del CRS
    """
    assert img.ndim == 3 and img.shape[0] == len(BANDAS)
    assert mask.shape == img.shape[1:]

    os.makedirs(carpeta, exist_ok=True)
    transform = from_origin(x_min, y_max, RESOLUCION, RESOLUCION)
    alto, ancho = mask.shape

    perfil = dict(driver="GTiff", height=alto, width=ancho,
                  crs=crs, transform=transform, compress="deflate")

    ruta_img = os.path.join(carpeta, f"{id_recorte}.tif")
    with rasterio.open(ruta_img, "w", count=len(BANDAS), dtype="float32", **perfil) as dst:
        dst.write(img.astype(np.float32))
        dst.descriptions = tuple(BANDAS)

    ruta_mask = os.path.join(carpeta, f"{id_recorte}_mask.tif")
    with rasterio.open(ruta_mask, "w", count=1, dtype="uint8", **perfil) as dst:
        dst.write(mask.astype(np.uint8), 1)


def leer_recorte(carpeta, id_recorte):
    """
    Lee un recorte. Devuelve (img, mask):
        img:  array (6, H, W) float32
        mask: array (H, W) uint8
    """
    with rasterio.open(os.path.join(carpeta, f"{id_recorte}.tif")) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(os.path.join(carpeta, f"{id_recorte}_mask.tif")) as src:
        mask = src.read(1)
    return img, mask


def leer_indice(carpeta):
    """Lee indice.csv de una carpeta de recortes."""
    return pd.read_csv(os.path.join(carpeta, "indice.csv"), dtype={"nota": str})


def generar_recortes_falsos(carpeta, n=8, tamano=TAMANO, semilla=42):
    """
    Genera n recortes falsos con el formato real, para probar sin imágenes.

    El fondo es suelo con ruido; las locaciones son rectángulos claros
    y las piletas, círculos oscuros (agua: baja reflectancia en NIR y SWIR).
    Devuelve el índice como DataFrame y lo guarda en carpeta/indice.csv.
    """
    rng = np.random.default_rng(semilla)

    # reflectancia típica por banda (B2, B3, B4, B8, B11, B12)
    suelo = np.array([0.08, 0.11, 0.14, 0.20, 0.28, 0.22])
    locacion = np.array([0.18, 0.22, 0.26, 0.30, 0.38, 0.33])
    agua = np.array([0.06, 0.07, 0.05, 0.03, 0.01, 0.01])

    filas_y, columnas_x = np.mgrid[0:tamano, 0:tamano]
    filas = []

    for i in range(n):
        img = suelo[:, None, None] + rng.normal(0, 0.02, (len(BANDAS), tamano, tamano))
        mask = np.full((tamano, tamano), FONDO, dtype=np.uint8)

        for _ in range(rng.integers(1, 4)):
            # locación: rectángulo de 5 a 25 píxeles de lado (50 a 250 m)
            alto, ancho = rng.integers(5, 26, size=2)
            y0 = rng.integers(0, tamano - alto)
            x0 = rng.integers(0, tamano - ancho)
            img[:, y0:y0 + alto, x0:x0 + ancho] = locacion[:, None, None]
            mask[y0:y0 + alto, x0:x0 + ancho] = LOCACION

            # a veces, una pileta adentro de la locación
            if rng.random() < 0.5:
                radio = max(2, min(alto, ancho) // 4)
                cy, cx = y0 + alto // 2, x0 + ancho // 2
                circulo = (filas_y - cy) ** 2 + (columnas_x - cx) ** 2 <= radio ** 2
                img[:, circulo] = agua[:, None]
                mask[circulo] = PILETA

        # algunos píxeles sin etiqueta, para probar el ignore_index
        mask[:2, :] = SIN_ETIQUETA

        img = np.clip(img, 0, 1).astype(np.float32)

        split = SPLITS[i % len(SPLITS)]
        zona = "externa" if split == "test_externo" else "anelo"
        bloque = -1 if zona == "externa" else int(rng.integers(0, 16))
        id_recorte = f"falso_{i:04d}"
        x_min, y_max = 330000 + i * tamano * RESOLUCION, 5770000

        guardar_recorte(carpeta, id_recorte, img, mask, x_min, y_max)
        filas.append(dict(
            id=id_recorte, zona=zona, bloque=bloque, split=split, mes="2021-03",
            x_min=x_min, y_max=y_max, crs=CRS, validada=False, nota="falso",
        ))

    indice = pd.DataFrame(filas, columns=COLUMNAS_INDICE)
    indice.to_csv(os.path.join(carpeta, "indice.csv"), index=False)
    return indice
