"""
Issue #11: bloques de cada zona y grilla fija de 4×4 encuadres por bloque.

Un encuadre es un cuadrado de 256×256 px (2,56 km) sobre el terreno, contenido
entero en un bloque y sin fecha (ver GLOSSARY.md y docs/METODO.md).

Los bloques se definen en grados (como en data/armar_dataset.py) y las imágenes
están en UTM 19S, así que el borde de un bloque no cae justo sobre la grilla de
píxeles. Para cada bloque se toma el rectángulo UTM más grande que entra adentro,
se ajusta hacia adentro a la grilla de 10 m de las imágenes de GEE, y se reparten
los 4 encuadres de cada lado de punta a punta, solapados entre sí (60–90 px).
Nunca salen del bloque.

Por qué 4×4 y no 3×3: un bloque mide ~830 px de alto y 3 encuadres ocupan 768,
así que quedaban franjas de ~300 m sin cubrir; 20 de las 172 locaciones nuevas
caían ahí y no tenían encuadre. Con 4×4 todas tienen uno. El solapamiento es
dentro del bloque (mismo split), así que no hay fuga.
"""
import json
import math

import numpy as np
import pandas as pd
from rasterio.warp import transform

from vmseg import gee
from vmseg.recorte import CRS, RESOLUCION, TAMANO

# (columnas, filas) de la grilla de bloques. En la zona externa los bloques
# solo sirven para repartir el etiquetado; con 4 columnas un bloque mediría
# 656 px de ancho, y con 3 mide ~870.
GRILLA_BLOQUES = {"anelo": (4, 4), "externa": (3, 4)}
POR_LADO = 4  # encuadres por lado de bloque
PUNTOS_POR_BORDE = 21  # para seguir la curvatura del borde al pasarlo a UTM


def _zonas():
    with open(gee.RUTA_ZONAS, encoding="utf-8") as f:
        return json.load(f)


def bloques(zona):
    """{bloque: (lon_min, lat_min, lon_max, lat_max)}. Bloque = fila * columnas + columna, desde el suroeste."""
    lon_min, lat_min, lon_max, lat_max = gee.recuadro(zona)
    ncol, nfil = GRILLA_BLOQUES[zona]
    dlon, dlat = (lon_max - lon_min) / ncol, (lat_max - lat_min) / nfil
    return {
        f * ncol + c: (lon_min + c * dlon, lat_min + f * dlat, lon_min + (c + 1) * dlon, lat_min + (f + 1) * dlat)
        for f in range(nfil)
        for c in range(ncol)
    }


def bloque_de(zona, lon, lat):
    """Bloque de cada punto (misma cuenta que data/armar_dataset.py)."""
    lon_min, lat_min, lon_max, lat_max = gee.recuadro(zona)
    ncol, nfil = GRILLA_BLOQUES[zona]
    c = np.clip(((np.asarray(lon) - lon_min) / (lon_max - lon_min) * ncol).astype(int), 0, ncol - 1)
    f = np.clip(((np.asarray(lat) - lat_min) / (lat_max - lat_min) * nfil).astype(int), 0, nfil - 1)
    return f * ncol + c


def split_de_bloque(zona, bloque):
    """Split de un bloque. El bloque 0 de Añelo no tiene locaciones: sus negativos van a train."""
    if zona == "externa":
        return "test_externo"
    return _zonas()["split_por_bloque"].get(str(bloque), "train")


def rectangulo_interior(lon_min, lat_min, lon_max, lat_max):
    """Rectángulo UTM (x_min, y_min, x_max, y_max) más grande dentro de la caja en grados, en la grilla de 10 m."""
    t = np.linspace(0, 1, PUNTOS_POR_BORDE)
    lons, lats = lon_min + t * (lon_max - lon_min), lat_min + t * (lat_max - lat_min)

    def utm(lon, lat):
        return transform("EPSG:4326", CRS, list(lon), list(lat))

    x_oeste, _ = utm(np.full_like(lats, lon_min), lats)
    x_este, _ = utm(np.full_like(lats, lon_max), lats)
    _, y_sur = utm(lons, np.full_like(lons, lat_min))
    _, y_norte = utm(lons, np.full_like(lons, lat_max))
    r = RESOLUCION
    return (
        math.ceil(max(x_oeste) / r) * r,
        math.ceil(max(y_sur) / r) * r,
        math.floor(min(x_este) / r) * r,
        math.floor(min(y_norte) / r) * r,
    )


def posiciones(desde, hasta):
    """Comienzo de los POR_LADO encuadres repartidos de punta a punta entre desde y hasta (en metros)."""
    sobra = (hasta - desde) // RESOLUCION - TAMANO  # en píxeles
    assert sobra >= 0, "el bloque es más chico que un encuadre"
    return [desde + RESOLUCION * round(i * sobra / (POR_LADO - 1)) for i in range(POR_LADO)]


def encuadres(zona):
    """
    Tabla de encuadres de una zona: encuadre, zona, bloque, split, x_min, y_max.

    Dentro de cada bloque, el encuadre e0 es el de arriba a la izquierda y e15 el de
    abajo a la derecha (e = fila * 4 + columna, filas de norte a sur, como en la imagen).
    """
    filas = []
    for bloque, caja in bloques(zona).items():
        x_min, y_min, x_max, y_max = rectangulo_interior(*caja)
        xs = posiciones(x_min, x_max)
        # de norte a sur: el borde superior baja desde y_max
        ys = [y_max - (y - y_min) for y in posiciones(y_min, y_max)]
        for f, y in enumerate(ys):
            for c, x in enumerate(xs):
                filas.append(dict(
                    encuadre=f"{zona}_b{bloque:02d}_e{f * POR_LADO + c}",
                    zona=zona, bloque=bloque, split=split_de_bloque(zona, bloque),
                    x_min=int(x), y_max=int(y),
                ))
    return pd.DataFrame(filas)
