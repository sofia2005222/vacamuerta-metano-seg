"""
Acceso a Sentinel-2 y Cloud Score+ en Google Earth Engine (ADR-0010).

Requiere el extra `gee` (`pip install -e .[gee]`) y una cuenta de GEE con un
proyecto de Cloud registrado. Solo lo usa quien genera las imágenes mensuales.
"""
import json
import os

import pandas as pd

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

BANDA_NUBES = "cs_cdf"
UMBRAL_NUBES = 0.60
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BUCKET = "vmseg-imagenes"
CRS = "EPSG:32719"   # UTM 19 sur, el del mosaico T19HET

# nombre de la zona -> clave en data/zonas_estudio.json
ZONAS = {"anelo": "anelo_train_val_test", "externa": "test_externo"}

RUTA_ZONAS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "zonas_estudio.json"
)


def recuadro(zona, ruta=RUTA_ZONAS):
    """Recuadro (lon_min, lat_min, lon_max, lat_max) de una zona de estudio."""
    with open(ruta, encoding="utf-8") as f:
        z = json.load(f)[ZONAS[zona]]
    return z["lon_min"], z["lat_min"], z["lon_max"], z["lat_max"]


def inicializar(proyecto=None):
    """Inicializa Earth Engine con el proyecto de Cloud (o $EE_PROJECT)."""
    import ee

    ee.Initialize(project=proyecto or os.environ.get("EE_PROJECT"))
    return ee


def ids_escenas(ee, coleccion, zona, inicio, fin):
    """`system:index` de las escenas de una colección que tocan la zona en [inicio, fin)."""
    geom = ee.Geometry.Rectangle(list(recuadro(zona)))
    col = ee.ImageCollection(coleccion).filterBounds(geom).filterDate(inicio, fin)
    return col.aggregate_array("system:index").getInfo()


def mes_de_escena(id_escena):
    """'20190105T143751_20190105T144421_T19HEB' -> '2019-01'."""
    return f"{id_escena[:4]}-{id_escena[4:6]}"


def tabla_cobertura(ids_s2, ids_cs, inicio, fin):
    """
    Cuenta por mes las escenas de Sentinel-2 y cuántas tienen Cloud Score+.

    Cloud Score+ usa el mismo `system:index` que la escena de Sentinel-2.
    Devuelve una fila por mes en [inicio, fin), incluidos los meses sin escenas.
    """
    con_cs = set(ids_cs)
    meses = pd.period_range(inicio, pd.Period(fin, "M") - 1, freq="M").strftime("%Y-%m")
    filas = {m: [0, 0] for m in meses}
    for i in ids_s2:
        m = mes_de_escena(i)
        if m in filas:
            filas[m][0] += 1
            filas[m][1] += i in con_cs
    tabla = pd.DataFrame(
        [(m, n, c) for m, (n, c) in filas.items()],
        columns=["mes", "escenas_s2", "con_cloudscore"],
    )
    tabla["sin_cloudscore"] = tabla["escenas_s2"] - tabla["con_cloudscore"]
    return tabla


def escenas_mes(ee, zona, mes):
    """Escenas de Sentinel-2 de un mes ('2021-07') en una zona, con la banda de Cloud Score+."""
    geom = ee.Geometry.Rectangle(list(recuadro(zona)))
    inicio = ee.Date(f"{mes}-01")
    fin = inicio.advance(1, "month")
    escenas = ee.ImageCollection(S2).filterBounds(geom).filterDate(inicio, fin)
    return escenas.linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS), [BANDA_NUBES])

def imagen_mensual(ee, zona, mes):
    """Mediana de los píxeles despejados del mes, con las 6 bandas de BANDAS."""

    def despejada(img):
        return img.updateMask(img.select(BANDA_NUBES).gte(UMBRAL_NUBES))

    return escenas_mes(ee, zona, mes).map(despejada).median().select(BANDAS)

def exportar_mes(ee, zona, mes, bucket=BUCKET):
    """Lanza la exportación de la imagen mensual a gs://<bucket>/<zona>/<mes>.tif."""
    tarea = ee.batch.Export.image.toCloudStorage(
        image=imagen_mensual(ee,zona,mes).toUint16(),            
        description=f"{zona}_{mes}",      
        bucket=bucket,
        fileNamePrefix=f"{zona}/{mes}",   
        region=ee.Geometry.Rectangle(list(recuadro(zona))),           
        scale=10,
        crs=CRS,
        maxPixels=1e9,
    )
    tarea.start()
    return tarea
