"""
Issue #11: qué pares se etiquetan (ver docs/METODO.md, sección "Encuadres y pares").

Positivos: un par por cada locación nueva (primer pozo entre julio de 2019 y
diciembre de 2023), en el encuadre que la contiene, con T1 = mes del inicio de
perforación − 6 y T2 = mes del inicio de perforación + 1.
"""
import os

import geopandas as gpd
import pandas as pd

from vmseg import gee
from vmseg.exportar import FIN
from vmseg.par import COLUMNAS_INDICE_PARES
from vmseg.recorte import CRS, RESOLUCION, TAMANO

MESES_ANTES = 6
MESES_DESPUES = 1
DESDE_NUEVAS, HASTA_NUEVAS = "2019-07-01", "2023-12-31"
LADO = TAMANO * RESOLUCION  # metros
RUTA_LOCACIONES = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "locaciones_vaca_muerta.geojson")
COLUMNA_ZONA = {"anelo": "en_zona_anelo", "externa": "en_zona_test_externo"}


def locaciones_nuevas(zona, ruta=RUTA_LOCACIONES):
    """Locaciones nuevas de una zona, con x, y en metros del CRS de las imágenes."""
    loc = gpd.read_file(ruta).to_crs(CRS)
    fecha = pd.to_datetime(loc["fecha_primer_pozo"])
    loc = loc[loc[COLUMNA_ZONA[zona]] & fecha.between(DESDE_NUEVAS, HASTA_NUEVAS)].copy()
    loc["x"], loc["y"] = loc.geometry.x, loc.geometry.y
    return pd.DataFrame(loc.drop(columns="geometry"))


def meses_par(fecha_perforacion):
    """(mes_t1, mes_t2, nota) de un positivo. T2 no puede pasar del último mes exportado."""
    mes = pd.Period(fecha_perforacion, freq="M")
    t1, t2 = mes - MESES_ANTES, mes + MESES_DESPUES
    nota = ""
    if t2 > pd.Period(FIN, freq="M"):
        t2 = pd.Period(FIN, freq="M")
        nota = f"T2 = {FIN}: no hay imágenes después"
    return str(t1), str(t2), nota


def margen(x, y, encuadres):
    """Distancia (m) del punto al borde más cercano de cada encuadre; negativa si queda afuera."""
    return pd.concat([
        x - encuadres["x_min"], encuadres["x_min"] + LADO - x,
        encuadres["y_max"] - y, y - (encuadres["y_max"] - LADO),
    ], axis=1).min(axis=1)


def encuadre_de(x, y, encuadres):
    """Encuadre que contiene el punto, el que lo deja más lejos del borde; None si ninguno lo contiene."""
    m = margen(x, y, encuadres)
    if m.max() < 0:
        return None
    return encuadres.loc[m.idxmax()]


def pares_positivos(locaciones, encuadres):
    """
    Un par por locación nueva. `locaciones` tiene locacion_id, x, y (en metros del CRS)
    y fecha_primer_pozo. Las que no caen en ningún encuadre no dan par.
    """
    filas = []
    for loc in locaciones.itertuples():
        e = encuadre_de(loc.x, loc.y, encuadres)
        if e is None:
            continue
        mes_t1, mes_t2, nota = meses_par(loc.fecha_primer_pozo)
        filas.append(dict(
            id=f"{e.encuadre}_{loc.locacion_id}", encuadre=e.encuadre, locacion_id=loc.locacion_id,
            zona=e.zona, bloque=e.bloque, split=e.split, mes_t1=mes_t1, mes_t2=mes_t2,
            x_min=e.x_min, y_max=e.y_max, crs=CRS, validada=False, nota=nota,
        ))
    return pd.DataFrame(filas, columns=COLUMNAS_INDICE_PARES)
