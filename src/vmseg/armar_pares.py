"""
Issue #11: qué pares se etiquetan (ver docs/METODO.md, sección "Encuadres y pares").

Positivos: un par por cada locación nueva (primer pozo entre julio de 2019 y
diciembre de 2023), en el encuadre que la contiene, con T1 = mes del inicio de
perforación − 6 y T2 = mes del inicio de perforación + 1.

Negativos: pares en que no aparece ninguna locación entre T1 y T2, con T2 = T1 + 7
meses (como los positivos, y así caen siempre en estaciones distintas). Se usan
todos los pozos del padrón (data/pozos_todos.csv), no solo los de Vaca Muerta.
Un negativo por encuadre como máximo, cantidad por split proporcional a los
positivos, sin meses dudosos del índice y elegidos al azar con semilla fija.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from vmseg import gee
from vmseg.exportar import FIN, INICIO
from vmseg.indice import RUTA_INDICE
from vmseg.par import COLUMNAS_INDICE_PARES
from vmseg.recorte import CRS, RESOLUCION, TAMANO

MESES_ANTES = 6
MESES_DESPUES = 1
DESDE_NUEVAS, HASTA_NUEVAS = "2019-07-01", "2023-12-31"
LADO = TAMANO * RESOLUCION  # metros
RUTA_LOCACIONES = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "locaciones_vaca_muerta.geojson")
COLUMNA_ZONA = {"anelo": "en_zona_anelo", "externa": "en_zona_test_externo"}
RUTA_POZOS_TODOS = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "pozos_todos.csv")

SEPARACION_NEGATIVOS = 7  # meses entre T1 y T2, igual que en los positivos
APARECE_ANTES = 6  # meses antes del inicio de perforación en que ya se puede ver la locación
MARGEN_NEGATIVOS = 150  # m alrededor del encuadre: una locación de afuera puede asomar
TOTAL_NEGATIVOS = 130  # meta provisoria (docs/METODO.md)
SEMILLA = 11


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


def pozos_todos(zona, ruta=RUTA_POZOS_TODOS):
    """Pozos de una zona (cualquier formación), con x, y en metros del CRS y mes de inicio de perforación (NaT si no hay)."""
    p = pd.read_csv(ruta)
    p = p[p["zona"] == zona]
    p = gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p["lon"], p["lat"]), crs="EPSG:4326").to_crs(CRS)
    return pd.DataFrame(dict(
        x=p.geometry.x.values, y=p.geometry.y.values,
        mes=pd.PeriodIndex(pd.to_datetime(p["fecha_inicio_perforacion"]), freq="M"),
    ))


def meses_invalidos(zona, ruta=RUTA_INDICE):
    """Meses de la zona marcados como nublados o con pocas observaciones despejadas."""
    idx = pd.read_csv(ruta)
    malos = idx[(idx["zona"] == zona) & (idx["nublado"] | idx["pocas_despejadas"])]
    return set(malos["mes"])


def cantidades_por_split(positivos, total=TOTAL_NEGATIVOS):
    """Negativos por split, proporcionales a los positivos (redondeo por mayor resto, suma `total`)."""
    cuota = positivos["split"].value_counts().sort_index() * total / len(positivos)
    n = np.floor(cuota).astype(int)
    faltan = total - n.sum()
    for split in (cuota - n).sort_values(ascending=False).index[:faltan]:
        n[split] += 1
    return {split: int(v) for split, v in n.items()}


def meses_t1_posibles(encuadre, pozos, invalidos):
    """
    Meses T1 que dan un negativo en el encuadre (T2 = T1 + 7). Se descarta un T1 si T1 o T2
    son inválidos, o si algún pozo a MARGEN_NEGATIVOS del encuadre empezó a perforarse
    después de T1 y hasta APARECE_ANTES meses después de T2 (su locación puede verse en T2
    y no en T1). Un pozo sin fecha en el encuadre lo descarta entero.
    """
    x0, y1 = encuadre["x_min"] - MARGEN_NEGATIVOS, encuadre["y_max"] + MARGEN_NEGATIVOS
    x1, y0 = encuadre["x_min"] + LADO + MARGEN_NEGATIVOS, encuadre["y_max"] - LADO - MARGEN_NEGATIVOS
    cerca = pozos[pozos["x"].between(x0, x1) & pozos["y"].between(y0, y1)]
    if cerca["mes"].isna().any():
        return []
    meses = list(cerca["mes"])
    t1s = []
    for t1 in pd.period_range(INICIO, pd.Period(FIN, freq="M") - SEPARACION_NEGATIVOS, freq="M"):
        t2 = t1 + SEPARACION_NEGATIVOS
        if str(t1) in invalidos or str(t2) in invalidos:
            continue
        if any(t1 < m <= t2 + APARECE_ANTES for m in meses):
            continue
        t1s.append(str(t1))
    return t1s


def pares_negativos(encuadres, pozos, invalidos, cantidades, semilla=SEMILLA):
    """
    Negativos de una zona: `cantidades[split]` encuadres al azar de cada split, uno por
    encuadre, cada uno con un T1 al azar entre los posibles. Falla si no alcanzan.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for split in sorted(cantidades):
        candidatos = encuadres[encuadres["split"] == split].sort_values("encuadre")
        elegidos = 0
        for i in rng.permutation(len(candidatos)):
            if elegidos == cantidades[split]:
                break
            e = candidatos.iloc[i]
            t1s = meses_t1_posibles(e, pozos, invalidos)
            if not t1s:
                continue
            t1 = pd.Period(t1s[rng.integers(len(t1s))], freq="M")
            filas.append(dict(
                id=f"{e.encuadre}_neg", encuadre=e.encuadre, locacion_id=-1,
                zona=e.zona, bloque=e.bloque, split=split,
                mes_t1=str(t1), mes_t2=str(t1 + SEPARACION_NEGATIVOS),
                x_min=e.x_min, y_max=e.y_max, crs=CRS, validada=False, nota="",
            ))
            elegidos += 1
        if elegidos < cantidades[split]:
            raise ValueError(f"{split}: hay {elegidos} encuadres posibles y se piden {cantidades[split]}")
    return pd.DataFrame(filas, columns=COLUMNAS_INDICE_PARES)
