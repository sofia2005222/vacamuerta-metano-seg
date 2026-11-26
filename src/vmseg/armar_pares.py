"""
Issue #11: qué pares se etiquetan (ver docs/METODO.md, sección "Encuadres y pares").

Positivos: un par por cada locación nueva (primer pozo entre julio de 2019 y
diciembre de 2023), en el encuadre que la contiene, con T1 = mes del inicio de
perforación − 6 y T2 = mes del inicio de perforación + 1.

También son positivos las locaciones nuevas de otras formaciones (pozos de
data/pozos_todos.csv a más de CERCA_DE_VM_M de toda locación de Vaca Muerta,
agrupados a DIST_LOCACION_M), pero solo en los bloques de train y val de la Zona
Añelo: el backtesting sigue siendo solo sobre Vaca Muerta. Su locacion_id es el
idpozo de su primer pozo (los de Vaca Muerta no pasan de ~1.200, los idpozo
empiezan en ~8.000).

Negativos: pares en que no aparece ninguna locación entre T1 y T2, con T2 = T1 + 7
meses (como los positivos, y así caen siempre en estaciones distintas). Se usan
todos los pozos del padrón (data/pozos_todos.csv), no solo los de Vaca Muerta.
Un negativo por encuadre como máximo, cantidad por split proporcional a los
positivos, sin meses dudosos del índice y elegidos al azar con semilla fija.

Uso (escribe data/indice_pares.csv):
    python -m vmseg.armar_pares
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from vmseg import gee
from vmseg.encuadres import bloque_de, encuadres, split_de_bloque
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
RUTA_INDICE_PARES = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "indice_pares.csv")
RUTA_CORRECCIONES = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "correcciones_pares.csv")
ZONAS = ["anelo", "externa"]

DIST_LOCACION_M = 60  # igual que en data/armar_dataset.py
CERCA_DE_VM_M = 100  # un pozo de otra formación a menos de esto de una locación del padrón es de esa locación
SPLITS_OTRAS_FORMACIONES = {"train", "val"}
# Los positivos de otras formaciones los valida una sola persona, sin importar el bloque, junto
# con su parte del acuerdo entre etiquetadoras (#19).
RESPONSABLE_OTRAS_FORMACIONES = "sofi"

SEPARACION_NEGATIVOS = 7  # meses entre T1 y T2, igual que en los positivos
APARECE_ANTES = 6  # meses antes del inicio de perforación en que ya se puede ver la locación
MARGEN_NEGATIVOS = 150  # m alrededor del encuadre: una locación de afuera puede asomar
TOTAL_NEGATIVOS = 130  # meta provisoria (docs/METODO.md)
SEMILLA = 11

# Quién etiqueta cada bloque: {zona: {persona: [bloques]}}. Cada bloque entero es de una
# sola persona. Se eligió con una búsqueda local que empareja el total (100/99/99), los
# positivos (56 cada una) y cada split entre las tres, para que la forma de marcar de
# una no quede concentrada en un split (val tiene solo 2 bloques: a Bian no le toca y
# lleva más test_anelo). Queda fijo para que no cambie al regenerar el índice.
REPARTO = {
    "anelo": {"sofi": [6, 7, 10, 11], "agus": [0, 1, 2, 3, 4, 13], "bian": [5, 8, 9, 12, 14, 15]},
    "externa": {"sofi": [0, 2, 3, 9, 11], "agus": [4, 7, 8, 10], "bian": [1, 5, 6]},
}


def locaciones_nuevas(zona, ruta=RUTA_LOCACIONES):
    """Locaciones nuevas de una zona, con x, y en metros del CRS de las imágenes."""
    loc = gpd.read_file(ruta).to_crs(CRS)
    fecha = pd.to_datetime(loc["fecha_primer_pozo"])
    loc = loc[loc[COLUMNA_ZONA[zona]] & fecha.between(DESDE_NUEVAS, HASTA_NUEVAS)].copy()
    loc["x"], loc["y"] = loc.geometry.x, loc.geometry.y
    return pd.DataFrame(loc.drop(columns="geometry"))


def locaciones_otras_formaciones(locaciones, pozos):
    """
    Locaciones de pozos de otras formaciones que no son de ninguna locación de Vaca Muerta:
    pozos con fecha agrupados a DIST_LOCACION_M (como en armar_dataset.py), sin los grupos a
    menos de CERCA_DE_VM_M de una locación del padrón (de cualquier fecha).

    locaciones: GeoDataFrame de data/locaciones_vaca_muerta.geojson.
    pozos: DataFrame de data/pozos_todos.csv.
    Devuelve un GeoDataFrame (EPSG:4326) con locacion_id (idpozo del primer pozo),
    fecha_primer_pozo, formacion y sigla.
    """
    columnas = ["locacion_id", "fecha_primer_pozo", "formacion", "sigla", "geometry"]
    otros = pozos[~pozos["vaca_muerta"] & pozos["fecha_inicio_perforacion"].notna()].copy()
    if otros.empty:
        return gpd.GeoDataFrame(columns=columnas, geometry="geometry", crs="EPSG:4326")
    rad = np.radians(otros[["lat", "lon"]].values)
    otros["grupo"] = DBSCAN(eps=DIST_LOCACION_M / 6_371_000, min_samples=1,
                            metric="haversine", algorithm="ball_tree").fit_predict(rad)
    unir = lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))
    primero = otros.sort_values(["fecha_inicio_perforacion", "idpozo"]).groupby("grupo")["idpozo"].first()
    g = otros.groupby("grupo").agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                   fecha_primer_pozo=("fecha_inicio_perforacion", "min"),
                                   formacion=("formacion", unir), sigla=("sigla", unir))
    g["locacion_id"] = primero
    g = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326")

    # Los que caen sobre una locación de Vaca Muerta son de esa locación.
    cerca = gpd.sjoin_nearest(g.to_crs(CRS), locaciones[["geometry"]].to_crs(CRS),
                              max_distance=CERCA_DE_VM_M, how="left")
    lejos = cerca[cerca["index_right"].isna()].index.unique()
    return g.loc[lejos, columnas].reset_index(drop=True)


def locaciones_nuevas_otras_formaciones(zona, ruta_locaciones=RUTA_LOCACIONES, ruta_pozos=RUTA_POZOS_TODOS):
    """Locaciones nuevas de otras formaciones en los bloques de SPLITS_OTRAS_FORMACIONES de la zona, con x, y."""
    pozos = pd.read_csv(ruta_pozos)
    otras = locaciones_otras_formaciones(gpd.read_file(ruta_locaciones), pozos[pozos["zona"] == zona])
    fecha = pd.to_datetime(otras["fecha_primer_pozo"])
    otras = otras[fecha.between(DESDE_NUEVAS, HASTA_NUEVAS)]
    bloques = bloque_de(zona, otras.geometry.x.values, otras.geometry.y.values)
    otras = otras[[split_de_bloque(zona, int(b)) in SPLITS_OTRAS_FORMACIONES for b in bloques]].to_crs(CRS)
    otras["x"], otras["y"] = otras.geometry.x, otras.geometry.y
    return pd.DataFrame(otras.drop(columns="geometry"))


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

    Si dos locaciones caen en el mismo encuadre con los mismos T1 y T2, los dos pares
    serían la misma imagen con la misma máscara (se marca toda locación nueva): queda
    el de la primera y la otra se anota en la nota.
    """
    filas = []
    por_par = {}  # (encuadre, mes_t1, mes_t2) -> fila
    for loc in locaciones.itertuples():
        e = encuadre_de(loc.x, loc.y, encuadres)
        if e is None:
            continue
        mes_t1, mes_t2, nota = meses_par(loc.fecha_primer_pozo)
        clave = (e.encuadre, mes_t1, mes_t2)
        if clave in por_par:
            fila = por_par[clave]
            fila["nota"] = "; ".join(filter(None, [fila["nota"], f"también la locación {loc.locacion_id}"]))
            continue
        por_par[clave] = dict(
            id=f"{e.encuadre}_{loc.locacion_id}", encuadre=e.encuadre, locacion_id=loc.locacion_id,
            zona=e.zona, bloque=e.bloque, split=e.split, mes_t1=mes_t1, mes_t2=mes_t2,
            x_min=e.x_min, y_max=e.y_max, crs=CRS, validada=False, nota=nota,
        )
        filas.append(por_par[clave])
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


def anotar_meses_dudosos(pares, invalidos):
    """Agrega a la nota de cada par si T1 o T2 es un mes dudoso de su zona (`invalidos[zona]`)."""
    pares = pares.reset_index(drop=True)  # con índices repetidos, .at escribiría en varias filas
    for i, par in pares.iterrows():
        dudosos = [f"{t} {par[col]}" for t, col in [("T1", "mes_t1"), ("T2", "mes_t2")] if par[col] in invalidos[par["zona"]]]
        if dudosos:
            aviso = f"{' y '.join(dudosos)}: pocas observaciones despejadas, mirar con atención"
            pares.at[i, "nota"] = "; ".join(filter(None, [par["nota"], aviso]))
    return pares


def aplicar_correcciones(indice, correcciones):
    """
    Correcciones que salen de validar (data/correcciones_pares.csv: id, accion, nota):
      t1_menos_12   la locación ya se veía en T1: T1 pasa de −6 a −12 meses (no antes de enero de 2019)
      t1_menos_24   se desmontó más de un año antes del pozo: T1 pasa a −24 meses (no antes de enero de 2019)
      preexistente  ya estaba en enero de 2019 (o ni con −24 hay un antes limpio): el par sale (se anota para #15)
    El id del par no cambia; hay que volver a recortarlo y proponerlo (docs/VALIDACION.md).
    """
    corrimiento = {"t1_menos_12": 12, "t1_menos_24": 24}  # meses antes del inicio de perforación
    acciones = {*corrimiento, "preexistente"}
    desconocidas = set(correcciones["accion"]) - acciones
    if desconocidas:
        raise ValueError(f"acciones desconocidas en las correcciones: {desconocidas}")
    indice = indice.copy()
    for c in correcciones.itertuples():
        fila = indice["id"] == c.id
        if c.accion == "preexistente":
            indice = indice[~fila]
            continue
        meses = corrimiento[c.accion]
        t1 = max(pd.Period(indice.loc[fila, "mes_t1"].iloc[0], freq="M") - (meses - MESES_ANTES),
                 pd.Period(INICIO, freq="M"))
        indice.loc[fila, "mes_t1"] = str(t1)
        aviso = "; ".join(filter(None, [f"T1 corrido a −{meses} meses: la locación ya se veía", c.nota]))
        indice.loc[fila, "nota"] = indice.loc[fila, "nota"].map(lambda n: "; ".join(filter(None, [n, aviso])))
    return indice.reset_index(drop=True)


def armar_indice(semilla=SEMILLA, ruta_correcciones=RUTA_CORRECCIONES):
    """
    Positivos y negativos de las dos zonas, ordenados por zona, bloque y encuadre. Las correcciones
    se aplican al final, para que la cantidad de negativos y cuáles son no cambien.
    """
    encuadres_z = {z: encuadres(z) for z in ZONAS}
    invalidos = {z: meses_invalidos(z) for z in ZONAS}
    positivos = pd.concat([pares_positivos(locaciones_nuevas(z), encuadres_z[z]) for z in ZONAS])
    # Los negativos se reparten con los positivos de Vaca Muerta solamente, para que no cambien.
    cantidades = cantidades_por_split(positivos)
    # Las de otras formaciones van después, así un par que ya existe no cambia de locación.
    otras = {z: locaciones_nuevas_otras_formaciones(z) for z in ZONAS}
    positivos = pd.concat([
        pares_positivos(pd.concat([locaciones_nuevas(z), otras[z]], ignore_index=True), encuadres_z[z])
        for z in ZONAS
    ])
    negativos = [
        pares_negativos(encuadres_z[z], pozos_todos(z), invalidos[z],
                        {s: n for s, n in cantidades.items() if s in set(encuadres_z[z]["split"])}, semilla)
        for z in ZONAS
    ]
    indice = anotar_meses_dudosos(pd.concat([positivos, *negativos], ignore_index=True), invalidos)
    responsable = {(z, b): p for z, personas in REPARTO.items() for p, bloques in personas.items() for b in bloques}
    indice["responsable"] = [responsable[(z, b)] for z, b in zip(indice["zona"], indice["bloque"])]
    ids_otras = set(pd.concat([o["locacion_id"] for o in otras.values()]))
    indice.loc[indice["locacion_id"].isin(ids_otras), "responsable"] = RESPONSABLE_OTRAS_FORMACIONES
    if os.path.exists(ruta_correcciones):
        indice = aplicar_correcciones(indice, pd.read_csv(ruta_correcciones, dtype=str).fillna(""))
    return indice.sort_values(["zona", "bloque", "encuadre", "mes_t1"], ignore_index=True)


if __name__ == "__main__":
    indice = armar_indice()
    indice.to_csv(RUTA_INDICE_PARES, index=False)
    tipo = indice["locacion_id"].ge(0).map({True: "positivos", False: "negativos"})
    print(f"{len(indice)} pares en {RUTA_INDICE_PARES}")
    print(pd.crosstab(indice["split"], tipo, margins=True))
