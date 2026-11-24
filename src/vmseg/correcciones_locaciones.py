"""
Correcciones del padrón de locaciones encontradas al armar y validar los pares (issue #15).

`data/correcciones_locaciones.csv` tiene una fila por corrección:

- `preexistente`: la locación ya estaba desmontada en enero de 2019, o se desmontó tanto antes del
  pozo que no hay un "antes" limpio. No es una locación nueva y sale del backtesting.
- `misma_locacion`: la agrupación a 60 m partió una locación en dos; `locacion_id_a` es parte de
  `locacion_id_b`.

Las correcciones **se marcan, no se borran filas**: se agregan las columnas `correccion` y
`misma_que` al padrón. Si se sacaran filas, `armar_pares` armaría otra cantidad de positivos y,
con ella, otros negativos, y se perderían pares ya validados. Para el backtesting se usan las
locaciones con `correccion` vacía (`locaciones_validas`).

    python -m vmseg.correcciones_locaciones     # marca el padrón de data/ sin regenerarlo
"""
import json
import os

import pandas as pd

CARPETA_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RUTA_CORRECCIONES_LOCACIONES = os.path.join(CARPETA_DATA, "correcciones_locaciones.csv")
MOTIVOS = ("preexistente", "misma_locacion")


def leer_correcciones(ruta=RUTA_CORRECCIONES_LOCACIONES):
    corr = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    desconocidos = set(corr["motivo"]) - set(MOTIVOS)
    if desconocidos:
        raise ValueError(f"motivos desconocidos en {ruta}: {desconocidos}")
    if corr["locacion_id_a"].duplicated().any():
        raise ValueError(f"locación con dos correcciones en {ruta}")
    sin_b = (corr["motivo"] == "misma_locacion") & (corr["locacion_id_b"] == "")
    if sin_b.any():
        raise ValueError("misma_locacion necesita locacion_id_b")
    return corr


def marcar_correcciones(loc, corr):
    """
    Agrega a `loc` (una fila por locación) las columnas `correccion` (vacía, "preexistente" o
    "misma_locacion") y `misma_que` (la locación de la que es parte, o NA). Si una locación
    de las correcciones no está en el padrón, es un error.
    """
    ids = loc["locacion_id"].astype(int)
    faltan = set(corr["locacion_id_a"].astype(int)) - set(ids)
    if faltan:
        raise ValueError(f"locaciones de las correcciones que no están en el padrón: {sorted(faltan)}")
    por_id = corr.assign(a=corr["locacion_id_a"].astype(int)).set_index("a")
    loc = loc.copy()
    loc["correccion"] = ids.map(por_id["motivo"]).fillna("").values
    loc["misma_que"] = ids.map(por_id["locacion_id_b"].replace("", None)).astype("Int64").values
    return loc


def locaciones_validas(loc):
    """Las locaciones sin corrección: las que cuentan en el backtesting."""
    return loc[loc["correccion"].fillna("") == ""]


def _marcar_archivos(carpeta=CARPETA_DATA, corr=None):
    """Marca locaciones_vaca_muerta.csv y .geojson tal como están (texto idéntico salvo las columnas nuevas)."""
    corr = leer_correcciones() if corr is None else corr
    ruta_csv = os.path.join(carpeta, "locaciones_vaca_muerta.csv")
    loc = pd.read_csv(ruta_csv, dtype=str, keep_default_na=False).drop(columns=["correccion", "misma_que"],
                                                                       errors="ignore")
    loc = marcar_correcciones(loc, corr)
    loc.to_csv(ruta_csv, index=False)

    ruta_geojson = os.path.join(carpeta, "locaciones_vaca_muerta.geojson")
    with open(ruta_geojson, encoding="utf-8") as f:
        g = json.load(f)
    marcas = loc.set_index(loc["locacion_id"].astype(int))[["correccion", "misma_que"]]
    for feat in g["features"]:
        p = feat["properties"]
        c, m = marcas.loc[int(p["locacion_id"])]
        p["correccion"] = c
        p["misma_que"] = None if pd.isna(m) else int(m)
    with open(ruta_geojson, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False)
    return loc


if __name__ == "__main__":
    loc = _marcar_archivos()
    print(loc.loc[loc["correccion"] != "", ["locacion_id", "correccion", "misma_que", "split"]].to_string(index=False))
