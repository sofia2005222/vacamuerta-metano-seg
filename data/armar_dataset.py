"""
Arma el dataset de pozos y locaciones de Vaca Muerta a partir de los datos
abiertos de la Secretaría de Energía (Capítulo IV, licencia CC-BY-4.0).

Entradas (descargar de datos.gob.ar, dataset "Producción de petróleo y gas por pozo (Capítulo IV)"):
  - capitulo-iv-pozos.csv
  - padrn-de-pozos-de-captulo-iv-con-fecha-de-primera-produccin.csv

Salidas:
  - pozos_vaca_muerta.csv / .geojson        -> un registro por pozo
  - locaciones_vaca_muerta.csv / .geojson   -> pozos agrupados en locaciones (pads)
  - zonas_estudio.json                      -> recuadros de las zonas y división por bloques
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ENTRADA_POZOS = "capitulo-iv-pozos.csv"
ENTRADA_FECHAS = "padrn-de-pozos-de-captulo-iv-con-fecha-de-primera-produccin.csv"
SALIDA = "./"

# Pozos a menos de esta distancia se consideran parte de la misma locación.
# En Vaca Muerta las bocas de pozo de un pad están separadas ~10 m.
DIST_LOCACION_M = 60

# Zona de estudio alrededor de Añelo (Loma Campana, Bandurria Sur,
# La Amarga Chica, Bajada del Palo Oeste, Cruz de Lorena). ~30 x 33 km.
ZONA = dict(lon_min=-68.85, lon_max=-68.50, lat_min=-38.42, lat_max=-38.12)

# Zona de test externa (oeste, zona gasífera: Fortín de Piedra, Aguada Pichana
# Este, La Calera). Otras operadoras (Tecpetrol, Total, Pluspetrol). Nunca se
# usa para entrenar: mide si el modelo generaliza a otra zona y otra empresa.
ZONA_TEST = dict(lon_min=-69.25, lon_max=-68.95, lat_min=-38.60, lat_max=-38.30)

# La zona de Añelo se divide en una grilla de N x N bloques (~7,5 km de lado)
# y cada bloque entero va a train, val o test (división espacial, sin fuga
# de información entre recortes vecinos). SEMILLA elegida para que quede ~70/10/20.
N_BLOQUES = 4
SEMILLA = 3

# 1. Filtrar Vaca Muerta no convencional
df = pd.read_csv(ENTRADA_POZOS, low_memory=False)
vm = df[
    df["formacion"].str.contains("vaca muerta", case=False, na=False)
    & (df["tipo_recurso"] == "NO CONVENCIONAL")
].copy()

# 2. Coordenadas desde la columna geojson (EPSG:4326)
coords = vm["geojson"].apply(lambda g: json.loads(g)["coordinates"])
vm["lon"] = coords.str[0].round(6)
vm["lat"] = coords.str[1].round(6)

# 3. Cruce con fecha de primera producción
fp = pd.read_csv(ENTRADA_FECHAS)
vm = vm.merge(fp, on="idpozo", how="left")
vm["fecha_primera_produccion"] = pd.to_datetime(
    dict(year=vm["anio"], month=vm["mes"], day=1), errors="coerce"
)

renombrar = {
    "adjiv_fecha_inicio_perf": "fecha_inicio_perforacion",
    "adjiv_fecha_fin_perf": "fecha_fin_perforacion",
    "adjiv_fecha_inicio_term": "fecha_inicio_terminacion",
    "adjiv_fecha_fin_term": "fecha_fin_terminacion",
}
vm = vm.rename(columns=renombrar)
for c in renombrar.values():
    vm[c] = pd.to_datetime(vm[c], errors="coerce")

# 4. Agrupar pozos en locaciones (lo que se ve desde el satélite es la locación)
rad = np.radians(vm[["lat", "lon"]].values)
vm["locacion_id"] = DBSCAN(
    eps=DIST_LOCACION_M / 6_371_000, min_samples=1,
    metric="haversine", algorithm="ball_tree",
).fit_predict(rad)

vm["en_zona_anelo"] = vm["lon"].between(ZONA["lon_min"], ZONA["lon_max"]) & vm["lat"].between(
    ZONA["lat_min"], ZONA["lat_max"]
)

unir = lambda s: "; ".join(sorted(s.dropna().astype(str).unique()))
loc = vm.groupby("locacion_id").agg(
    lon=("lon", "mean"),
    lat=("lat", "mean"),
    n_pozos=("idpozo", "size"),
    fecha_primer_pozo=("fecha_inicio_perforacion", "min"),
    fecha_ultimo_pozo=("fecha_inicio_perforacion", "max"),
    primera_produccion=("fecha_primera_produccion", "min"),
    empresa=("empresa", unir),
    area=("area", unir),
    siglas=("sigla", unir),
    en_zona_anelo=("en_zona_anelo", "max"),
).reset_index()
loc[["lon", "lat"]] = loc[["lon", "lat"]].round(6)
loc["anio_primer_pozo"] = loc["fecha_primer_pozo"].dt.year.astype("Int64")
loc["nueva_2019_2023"] = loc["anio_primer_pozo"].between(2019, 2023).fillna(False)
loc["ampliada_luego"] = (loc["fecha_ultimo_pozo"] - loc["fecha_primer_pozo"]).dt.days > 365

# 4b. División espacial: bloques en Añelo + zona de test externa
loc["en_zona_test_externo"] = loc["lon"].between(ZONA_TEST["lon_min"], ZONA_TEST["lon_max"]) & loc[
    "lat"
].between(ZONA_TEST["lat_min"], ZONA_TEST["lat_max"])

bx = ((loc["lon"] - ZONA["lon_min"]) / (ZONA["lon_max"] - ZONA["lon_min"]) * N_BLOQUES).astype(int).clip(0, N_BLOQUES - 1)
by = ((loc["lat"] - ZONA["lat_min"]) / (ZONA["lat_max"] - ZONA["lat_min"]) * N_BLOQUES).astype(int).clip(0, N_BLOQUES - 1)
loc["bloque_anelo"] = (by * N_BLOQUES + bx).where(loc["en_zona_anelo"]).astype("Int64")

conteo = loc.loc[loc["en_zona_anelo"]].groupby("bloque_anelo").size()
orden = np.random.default_rng(SEMILLA).permutation(conteo.index.values)
total, acumulado, split_bloque = conteo.sum(), 0, {}
for b in orden:
    frac = acumulado / total
    split_bloque[b] = "test_anelo" if frac < 0.20 else ("val" if frac < 0.30 else "train")
    acumulado += conteo[b]

loc["split"] = loc["bloque_anelo"].map(split_bloque)
loc.loc[loc["en_zona_test_externo"], "split"] = "test_externo"

# 5. Exportar
cols_pozos = [
    "idpozo", "sigla", "locacion_id", "empresa", "area", "yacimiento", "provincia",
    "lon", "lat", "tipopozo", "tipoestado", "sub_tipo_recurso", "profundidad",
    "fecha_inicio_perforacion", "fecha_fin_perforacion", "fecha_inicio_terminacion",
    "fecha_fin_terminacion", "fecha_primera_produccion", "en_zona_anelo",
]
pozos = vm[cols_pozos].sort_values(["locacion_id", "fecha_inicio_perforacion"]).copy()
fechas_pozos = list(renombrar.values()) + ["fecha_primera_produccion"]
for c in fechas_pozos:
    pozos[c] = pozos[c].dt.strftime("%Y-%m-%d")

loc = loc.sort_values("fecha_primer_pozo")
for c in ["fecha_primer_pozo", "fecha_ultimo_pozo", "primera_produccion"]:
    loc[c] = loc[c].dt.strftime("%Y-%m-%d")

pozos.to_csv(SALIDA + "pozos_vaca_muerta.csv", index=False)
loc.to_csv(SALIDA + "locaciones_vaca_muerta.csv", index=False)


def a_geojson(tabla, ruta):
    props = [c for c in tabla.columns if c not in ("lon", "lat")]
    feats = []
    for _, r in tabla.iterrows():
        p = {}
        for k in props:
            v = r[k]
            if pd.isna(v):
                p[k] = None
            elif isinstance(v, (np.integer,)):
                p[k] = int(v)
            elif isinstance(v, (np.floating,)):
                p[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                p[k] = bool(v)
            else:
                p[k] = v
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": p,
        })
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)


a_geojson(pozos, SALIDA + "pozos_vaca_muerta.geojson")
a_geojson(loc.drop(columns=["siglas"]), SALIDA + "locaciones_vaca_muerta.geojson")
with open(SALIDA + "zonas_estudio.json", "w") as f:
    json.dump({"anelo_train_val_test": ZONA, "test_externo": ZONA_TEST,
               "n_bloques": N_BLOQUES, "semilla": SEMILLA,
               "split_por_bloque": {int(k): v for k, v in split_bloque.items()}}, f, indent=1)

print(f"Pozos Vaca Muerta: {len(pozos)} | Locaciones: {len(loc)}")
print(loc.groupby("split").agg(locaciones=("locacion_id", "size"), nuevas_2019_2023=("nueva_2019_2023", "sum")))
