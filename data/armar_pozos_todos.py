"""
Arma la capa con todos los pozos del padrón (todas las formaciones y fechas)
que caen en las zonas de estudio, a partir de los datos abiertos de la
Secretaría de Energía (Capítulo IV, licencia CC-BY-4.0).

Sirve para lo que el padrón de Vaca Muerta no cubre: una locación de un pozo
convencional o de otra formación se ve igual desde el satélite. Se usa para
elegir los pares negativos (que no aparezca ninguna locación entre T1 y T2) y
para clasificar las detecciones sin registro del backtesting.

Va aparte de armar_dataset.py para no regenerar las locaciones de Vaca Muerta
con otra descarga del padrón.

Entrada (descargar de datos.gob.ar, dataset "Producción de petróleo y gas por pozo (Capítulo IV)"):
  - capitulo-iv-pozos.csv

Salida:
  - pozos_todos.csv  -> un registro por pozo, solo los de las zonas (con margen)

Uso (desde la carpeta data/):
    python armar_pozos_todos.py
"""
import json

import pandas as pd

ENTRADA_POZOS = "capitulo-iv-pozos.csv"
ZONAS = "zonas_estudio.json"
SALIDA = "./"

# Margen alrededor de cada zona, en grados (~1 km). Alcanza de sobra para los
# 150 m alrededor de los encuadres del borde.
MARGEN = 0.01

# Nombre corto de cada zona (igual que en src/vmseg/gee.py) -> clave en zonas_estudio.json
CLAVES = {"anelo": "anelo_train_val_test", "externa": "test_externo"}

df = pd.read_csv(ENTRADA_POZOS, low_memory=False)

# Coordenadas desde la columna geojson (EPSG:4326)
coords = df["geojson"].apply(lambda g: json.loads(g)["coordinates"])
df["lon"] = coords.str[0].round(6)
df["lat"] = coords.str[1].round(6)

# Zona de cada pozo (o vacío si no cae en ninguna)
with open(ZONAS, encoding="utf-8") as f:
    zonas = json.load(f)
df["zona"] = None
for nombre, clave in CLAVES.items():
    z = zonas[clave]
    adentro = df["lon"].between(z["lon_min"] - MARGEN, z["lon_max"] + MARGEN) & df["lat"].between(
        z["lat_min"] - MARGEN, z["lat_max"] + MARGEN
    )
    df.loc[adentro, "zona"] = nombre
df = df[df["zona"].notna()].copy()

# Mismo criterio que armar_dataset.py, para saber cuáles ya están en el padrón de Vaca Muerta
df["vaca_muerta"] = df["formacion"].str.contains("vaca muerta", case=False, na=False) & (
    df["tipo_recurso"] == "NO CONVENCIONAL"
)

# Fecha vacía = el padrón no la tiene (ver docs/DATASET.md)
df["fecha_inicio_perforacion"] = pd.to_datetime(df["adjiv_fecha_inicio_perf"], errors="coerce").dt.strftime(
    "%Y-%m-%d"
)

cols = [
    "idpozo", "sigla", "zona", "lon", "lat", "fecha_inicio_perforacion",
    "formacion", "tipo_recurso", "vaca_muerta", "tipopozo", "tipoestado", "empresa",
]
df[cols].sort_values(["zona", "fecha_inicio_perforacion"]).to_csv(SALIDA + "pozos_todos.csv", index=False)
print(df.groupby(["zona", "vaca_muerta"]).size())
