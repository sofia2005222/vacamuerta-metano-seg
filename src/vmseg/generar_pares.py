"""
Issue #11: genera los pares de data/indice_pares.csv a partir de las imágenes mensuales
del bucket y deja todo listo para validar (docs/VALIDACION.md).

    python -m vmseg.generar_pares recortar      T1, T2 y máscara sin etiquetar (255) de cada par,
                                                 más 512 px de T2 alrededor (contexto para SAM)
    python -m vmseg.generar_pares proponer      SAM sobre T2 y caída de NDVI de cada par
    python -m vmseg.generar_pares empaquetar    un zip por responsable con sus pares
    python -m vmseg.generar_pares subir         pares, índice y paquetes a gs://vmseg-imagenes/pares/
    python -m vmseg.generar_pares bajar         lo que falte en data/pares/ desde el bucket (Colab)
    python -m vmseg.generar_pares estado        bajá las máscaras del bucket y contá las validadas

Todo se guarda en data/pares/ (no se sube al repo). Con ids al final, cada paso hace solo esos
pares; sin ids, hace los que falten (recortar, proponer) o todos (el resto).

Para leer y escribir el bucket usa las credenciales de Earth Engine (`earthengine authenticate`
pide acceso a Cloud Storage), así no hace falta `gcloud`.
"""
import argparse
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from vmseg import gee
from vmseg.armar_pares import RUTA_INDICE_PARES, RUTA_LOCACIONES, RUTA_POZOS_TODOS, locaciones_otras_formaciones
from vmseg.mascaras_propuestas import leer_contexto
from vmseg.par import COLUMNAS_INDICE_PARES, guardar_tif, leer_indice_pares
from vmseg.recorte import CRS, RESOLUCION, SIN_ETIQUETA, TAMANO

CARPETA = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "pares")
CONTEXTO = "contexto"  # subcarpeta con <id>_t2_contexto.tif
PAQUETES = "paquetes"  # subcarpeta con <persona>.zip
PREFIJO = "pares"  # carpeta en el bucket
PROYECTO_BUCKET = "tp-final-cv-514618"
CHECKPOINT_SAM = os.path.join(os.path.expanduser("~"), ".cache", "sam", "sam_vit_h_4b8939.pth")
SUFIJOS_PAQUETE = ["_t1.tif", "_t2.tif", "_mask.tif", "_dndvi.tif", "_propuesta.gpkg"]


# ---------------------------------------------------------------- bucket

def credenciales():
    """
    Credenciales de Earth Engine (incluyen Cloud Storage). Si no hay (por ejemplo en Colab,
    después de `google.colab.auth.authenticate_user()`), las de la cuenta por defecto.

    Se sacan el proyecto de cuota de Earth Engine (ee-sbottino): no tiene facturación y Cloud
    Storage rechaza las escrituras con "billing account ... disabled". Sin proyecto de cuota,
    las paga el dueño del bucket.
    """
    try:
        import ee
        cred = ee.data.get_persistent_credentials()
        if cred is not None:
            return cred.with_quota_project(None)
    except Exception:
        pass
    import google.auth
    return google.auth.default()[0]


def entorno_gdal(cred=None):
    """Variables para que GDAL lea gs:// (/vsigs/) con las credenciales de Earth Engine."""
    cred = cred or credenciales()
    return dict(GS_OAUTH2_REFRESH_TOKEN=cred.refresh_token, GS_OAUTH2_CLIENT_ID=cred.client_id,
                GS_OAUTH2_CLIENT_SECRET=cred.client_secret)


def bucket():
    from google.cloud import storage
    return storage.Client(project=PROYECTO_BUCKET, credentials=credenciales()).bucket(gee.BUCKET)


def ruta_mes(zona, mes):
    """Imagen mensual en el bucket, para leer con GDAL."""
    return f"/vsigs/{gee.BUCKET}/{zona}/{mes}.tif"


# ---------------------------------------------------------------- recortar

def a_reflectancia(crudo):
    """uint16 de GEE (valor × 10000) → float32 en [0, 1]; píxel con las 6 bandas en 0 (sin dato) → NaN."""
    img = np.clip(crudo.astype(np.float32) / 10000, 0, 1)
    img[:, (crudo == 0).all(axis=0)] = np.nan
    return img


def leer_recorte(src, x_min, y_max, tamano=TAMANO):
    """Ventana de tamano × tamano px con esquina (x_min, y_max); tiene que caer en la grilla de la imagen."""
    t = src.transform
    col, fila = (x_min - t.c) / t.a, (y_max - t.f) / t.e
    if not (float(col).is_integer() and float(fila).is_integer()):
        raise ValueError(f"({x_min}, {y_max}) no cae en la grilla de {src.name}")
    ventana = rasterio.windows.Window(int(col), int(fila), tamano, tamano)
    return src.read(window=ventana, boundless=True, fill_value=0)


def recortar(indice, carpeta=CARPETA, ruta_mes=ruta_mes, env=None):
    """
    Para cada par: <id>_t1.tif y <id>_t2.tif (formato de FORMATO_RECORTE.md), <id>_mask.tif en
    SIN_ETIQUETA y contexto/<id>_t2_contexto.tif (512 px de T2, valores crudos, para SAM).
    Abre cada imagen mensual una sola vez.
    """
    os.makedirs(os.path.join(carpeta, CONTEXTO), exist_ok=True)
    usos = pd.concat([
        indice.assign(mes=indice["mes_t1"], cual="t1"),
        indice.assign(mes=indice["mes_t2"], cual="t2"),
    ])
    with rasterio.Env(**(env or {})):
        for (zona, mes), grupo in usos.groupby(["zona", "mes"]):
            ruta = ruta_mes(zona, mes)
            with rasterio.open(ruta) as src:
                for par in grupo.itertuples():
                    base = os.path.join(carpeta, par.id)
                    img = a_reflectancia(leer_recorte(src, par.x_min, par.y_max))
                    guardar_tif(f"{base}_{par.cual}.tif", img, par.x_min, par.y_max, CRS, "float32")
            if (grupo["cual"] == "t2").any():
                for par in grupo[grupo["cual"] == "t2"].itertuples():
                    ctx, t, _, _ = leer_contexto(ruta, par.x_min, par.y_max, TAMANO)
                    guardar_tif(os.path.join(carpeta, CONTEXTO, f"{par.id}_t2_contexto.tif"),
                                ctx, t.c, t.f, CRS, "uint16")
            print(f"{zona} {mes}: {len(grupo)} recortes")
    for par in indice.itertuples():
        mask = np.full((1, TAMANO, TAMANO), SIN_ETIQUETA, np.uint8)
        guardar_tif(os.path.join(carpeta, f"{par.id}_mask.tif"), mask, par.x_min, par.y_max, CRS, "uint8")


# ---------------------------------------------------------------- proponer

def padron_para_clic(locaciones, pozos):
    """
    Puntos donde SAM hace clic: las locaciones de Vaca Muerta más los pozos de otras formaciones
    que no son de ninguna de ellas (agrupados a 60 m, como en armar_dataset.py). Estos últimos van
    con locacion_id = -1 y una nota, porque no están en el padrón de locaciones.

    locaciones: GeoDataFrame de data/locaciones_vaca_muerta.geojson.
    pozos: DataFrame de data/pozos_todos.csv.
    """
    vm = locaciones[["locacion_id", "fecha_primer_pozo", "geometry"]].assign(nota="")
    g = locaciones_otras_formaciones(locaciones, pozos)
    if g.empty:
        return vm
    g = gpd.GeoDataFrame(dict(
        locacion_id=-1, fecha_primer_pozo=g["fecha_primer_pozo"].values,
        nota=[f"pozo de otra formación ({f}, {s}), no está en el padrón de locaciones"
              for f, s in zip(g["formacion"], g["sigla"])],
    ), geometry=g.geometry.values, crs="EPSG:4326")
    return pd.concat([vm.to_crs("EPSG:4326"), g], ignore_index=True)


def proponer_todos(indice, carpeta=CARPETA, checkpoint=CHECKPOINT_SAM):
    """SAM sobre T2 y caída de NDVI de cada par (ver mascaras_propuestas.proponer_par)."""
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    from vmseg.mascaras_propuestas import proponer_par, segmentador_sam

    padron = padron_para_clic(gpd.read_file(RUTA_LOCACIONES), pd.read_csv(RUTA_POZOS_TODOS))
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint).to(dispositivo)
    segmentar = segmentador_sam(SamPredictor(sam))
    for n, par in enumerate(indice.itertuples(), 1):
        contexto = os.path.join(carpeta, CONTEXTO, f"{par.id}_t2_contexto.tif")
        prop = proponer_par(carpeta, par.id, padron, segmentar, imagen_mensual_t2=contexto)
        print(f"[{n}/{len(indice)}] {par.id}: {len(prop)} polígonos", flush=True)


# ---------------------------------------------------------------- empaquetar y subir

def empaquetar(indice, carpeta=CARPETA):
    """data/pares/paquetes/<persona>.zip con los archivos de sus pares y su indice.csv."""
    os.makedirs(os.path.join(carpeta, PAQUETES), exist_ok=True)
    rutas = {}
    for persona, suyos in indice.groupby("responsable"):
        ruta = os.path.join(carpeta, PAQUETES, f"{persona}.zip")
        with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"pares_{persona}/indice.csv", suyos.to_csv(index=False))
            for id_par in suyos["id"]:
                for sufijo in SUFIJOS_PAQUETE:
                    z.write(os.path.join(carpeta, id_par + sufijo), f"pares_{persona}/{id_par}{sufijo}")
        rutas[persona] = ruta
        print(f"{ruta}: {len(suyos)} pares, {os.path.getsize(ruta) / 1e6:.0f} MB")
    return rutas


EDITABLES = ("_mask.tif", "_propuesta.gpkg")  # los devuelven las que validan


def archivos_del_par(id_par):
    """Rutas relativas a la carpeta de pares (y al prefijo del bucket) de todos los archivos de un par."""
    return [f"{id_par}{s}" for s in SUFIJOS_PAQUETE] + [f"{CONTEXTO}/{id_par}_t2_contexto.tif"]


def archivos_a_subir(ids, en_bucket, locales, regenerados=(), propuestas_nuevas=()):
    """
    Archivos de los pares a subir: los que existen localmente y no están en el bucket. Lo que ya
    está (la máscara y la propuesta pueden estar validadas) solo se reemplaza en los pares
    `regenerados` (pedidos por id: el par entero) y, para la propuesta y la caída de NDVI, en los
    pares de `propuestas_nuevas` (proponer --todos).
    """
    rehechos = ("_propuesta.gpkg", "_dndvi.tif")
    elegidos = []
    for i in ids:
        for f in archivos_del_par(i):
            if f not in locales:
                continue
            if f in en_bucket and i not in regenerados and not (i in propuestas_nuevas and f.endswith(rehechos)):
                continue
            elegidos.append(f)
    return elegidos


def _locales(carpeta):
    return {os.path.relpath(os.path.join(r, f), carpeta).replace(os.sep, "/")
            for r, _, fs in os.walk(carpeta) for f in fs}


def _en_bucket(b):
    return {bl.name[len(PREFIJO) + 1:] for bl in b.list_blobs(prefix=f"{PREFIJO}/")}


def subir(indice, carpeta=CARPETA, regenerados=(), propuestas_nuevas=()):
    """Sube a gs://<bucket>/pares/ los archivos de los pares (ver archivos_a_subir) y los paquetes."""
    b = bucket()
    en_bucket, locales = _en_bucket(b), _locales(carpeta)
    archivos = archivos_a_subir(indice["id"], en_bucket, locales, regenerados, propuestas_nuevas)
    archivos += sorted(f for f in locales if f.startswith(f"{PAQUETES}/"))
    for n, f in enumerate(archivos, 1):
        b.blob(f"{PREFIJO}/{f}").upload_from_filename(os.path.join(carpeta, f))
        if n % 100 == 0 or n == len(archivos):
            print(f"{n}/{len(archivos)} archivos subidos", flush=True)
    if "indice.csv" not in en_bucket:  # después lo actualiza `estado`
        b.blob(f"{PREFIJO}/indice.csv").upload_from_filename(os.path.join(carpeta, "indice.csv"))


def bajar(indice, carpeta=CARPETA):
    """Baja del bucket los archivos de los pares que falten en la carpeta local (por ejemplo, en Colab)."""
    b = bucket()
    en_bucket, locales = _en_bucket(b), _locales(carpeta)
    faltan = [f for i in indice["id"] for f in archivos_del_par(i) if f in en_bucket and f not in locales]
    os.makedirs(os.path.join(carpeta, CONTEXTO), exist_ok=True)
    for n, f in enumerate(faltan, 1):
        b.blob(f"{PREFIJO}/{f}").download_to_filename(os.path.join(carpeta, f))
        if n % 100 == 0 or n == len(faltan):
            print(f"{n}/{len(faltan)} archivos bajados", flush=True)


# ---------------------------------------------------------------- estado

def esta_validada(mask):
    """Una máscara está validada si ya no tiene píxeles SIN_ETIQUETA (validar_par la reescribe entera)."""
    return not (mask == SIN_ETIQUETA).any()


def estado(indice, carpeta=CARPETA):
    """Baja las máscaras del bucket, marca `validada` en indice.csv (local y del bucket) y cuenta por persona."""
    b = bucket()
    validada = []
    for id_par in indice["id"]:
        ruta = os.path.join(carpeta, f"{id_par}_mask.tif")
        b.blob(f"{PREFIJO}/{id_par}_mask.tif").download_to_filename(ruta)
        with rasterio.open(ruta) as src:
            validada.append(esta_validada(src.read(1)))
    indice = indice.assign(validada=validada)
    indice.to_csv(os.path.join(carpeta, "indice.csv"), index=False)
    b.blob(f"{PREFIJO}/indice.csv").upload_from_filename(os.path.join(carpeta, "indice.csv"))
    print(indice.groupby("responsable")["validada"].agg(validadas="sum", pares="size"))
    return indice


# ---------------------------------------------------------------- línea de comandos

def _main(argv=None):
    p = argparse.ArgumentParser(prog="python -m vmseg.generar_pares")
    p.add_argument("paso", choices=["recortar", "proponer", "empaquetar", "subir", "bajar", "estado"])
    p.add_argument("ids", nargs="*", help="solo estos pares")
    p.add_argument("--todos", action="store_true",
                   help="proponer: rehace también las propuestas que ya existen; subir: las reemplaza en el bucket")
    args = p.parse_args(argv)

    os.makedirs(CARPETA, exist_ok=True)
    indice = pd.read_csv(RUTA_INDICE_PARES, dtype={"nota": str})[COLUMNAS_INDICE_PARES]
    ruta_local = os.path.join(CARPETA, "indice.csv")
    if os.path.exists(ruta_local):  # conservar lo ya validado
        validada = leer_indice_pares(CARPETA).set_index("id")["validada"]
        indice["validada"] = indice["id"].map(validada).fillna(False).astype(bool)
    indice.to_csv(ruta_local, index=False)

    elegidos = indice[indice["id"].isin(args.ids)] if args.ids else indice
    if args.paso == "recortar":
        if not args.ids:
            elegidos = indice[[not os.path.exists(os.path.join(CARPETA, f"{i}_t2.tif")) for i in indice["id"]]]
        recortar(elegidos, env=entorno_gdal())
    elif args.paso == "proponer":
        if not args.ids and not args.todos:
            elegidos = indice[[not os.path.exists(os.path.join(CARPETA, f"{i}_propuesta.gpkg")) for i in indice["id"]]]
        proponer_todos(elegidos)
    elif args.paso == "empaquetar":
        empaquetar(elegidos)
    elif args.paso == "subir":
        subir(elegidos, regenerados=set(args.ids), propuestas_nuevas=set(elegidos["id"]) if args.todos else ())
    elif args.paso == "bajar":
        bajar(elegidos)
    else:
        estado(indice)


if __name__ == "__main__":
    _main()
