"""
Máscaras propuestas de los pares, para validar en QGIS (issue #9, ADR-0011, docs/METODO.md).

El circuito de etiquetado de un par es:

    <id>_t1.tif + <id>_t2.tif (+ imagen mensual de T2 para dar contexto a SAM)
        → [proponer_par]  SAM sobre T2 hace un clic en cada locación nueva del padrón
        → <id>_propuesta.gpkg   polígonos editables (clase, origen, locacion_id, verificar, nota)
          <id>_dndvi.tif        caída de NDVI entre T1 y T2 (capa de ayuda para ver qué cambió)
        → [QGIS]   scripts/qgis_abrir_par.py: T1, T2, dNDVI, la propuesta y el padrón.
                   Se borra lo que ya estaba en T1, se corrigen bordes y se agregan piletas
                   y locaciones nuevas que no están en el padrón.
        → [validar_par]  <id>_mask.tif en el formato de par y `validada=True` en el índice

Decisiones:
- La máscara marca solo lo que **apareció entre T1 y T2** (ADR-0011). Por eso SAM hace clic
  solo en las locaciones con primer pozo **después de T1**: las anteriores ya estaban.
- También se hace clic en las que se perforan hasta `MESES_DESPUES` meses después de T2, porque
  pueden estar ya desmontadas en T2 (en obra). Llevan `verificar=1`: quien valida confirma que se vean.
- SAM ve la imagen entera, no una ventana chica (issue #8: 20/20 con la imagen entera, 13/20 con
  una ventana de 640 m). Si hay imagen mensual de T2, `leer_contexto` le da 512 px alrededor del
  recorte; si no, ve el recorte T2 de 256 px.
- OpenStreetMap no tiene locaciones en la zona (issue #8): no se usa.

SAM no se importa acá: las funciones reciben `segmentar(rgb, fila, col) -> máscara`.
Así las pruebas corren sin GPU y el notebook decide cómo cargar SAM (ver `segmentador_sam`).
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import rowcol, xy
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import box, shape

from vmseg.par import guardar_tif, leer_indice_pares
from vmseg.recorte import CRS, FONDO, LOCACION, PILETA, RESOLUCION

MESES_DESPUES = 6        # locaciones perforadas poco después de T2 pueden estar ya en obra
CONTEXTO = 512           # lado, en píxeles, de la imagen que ve SAM si hay imagen mensual
CAMPOS = ["clase", "origen", "locacion_id", "verificar", "nota"]
MAX_HA = 10              # una locación con piletas no llega; más grande es que SAM agarró el fondo (#11)
MAX_PIXELES = int(MAX_HA * 10_000 / RESOLUCION**2)


# ---------------------------------------------------------------- imagen

def rgb8(img, percentiles=(2, 98)):
    """
    Pasa una imagen de 6 bandas (B2, B3, B4, B8, B11, B12), reflectancia en [0, 1],
    a RGB de 8 bits (alto, ancho, 3) estirado por percentiles, que es lo que espera SAM.
    """
    rgb = img[[2, 1, 0]].astype(np.float32)          # B4, B3, B2
    validos = rgb[np.isfinite(rgb)]
    if validos.size == 0:
        return np.zeros(rgb.shape[1:] + (3,), np.uint8)
    lo, hi = np.percentile(validos, percentiles)
    rgb = np.nan_to_num((rgb - lo) / max(hi - lo, 1e-6), nan=0.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def ndvi(img):
    """NDVI = (B8 − B4) / (B8 + B4). Alto = vegetación; cerca de 0 = suelo pelado."""
    rojo, nir = img[2].astype(np.float32), img[3].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - rojo) / (nir + rojo)


def caida_ndvi(img_t1, img_t2):
    """
    Cuánto bajó el NDVI entre T1 y T2 (NDVI_T1 − NDVI_T2). Positivo = se perdió vegetación,
    que es lo que pasa cuando se desmonta una locación.
    """
    return ndvi(img_t1) - ndvi(img_t2)


def leer_contexto(ruta, x_min, y_max, tamano, lado=CONTEXTO):
    """
    Lee de una imagen grande (la imagen mensual de la zona) una ventana de `lado` × `lado`
    píxeles centrada en el recorte con esquina (x_min, y_max), recortada a los bordes.

    Devuelve (img, transform, fila0, col0): fila0 y col0 son la posición del recorte
    adentro de la ventana.
    """
    with rasterio.open(ruta) as src:
        x_max = x_min + tamano * RESOLUCION
        y_min = y_max - tamano * RESOLUCION
        w = from_bounds(x_min, y_min, x_max, y_max, src.transform)
        f_rec, c_rec = int(round(w.row_off)), int(round(w.col_off))
        margen = (lado - tamano) // 2
        alto, ancho = min(lado, src.height), min(lado, src.width)
        f0 = max(0, min(f_rec - margen, src.height - alto))
        c0 = max(0, min(c_rec - margen, src.width - ancho))
        ventana = Window(c0, f0, ancho, alto)
        img = src.read(window=ventana).astype(np.float32)
        return img, src.window_transform(ventana), f_rec - f0, c_rec - c0


# ---------------------------------------------------------------- puntos del padrón

def _fin_de_mes(mes):
    return pd.Period(mes, freq="M").end_time.normalize()


def locaciones_nuevas_para_clic(locaciones, transform, crs, alto, ancho, mes_t1, mes_t2,
                                meses_despues=MESES_DESPUES):
    """
    Locaciones del padrón adentro de una imagen (transform, crs, alto, ancho) que pueden haber
    aparecido entre T1 y T2: primer pozo después del fin de `mes_t1` y hasta `meses_despues`
    meses después del fin de `mes_t2`.

    locaciones: GeoDataFrame del padrón (`data/locaciones_vaca_muerta.geojson`), en cualquier CRS.
    Si tiene una columna `nota` (por ejemplo, pozos de otras formaciones), pasa a los polígonos.
    Devuelve un DataFrame con locacion_id, fila, col, fecha_primer_pozo, verificar
    (1 si el primer pozo es posterior a T2: la locación puede no estar todavía) y nota.
    """
    columnas = ["locacion_id", "fila", "col", "fecha_primer_pozo", "verificar", "nota"]
    loc = locaciones[locaciones["fecha_primer_pozo"].notna()].copy()
    loc["fecha"] = pd.to_datetime(loc["fecha_primer_pozo"])
    fin_t1, fin_t2 = _fin_de_mes(mes_t1), _fin_de_mes(mes_t2)
    loc = loc[(loc["fecha"] > fin_t1) & (loc["fecha"] <= fin_t2 + pd.DateOffset(months=meses_despues))]
    if loc.empty:
        return pd.DataFrame(columns=columnas)

    loc = loc.to_crs(crs)
    filas, cols = rowcol(transform, loc.geometry.x.values, loc.geometry.y.values)
    loc["fila"], loc["col"] = np.asarray(filas), np.asarray(cols)
    loc = loc[loc["fila"].between(0, alto - 1) & loc["col"].between(0, ancho - 1)]
    return pd.DataFrame({
        "locacion_id": loc["locacion_id"].values,
        "fila": loc["fila"].values,
        "col": loc["col"].values,
        "fecha_primer_pozo": loc["fecha"].dt.strftime("%Y-%m-%d").values,
        "verificar": (loc["fecha"] > fin_t2).astype(int).values,
        "nota": loc["nota"].fillna("").values if "nota" in loc else "",
    }, columns=columnas)


# ---------------------------------------------------------------- SAM

def segmentador_sam(predictor):
    """
    Envuelve un `segment_anything.SamPredictor` en la función `segmentar(rgb, fila, col)`.
    Codifica cada imagen una sola vez y, de las 3 máscaras que propone SAM, devuelve la de mayor
    puntaje (igual que en la prueba del #8) entre las de hasta MAX_PIXELES: en los pares del #11
    la de mayor puntaje a veces era el fondo entero.
    """
    estado = {"imagen": None}

    def segmentar(rgb, fila, col):
        if estado["imagen"] is not rgb:
            predictor.set_image(rgb)
            estado["imagen"] = rgb
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[col, fila]]), point_labels=np.array([1]),
            multimask_output=True,
        )
        scores = np.where(masks.reshape(len(masks), -1).sum(axis=1) <= MAX_PIXELES, scores, -np.inf)
        return masks[int(np.argmax(scores))]

    return segmentar


def parte_del_clic(mask, fila, col):
    """
    La parte conexa de `mask` que contiene el clic (o la más cercana, si el clic cae afuera);
    vacía si es más grande que MAX_PIXELES. Saca las manchitas sueltas que a veces devuelve SAM.
    """
    etiquetas, n = ndimage.label(mask)
    if n == 0:
        return mask
    e = etiquetas[fila, col]
    if e == 0:
        filas, cols = np.nonzero(etiquetas)
        cerca = np.argmin((filas - fila) ** 2 + (cols - col) ** 2)
        e = etiquetas[filas[cerca], cols[cerca]]
    parte = etiquetas == e
    return parte if parte.sum() <= MAX_PIXELES else np.zeros_like(parte)


# ---------------------------------------------------------------- polígonos ↔ máscara

def mascara_a_poligonos(mask, transform, crs, clase=LOCACION, **atributos):
    """Vectoriza los píxeles True de `mask` en polígonos con los atributos de CAMPOS."""
    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask.astype(bool),
                                         transform=transform) if v == 1]
    fila = {"clase": clase, "origen": "", "locacion_id": -1, "verificar": 0, "nota": ""}
    fila.update(atributos)
    datos = pd.DataFrame([fila] * len(geoms), columns=CAMPOS)
    return gpd.GeoDataFrame(datos, geometry=geoms, crs=crs)


def poligonos_a_mascara(poligonos, transform, alto, ancho):
    """
    Rasteriza los polígonos validados a una máscara uint8 (0 fondo, 1 locación, 2 pileta).
    Las piletas se pintan después, así **la pileta le gana a la locación** donde se superponen.
    """
    mask = np.full((alto, ancho), FONDO, dtype=np.uint8)
    if poligonos is None or len(poligonos) == 0:
        return mask
    for clase in (LOCACION, PILETA):
        geoms = [g for g, c in zip(poligonos.geometry, poligonos["clase"])
                 if g is not None and not g.is_empty and int(c) == clase]
        if geoms:
            capa = rasterize(geoms, out_shape=(alto, ancho), transform=transform,
                             fill=0, default_value=1, dtype=np.uint8)
            mask[capa == 1] = clase
    return mask


def _vacia(crs):
    return gpd.GeoDataFrame(pd.DataFrame(columns=CAMPOS), geometry=[], crs=crs)


# ---------------------------------------------------------------- generador

def proponer(img_t2, transform, crs, mes_t1, mes_t2, locaciones, segmentar,
             recorte=None, meses_despues=MESES_DESPUES):
    """
    Máscara propuesta como polígonos editables, a partir de la imagen T2 que ve SAM.

    img_t2, transform, crs: imagen de 6 bandas de T2 (el recorte o un contexto más grande).
    locaciones: GeoDataFrame del padrón.
    segmentar: función (rgb, fila, col) -> máscara booleana (ver `segmentador_sam`).
    recorte: (fila0, col0, tamano) para quedarse con lo que cae en el recorte; None = todo.
    Devuelve un GeoDataFrame con CAMPOS (clase=1 locación) en `crs`.
    """
    alto, ancho = img_t2.shape[1:]
    rgb = rgb8(img_t2)
    puntos = locaciones_nuevas_para_clic(locaciones, transform, crs, alto, ancho,
                                         mes_t1, mes_t2, meses_despues)
    partes = []
    for p in puntos.itertuples():
        m = parte_del_clic(np.asarray(segmentar(rgb, int(p.fila), int(p.col)), dtype=bool),
                           int(p.fila), int(p.col))
        if not m.any():
            continue
        nota = "; ".join(filter(None, [
            p.nota, "primer pozo posterior a T2: confirmar que se vea" if p.verificar else ""]))
        partes.append(mascara_a_poligonos(m, transform, crs, origen="sam",
                                          locacion_id=int(p.locacion_id),
                                          verificar=int(p.verificar), nota=nota))
    if not partes:
        return _vacia(crs)
    propuesta = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=crs)

    if recorte is not None:
        fila0, col0, tamano = recorte
        x0, y0 = xy(transform, fila0, col0, offset="ul")
        x1, y1 = xy(transform, fila0 + tamano, col0 + tamano, offset="ul")
        caja = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        propuesta["geometry"] = propuesta.geometry.intersection(caja)
        propuesta = propuesta[~propuesta.geometry.is_empty & (propuesta.geometry.area > 0)]
    return propuesta.reset_index(drop=True)


def guardar_propuesta(propuesta, ruta):
    """Guarda la propuesta como GeoPackage (capa `mascara`), el formato que se edita en QGIS."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    propuesta = propuesta.copy()
    for col, tipo in [("clase", int), ("locacion_id", int), ("verificar", int)]:
        propuesta[col] = propuesta[col].astype(tipo)
    propuesta["origen"] = propuesta["origen"].astype(str)
    propuesta["nota"] = propuesta["nota"].fillna("").astype(str)
    propuesta.to_file(ruta, layer="mascara", driver="GPKG")


def _fila_indice(carpeta, id_par):
    indice = leer_indice_pares(carpeta)
    fila = indice[indice["id"] == id_par]
    if fila.empty:
        raise KeyError(f"{id_par} no está en {carpeta}/indice.csv")
    return fila.iloc[0]


def proponer_par(carpeta, id_par, locaciones, segmentar, imagen_mensual_t2=None):
    """
    Arma los archivos de validación de un par de la carpeta:
        <id>_propuesta.gpkg  máscara propuesta (SAM sobre T2)
        <id>_dndvi.tif       caída de NDVI entre T1 y T2
    imagen_mensual_t2: ruta a la imagen mensual de la zona en `mes_t2` (opcional). Si está,
    SAM ve 512 px alrededor del recorte; si no, ve el recorte T2.
    Devuelve la propuesta.
    """
    fila = _fila_indice(carpeta, id_par)
    base = os.path.join(carpeta, id_par)
    with rasterio.open(f"{base}_t1.tif") as src:
        img_t1 = src.read().astype(np.float32)
    with rasterio.open(f"{base}_t2.tif") as src:
        img_t2, t_rec, crs, tamano = src.read().astype(np.float32), src.transform, src.crs, src.height

    caida = caida_ndvi(img_t1, img_t2).astype(np.float32)
    guardar_tif(f"{base}_dndvi.tif", caida[None], t_rec.c, t_rec.f, crs, "float32")

    if imagen_mensual_t2:
        img, t, f0, c0 = leer_contexto(imagen_mensual_t2, t_rec.c, t_rec.f, tamano)
        recorte = (f0, c0, tamano)
    else:
        img, t, recorte = img_t2, t_rec, None
    propuesta = proponer(img, t, crs, str(fila["mes_t1"]), str(fila["mes_t2"]),
                         locaciones, segmentar, recorte=recorte)
    guardar_propuesta(propuesta, f"{base}_propuesta.gpkg")
    return propuesta


# ---------------------------------------------------------------- vuelta: QGIS → máscara

def validar_par(carpeta, id_par, ruta_gpkg=None):
    """
    Convierte el GeoPackage corregido en QGIS en `<id>_mask.tif` (misma grilla que el par)
    y marca `validada=True` en `indice.csv`. Devuelve la máscara.
    """
    base = os.path.join(carpeta, id_par)
    ruta_gpkg = ruta_gpkg or f"{base}_propuesta.gpkg"
    with rasterio.open(f"{base}_t2.tif") as src:
        transform, alto, ancho, crs = src.transform, src.height, src.width, src.crs

    poligonos = gpd.read_file(ruta_gpkg, layer="mascara")
    if poligonos.crs is not None and crs is not None and poligonos.crs != crs:
        poligonos = poligonos.to_crs(crs)
    poligonos = poligonos[poligonos["clase"].notna()]
    mask = poligonos_a_mascara(poligonos, transform, alto, ancho)
    guardar_tif(f"{base}_mask.tif", mask[None], transform.c, transform.f, crs, "uint8")

    indice = leer_indice_pares(carpeta)
    indice.loc[indice["id"] == id_par, "validada"] = True
    indice.to_csv(os.path.join(carpeta, "indice.csv"), index=False)
    return mask


# ---------------------------------------------------------------- línea de comandos

def _main(argv=None):
    """
    python -m vmseg.mascaras_propuestas validar <carpeta> [id ...]

    Sin ids, valida todos los pares de la carpeta que tengan <id>_propuesta.gpkg.
    """
    import argparse

    p = argparse.ArgumentParser(prog="python -m vmseg.mascaras_propuestas")
    sub = p.add_subparsers(dest="comando", required=True)
    v = sub.add_parser("validar", help="GeoPackage corregido en QGIS → <id>_mask.tif")
    v.add_argument("carpeta")
    v.add_argument("ids", nargs="*")
    args = p.parse_args(argv)

    ids = args.ids or sorted(f[:-len("_propuesta.gpkg")] for f in os.listdir(args.carpeta)
                             if f.endswith("_propuesta.gpkg"))
    for id_par in ids:
        mask = validar_par(args.carpeta, id_par)
        n_loc, n_pil = int((mask == LOCACION).sum()), int((mask == PILETA).sum())
        print(f"{id_par}: {n_loc} px de locación nueva, {n_pil} px de pileta nueva → {id_par}_mask.tif")


if __name__ == "__main__":
    _main()
