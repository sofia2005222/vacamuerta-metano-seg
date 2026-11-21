"""
Validar pares sin mover el mapa de QGIS: todo desde la terminal (docs/VALIDACION.md).

Uso, parada en la carpeta del repo:
    python scripts/revisar_par.py siguiente              próximo par sin validar: arma la figura y la abre
    python scripts/revisar_par.py lista                  pares sin validar (positivos primero)
    python scripts/revisar_par.py ver <id>               figura del par en revision/<id>.png
    python scripts/revisar_par.py zoom <id> <fila> <col> T1, T2 y caída de NDVI ampliados alrededor de ese píxel
    python scripts/revisar_par.py borrar <id> <n> [n…]   saca los polígonos n
    python scripts/revisar_par.py clase <id> <n> <1|2>   polígono n pasa a locación (1) o pileta (2)
    python scripts/revisar_par.py agregar <id> <fila> <col> [clase]
                                                         nueva locación: la mancha de caída de NDVI
                                                         que contiene ese píxel
    python scripts/revisar_par.py rectangulo <id> <fila_arriba> <col_izq> <fila_abajo> <col_der> [clase]
                                                         dibuja a mano un rectángulo (para locaciones en
                                                         obra que la detección automática no toma)
    python scripts/revisar_par.py nota <id> <n> <texto>  nota en el polígono n
    python scripts/revisar_par.py deshacer <id>          vuelve atrás el último cambio
    python scripts/revisar_par.py ok <id>                máscara final (como `mascaras_propuestas validar`)
    python scripts/revisar_par.py corregir <id> <t1_menos_12|preexistente> <nota>
                                                         la locación ya se ve en T1: fila en
                                                         data/correcciones_pares.csv (no se valida)

La carpeta de pares es pares_<persona>; por defecto pares_agus (cambiar con CARPETA=...).
"""
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from vmseg.mascaras_propuestas import validar_par
from vmseg.par import leer_indice_pares

CARPETA = os.environ.get("CARPETA", "pares_agus")
SALIDA = "revision"
UMBRAL_NDVI = 0.05      # caída de NDVI desde la que un píxel cuenta como desmonte
UMBRAL_BRILLO = 0.06    # aumento de brillo (banda roja, descontado el cambio típico de la imagen)


def cambio(t1, t2, dndvi):
    """
    Píxeles que cambiaron como un desmonte: cae el NDVI (se perdió vegetación) o sube mucho
    el brillo (suelo removido donde casi no había vegetación). El aumento de brillo se mide
    contra la mediana de toda la imagen, para que no cuente el cambio de estación.
    """
    sube = t2[2] - t1[2]
    sube = sube - np.nanmedian(sube)
    cae = np.nan_to_num(dndvi) - np.nanmedian(dndvi)          # también sin el cambio de estación
    return (cae > UMBRAL_NDVI) | (np.nan_to_num(sube) > UMBRAL_BRILLO)


def es_compacta(region):
    """Forma de locación: compacta y llena su rectángulo (no una franja ni una mancha deshilachada)."""
    from shapely.geometry import MultiPoint
    puntos = np.argwhere(region)
    rect = MultiPoint(puntos[:, ::-1] + 0.5).minimum_rotated_rectangle.buffer(0.5, join_style=2)
    lleno = region.sum() / max(rect.area, 1)
    x, y = rect.exterior.coords.xy
    lados = sorted(np.hypot(np.diff(x), np.diff(y))[:2])
    return lleno > 0.55 and lados[0] / max(lados[1], 1e-6) > 0.3


def leer_revision():
    """{id: decisión} de <carpeta>/revision.csv, el mismo archivo que usa `python -m vmseg.revisar`."""
    ruta = os.path.join(CARPETA, "revision.csv")
    if not os.path.exists(ruta):
        return {}
    r = pd.read_csv(ruta)
    return dict(zip(r["id"], r["decision"]))


def guardar_decision(id_par, decision):
    from datetime import datetime
    revision = leer_revision()
    revision[id_par] = decision
    pd.DataFrame({"id": list(revision), "decision": list(revision.values()),
                  "fecha": datetime.now().isoformat(timespec="seconds")}).to_csv(
        os.path.join(CARPETA, "revision.csv"), index=False)


def rutas(id_par):
    base = os.path.join(CARPETA, id_par)
    return {k: f"{base}_{k}.tif" for k in ("t1", "t2", "dndvi")} | {"gpkg": f"{base}_propuesta.gpkg"}


def leer_poligonos(id_par):
    return gpd.read_file(rutas(id_par)["gpkg"], layer="mascara")


def guardar_poligonos(id_par, poligonos):
    import shutil
    ruta = rutas(id_par)["gpkg"]
    shutil.copyfile(ruta, ruta.replace(".gpkg", "_anterior.gpkg"))   # para `deshacer`
    poligonos.reset_index(drop=True).to_file(ruta, layer="mascara", driver="GPKG")


def deshacer(id_par):
    """Vuelve la propuesta al estado anterior al último cambio (borrar, agregar, clase o nota)."""
    import shutil
    ruta = rutas(id_par)["gpkg"]
    anterior = ruta.replace(".gpkg", "_anterior.gpkg")
    if not os.path.exists(anterior):
        print("No hay ningún cambio para deshacer.")
        return
    shutil.move(anterior, ruta)
    print("Deshecho el último cambio. Si ya habías corrido `ok`, corrélo de nuevo.")
    ver(id_par)


def rgb(img):
    """Color natural (B4, B3, B2) estirado entre los percentiles 2 y 98."""
    x = np.stack([img[2], img[1], img[0]], -1)
    lo, hi = np.nanpercentile(x, (2, 98))
    return np.clip((x - lo) / (hi - lo), 0, 1)


def puntos_en(ruta_puntos, bounds, crs):
    """Puntos de un archivo (geojson o csv con lon/lat) que caen en el recorte."""
    if ruta_puntos.endswith(".csv"):
        df = pd.read_csv(ruta_puntos)
        lon = next(c for c in df.columns if c.lower() in ("lon", "longitud", "x"))
        lat = next(c for c in df.columns if c.lower() in ("lat", "latitud", "y"))
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    else:
        g = gpd.read_file(ruta_puntos)
    g = g.to_crs(crs)
    return g.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]


def ndvi(img):
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


def manchas_sin_poligono(id_par, t1, t2, dndvi, poligonos, transform, min_px=50, ndvi_max=0.10):
    """
    Manchas de caída de NDVI que quedan sin polígono y en T2 parecen suelo pelado
    (NDVI bajo, como una locación construida): candidatas a locación nueva que SAM no marcó.
    """
    from rasterio.features import geometry_mask
    tapado = np.zeros(dndvi.shape, bool)
    if len(poligonos):
        tapado = ~geometry_mask(list(poligonos.geometry.buffer(20)), out_shape=dndvi.shape, transform=transform)
    desmonte = ndimage.binary_fill_holes(ndimage.binary_opening(cambio(t1, t2, dndvi)))
    manchas, n = ndimage.label(desmonte)
    candidatas = []
    for k in range(1, n + 1):
        region = manchas == k
        if region.sum() < min_px or (region & tapado).sum() / region.sum() > 0.3:
            continue
        if ndvi(t2)[region].mean() > ndvi_max or not es_compacta(region):
            continue
        fila, col = np.argwhere(region).mean(0).round().astype(int)
        # un píxel que seguro esté adentro de la mancha, para `agregar`
        dentro = np.argwhere(region)[np.argmin(((np.argwhere(region) - [fila, col]) ** 2).sum(1))]
        candidatas.append(dict(fila=int(dentro[0]), col=int(dentro[1]), ha=region.sum() / 100,
                               rojo_t1=t1[2][region].mean(), rojo_t2=t2[2][region].mean(),
                               ndvi_t1=ndvi(t1)[region].mean(), ndvi_t2=ndvi(t2)[region].mean()))
    return candidatas


def otros_pozos(bounds, crs):
    """Pozos del padrón completo que no son de Vaca Muerta (otras formaciones), en el recorte."""
    ruta = "data/pozos_todos.csv"
    if not os.path.exists(ruta):
        return gpd.GeoDataFrame()
    df = pd.read_csv(ruta)
    df = df[~df["vaca_muerta"].astype(bool)]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(crs)
    return g.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]


def ver(id_par):
    r = rutas(id_par)
    with rasterio.open(r["t1"]) as s:
        t1, transform, crs, bounds = s.read(), s.transform, s.crs, s.bounds
    with rasterio.open(r["t2"]) as s:
        t2 = s.read()
    with rasterio.open(r["dndvi"]) as s:
        dndvi = s.read(1)
    poligonos = leer_poligonos(id_par)
    padron = puntos_en("data/locaciones_vaca_muerta.geojson", bounds, crs)
    otros = otros_pozos(bounds, crs)
    candidatas = manchas_sin_poligono(id_par, t1, t2, dndvi, poligonos, transform)
    fila_indice = leer_indice_pares(CARPETA).set_index("id").loc[id_par]

    def a_pixel(x, y):
        col, fila = ~transform * (x, y)
        return col, fila

    fig, axes = plt.subplots(1, 4, figsize=(28, 7.6))
    paneles = [(rgb(t1), f"T1 {fila_indice['mes_t1']} (antes)"),
               (rgb(t2), f"T2 {fila_indice['mes_t2']} (después)"),
               (dndvi, "Caída de NDVI (blanco = se perdió vegetación)"),
               (rgb(t2), "T2 con la propuesta (rojo = locación, azul = pileta)")]
    for k, (ax, (img, titulo)) in enumerate(zip(axes, paneles)):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
                  vmax=0.15 if img.ndim == 2 else None)
        ax.set_title(titulo, fontsize=15)
        ax.set_xticks(range(0, img.shape[1] + 1, 32))
        ax.set_yticks(range(0, img.shape[0] + 1, 32))
        ax.grid(True, color="yellow", alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=9)
        # el contorno de los polígonos va en todos los paneles, para comparar
        for n, fila in poligonos.iterrows():
            geoms = getattr(fila.geometry, "geoms", [fila.geometry])
            for g in geoms:
                xs, ys = zip(*[a_pixel(x, y) for x, y in g.exterior.coords])
                color = "red" if fila["clase"] == 1 else "cyan"
                ax.plot(xs, ys, color=color, lw=2 if k == 3 else 1, alpha=1 if k == 3 else 0.6)
            if k == 3:
                cx, cy = a_pixel(fila.geometry.centroid.x, fila.geometry.centroid.y)
                ax.text(cx + 10 * n, cy, str(n), color="white", fontsize=16, weight="bold", ha="center",
                        va="center", bbox=dict(facecolor="red", alpha=0.7, pad=1))
        for _, p in otros.iterrows():
            px, py = a_pixel(p.geometry.x, p.geometry.y)
            ax.plot(px, py, "^", color="orange", mec="black", ms=7)
            if k == 3:
                ax.text(px + 3, py + 6, f"{str(p['fecha_inicio_perforacion'])[:7]} {p['formacion']}",
                        color="orange", fontsize=8)
        if k == 2:
            for j, c in enumerate(candidatas):
                ax.plot(c["col"], c["fila"], "s", mfc="none", mec="magenta", ms=22, mew=2)
                ax.text(c["col"] + 12, c["fila"], f"?{j}", color="magenta", fontsize=14, weight="bold")
        for _, p in padron.iterrows():
            px, py = a_pixel(p.geometry.x, p.geometry.y)
            ax.plot(px, py, "o", mfc="none", mec="white", ms=7)
            if k == 3:
                ax.text(px + 3, py - 3, f"{p['locacion_id']} ({str(p['fecha_primer_pozo'])[:7]})",
                        color="white", fontsize=9)
    fig.suptitle(f"{id_par}  ·  split {fila_indice['split']}  ·  nota: {fila_indice['nota'] if isinstance(fila_indice['nota'], str) else '-'}",
                 fontsize=16)
    fig.tight_layout()
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, f"{id_par}.png")
    fig.savefig(ruta, dpi=70)
    plt.close(fig)

    print(f"Figura: {ruta}")
    if len(poligonos) == 0:
        print("Sin polígonos propuestos.")
    for j, c in enumerate(candidatas):
        print(f"  ?{j}: cambio SIN polígono, {c['ha']:.1f} ha en (fila {c['fila']}, col {c['col']}) | "
              f"brillo {c['rojo_t1']:.2f} -> {c['rojo_t2']:.2f}, NDVI {c['ndvi_t1']:.2f} -> {c['ndvi_t2']:.2f}")
        print(f"       si es una locación nueva: python scripts/revisar_par.py agregar {id_par} {c['fila']} {c['col']}")
        print(f"       para mirarla de cerca:    python scripts/revisar_par.py zoom {id_par} {c['fila']} {c['col']}")
    for n, fila in poligonos.iterrows():
        clase = {1: "locación", 2: "pileta"}.get(fila["clase"], fila["clase"])
        extra = f" | nota: {fila['nota']}" if isinstance(fila.get("nota"), str) and fila["nota"] else ""
        print(f"  polígono {n}: {clase}, {fila.geometry.area / 1e4:.1f} ha, "
              f"locación del padrón {fila.get('locacion_id')}{extra}")


def zoom(id_par, fila, col, lado=48):
    """T1, T2 y caída de NDVI ampliados en una ventana de `lado` píxeles alrededor de (fila, col)."""
    fila, col, lado = int(fila), int(col), int(lado)
    r = rutas(id_par)
    with rasterio.open(r["t1"]) as s:
        t1 = s.read()
    with rasterio.open(r["t2"]) as s:
        t2 = s.read()
    with rasterio.open(r["dndvi"]) as s:
        dndvi = s.read(1)
    f0, f1 = max(0, fila - lado // 2), min(t1.shape[1], fila + lado // 2)
    c0, c1 = max(0, col - lado // 2), min(t1.shape[2], col + lado // 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, img, titulo in zip(axes, [rgb(t1[:, f0:f1, c0:c1]), rgb(t2[:, f0:f1, c0:c1]), dndvi[f0:f1, c0:c1]],
                               ["T1 (antes)", "T2 (después)", "Caída de NDVI"]):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
                  vmax=0.15 if img.ndim == 2 else None, extent=(c0, c1, f1, f0))
        ax.plot(col, fila, "+", color="magenta", ms=20, mew=2)
        ax.set_title(titulo, fontsize=15)
    fig.suptitle(f"{id_par}: zoom en (fila {fila}, col {col}), {lado * 10} m de lado", fontsize=15)
    fig.tight_layout()
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, f"{id_par}_zoom_{fila}_{col}.png")
    fig.savefig(ruta, dpi=70)
    plt.close(fig)
    print(f"Zoom: {ruta}")
    os.system(f"open {ruta}")


def borrar(id_par, *numeros):
    poligonos = leer_poligonos(id_par)
    guardar_poligonos(id_par, poligonos.drop(index=[int(n) for n in numeros]))
    print(f"Borrados: {', '.join(numeros)}. Ojo: los polígonos que quedan se renumeran desde 0.")
    ver(id_par)


def clase(id_par, n, nueva):
    poligonos = leer_poligonos(id_par)
    poligonos.loc[int(n), "clase"] = int(nueva)
    guardar_poligonos(id_par, poligonos)
    ver(id_par)


def nota(id_par, n, *texto):
    poligonos = leer_poligonos(id_par)
    poligonos.loc[int(n), "nota"] = " ".join(texto)
    guardar_poligonos(id_par, poligonos)
    print("Nota guardada.")


def agregar(id_par, fila, col, nueva_clase="1"):
    """Agrega la mancha de cambio (caída de NDVI o aumento de brillo) que contiene el píxel (fila, col)."""
    fila, col = int(fila), int(col)
    r = rutas(id_par)
    with rasterio.open(r["dndvi"]) as s:
        dndvi, transform = s.read(1), s.transform
    with rasterio.open(r["t1"]) as s:
        t1 = s.read()
    with rasterio.open(r["t2"]) as s:
        t2 = s.read()
    desmonte = ndimage.binary_opening(cambio(t1, t2, dndvi))   # saca píxeles sueltos
    manchas, _ = ndimage.label(desmonte)
    etiqueta = manchas[fila, col]
    if etiqueta == 0:
        print(f"En ({fila}, {col}) no hay caída de NDVI: probá con un píxel más adentro de la locación.")
        return
    region = ndimage.binary_fill_holes(manchas == etiqueta)
    geom = [shape(g) for g, v in shapes(region.astype(np.uint8), mask=region, transform=transform) if v == 1]
    poligonos = leer_poligonos(id_par)
    nuevo = gpd.GeoDataFrame({"clase": [int(nueva_clase)], "origen": ["manual"], "locacion_id": [None],
                              "verificar": [""], "nota": ["agregada desde el cambio entre T1 y T2"]},
                             geometry=[max(geom, key=lambda g: g.area)], crs=poligonos.crs)
    guardar_poligonos(id_par, pd.concat([poligonos, nuevo], ignore_index=True))
    print(f"Agregada: {nuevo.geometry.area.iloc[0] / 1e4:.1f} ha")
    ver(id_par)


def rectangulo(id_par, fila0, col0, fila1, col1, nueva_clase="1"):
    """Agrega un polígono rectangular entre las filas y columnas dadas (en píxeles de la grilla)."""
    from shapely.geometry import box
    with rasterio.open(rutas(id_par)["t2"]) as s:
        transform = s.transform
    f0, f1 = sorted((int(fila0), int(fila1)))
    c0, c1 = sorted((int(col0), int(col1)))
    x0, y0 = transform * (c0, f0)
    x1, y1 = transform * (c1, f1)
    poligonos = leer_poligonos(id_par)
    nuevo = gpd.GeoDataFrame({"clase": [int(nueva_clase)], "origen": ["manual"], "locacion_id": [None],
                              "verificar": [""], "nota": ["rectángulo dibujado a mano"]},
                             geometry=[box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))],
                             crs=poligonos.crs)
    guardar_poligonos(id_par, pd.concat([poligonos, nuevo], ignore_index=True))
    print(f"Agregado rectángulo de {nuevo.geometry.area.iloc[0] / 1e4:.1f} ha")
    ver(id_par)


def ok(id_par):
    mask = validar_par(CARPETA, id_par)
    guardar_decision(id_par, "aceptar")
    print(f"Validado {id_par}: {int((mask == 1).sum())} píxeles de locación, {int((mask == 2).sum())} de pileta.")


def corregir(id_par, accion, *texto):
    if accion not in ("t1_menos_12", "preexistente"):
        print("La acción tiene que ser t1_menos_12 o preexistente.")
        return
    if id_par not in set(leer_indice_pares(CARPETA)["id"]):
        print(f"No existe el par {id_par} en {CARPETA}: revisá que esté bien escrito. No se anotó nada.")
        return
    ruta = "data/correcciones_pares.csv"
    if os.path.exists(ruta):
        correcciones = pd.read_csv(ruta, dtype=str)
        correcciones = correcciones[correcciones["id"] != id_par]   # si ya estaba, se reemplaza
    else:
        correcciones = pd.DataFrame(columns=["id", "accion", "nota"])
    fila = pd.DataFrame([{"id": id_par, "accion": accion, "nota": " ".join(texto)}])
    pd.concat([correcciones, fila], ignore_index=True).to_csv(ruta, index=False)
    guardar_decision(id_par, accion)
    print(f"Anotado en {ruta}: {id_par} -> {accion}. Este par no se valida; Sofi lo regenera.")


def pendientes():
    """Pares del índice sin decisión en revision.csv, primero los positivos."""
    indice = leer_indice_pares(CARPETA)
    decididos = set(leer_revision())
    faltan = indice[~indice["id"].isin(decididos)]
    return indice, faltan.assign(negativo=faltan["id"].str.endswith("_neg")).sort_values(["negativo", "id"])


def siguiente():
    indice, faltan = pendientes()
    if faltan.empty:
        print("¡No queda ningún par por validar!")
        return
    print(f"Faltan {len(faltan)} de {len(indice)}.")
    id_par = faltan["id"].iloc[0]
    ver(id_par)
    os.system(f"open {os.path.join(SALIDA, id_par + '.png')}")   # se abre en Vista Previa


def lista():
    indice, faltan = pendientes()
    print(f"Hechos: {len(indice) - len(faltan)} de {len(indice)}")
    for _, f in faltan.head(15).iterrows():
        nota_ = f" | {f['nota']}" if isinstance(f["nota"], str) else ""
        print(f"  {f['id']} ({f['split']}){nota_}")


if __name__ == "__main__":
    comando, *args = sys.argv[1:] or ["lista"]
    if args and comando not in ("lista", "siguiente") and not os.path.exists(rutas(args[0])["t2"]):
        sys.exit(f"No existe el par {args[0]} en {CARPETA}: revisá que esté bien escrito.")
    {"siguiente": siguiente, "lista": lista, "ver": ver, "zoom": zoom, "borrar": borrar, "clase": clase, "nota": nota,
     "agregar": agregar, "rectangulo": rectangulo, "deshacer": deshacer, "ok": ok, "corregir": corregir}[comando](*args)
