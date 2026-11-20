import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin, xy
from scipy import ndimage
from shapely.geometry import Point, box

from vmseg.mascaras_propuestas import (
    _main, caida_ndvi, guardar_propuesta, leer_contexto, locaciones_nuevas_para_clic,
    mascara_a_poligonos, ndvi, poligonos_a_mascara, proponer, proponer_par, rgb8, validar_par,
)
from vmseg.par import generar_pares_falsos, leer_indice_pares, leer_par
from vmseg.recorte import CRS, LOCACION, PILETA, RESOLUCION, SIN_ETIQUETA

X0, Y0 = 330000, 5770000


def _transform(x_min=X0, y_max=Y0):
    return from_origin(x_min, y_max, RESOLUCION, RESOLUCION)


def _padron(puntos_xy, fechas):
    """Padrón falso: una locación por punto (x, y en el CRS de los pares), en lon/lat como el geojson."""
    return gpd.GeoDataFrame(
        {"locacion_id": range(len(puntos_xy)), "fecha_primer_pozo": fechas},
        geometry=[Point(x, y) for x, y in puntos_xy], crs=CRS,
    ).to_crs("EPSG:4326")


def _sam_falso(region):
    """Imita a SAM: devuelve la región conexa de `region` que contiene el clic."""
    etiquetas, _ = ndimage.label(region)

    def segmentar(rgb, fila, col):
        e = etiquetas[fila, col]
        return etiquetas == e if e else np.zeros_like(region, bool)
    return segmentar


# ---------------------------------------------------------------- polígonos ↔ máscara

def test_poligonos_ida_y_vuelta_reproduce_la_mascara():
    t = _transform()
    mask = np.zeros((64, 64), bool)
    mask[10:20, 5:30] = True
    mask[40:55, 40:50] = True
    pol = mascara_a_poligonos(mask, t, CRS, origen="sam", locacion_id=3)
    assert len(pol) == 2 and set(pol["locacion_id"]) == {3}
    assert np.array_equal(poligonos_a_mascara(pol, t, 64, 64) == LOCACION, mask)


def test_pileta_le_gana_a_locacion():
    locacion = box(X0 + 100, Y0 - 300, X0 + 300, Y0 - 100)      # 20×20 px
    pileta = box(X0 + 150, Y0 - 250, X0 + 200, Y0 - 200)        # 5×5 px, adentro
    pol = gpd.GeoDataFrame({"clase": [PILETA, LOCACION]}, geometry=[pileta, locacion], crs=CRS)
    mask = poligonos_a_mascara(pol, _transform(), 64, 64)
    assert (mask == PILETA).sum() == 25
    assert (mask == LOCACION).sum() == 400 - 25


def test_sin_poligonos_da_todo_fondo():
    assert not poligonos_a_mascara(None, _transform(), 8, 8).any()


def test_propuesta_vacia_se_guarda_como_capa_de_poligonos(tmp_path):
    """Sin tipo de geometría (Unknown), QGIS no ofrece Add Polygon Feature y no se puede dibujar."""
    import pyogrio
    vacia = mascara_a_poligonos(np.zeros((8, 8), bool), _transform(), CRS)
    ruta = tmp_path / "x_neg_propuesta.gpkg"
    guardar_propuesta(vacia, ruta)
    assert pyogrio.read_info(ruta, layer="mascara")["geometry_type"] == "Polygon"


# ---------------------------------------------------------------- puntos del padrón

def test_clic_solo_en_locaciones_nuevas_entre_t1_y_t2():
    t = _transform()
    puntos = [(X0 + 105, Y0 - 105), (X0 + 205, Y0 - 305), (X0 + 305, Y0 - 405),
              (X0 + 405, Y0 - 505), (X0 + 9000, Y0 - 105)]
    fechas = ["2020-11-01",    # antes de T1: ya estaba → sin clic
              "2021-05-10",    # entre T1 y T2 → clic
              "2021-12-01",    # 4 meses después de T2 → clic, verificar
              "2023-01-01",    # mucho después → sin clic
              "2021-05-10"]    # afuera del recorte → sin clic
    p = locaciones_nuevas_para_clic(_padron(puntos, fechas), _transform(), CRS, 64, 64,
                                    "2021-01", "2021-08")
    assert list(p["locacion_id"]) == [1, 2]
    assert list(p["verificar"]) == [0, 1]
    assert (p.iloc[0]["fila"], p.iloc[0]["col"]) == (30, 20)


def test_locaciones_sin_fecha_se_ignoran():
    padron = _padron([(X0 + 105, Y0 - 105)], [None])
    assert locaciones_nuevas_para_clic(padron, _transform(), CRS, 64, 64, "2021-01", "2021-08").empty


# ---------------------------------------------------------------- generador

def test_proponer_recorta_al_recorte_dentro_del_contexto():
    t = _transform()
    img = np.full((6, 128, 128), 0.1, np.float32)
    region = np.zeros((128, 128), bool)
    region[30:50, 30:50] = True                    # cruza el borde del recorte (40–103)
    padron = _padron([xy(t, 45, 45)], ["2021-05-01"])
    prop = proponer(img, t, CRS, "2021-01", "2021-08", padron, _sam_falso(region),
                    recorte=(40, 40, 64))
    assert (poligonos_a_mascara(prop, t, 128, 128) == LOCACION).sum() == 10 * 10


def test_proponer_pasa_la_nota_del_padron_a_los_poligonos():
    # Pozos de otras formaciones: entran al padrón con locacion_id = -1 y una nota para quien valida.
    t = _transform()
    img = np.full((6, 64, 64), 0.1, np.float32)
    region = np.zeros((64, 64), bool)
    region[10:20, 10:20] = True
    padron = _padron([xy(t, 15, 15)], ["2021-05-01"])
    padron["locacion_id"] = -1
    padron["nota"] = "otra formación"
    prop = proponer(img, t, CRS, "2021-01", "2021-08", padron, _sam_falso(region))
    assert list(prop["locacion_id"]) == [-1] and list(prop["nota"]) == ["otra formación"]


def test_proponer_se_queda_con_la_parte_que_toca_el_clic():
    # SAM a veces devuelve la locación más manchitas sueltas por el encuadre (#11).
    t = _transform()
    img = np.full((6, 64, 64), 0.1, np.float32)
    mask = np.zeros((64, 64), bool)
    mask[10:20, 10:20] = True
    mask[40, 40] = mask[50:52, 5:7] = True
    padron = _padron([xy(t, 15, 15)], ["2021-05-01"])
    prop = proponer(img, t, CRS, "2021-01", "2021-08", padron, lambda rgb, f, c: mask)
    assert len(prop) == 1 and (poligonos_a_mascara(prop, t, 64, 64) == LOCACION).sum() == 100


def test_proponer_descarta_mascaras_mas_grandes_que_una_locacion():
    # Si SAM agarra el fondo (todo el encuadre), no se propone nada para ese clic.
    t = _transform()
    img = np.full((6, 128, 128), 0.1, np.float32)
    padron = _padron([xy(t, 15, 15)], ["2021-05-01"])
    prop = proponer(img, t, CRS, "2021-01", "2021-08", padron, lambda rgb, f, c: np.ones((128, 128), bool))
    assert prop.empty


def test_segmentador_sam_elige_la_mejor_mascara_del_tamano_de_una_locacion():
    from vmseg.mascaras_propuestas import MAX_PIXELES, segmentador_sam
    chica = np.zeros((128, 128), bool)
    chica[:10, :10] = True
    fondo = np.ones((128, 128), bool)            # más de MAX_PIXELES

    class Predictor:
        def set_image(self, rgb): pass
        def predict(self, **_): return np.stack([chica, fondo, chica]), np.array([0.5, 0.9, 0.4]), None

    assert fondo.sum() > MAX_PIXELES
    assert segmentador_sam(Predictor())(np.zeros((128, 128, 3), np.uint8), 5, 5).sum() == 100


def test_circuito_completo_con_pares_falsos(tmp_path):
    """proponer_par → (QGIS, acá sin cambios) → validar_par → máscara de cambio en formato de par."""
    indice = generar_pares_falsos(tmp_path, n=4, tamano=64, semilla=3, frac_negativos=0)
    fila = indice[indice["locacion_id"] >= 0].iloc[0]
    id_par = fila["id"]
    img_t1, img_t2, mask_real = leer_par(tmp_path, id_par)
    t = _transform(fila["x_min"], fila["y_max"])

    # padrón: un punto en cada locación nueva (las viejas no están en el padrón como nuevas)
    nueva = np.isin(mask_real, [LOCACION, PILETA])
    etiquetas, n = ndimage.label(nueva)
    centros = ndimage.center_of_mass(nueva, etiquetas, range(1, n + 1))
    padron = _padron([xy(t, int(round(f)), int(round(c))) for f, c in centros], ["2021-05-01"] * n)

    prop = proponer_par(tmp_path, id_par, padron, _sam_falso(nueva))
    assert len(prop) >= n and set(prop["origen"]) == {"sam"}
    assert (tmp_path / f"{id_par}_propuesta.gpkg").exists()

    # la capa de ayuda: el NDVI no cambia donde no hubo obra
    with rasterio.open(tmp_path / f"{id_par}_dndvi.tif") as src:
        dndvi = src.read(1)
    assert dndvi.shape == (64, 64)
    assert np.allclose(dndvi[~nueva], caida_ndvi(img_t1, img_t2)[~nueva], equal_nan=True)

    mask = validar_par(tmp_path, id_par)
    _, _, leida = leer_par(tmp_path, id_par)
    assert np.array_equal(mask, leida)
    valida = mask_real != SIN_ETIQUETA
    # SAM falso marca locación + pileta nuevas juntas como locación
    assert np.array_equal((leida == LOCACION)[valida], nueva[valida])
    assert bool(leer_indice_pares(tmp_path).set_index("id").loc[id_par, "validada"]) is True


def test_proponer_par_con_imagen_mensual_de_contexto(tmp_path):
    indice = generar_pares_falsos(tmp_path, n=1, tamano=64, semilla=0, frac_negativos=0)
    fila = indice.iloc[0]
    # imagen mensual de la zona: 200×200 px que contienen al par
    zona = tmp_path / "zona_t2.tif"
    t_zona = _transform(fila["x_min"] - 500, fila["y_max"] + 500)
    with rasterio.open(zona, "w", driver="GTiff", height=200, width=200, count=6,
                       dtype="float32", crs=CRS, transform=t_zona) as dst:
        dst.write(np.full((6, 200, 200), 0.1, np.float32))
    vistos = []

    def segmentar(rgb, fila_, col_):
        vistos.append(rgb.shape)
        return np.zeros(rgb.shape[:2], bool)

    t = _transform(fila["x_min"], fila["y_max"])
    padron = _padron([xy(t, 30, 30)], ["2021-05-01"])
    proponer_par(tmp_path, fila["id"], padron, segmentar, imagen_mensual_t2=zona)
    assert vistos == [(200, 200, 3)]           # SAM vio el contexto, no el recorte de 64 px


def test_leer_contexto_centra_el_recorte(tmp_path):
    ruta = tmp_path / "zona.tif"
    with rasterio.open(ruta, "w", driver="GTiff", height=1000, width=1000, count=6,
                       dtype="float32", crs=CRS, transform=_transform()) as dst:
        dst.write(np.zeros((6, 1000, 1000), np.float32))
    x_min, y_max = X0 + 400 * RESOLUCION, Y0 - 300 * RESOLUCION
    img, t_ctx, f0, c0 = leer_contexto(ruta, x_min, y_max, 256, lado=512)
    assert img.shape == (6, 512, 512) and (f0, c0) == (128, 128)
    assert xy(t_ctx, f0, c0, offset="ul") == (x_min, y_max)
    # cerca del borde, la ventana se corre pero el recorte sigue adentro
    img, _, f0, c0 = leer_contexto(ruta, X0, Y0, 256, lado=512)
    assert (f0, c0) == (0, 0) and img.shape == (6, 512, 512)


# ---------------------------------------------------------------- imagen y línea de comandos

def test_rgb8_y_ndvi():
    img = np.zeros((6, 4, 4), np.float32)
    img[2], img[3] = 0.1, 0.5                   # rojo bajo, NIR alto: vegetación
    assert rgb8(img).shape == (4, 4, 3) and rgb8(img).dtype == np.uint8
    assert np.allclose(ndvi(img), (0.5 - 0.1) / (0.5 + 0.1))
    despues = img.copy()
    despues[2], despues[3] = 0.3, 0.3           # suelo pelado: NDVI 0
    assert np.allclose(caida_ndvi(img, despues), ndvi(img))


def test_linea_de_comandos_valida_todos(tmp_path, capsys):
    indice = generar_pares_falsos(tmp_path, n=2, tamano=32, semilla=1)
    t = _transform(indice["x_min"][0], indice["y_max"][0])
    pol = gpd.GeoDataFrame({"clase": [LOCACION], "origen": ["manual"], "locacion_id": [-1],
                            "verificar": [0], "nota": [""]},
                           geometry=[box(*xy(t, 10, 0, offset="ul"), *xy(t, 0, 10, offset="ul"))],
                           crs=CRS)
    guardar_propuesta(pol, tmp_path / f"{indice['id'][0]}_propuesta.gpkg")
    _main(["validar", str(tmp_path)])
    assert "100 px de locación nueva" in capsys.readouterr().out
    assert leer_indice_pares(tmp_path)["validada"].tolist() == [True, False]
