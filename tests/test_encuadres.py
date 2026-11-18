import geopandas as gpd
import numpy as np
from rasterio.warp import transform

from vmseg.encuadres import bloque_de, bloques, encuadres
from vmseg.gee import RUTA_ZONAS
from vmseg.recorte import CRS, RESOLUCION, TAMANO

LADO = TAMANO * RESOLUCION  # 2560 m


def test_cantidad_de_encuadres_por_zona():
    assert len(encuadres("anelo")) == 16 * 16
    assert len(encuadres("externa")) == 12 * 16
    assert encuadres("anelo")["encuadre"].is_unique


def test_encuadres_caen_en_la_grilla_de_10_m():
    for zona in ["anelo", "externa"]:
        e = encuadres(zona)
        assert (e["x_min"] % RESOLUCION == 0).all()
        assert (e["y_max"] % RESOLUCION == 0).all()


def test_ningun_encuadre_cruza_el_borde_de_su_bloque():
    for zona in ["anelo", "externa"]:
        cajas = bloques(zona)
        for _, e in encuadres(zona).iterrows():
            xs = [e.x_min, e.x_min + LADO, e.x_min, e.x_min + LADO]
            ys = [e.y_max, e.y_max, e.y_max - LADO, e.y_max - LADO]
            lon, lat = transform(CRS, "EPSG:4326", xs, ys)
            lon_min, lat_min, lon_max, lat_max = cajas[e.bloque]
            assert lon_min <= min(lon) and max(lon) <= lon_max, e.encuadre
            assert lat_min <= min(lat) and max(lat) <= lat_max, e.encuadre


def test_bloque_de_coincide_con_el_de_armar_dataset():
    loc = gpd.read_file(RUTA_ZONAS.replace("zonas_estudio.json", "locaciones_vaca_muerta.geojson"))
    loc = loc[loc["en_zona_anelo"]]
    calculado = bloque_de("anelo", loc.geometry.x.values, loc.geometry.y.values)
    assert np.array_equal(calculado, loc["bloque_anelo"].astype(int).values)


def test_split_de_los_encuadres():
    a = encuadres("anelo")
    assert set(a["split"]) == {"train", "val", "test_anelo"}
    assert (a.loc[a["bloque"] == 0, "split"] == "train").all()  # bloque 0: sin locaciones, negativos de train
    assert (encuadres("externa")["split"] == "test_externo").all()
