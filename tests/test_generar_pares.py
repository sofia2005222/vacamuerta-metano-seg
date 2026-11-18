import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from vmseg.generar_pares import a_reflectancia, archivos_a_subir, empaquetar, esta_validada, padron_para_clic, recortar
from vmseg.par import COLUMNAS_INDICE_PARES, leer_par
from vmseg.recorte import CRS, SIN_ETIQUETA

X0, Y0 = 500000, 5760000  # esquina de la imagen mensual falsa


def imagen_mensual(ruta, valor, lado=600):
    """Imagen mensual falsa: uint16 como las de GEE, con una franja sin dato (0) a la izquierda."""
    datos = np.full((6, lado, lado), valor, np.uint16)
    datos[:, :, :10] = 0
    with rasterio.open(ruta, "w", driver="GTiff", height=lado, width=lado, count=6, dtype="uint16",
                       crs=CRS, transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(datos)


def indice(**cambios):
    fila = dict(id="anelo_b05_e0_738", encuadre="anelo_b05_e0", locacion_id=738, zona="anelo", bloque=5,
                split="train", mes_t1="2020-09", mes_t2="2021-04", x_min=X0, y_max=Y0 - 1000, crs=CRS,
                validada=False, nota="", responsable="sofi")
    fila.update(cambios)
    return pd.DataFrame([fila], columns=COLUMNAS_INDICE_PARES)


def test_a_reflectancia_divide_recorta_y_pone_nan_sin_dato():
    crudo = np.array([[[0, 500, 12000]]] * 6, np.uint16)
    img = a_reflectancia(crudo)
    assert img.dtype == np.float32
    assert np.isnan(img[:, 0, 0]).all()
    assert img[0, 0, 1] == pytest.approx(0.05) and img[0, 0, 2] == 1.0


def test_recortar_arma_el_par_con_la_mascara_sin_etiquetar(tmp_path):
    imagen_mensual(tmp_path / "2020-09.tif", 1000)
    imagen_mensual(tmp_path / "2021-04.tif", 3000)
    carpeta = tmp_path / "pares"
    recortar(indice(), carpeta=str(carpeta), ruta_mes=lambda zona, mes: str(tmp_path / f"{mes}.tif"))

    t1, t2, mask = leer_par(str(carpeta), "anelo_b05_e0_738")
    assert t1.shape == t2.shape == (6, 256, 256)
    assert np.isnan(t1[:, :, :10]).all() and np.nanmax(t1) == pytest.approx(0.1)
    assert np.nanmax(t2) == pytest.approx(0.3)
    assert (mask == SIN_ETIQUETA).all()
    with rasterio.open(carpeta / "anelo_b05_e0_738_t1.tif") as src:
        assert (src.transform.c, src.transform.f) == (X0, Y0 - 1000)
    with rasterio.open(carpeta / "contexto" / "anelo_b05_e0_738_t2_contexto.tif") as src:
        assert src.width == 512 and src.read(1).max() == 3000


def test_recortar_falla_si_el_encuadre_no_cae_en_la_grilla(tmp_path):
    imagen_mensual(tmp_path / "m.tif", 1000)
    with pytest.raises(ValueError):
        recortar(indice(x_min=X0 + 5), carpeta=str(tmp_path / "p"), ruta_mes=lambda z, m: str(tmp_path / "m.tif"))


def test_padron_para_clic_suma_pozos_de_otras_formaciones_lejos_de_vaca_muerta():
    locs = gpd.GeoDataFrame(dict(locacion_id=[7], fecha_primer_pozo=["2021-03-01"]),
                            geometry=[Point(-68.7, -38.3)], crs="EPSG:4326")
    pozos = pd.DataFrame(dict(
        lon=[-68.7, -68.65, -68.6502, -68.6], lat=[-38.3, -38.3, -38.3, -38.3],
        fecha_inicio_perforacion=["2021-01-05", "2021-02-01", "2021-03-01", None],
        vaca_muerta=[False, False, False, False],
        formacion=["lotena", "centenario", "centenario", "rayoso"], sigla=["A", "B", "C", "D"],
    ))
    p = padron_para_clic(locs, pozos)
    # A cae sobre la locación 7; B y C (a 17 m) son una sola locación nueva; D no tiene fecha.
    assert len(p) == 2
    otra = p[p["locacion_id"] == -1].iloc[0]
    assert otra["fecha_primer_pozo"] == "2021-02-01" and "centenario" in otra["nota"]


def test_esta_validada():
    assert not esta_validada(np.full((4, 4), SIN_ETIQUETA, np.uint8))
    assert esta_validada(np.zeros((4, 4), np.uint8))


def test_empaquetar_un_zip_por_responsable(tmp_path):
    idx = pd.concat([indice(), indice(id="externa_b01_e9_720", responsable="bian")], ignore_index=True)
    for i in idx["id"]:
        for s in ["_t1.tif", "_t2.tif", "_mask.tif", "_dndvi.tif", "_propuesta.gpkg"]:
            (tmp_path / f"{i}{s}").write_bytes(b"x")
    rutas = empaquetar(idx, carpeta=str(tmp_path))
    with zipfile.ZipFile(rutas["bian"]) as z:
        nombres = set(z.namelist())
        assert "pares_bian/indice.csv" in nombres and "pares_bian/externa_b01_e9_720_t2.tif" in nombres
        assert not any("anelo" in n for n in nombres)


def test_subir_no_pisa_mascaras_ni_propuestas_que_ya_estan_en_el_bucket():
    en_bucket = {"a_mask.tif", "a_propuesta.gpkg", "a_t1.tif"}
    assert archivos_a_subir(["a"], en_bucket) == ["a_t1.tif", "a_t2.tif", "a_dndvi.tif"]
    # Un par regenerado (pedido por id) sí se reemplaza entero.
    assert len(archivos_a_subir(["a"], en_bucket, regenerados={"a"})) == 5
