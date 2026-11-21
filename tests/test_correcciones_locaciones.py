import geopandas as gpd
import pandas as pd
import pytest

from vmseg.correcciones_locaciones import (
    CARPETA_DATA, leer_correcciones, locaciones_validas, marcar_correcciones, _marcar_archivos,
)

LOC = pd.DataFrame(dict(locacion_id=[1, 2, 3, 4], split=["train"] * 4))


def corr(*filas):
    return pd.DataFrame(filas, columns=["locacion_id_a", "locacion_id_b", "motivo", "quien", "nota"])


def test_marca_sin_borrar_filas():
    loc = marcar_correcciones(LOC, corr(("2", "", "preexistente", "sofi", ""), ("3", "1", "misma_locacion", "agus", "")))
    assert len(loc) == 4
    assert list(loc["correccion"]) == ["", "preexistente", "misma_locacion", ""]
    assert loc["misma_que"].isna().tolist() == [True, True, False, True] and loc["misma_que"].iloc[2] == 1
    assert list(locaciones_validas(loc)["locacion_id"]) == [1, 4]


def test_locacion_que_no_esta_en_el_padron_es_error():
    with pytest.raises(ValueError):
        marcar_correcciones(LOC, corr(("9", "", "preexistente", "sofi", "")))


def test_leer_rechaza_motivos_desconocidos_y_misma_locacion_sin_b(tmp_path):
    for filas in [[("2", "", "borrar", "sofi", "")], [("2", "", "misma_locacion", "sofi", "")],
                  [("2", "", "preexistente", "sofi", ""), ("2", "", "preexistente", "bian", "")]]:
        ruta = tmp_path / "c.csv"
        corr(*filas).to_csv(ruta, index=False)
        with pytest.raises(ValueError):
            leer_correcciones(ruta)


def test_marcar_archivos_es_idempotente_y_no_toca_el_resto(tmp_path):
    loc = pd.DataFrame(dict(locacion_id=["1", "2"], lon=["-68.5", "-68.6"], lat=["-38.2", "-38.3"], split=["train", ""]))
    loc.to_csv(tmp_path / "locaciones_vaca_muerta.csv", index=False)
    gpd.GeoDataFrame(loc.astype({"locacion_id": int}), geometry=gpd.points_from_xy(loc.lon.astype(float), loc.lat.astype(float)),
                     crs=4326).drop(columns=["lon", "lat"]).to_file(tmp_path / "locaciones_vaca_muerta.geojson")
    c = corr(("2", "1", "misma_locacion", "agus", ""))
    _marcar_archivos(tmp_path, c)
    primera = (tmp_path / "locaciones_vaca_muerta.csv").read_text()
    _marcar_archivos(tmp_path, c)
    assert (tmp_path / "locaciones_vaca_muerta.csv").read_text() == primera
    assert primera.splitlines()[2] == "2,-68.6,-38.3,,misma_locacion,1"
    g = gpd.read_file(tmp_path / "locaciones_vaca_muerta.geojson")
    assert list(g["correccion"]) == ["", "misma_locacion"] and g["misma_que"].iloc[1] == 1


def test_padron_real_marcado_con_los_preexistentes_de_los_pares():
    """Cada par sacado como preexistente tiene su locación corregida en el padrón."""
    from vmseg.armar_pares import RUTA_CORRECCIONES
    pares = pd.read_csv(RUTA_CORRECCIONES)
    ids = set(pares.loc[pares["accion"] == "preexistente", "id"].str.rsplit("_", n=1).str[1].astype(int))
    padron = pd.read_csv(f"{CARPETA_DATA}/locaciones_vaca_muerta.csv", keep_default_na=False)
    marcadas = set(padron.loc[padron["correccion"] != "", "locacion_id"])
    assert ids == marcadas == set(leer_correcciones()["locacion_id_a"].astype(int))
