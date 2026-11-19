import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from vmseg.mascaras_propuestas import guardar_propuesta
from vmseg.par import generar_pares_falsos, guardar_tif, leer_indice_pares, leer_par
from vmseg.recorte import CRS, LOCACION, SIN_ETIQUETA
from vmseg.revisar import Revisor, decidir, leer_revision, pares_a_corregir, pendientes


@pytest.fixture
def carpeta(tmp_path):
    """Pares como los de un zip de validación: máscara sin etiquetar y propuesta de SAM."""
    c = tmp_path / "pares_sofi"
    indice = generar_pares_falsos(str(c), n=3)
    indice["validada"] = False
    indice.to_csv(c / "indice.csv", index=False)
    for f in indice.itertuples():
        guardar_tif(str(c / f"{f.id}_mask.tif"), np.full((1, 256, 256), SIN_ETIQUETA), f.x_min, f.y_max, CRS, "uint8")
        prop = gpd.GeoDataFrame(dict(clase=[1], origen=["sam"], locacion_id=[7], verificar=[0], nota=[""]),
                                geometry=[box(f.x_min, f.y_max - 100, f.x_min + 100, f.y_max)], crs=CRS)
        guardar_propuesta(prop, str(c / f"{f.id}_propuesta.gpkg"))
    return str(c)


@pytest.fixture
def correcciones(tmp_path):
    ruta = tmp_path / "correcciones_pares.csv"
    ruta.write_text("id,accion,nota\n")
    return str(ruta)


def _ids(carpeta):
    return list(leer_indice_pares(carpeta)["id"])


def test_aceptar_escribe_la_mascara_de_la_propuesta_y_valida(carpeta, correcciones):
    a = _ids(carpeta)[0]
    decidir(carpeta, a, "aceptar", correcciones)
    _, _, mask = leer_par(carpeta, a)
    assert (mask == LOCACION).sum() == 100 and (mask != SIN_ETIQUETA).all()
    assert leer_indice_pares(carpeta).set_index("id").loc[a, "validada"]
    assert leer_revision(carpeta) == {a: "aceptar"}


def test_t1_y_preexistente_van_a_correcciones(carpeta, correcciones):
    a, b, _ = _ids(carpeta)
    decidir(carpeta, a, "t1_menos_12", correcciones)
    decidir(carpeta, b, "preexistente", correcciones)
    c = pd.read_csv(correcciones)
    assert list(zip(c["id"], c["accion"])) == [(a, "t1_menos_12"), (b, "preexistente")]


def test_cambiar_de_decision_deshace_la_anterior(carpeta, correcciones):
    a = _ids(carpeta)[0]
    decidir(carpeta, a, "aceptar", correcciones)
    decidir(carpeta, a, "t1_menos_12", correcciones)
    _, _, mask = leer_par(carpeta, a)
    assert (mask == SIN_ETIQUETA).all() and not leer_indice_pares(carpeta).set_index("id").loc[a, "validada"]
    decidir(carpeta, a, "corregir", correcciones)
    assert pd.read_csv(correcciones).empty
    assert leer_revision(carpeta) == {a: "corregir"}


def test_un_negativo_no_puede_ir_a_correcciones(carpeta, correcciones):
    with pytest.raises(ValueError):
        decidir(carpeta, "anelo_b02_e0_neg", "t1_menos_12", correcciones)


def test_pendientes_primero_los_nuevos_y_al_final_los_que_estaban_para_corregir(carpeta, correcciones):
    a, b, c = _ids(carpeta)
    decidir(carpeta, a, "corregir", correcciones)
    decidir(carpeta, b, "aceptar", correcciones)
    assert pendientes(carpeta) == [c, a]
    assert pares_a_corregir(carpeta) == [a]


def test_revisor_con_teclas(carpeta, correcciones):
    a, b, c = _ids(carpeta)
    r = Revisor(carpeta, ruta_correcciones=correcciones, mostrar=False)
    r.tecla("a")            # acepta a, pasa a b
    r.tecla("c")            # b para corregir, pasa a c
    r.tecla("left")         # vuelve a b
    r.tecla("a")            # ahora b se acepta
    r.tecla("x")            # tecla desconocida: no hace nada
    assert r.actual == c
    assert leer_revision(carpeta) == {a: "aceptar", b: "aceptar"}
