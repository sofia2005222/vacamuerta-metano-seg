import pandas as pd

from vmseg.armar_pares import encuadre_de, meses_par, pares_positivos

ENCUADRES = pd.DataFrame([
    dict(encuadre="anelo_b05_e0", zona="anelo", bloque=5, split="train", x_min=0, y_max=2560),
    dict(encuadre="anelo_b05_e1", zona="anelo", bloque=5, split="train", x_min=2000, y_max=2560),
])


def test_meses_par_menos_6_y_mas_1():
    assert meses_par("2021-03-15") == ("2020-09", "2021-04", "")


def test_meses_par_no_se_pasa_del_ultimo_mes_exportado():
    t1, t2, nota = meses_par("2023-12-04")
    assert (t1, t2) == ("2023-06", "2023-12")
    assert nota != ""


def test_encuadre_de_elige_el_que_deja_la_locacion_mas_lejos_del_borde():
    # x = 2100: en e0 queda a 460 m del borde derecho, en e1 a 100 m del izquierdo
    assert encuadre_de(2100, 1280, ENCUADRES)["encuadre"] == "anelo_b05_e0"
    # x = 2500: en e0 a 60 m, en e1 a 500 m
    assert encuadre_de(2500, 1280, ENCUADRES)["encuadre"] == "anelo_b05_e1"


def test_encuadre_de_devuelve_none_si_ninguno_la_contiene():
    assert encuadre_de(9000, 1280, ENCUADRES) is None


def test_pares_positivos_un_par_por_locacion():
    locs = pd.DataFrame([
        dict(locacion_id=738, x=2500, y=1280, fecha_primer_pozo="2021-03-15"),
        dict(locacion_id=12, x=100, y=100, fecha_primer_pozo="2020-01-02"),
    ])
    p = pares_positivos(locs, ENCUADRES)
    assert list(p["id"]) == ["anelo_b05_e1_738", "anelo_b05_e0_12"]
    assert list(p["mes_t1"]) == ["2020-09", "2019-07"]
    assert list(p["mes_t2"]) == ["2021-04", "2020-02"]
    assert (p["split"] == "train").all() and not p["validada"].any()
    assert list(p["x_min"]) == [2000, 0]
