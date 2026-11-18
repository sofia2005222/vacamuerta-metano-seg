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


def test_pares_positivos_junta_locaciones_con_el_mismo_par():
    # Dos locaciones en el mismo encuadre con el mismo mes de perforación dan la misma imagen:
    # queda un solo par, con la otra locación en la nota.
    locs = pd.DataFrame([
        dict(locacion_id=810, x=500, y=1280, fecha_primer_pozo="2021-05-02"),
        dict(locacion_id=704, x=900, y=1280, fecha_primer_pozo="2021-05-30"),
    ])
    p = pares_positivos(locs, ENCUADRES)
    assert list(p["id"]) == ["anelo_b05_e0_810"]
    assert "704" in p["nota"].iloc[0]


# --- Negativos ---

from vmseg.armar_pares import cantidades_por_split, meses_t1_posibles, pares_negativos  # noqa: E402

ENCUADRE = ENCUADRES.iloc[0]  # x de 0 a 2560, y de 0 a 2560
SIN_POZOS = pd.DataFrame(dict(x=[], y=[], mes=pd.PeriodIndex([], freq="M")))


def pozos(*filas):
    """Pozos de prueba: (x, y, mes o None)."""
    x, y, mes = zip(*filas)
    return pd.DataFrame(dict(x=x, y=y, mes=pd.PeriodIndex(mes, freq="M")))


def estacion(mes):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[pd.Period(mes).month]


def test_cantidades_por_split_proporcional_a_los_positivos():
    positivos = pd.DataFrame(dict(split=["train"] * 71 + ["val"] * 18 + ["test_anelo"] * 32 + ["test_externo"] * 51))
    assert cantidades_por_split(positivos, 130) == {"test_anelo": 24, "test_externo": 38, "train": 54, "val": 14}


def test_sin_pozos_todos_los_meses_sirven_y_t2_es_7_meses_despues():
    t1s = meses_t1_posibles(ENCUADRE, SIN_POZOS, set())
    assert len(t1s) == 53 and t1s[0] == "2019-01" and t1s[-1] == "2023-05"


def test_t1_y_t2_caen_siempre_en_estaciones_distintas():
    for t1 in meses_t1_posibles(ENCUADRE, SIN_POZOS, set()):
        assert estacion(t1) != estacion(pd.Period(t1) + 7)


def test_pozo_a_100_m_del_borde_descarta_los_intervalos_en_que_aparece_su_locacion():
    # Perforado en marzo de 2021: la locación puede aparecer desde 6 meses antes.
    # Se descarta todo T1 anterior al pozo con T2 + 6 meses que lo alcance.
    t1s = meses_t1_posibles(ENCUADRE, pozos((2660, 1280, "2021-03")), set())
    assert "2020-01" in t1s and "2021-03" in t1s
    assert "2020-02" not in t1s and "2021-02" not in t1s


def test_pozo_a_200_m_del_borde_no_afecta():
    assert len(meses_t1_posibles(ENCUADRE, pozos((2760, 1280, "2021-03")), set())) == 53


def test_pozo_sin_fecha_descarta_el_encuadre():
    assert meses_t1_posibles(ENCUADRE, pozos((1280, 1280, None)), set()) == []


def test_meses_invalidos_no_se_usan_ni_de_t1_ni_de_t2():
    t1s = meses_t1_posibles(ENCUADRE, SIN_POZOS, {"2020-07"})
    assert "2020-07" not in t1s and "2019-12" not in t1s and len(t1s) == 51


def test_pares_negativos_uno_por_encuadre_y_reproducible():
    encuadres = pd.DataFrame([
        dict(encuadre=f"anelo_b05_e{i}", zona="anelo", bloque=5, split="train", x_min=i * 10000, y_max=2560)
        for i in range(5)
    ])
    con_pozo_sin_fecha = pozos((1280, 1280, None))  # cae en e0
    p = pares_negativos(encuadres, con_pozo_sin_fecha, set(), {"train": 3}, semilla=1)
    assert len(p) == 3 and p["encuadre"].is_unique
    assert "anelo_b05_e0" not in set(p["encuadre"])
    assert (p["locacion_id"] == -1).all() and (p["id"] == p["encuadre"] + "_neg").all()
    assert (p["mes_t2"].map(pd.Period) - p["mes_t1"].map(pd.Period)).map(lambda d: d.n).eq(7).all()
    assert p.equals(pares_negativos(encuadres, con_pozo_sin_fecha, set(), {"train": 3}, semilla=1))
