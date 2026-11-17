from vmseg.gee import mes_de_escena, recuadro, tabla_cobertura


def test_mes_de_escena():
    assert mes_de_escena("20190105T143751_20190105T144421_T19HEB") == "2019-01"


def test_recuadro_anelo():
    lon_min, lat_min, lon_max, lat_max = recuadro("anelo")
    assert lon_min < lon_max and lat_min < lat_max
    assert (lon_min, lat_max) == (-68.85, -38.12)


def test_tabla_cobertura_cuenta_por_mes_e_incluye_meses_vacios():
    ids_s2 = [
        "20190105T143751_20190105T144421_T19HEB",
        "20190110T143751_20190110T144421_T19HEB",
        "20190304T143751_20190304T144421_T19HEB",
    ]
    ids_cs = ids_s2[:1] + ids_s2[2:]

    t = tabla_cobertura(ids_s2, ids_cs, "2019-01-01", "2019-04-01")

    assert list(t["mes"]) == ["2019-01", "2019-02", "2019-03"]
    assert list(t["escenas_s2"]) == [2, 0, 1]
    assert list(t["con_cloudscore"]) == [1, 0, 1]
    assert list(t["sin_cloudscore"]) == [1, 0, 0]


def test_tabla_cobertura_ignora_escenas_fuera_de_rango():
    t = tabla_cobertura(["20240101T000000_x_T19HEB"], [], "2019-01-01", "2019-02-01")
    assert t["escenas_s2"].sum() == 0


def test_meses_del_periodo_son_60():
    from vmseg.exportar import meses

    m = meses()
    assert len(m) == 60
    assert (m[0], m[1], m[-1]) == ("2019-01", "2019-02", "2023-12")


def test_tabla_indice_marca_los_meses_malos_y_ordena():
    from vmseg.indice import tabla_indice

    filas = [
        ("externa", "2019-02", 10, 100.0, 2),
        ("anelo", "2019-02", 12, 80.126, 0),
        ("anelo", "2019-01", 13, 99.5, 3),
    ]

    t = tabla_indice(filas, umbral=99.0, min_despejadas=3)

    assert list(t["zona"]) == ["anelo", "anelo", "externa"]
    assert list(t["mes"]) == ["2019-01", "2019-02", "2019-02"]
    assert list(t["pct_validos"]) == [99.5, 80.13, 100.0]
    assert list(t["nublado"]) == [False, True, False]
    assert list(t["pocas_despejadas"]) == [False, True, True]
