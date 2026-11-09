"""
Issue #1: ¿Cloud Score+ cubre 2019–2023 en las dos zonas?

Uso:
    python -m vmseg.cobertura --proyecto <proyecto-de-cloud>

Imprime, por zona y mes, cuántas escenas de Sentinel-2 L2A tocan la zona y
cuántas tienen Cloud Score+, en una tabla Markdown para pegar en el issue.
"""
import argparse

import pandas as pd

from vmseg import gee

INICIO, FIN = "2019-01-01", "2024-01-01"


def cobertura(ee, inicio=INICIO, fin=FIN):
    tablas = []
    for zona in gee.ZONAS:
        ids_s2 = gee.ids_escenas(ee, gee.S2, zona, inicio, fin)
        ids_cs = gee.ids_escenas(ee, gee.CLOUD_SCORE_PLUS, zona, inicio, fin)
        t = gee.tabla_cobertura(ids_s2, ids_cs, inicio, fin)
        t.insert(0, "zona", zona)
        tablas.append(t)
    return pd.concat(tablas, ignore_index=True)


def a_markdown(df):
    """Tabla Markdown sin depender de `tabulate`."""
    filas = [list(df.columns), ["---"] * len(df.columns)] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(f) + " |" for f in filas)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--proyecto", help="proyecto de Cloud registrado en GEE (o $EE_PROJECT)")
    p.add_argument("--csv", help="además, guardar la tabla en este CSV")
    args = p.parse_args()

    ee = gee.inicializar(args.proyecto)
    tabla = cobertura(ee)
    if args.csv:
        tabla.to_csv(args.csv, index=False)

    faltan = tabla[tabla["sin_cloudscore"] > 0]
    vacios = tabla[tabla["escenas_s2"] == 0]
    totales = tabla.groupby("zona", as_index=False)[["escenas_s2", "con_cloudscore", "sin_cloudscore"]].sum()
    print(a_markdown(totales))
    print(f"\nMeses con escenas sin Cloud Score+: {len(faltan)}")
    print(f"Meses sin ninguna escena: {len(vacios)}\n")
    print(a_markdown(tabla))


if __name__ == "__main__":
    main()
