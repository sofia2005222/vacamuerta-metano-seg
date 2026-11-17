"""
Issue #4: exportar las imágenes mensuales 2019–2023 de las dos zonas.

Uso:
    python -m vmseg.exportar --proyecto <proyecto-de-cloud>

Lanza una tarea de Earth Engine por zona y mes (120 en total). Las tareas
corren en los servidores de Google y dejan gs://<bucket>/<zona>/<mes>.tif.
"""
import argparse

import pandas as pd

from vmseg import gee

INICIO, FIN = "2019-01", "2023-12"


def meses(inicio=INICIO, fin=FIN):
    """Lista de meses entre inicio y fin, los dos incluidos: ['2019-01', ..., '2023-12']."""
    return list(pd.period_range(inicio, fin, freq="M").strftime("%Y-%m"))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--proyecto", help="proyecto de Cloud registrado en GEE (o $EE_PROJECT)")
    args = p.parse_args()

    ee = gee.inicializar(args.proyecto)
    for zona in gee.ZONAS:
        for mes in meses():
            gee.exportar_mes(ee, zona, mes)
            print(zona, mes)


if __name__ == "__main__":
    main()
