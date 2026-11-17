"""
Issue #4: índice de las imágenes mensuales, con su porcentaje de píxeles válidos.

Uso:
    python -m vmseg.indice --proyecto <proyecto-de-cloud>

Escribe data/indice_imagenes.csv con una fila por zona y mes. Un mes queda
marcado como `nublado` si sus píxeles válidos no llegan a UMBRAL_VALIDOS, y
como `pocas_despejadas` si algún píxel tiene menos de MIN_DESPEJADAS
observaciones despejadas (la mediana de 1 o 2 es poco confiable).
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from vmseg import gee
from vmseg.exportar import meses

UMBRAL_VALIDOS = 99.0  # % mínimo de píxeles válidos para usar el mes
MIN_DESPEJADAS = 3  # observaciones despejadas que tiene que tener todo píxel
RUTA_INDICE = os.path.join(os.path.dirname(gee.RUTA_ZONAS), "indice_imagenes.csv")


def tabla_indice(filas, umbral=UMBRAL_VALIDOS, min_despejadas=MIN_DESPEJADAS):
    """Arma el índice a partir de filas (zona, mes, escenas, pct_validos, min_despejadas)."""
    tabla = pd.DataFrame(filas, columns=["zona", "mes", "escenas", "pct_validos", "min_despejadas"])
    tabla["pct_validos"] = tabla["pct_validos"].round(2)
    tabla["nublado"] = tabla["pct_validos"] < umbral
    tabla["pocas_despejadas"] = tabla["min_despejadas"] < min_despejadas
    return tabla.sort_values(["zona", "mes"], ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--proyecto", help="proyecto de Cloud registrado en GEE (o $EE_PROJECT)")
    p.add_argument("--csv", default=RUTA_INDICE, help="dónde guardar el índice")
    args = p.parse_args()

    ee = gee.inicializar(args.proyecto)
    pares = [(zona, mes) for zona in gee.ZONAS for mes in meses()]
    # las consultas son independientes: se piden de a 8 para no esperar 120 en fila
    with ThreadPoolExecutor(8) as hilos:
        resultados = list(hilos.map(lambda zm: gee.validos_mes(ee, *zm), pares))

    tabla = tabla_indice([zm + r for zm, r in zip(pares, resultados)])
    tabla.to_csv(args.csv, index=False, lineterminator="\n")
    print(tabla.to_string(index=False))
    print(f"\nMeses nublados (< {UMBRAL_VALIDOS} % de píxeles válidos): {int(tabla['nublado'].sum())}")


if __name__ == "__main__":
    main()
