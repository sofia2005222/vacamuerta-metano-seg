"""
Revisión rápida de los pares antes de QGIS (issues #12, #13 y #14, docs/VALIDACION.md).

    python -m vmseg.revisar pares_<persona>

Muestra cada par pendiente con T1, T2 y la caída de NDVI, los polígonos de SAM, el padrón y los
pozos de otras formaciones, y se decide con una tecla:

    A  la propuesta está bien: escribe <id>_mask.tif y marca validada=True
    C  hay que corregirla en QGIS (scripts/qgis_abrir_par.py los abre en fila)
    T  la locación ya se ve en T1: fila t1_menos_12 en data/correcciones_pares.csv
    P  la locación ya estaba en enero de 2019: fila preexistente
    ←  volver al anterior · →  saltear · Q  salir

Las decisiones se guardan en <carpeta>/revision.csv. Decidir de nuevo un par deshace lo anterior
(si estaba aceptado, la máscara vuelve a quedar sin etiquetar; si estaba en correcciones, sale).
Los pares para corregir vuelven a aparecer al final: después de editarlos en QGIS, se aceptan con A.
"""
import argparse
import os
import textwrap
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from vmseg.armar_pares import RUTA_CORRECCIONES, RUTA_LOCACIONES, RUTA_POZOS_TODOS
from vmseg.mascaras_propuestas import rgb8, validar_par
from vmseg.par import guardar_tif, leer_indice_pares
from vmseg.recorte import LOCACION, PILETA, SIN_ETIQUETA

REVISION = "revision.csv"
DECISIONES = ("aceptar", "corregir", "t1_menos_12", "preexistente")
A_CORRECCIONES = ("t1_menos_12", "preexistente")
TECLAS = {"a": "aceptar", "c": "corregir", "t": "t1_menos_12", "p": "preexistente"}
AYUDA = "A aceptar · C corregir en QGIS · T ya se ve en T1 · P ya estaba en 2019 · ← volver · → saltear · Q salir"


# ---------------------------------------------------------------- decisiones

def leer_revision(carpeta):
    """{id: decisión} de <carpeta>/revision.csv (vacío si todavía no se revisó nada)."""
    ruta = os.path.join(carpeta, REVISION)
    if not os.path.exists(ruta):
        return {}
    r = pd.read_csv(ruta)
    return dict(zip(r["id"], r["decision"]))


def _guardar_revision(carpeta, revision):
    pd.DataFrame({"id": list(revision), "decision": list(revision.values()),
                  "fecha": datetime.now().isoformat(timespec="seconds")}).to_csv(
        os.path.join(carpeta, REVISION), index=False)


def _validadas(carpeta):
    indice = leer_indice_pares(carpeta)
    return list(indice["id"]), dict(zip(indice["id"], indice["validada"].astype(str).str.lower() == "true"))


def _desvalidar(carpeta, id_par):
    """Vuelve la máscara a SIN_ETIQUETA y validada=False."""
    base = os.path.join(carpeta, id_par)
    with rasterio.open(f"{base}_t2.tif") as src:
        t, alto, ancho, crs = src.transform, src.height, src.width, src.crs
    guardar_tif(f"{base}_mask.tif", np.full((1, alto, ancho), SIN_ETIQUETA), t.c, t.f, crs, "uint8")
    indice = leer_indice_pares(carpeta)
    indice.loc[indice["id"] == id_par, "validada"] = False
    indice.to_csv(os.path.join(carpeta, "indice.csv"), index=False)


def decidir(carpeta, id_par, decision, ruta_correcciones=RUTA_CORRECCIONES):
    """Registra la decisión sobre un par y la aplica, deshaciendo la anterior si había."""
    if decision not in DECISIONES:
        raise ValueError(f"decisión desconocida: {decision}")
    if decision in A_CORRECCIONES and id_par.endswith("_neg"):
        raise ValueError("un negativo no tiene locación del padrón: si apareció algo, C y marcalo en QGIS")

    revision = leer_revision(carpeta)
    anterior = revision.get(id_par)
    if anterior == "aceptar":
        _desvalidar(carpeta, id_par)
    if anterior in A_CORRECCIONES:
        corr = pd.read_csv(ruta_correcciones)
        corr[corr["id"] != id_par].to_csv(ruta_correcciones, index=False)

    if decision == "aceptar":
        validar_par(carpeta, id_par)
    elif decision in A_CORRECCIONES:
        corr = pd.read_csv(ruta_correcciones)
        fila = pd.DataFrame([{"id": id_par, "accion": decision,
                              "nota": f"revisión rápida ({os.path.basename(os.path.abspath(carpeta))})"}])
        pd.concat([corr, fila], ignore_index=True).to_csv(ruta_correcciones, index=False)

    revision[id_par] = decision
    _guardar_revision(carpeta, revision)


def pendientes(carpeta):
    """Pares sin validar: primero los que nadie miró y al final los que estaban para corregir."""
    ids, validada = _validadas(carpeta)
    revision = leer_revision(carpeta)
    nuevos = [i for i in ids if not validada[i] and i not in revision]
    return nuevos + pares_a_corregir(carpeta)


def pares_a_corregir(carpeta):
    ids, validada = _validadas(carpeta)
    revision = leer_revision(carpeta)
    return [i for i in ids if revision.get(i) == "corregir" and not validada[i]]


# ---------------------------------------------------------------- ventana

def _leer(ruta):
    with rasterio.open(ruta) as src:
        return src.read().astype(np.float32), src.bounds, src.crs


class Revisor:
    """Muestra los pares pendientes de una carpeta de a uno y aplica la decisión de cada tecla."""

    def __init__(self, carpeta, ruta_correcciones=RUTA_CORRECCIONES, mostrar=True,
                 ruta_locaciones=RUTA_LOCACIONES, ruta_pozos=RUTA_POZOS_TODOS):
        self.carpeta, self.ruta_correcciones, self.mostrar = carpeta, ruta_correcciones, mostrar
        self.ids = pendientes(carpeta)
        self.indice = leer_indice_pares(carpeta).set_index("id")
        self.pos = 0
        self.mensaje = ""
        if mostrar:
            self._preparar(ruta_locaciones, ruta_pozos)
            self.dibujar()

    @property
    def actual(self):
        return self.ids[self.pos] if self.pos < len(self.ids) else None

    def tecla(self, key):
        self.mensaje = ""
        if key in TECLAS and self.actual is not None:
            try:
                decidir(self.carpeta, self.actual, TECLAS[key], self.ruta_correcciones)
                self.pos += 1
            except ValueError as e:
                self.mensaje = str(e)
        elif key == "left":
            self.pos = max(0, self.pos - 1)
        elif key == "right" and self.actual is not None:
            self.pos += 1
        elif key == "q" and self.mostrar:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            return
        else:
            return
        if self.mostrar:
            self.dibujar()

    # -- dibujo

    def _preparar(self, ruta_locaciones, ruta_pozos):
        import matplotlib.pyplot as plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:  # que no choquen con A C T P ← → Q
            plt.rcParams[k] = [t for t in plt.rcParams[k] if t not in ("a", "c", "t", "p", "q", "left", "right")]
        _, _, crs = _leer(os.path.join(self.carpeta, f"{self.indice.index[0]}_t2.tif"))
        self.padron = gpd.read_file(ruta_locaciones).to_crs(crs)
        pozos = pd.read_csv(ruta_pozos)
        pozos = pozos[~pozos["vaca_muerta"].astype(str).str.lower().eq("true")]
        self.otros = gpd.GeoDataFrame(pozos, geometry=gpd.points_from_xy(pozos["lon"], pozos["lat"]),
                                      crs="EPSG:4326").to_crs(crs)
        self.fig, self.axs = plt.subplots(1, 3, figsize=(18, 7), sharex=True, sharey=True)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.03)
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.tecla(e.key))
        self.fig.canvas.manager.set_window_title(f"Revisión rápida · {self.carpeta}")

    def dibujar(self):
        import matplotlib.pyplot as plt
        for ax in self.axs:
            ax.clear()
            ax.set_xticks([]), ax.set_yticks([])
        if self.actual is None:
            self._dibujar_fin()
            self.fig.canvas.draw_idle()
            return

        i, fila = self.actual, self.indice.loc[self.actual]
        base = os.path.join(self.carpeta, i)
        t1, b, _ = _leer(f"{base}_t1.tif")
        t2, _, _ = _leer(f"{base}_t2.tif")
        dndvi, _, _ = _leer(f"{base}_dndvi.tif")
        ext = (b.left, b.right, b.bottom, b.top)
        self.axs[0].imshow(rgb8(t1), extent=ext)
        self.axs[1].imshow(rgb8(t2), extent=ext)
        self.axs[2].imshow(np.nan_to_num(dndvi[0]), extent=ext, cmap="gray", vmin=0, vmax=0.15)
        self.axs[0].set_title(f"T1 {fila['mes_t1']} (antes)")
        self.axs[1].set_title(f"T2 {fila['mes_t2']} (después)")
        self.axs[2].set_title("caída de NDVI (blanco = se desmontó)")

        prop = gpd.read_file(f"{base}_propuesta.gpkg", layer="mascara")
        padron = self.padron.cx[b.left:b.right, b.bottom:b.top]
        otros = self.otros.cx[b.left:b.right, b.bottom:b.top]
        for ax in self.axs:
            for clase, color in [(LOCACION, "red"), (PILETA, "deepskyblue")]:
                p = prop[prop["clase"] == clase]
                if len(p):
                    p.boundary.plot(ax=ax, color=color, linewidth=1.3)
            if len(padron):
                ax.scatter(padron.geometry.x, padron.geometry.y, s=14, c="white", edgecolors="black", linewidths=0.6)
            if len(otros):
                ax.scatter(otros.geometry.x, otros.geometry.y, s=18, marker="^", c="orange",
                           edgecolors="black", linewidths=0.5)
            ax.set_xlim(b.left, b.right), ax.set_ylim(b.bottom, b.top)
        for p in padron.itertuples():
            self.axs[1].annotate(str(p.fecha_primer_pozo)[:7], (p.geometry.x, p.geometry.y), fontsize=7,
                                 color="white", xytext=(3, 3), textcoords="offset points")
        for p in prop[prop["verificar"] == 1].itertuples():
            c = p.geometry.representative_point()
            self.axs[1].annotate("¿se ve?", (c.x, c.y), color="yellow", fontsize=9, ha="center")

        negativo = i.endswith("_neg")
        anterior = leer_revision(self.carpeta).get(i)
        notas = [str(fila["nota"])] if pd.notna(fila["nota"]) and str(fila["nota"]) else []
        notas += sorted({n for n in prop["nota"].dropna() if n})
        lineas = [f"[{self.pos + 1}/{len(self.ids)}]  {i}  ·  {fila['split']}  ·  "
                  + ("NEGATIVO: no debería aparecer nada" if negativo else f"{len(prop)} polígono(s) de SAM")
                  + (f"  ·  antes: {anterior}" if anterior else "")]
        lineas += textwrap.wrap(" | ".join(notas), 150)
        self.fig.suptitle("\n".join(lineas), fontsize=11)
        self._pie()
        self.fig.canvas.draw_idle()

    def _pie(self):
        for t in list(self.fig.texts):
            if t is not self.fig._suptitle:
                t.remove()
        self.fig.text(0.5, 0.02, AYUDA, ha="center", fontsize=11)
        if self.mensaje:
            self.fig.text(0.5, 0.06, self.mensaje, ha="center", color="red", fontsize=11)

    def _dibujar_fin(self):
        r = pd.Series(leer_revision(self.carpeta)).value_counts()
        a_corregir = len(pares_a_corregir(self.carpeta))
        self.fig.suptitle("Listo: no quedan pares sin mirar", fontsize=13)
        self.axs[1].text(0.5, 0.5, "\n".join(f"{d}: {n}" for d, n in r.items())
                         + (f"\n\n{a_corregir} para corregir en QGIS" if a_corregir else "")
                         + "\n\n← para volver · Q para salir", ha="center", va="center", fontsize=13,
                         transform=self.axs[1].transAxes)
        self._pie()


def resumen(carpeta):
    ids, validada = _validadas(carpeta)
    r = pd.Series(leer_revision(carpeta)).value_counts().to_dict()
    a_corregir = pares_a_corregir(carpeta)
    print(f"{sum(validada.values())}/{len(ids)} pares validados · decisiones: {r}")
    if a_corregir:
        print(f"{len(a_corregir)} para corregir en QGIS (scripts/qgis_abrir_par.py los abre en orden); "
              "después volvé a correr la revisión y aceptalos con A.")


def _main(argv=None):
    p = argparse.ArgumentParser(prog="python -m vmseg.revisar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", help="carpeta de tus pares, por ejemplo pares_sofi")
    args = p.parse_args(argv)
    if not pendientes(args.carpeta):
        resumen(args.carpeta)
        return
    import matplotlib.pyplot as plt
    Revisor(args.carpeta)
    plt.show()
    resumen(args.carpeta)


if __name__ == "__main__":
    _main()
