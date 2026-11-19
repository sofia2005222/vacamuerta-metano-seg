"""
Abre un par en QGIS listo para validar su máscara propuesta (issue #9, ADR-0011).

Cómo se usa:
  1. En QGIS: Plugins → Python Console → ícono "Show Editor" → abrir este archivo → ▶ Run Script.
  2. Elegí `revision.csv` de tu carpeta de pares: abre el primer par que marcaste con C en la
     revisión rápida (python -m vmseg.revisar). Para el próximo, escribí `siguiente()` en la consola.
     También podés elegir cualquier archivo de un par (por ejemplo <id>_t2.tif) para abrir ese.
  3. Se arma y se guarda el proyecto <id>.qgz con estas capas (de arriba hacia abajo):
       - pozos de otras formaciones: triángulos naranjas con el mes y la formación (no están en el
         padrón de locaciones, pero sus locaciones se ven igual y van en la máscara),
       - padrón: puntos con el mes del primer pozo,
       - máscara (editar): la propuesta (rojo = locación nueva, azul = pileta nueva;
         "¿se ve?" = locación perforada después de T2: confirmá que ya esté),
       - caída de NDVI (apagada): blanco = se perdió vegetación entre T1 y T2,
       - T2 (después) y T1 (antes, apagada).
     Prendé y apagá T1 / T2 / NDVI con las casillas del panel Layers para ver qué cambió.
  4. Corregí la capa "máscara (editar)": lápiz (Toggle Editing) → Vertex Tool para mover bordes,
     Add Polygon Feature para dibujar (clase 1 = locación nueva, 2 = pileta nueva),
     Delete Selected para borrar **lo que ya estaba en T1**. Guardá con "Save Layer Edits".
  5. Cuando termines, volvé a correr la revisión rápida: los pares corregidos aparecen al final
     y se aceptan con A (eso escribe la máscara final).

Si QGIS no encuentra el padrón, cambiá RUTA_PADRON.
"""
import csv
import os

from qgis.core import (
    QgsCategorizedSymbolRenderer, QgsContrastEnhancement, QgsDefaultValue,
    QgsEditorWidgetSetup, QgsFillSymbol, QgsMarkerSymbol, QgsMultiBandColorRenderer,
    QgsPalLayerSettings, QgsProject, QgsRasterLayer, QgsRasterMinMaxOrigin,
    QgsRendererCategory, QgsSingleBandGrayRenderer, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.utils import iface

# padrón de locaciones del repo (si los pares están en <repo>/pares_<persona>/, se encuentra solo)
RUTA_PADRON = ""
# pozos de todas las formaciones (data/pozos_todos.csv): se muestran los que no son de Vaca Muerta
RUTA_POZOS_TODOS = ""
SUFIJOS = ("_t1.tif", "_t2.tif", "_mask.tif", "_dndvi.tif", "_propuesta.gpkg")


def capa_rgb(ruta, nombre):
    capa = QgsRasterLayer(ruta, nombre)
    capa.setRenderer(QgsMultiBandColorRenderer(capa.dataProvider(), 3, 2, 1))   # B4, B3, B2
    capa.setContrastEnhancement(QgsContrastEnhancement.StretchToMinimumMaximum,
                                QgsRasterMinMaxOrigin.CumulativeCut)
    return capa


def capa_dndvi(ruta):
    capa = QgsRasterLayer(ruta, "caída de NDVI (blanco = se desmontó)")
    proveedor = capa.dataProvider()
    gris = QgsSingleBandGrayRenderer(proveedor, 1)
    contraste = QgsContrastEnhancement(proveedor.dataType(1))
    contraste.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
    contraste.setMinimumValue(0.0)
    contraste.setMaximumValue(0.15)
    gris.setContrastEnhancement(contraste)
    capa.setRenderer(gris)
    return capa


def apagar(proyecto, capa):
    nodo = proyecto.layerTreeRoot().findLayer(capa.id())
    if nodo is not None:
        nodo.setItemVisibilityChecked(False)


def a_corregir(carpeta):
    """Pares marcados con C en la revisión rápida (revision.csv) que todavía no se validaron."""
    ruta = os.path.join(carpeta, "revision.csv")
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        corregir = {r["id"] for r in csv.DictReader(f) if r["decision"] == "corregir"}
    with open(os.path.join(carpeta, "indice.csv"), encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f)
                if r["id"] in corregir and r["validada"].strip().lower() != "true"]


def abrir(carpeta, id_par):
    """Arma y guarda el proyecto <id>.qgz con las capas del par."""
    base = os.path.join(carpeta, id_par)
    ruta_padron = RUTA_PADRON or os.path.join(carpeta, "..", "data", "locaciones_vaca_muerta.geojson")
    ruta_pozos = RUTA_POZOS_TODOS or os.path.join(carpeta, "..", "data", "pozos_todos.csv")

    proyecto = QgsProject.instance()
    proyecto.clear()

    # se agregan de abajo hacia arriba
    t1 = capa_rgb(f"{base}_t1.tif", "T1 (antes)")
    t2 = capa_rgb(f"{base}_t2.tif", "T2 (después)")
    proyecto.addMapLayer(t1)
    proyecto.addMapLayer(t2)
    proyecto.setCrs(t2.crs())
    apagar(proyecto, t1)

    if os.path.exists(f"{base}_dndvi.tif"):
        dndvi = capa_dndvi(f"{base}_dndvi.tif")
        proyecto.addMapLayer(dndvi)
        apagar(proyecto, dndvi)

    ruta_gpkg = f"{base}_propuesta.gpkg"
    if os.path.exists(ruta_gpkg):
        mascara = QgsVectorLayer(f"{ruta_gpkg}|layername=mascara", "máscara (editar)", "ogr")

        def relleno(rgb):
            return QgsFillSymbol.createSimple(
                {"color": f"{rgb},60", "outline_color": rgb, "outline_width": "0.6"})

        mascara.setRenderer(QgsCategorizedSymbolRenderer("clase", [
            QgsRendererCategory(1, relleno("230,30,30"), "locación nueva"),
            QgsRendererCategory(2, relleno("30,120,255"), "pileta nueva"),
        ]))
        etiqueta = QgsPalLayerSettings()
        etiqueta.fieldName = "if(\"verificar\" = 1, '¿se ve?', '')"
        etiqueta.isExpression = True
        mascara.setLabeling(QgsVectorLayerSimpleLabeling(etiqueta))
        mascara.setLabelsEnabled(True)

        campos = mascara.fields()
        for campo, valor in [("clase", "1"), ("origen", "'manual'"),
                             ("locacion_id", "-1"), ("verificar", "0"), ("nota", "''")]:
            i = campos.indexOf(campo)
            if i >= 0:
                mascara.setDefaultValueDefinition(i, QgsDefaultValue(valor))
        i = campos.indexOf("clase")
        if i >= 0:
            mascara.setEditorWidgetSetup(i, QgsEditorWidgetSetup(
                "ValueMap", {"map": [{"locación nueva": "1"}, {"pileta nueva": "2"}]}))
        proyecto.addMapLayer(mascara)
    else:
        mascara = None
        QMessageBox.warning(None, "Falta la propuesta",
                            f"No encontré {os.path.basename(ruta_gpkg)} al lado del par.")

    if os.path.exists(ruta_padron):
        padron = QgsVectorLayer(ruta_padron, "padrón (mes del 1er pozo)", "ogr")
        padron.renderer().setSymbol(QgsMarkerSymbol.createSimple(
            {"name": "circle", "color": "255,255,255", "outline_color": "0,0,0", "size": "1.6"}))
        texto = QgsPalLayerSettings()
        texto.fieldName = "left(to_string(\"fecha_primer_pozo\"), 7)"
        texto.isExpression = True
        padron.setLabeling(QgsVectorLayerSimpleLabeling(texto))
        padron.setLabelsEnabled(True)
        proyecto.addMapLayer(padron)

    if os.path.exists(ruta_pozos):
        uri = ("file:///" + os.path.abspath(ruta_pozos).replace("\\", "/")
               + "?delimiter=,&xField=lon&yField=lat&crs=EPSG:4326")
        otros = QgsVectorLayer(uri, "pozos de otras formaciones (mes, formación)", "delimitedtext")
        otros.setSubsetString("lower(to_string(\"vaca_muerta\")) = 'false'")
        otros.renderer().setSymbol(QgsMarkerSymbol.createSimple(
            {"name": "triangle", "color": "255,150,0", "outline_color": "0,0,0", "size": "2"}))
        texto = QgsPalLayerSettings()
        texto.fieldName = ("coalesce(left(to_string(\"fecha_inicio_perforacion\"), 7), 'sin fecha')"
                           " || ' ' || \"formacion\"")
        texto.isExpression = True
        otros.setLabeling(QgsVectorLayerSimpleLabeling(texto))
        otros.setLabelsEnabled(True)
        proyecto.addMapLayer(otros)

    # el mapa en el CRS de los pares (UTM 19S) y centrado en el par: si no, QGIS toma el del
    # padrón (lon/lat) y el par queda chiquito en un mapa de toda la provincia
    proyecto.setCrs(t2.crs())
    lienzo = iface.mapCanvas()
    lienzo.setDestinationCrs(t2.crs())
    lienzo.setExtent(t2.extent())
    lienzo.refresh()
    if mascara is not None:
        iface.setActiveLayer(mascara)
    proyecto.write(f"{base}.qgz")
    print("Proyecto guardado:", f"{base}.qgz")


COLA = {"carpeta": None, "ids": [], "pos": -1}


def siguiente():
    """Abre el próximo par para corregir (escribilo en la consola)."""
    COLA["pos"] += 1
    if COLA["pos"] >= len(COLA["ids"]):
        print("No quedan pares para corregir. Volvé a correr la revisión rápida y aceptalos con A.")
        return
    id_par = COLA["ids"][COLA["pos"]]
    print(f"[{COLA['pos'] + 1}/{len(COLA['ids'])}] {id_par}")
    abrir(COLA["carpeta"], id_par)


ruta_elegida, _ = QFileDialog.getOpenFileName(
    None, "Elegí revision.csv (pares para corregir) o un archivo de un par", "",
    "Pares (revision.csv *.tif *.gpkg)")
if ruta_elegida:
    carpeta = os.path.dirname(ruta_elegida)
    nombre = os.path.basename(ruta_elegida)
    COLA.update(carpeta=carpeta, ids=a_corregir(carpeta), pos=-1)
    if nombre == "revision.csv":
        print(f"{len(COLA['ids'])} pares para corregir. Para pasar al próximo: siguiente()")
        siguiente()
    else:
        id_par = next((nombre[:-len(s)] for s in SUFIJOS if nombre.endswith(s)), os.path.splitext(nombre)[0])
        if id_par in COLA["ids"]:
            COLA["pos"] = COLA["ids"].index(id_par)
        abrir(carpeta, id_par)
