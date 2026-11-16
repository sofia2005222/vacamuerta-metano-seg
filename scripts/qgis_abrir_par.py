"""
Abre un par en QGIS listo para validar su máscara propuesta (issue #9, ADR-0011).

Cómo se usa (una vez por par):
  1. En QGIS: Plugins → Python Console → ícono "Show Editor" → abrir este archivo → ▶ Run Script.
  2. Elegí cualquier archivo del par (por ejemplo <id>_t2.tif). En la misma carpeta tienen que estar
     <id>_t1.tif, <id>_t2.tif, <id>_dndvi.tif y <id>_propuesta.gpkg (los arma el notebook
     notebooks/generar_mascaras_propuestas.ipynb).
  3. Se arma y se guarda el proyecto <id>.qgz con estas capas (de arriba hacia abajo):
       - padrón: puntos con el mes del primer pozo,
       - máscara (editar): la propuesta (rojo = locación nueva, azul = pileta nueva;
         "¿se ve?" = locación perforada después de T2: confirmá que ya esté),
       - caída de NDVI (apagada): blanco = se perdió vegetación entre T1 y T2,
       - T2 (después) y T1 (antes, apagada).
     Prendé y apagá T1 / T2 / NDVI con las casillas del panel Layers para ver qué cambió.
  4. Corregí la capa "máscara (editar)": lápiz (Toggle Editing) → Vertex Tool para mover bordes,
     Add Polygon Feature para dibujar (clase 1 = locación nueva, 2 = pileta nueva),
     Delete Selected para borrar **lo que ya estaba en T1**. Guardá con "Save Layer Edits".
  5. Convertí a máscara final (terminal, en la carpeta del repo):
       python -m vmseg.mascaras_propuestas validar <carpeta> <id>

Si QGIS no encuentra el padrón, cambiá RUTA_PADRON.
"""
import os

from qgis.core import (
    QgsCategorizedSymbolRenderer, QgsContrastEnhancement, QgsDefaultValue,
    QgsEditorWidgetSetup, QgsFillSymbol, QgsMarkerSymbol, QgsMultiBandColorRenderer,
    QgsPalLayerSettings, QgsProject, QgsRasterLayer, QgsRasterMinMaxOrigin,
    QgsRendererCategory, QgsSingleBandGrayRenderer, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.utils import iface

# padrón de locaciones del repo (si los pares están en <repo>/pares_prueba/, se encuentra solo)
RUTA_PADRON = ""
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


ruta_elegida, _ = QFileDialog.getOpenFileName(None, "Elegí un archivo del par (<id>_t2.tif)", "",
                                              "Archivos del par (*.tif *.gpkg)")
if ruta_elegida:
    carpeta = os.path.dirname(ruta_elegida)
    nombre = os.path.basename(ruta_elegida)
    id_par = next((nombre[:-len(s)] for s in SUFIJOS if nombre.endswith(s)), os.path.splitext(nombre)[0])
    base = os.path.join(carpeta, id_par)
    if not RUTA_PADRON:
        RUTA_PADRON = os.path.join(carpeta, "..", "data", "locaciones_vaca_muerta.geojson")

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

    if os.path.exists(RUTA_PADRON):
        padron = QgsVectorLayer(RUTA_PADRON, "padrón (mes del 1er pozo)", "ogr")
        padron.renderer().setSymbol(QgsMarkerSymbol.createSimple(
            {"name": "circle", "color": "255,255,255", "outline_color": "0,0,0", "size": "1.6"}))
        texto = QgsPalLayerSettings()
        texto.fieldName = "left(to_string(\"fecha_primer_pozo\"), 7)"
        texto.isExpression = True
        padron.setLabeling(QgsVectorLayerSimpleLabeling(texto))
        padron.setLabelsEnabled(True)
        proyecto.addMapLayer(padron)

    if mascara is not None:
        iface.setActiveLayer(mascara)
    iface.mapCanvas().setExtent(t2.extent())
    iface.mapCanvas().refresh()
    proyecto.write(f"{base}.qgz")
    print("Proyecto guardado:", f"{base}.qgz")
