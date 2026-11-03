# Datos: de dónde sale cada cosa y en qué estado está

Leyenda: ✔ confirmado (existe, verificado por búsqueda) · ❓ sin verificar (hay que comprobarlo la primera vez que se use).

## 1. Imágenes: Sentinel-2 L2A (Copernicus)

- Resolución 10 m en las bandas azul, verde, rojo y NIR. Gratis y abierto.
- Se descarga como **compuesto de mediana** de varias fechas con baja nubosidad (la mediana saca nubes y sombras).
- Vía: catálogo STAC público de AWS (*Earth Search*), implementado en `src/vmseg/stac.py`. ❓ **No probado contra la red real.** Alternativas si falla: Microsoft Planetary Computer (mismo estilo, assets `B02/B03/B04/B08/SCL`) o Copernicus Data Space.
- Las bandas se guardan en este orden: **azul, verde, rojo, NIR** (por eso `data.channels=[2,1,0]` es RGB).

## 2. Etiquetas débiles: padrón oficial de pozos

- Fuente: Secretaría de Energía de la Nación, en datos.gob.ar, dataset *Producción de petróleo y gas por pozo – Capítulo IV*. ✔ El dataset lista un recurso **"Shapefile de Pozos"** (proyección EPSG:4326).
  <https://datos.gob.ar/dataset/energia-produccion-petroleo-gas-por-pozo-capitulo-iv>
- ❓ No se verificó: nombres de columnas, fecha de última actualización, ni si incluye los pozos no convencionales más recientes. **Mirar el mapa de puntos sobre la imagen antes de entrenar.** `prepare-wells` imprime las columnas y filtra solo por caja geográfica.
- Es un punto por pozo: la locación se aproxima con un disco (ver `src/vmseg/labels.py`).

## 3. Control cruzado (opcional): OGIM

- Base global de infraestructura de petróleo y gas (EDF/MethaneSAT), abierta: <https://zenodo.org/records/13259749>; también en Google Earth Engine. ✔ Existe, con capas de pozos y baterías de tanques, entre otras.
- ❓ No se confirmó la cobertura de Neuquén. Sirve para comparar contra el padrón oficial, no como fuente principal.

## 4. Preentrenamiento (opcional): Alberta Wells Dataset

- ✔ <https://zenodo.org/records/13743323>: unos 188.000 parches de Planet (≈ 94.000 con pozos, más de 213.000 pozos), con máscaras de segmentación y cajas. Resolución ≈ 3 m/píxel.
- ❓ No se inspeccionó el formato ni el tamaño de descarga. Ver `docs/ALBERTA.md` por las diferencias con este proyecto.

## 5. Conjunto de prueba manual (lo construye el grupo)

- ~100–150 parches de val/test etiquetados a mano en QGIS. Ver `docs/LABELING.md`. Es la parte más cara en horas y la que le da valor científico al TP.

## 6. Coordenada del caso de estudio

- `configs/aois.yaml → sierra_barrosa` está centrada en la coordenada (−38.76047, −68.83698) que la prensa atribuye al reporte de PNUMA-IMEO. Sirve como **caso cualitativo** en el póster: ¿el modelo segmenta las instalaciones de ese punto? Confirmar la coordenada en la fuente original.

## Licencias y atribución

- Sentinel-2: citar "Contiene datos modificados de Copernicus Sentinel".
- Revisar la licencia de cada fuente antes de redistribuir datos. El repo **no** incluye datos (`.gitignore`).
- Se puede mirar Google Earth como referencia visual para entender qué es un pad, pero **no** se exportan sus imágenes al dataset: se etiqueta sobre Sentinel-2.
