# Método: pipeline e hipótesis

## Pipeline del proyecto

1. **Imágenes:** Sentinel-2 (gratis, desde 2017, revisita de 5 días, 10 m/píxel en RGB+NIR, 20 m en SWIR). Neuquén es árido: pocas nubes. Se descargan desde Google Earth Engine ([ADR-0010](adr/0010-google-earth-engine.md)). La imagen de cada mes es la **mediana** de los píxeles despejados del mes, con las nubes enmascaradas con Cloud Score+ (umbral 0,60). Es el procedimiento del ejemplo oficial del catálogo de Earth Engine; Cloud Score+: Pasquarella et al. (2023), CVPR Workshops, doi:10.1109/CVPRW59228.2023.00206.
2. **Etiquetado:** en todos los splits, **SAM** arma la máscara propuesta usando el centro de cada locación del padrón como clic, y una persona la valida y corrige ([ADR-0008](adr/0008-mascaras-dibujadas-sobre-imagen.md)). Clases: `locación` (terminada o en obra) y `pileta` (y fondo); donde hay pileta sobre una locación, se pinta pileta. **Caminos quedan afuera**: a 10 m/píxel son muy finos. Los puntos oficiales indican dónde buscar. Donde OpenStreetMap ya tiene el polígono de la locación (`industrial=wellsite` u `oil`), ese polígono es la máscara propuesta; hay que citar OSM (licencia ODbL). Las máscaras propuestas se generan en Colab con `segment-geospatial` y se validan en QGIS. Antes de etiquetar, se prueba SAM con 20 locaciones, porque a 10 m/píxel puede andar mal; si anda mal, se dibuja en QGIS partiendo de OSM o de cero. Las piletas no están en el padrón: la persona hace un clic en cada una y SAM arma el contorno. Si hay duda, se anota en un campo `nota`. Cada recorte se etiqueta sobre la imagen de un mes elegido al azar entre 2019 y 2023, así se mezclan locaciones terminadas, en obra y todavía inexistentes. Un mismo lugar puede aparecer en varias fechas, pero siempre en el mismo split. Meta provisoria: **300 recortes**, 240 en la Zona Añelo (~170 `train`, 25 `val`, 45 `test_anelo`, según bloques) y 60 en la Zona externa; se ajusta según lo que responda el profesor. Meta inicial: 200–400 recortes de 256×256 px (consultar con el profesor el número).
3. **Modelo:** U-Net (librería `segmentation_models_pytorch`) con encoder preentrenado en ImageNet. Comparar desde cero vs. encoder congelado vs. fine-tuning.
4. **Evaluación:** IoU y F1 por clase, matriz de confusión a nivel píxel, precisión y recall por objeto, ejemplos buenos/malos, comparación test Añelo vs. test externo (detalle en [EVALUACION.md](EVALUACION.md)).
5. **Despliegue / backtesting:** correr el modelo sobre imágenes mensuales 2019–2023, encontrar la primera detección de cada locación nueva y compararla con `fecha_primer_pozo` (inicio de perforación). Métricas: % de locaciones nuevas capturadas y **meses de anticipación** (mediana). Criterios en [EVALUACION.md](EVALUACION.md).

## Hipótesis y experimentos

- **H1 (bandas):** agregar las bandas infrarrojas (NIR, SWIR) a RGB mejora la distinción entre locaciones y suelo desnudo natural de la estepa. Experimento: **3 bandas (RGB) vs. 6 bandas (RGB + NIR + SWIR B11 y B12, remuestreadas a 10 m)**; requiere adaptar la primera capa del encoder. Si sobra tiempo, se agrega la variante intermedia RGB + NIR.
- **H2 (transfer learning):** fine-tuning > encoder congelado > desde cero, sobre todo con pocos datos.
- **H3 (generalización):** el desempeño cae en la zona de test externa (otras operadoras y otro paisaje). La diferencia entre test Añelo y test externo mide la generalización.
- **Extra opcional:** agregar unas pocas imágenes etiquetadas de la zona externa al entrenamiento y medir cuánto se recupera el desempeño (adaptación con pocos datos).


## Fuera de alcance (por ahora)

Quedan como trabajo futuro, para no sobredimensionar el TP:
- Pérdida robusta a ruido de etiqueta (pensada para etiquetas débiles).
- Preentrenamiento con el Alberta Wells Dataset.
- Comparación con el modelo de ESRI ("Well Pad Detection – Permian Basin"), que requiere ArcGIS.
- SAM como línea de base: se usa solo para armar máscaras propuestas.

## Acuerdo entre etiquetadoras

Se eligen 10 recortes y cada uno se valida dos veces: la dueña del bloque y otra persona, en rotación (Sofi → Agus → Bian → Sofi). El IoU entre las dos validaciones es el techo realista del modelo.
