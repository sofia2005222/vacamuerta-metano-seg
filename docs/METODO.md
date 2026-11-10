# Método: pipeline e hipótesis

## Pipeline del proyecto

1. **Imágenes:** Sentinel-2 (gratis, desde 2017, revisita de 5 días, 10 m/píxel en RGB+NIR, 20 m en SWIR). Neuquén es árido: pocas nubes. Se descargan desde Google Earth Engine ([ADR-0010](adr/0010-google-earth-engine.md)). La imagen de cada mes es la **mediana** de los píxeles despejados del mes, con las nubes enmascaradas con Cloud Score+ (umbral 0,60). Es el procedimiento del ejemplo oficial del catálogo de Earth Engine; Cloud Score+: Pasquarella et al. (2023), CVPR Workshops, doi:10.1109/CVPRW59228.2023.00206.
2. **Encuadres y pares:** cada bloque se cubre con una grilla fija de 3×3 encuadres de 256×256 px (2,56 km), con un poco de solapamiento entre encuadres del mismo bloque y ninguno cruzando el borde del bloque. En la Zona Añelo son 144 encuadres. La Zona externa también se divide en bloques, que solo sirven para repartir el etiquetado (todos son `test_externo`). La unidad de etiquetado y de entrada al modelo es un **par**: un encuadre en dos imágenes mensuales, una antes (T1) y otra después (T2) ([ADR-0011](adr/0011-modelo-bitemporal-siames.md)). Todos los pares de un encuadre quedan en el split de su bloque.
   - **Positivos:** uno por cada locación nueva del padrón (172 en total, ver [EVALUACION.md](EVALUACION.md)), en el encuadre que la contiene (si cae en el solapamiento de dos, en uno solo). T1 es el mes del inicio de perforación menos 6 y T2, el mes del inicio de perforación más 1. Como la locación nueva arranca en julio de 2019, T1 nunca cae antes del comienzo del archivo. Si en T1 la locación ya se ve, T1 se corre a 12 meses antes; si ya estaba en enero de 2019, es una **locación preexistente** y sale del dataset (se anota para las correcciones del padrón). Los desfasajes salen de la revisión manual de 3 locaciones, que aparecieron entre 5 días y 4,5 meses antes del inicio de perforación.
   - **Plan B si los desfasajes fijos fallan seguido** (por ejemplo, en más de 1 de cada 5 de los primeros 20 pares la locación ya está en T1): una **serie mensual de NDVI** por locación. Para cada locación nueva se calcula en Earth Engine el NDVI medio en un círculo de 50 m alrededor del punto del padrón, menos el de un anillo de 150–300 m alrededor (para que la estación no confunda), en los 60 meses. El mes de construcción es el primero en que esa diferencia cae y se mantiene abajo dos meses seguidos; T1 es el mes anterior y T2, uno o dos meses después. Si ya está abajo desde el primer mes, se marca como posible preexistente. Se valida con las 4 locaciones revisadas a mano (809: junio de 2022; 667: junio o julio de 2019; 738: septiembre de 2021; 882: preexistente).
   - **Negativos:** pares de encuadres sin locaciones nuevas en ese intervalo, con T1 y T2 de **estaciones distintas** (por ejemplo, verano e invierno), para que el modelo aprenda que la humedad del suelo o la luz no son una locación. Incluyen los encuadres del bloque 0.
   - Meta provisoria: **300 pares** (los 172 positivos y unos 130 negativos).
3. **Etiquetado:** la máscara de un par marca todo lo que **apareció entre T1 y T2**: locación (terminada o en obra) y pileta. Lo que ya estaba en T1 es fondo. **Se marca toda locación nueva visible en el par, esté o no en nuestro dataset** (por ejemplo, las de pozos convencionales u otras formaciones), no solo la que originó el par. Donde hay pileta sobre una locación, se pinta pileta. **Caminos quedan afuera**: a 10 m/píxel son muy finos ([ADR-0005](adr/0005-clases-sin-caminos.md)).
   - **Máscara propuesta:** **SAM** sobre la imagen T2 entera (no sobre una ventana chica), con el centro de cada locación nueva del padrón como clic. La prueba con 20 locaciones dio 20 bien con la imagen entera y solo 13 con una ventana de 640 m (issue #8). Las máscaras propuestas se generan en Colab con `segment-geospatial`.
   - **Resta de NDVI** entre T1 y T2 (NDVI = (B8 − B4) / (B8 + B4)): ayuda a ver qué cambió. Una locación nueva baja el NDVI porque se saca la vegetación. Sirve para encontrar locaciones nuevas que no están en el padrón y para confirmar que la del padrón no existía en T1.
   - **Validación en QGIS:** un proyecto por par con T1, T2 y la resta de NDVI como capas, la máscara propuesta y los centros del padrón. Quien valida prende y apaga las capas para comparar, corrige los bordes, agrega con un clic de SAM o a mano las locaciones nuevas que SAM no recibió, borra lo que ya estaba en T1 y marca las piletas (no están en el padrón: un clic en cada una y SAM arma el contorno). Si hay duda, se anota en un campo `nota`.
   - Si la resta de NDVI no muestra cambio en una locación nueva del padrón, el par se revisa a mano: puede ser un positivo con poco contraste (por ejemplo, en invierno) o una locación preexistente. Nunca se descarta automáticamente.
   - OpenStreetMap no tiene polígonos de locaciones en esta zona (issue #8), así que no se usa como fuente.
4. **Modelo** ([ADR-0011](adr/0011-modelo-bitemporal-siames.md)):
   - **Principal:** red siamesa estilo FC-Siam-diff (Daudt et al., 2018). Un mismo encoder de `segmentation_models_pytorch`, preentrenado en ImageNet, procesa T1 y T2 por separado; el decoder recibe la diferencia de las representaciones de cada nivel y predice la máscara.
   - **Línea de base:** U-Net con fusión temprana. Las dos fechas se apilan como canales (6 + 6) en una sola entrada.
   - Modos del encoder para H2: desde cero, encoder congelado y fine-tuning. En el modo "encoder congelado" la **primera capa siempre se entrena**, con 3 y con 6 bandas: los pesos de los canales que no son RGB se inicializan con el promedio de los de RGB y el resto del encoder queda congelado. Así la comparación de H1 en ese modo es justa.
5. **Evaluación:** IoU y F1 por clase, matriz de confusión a nivel píxel, precisión y recall por objeto, ejemplos buenos/malos, comparación test Añelo vs. test externo (detalle en [EVALUACION.md](EVALUACION.md)).
6. **Despliegue / backtesting:** para cada mes *m* de 2019–2023 se corre el modelo sobre el par (*m* − 6, *m*) de las imágenes mensuales. Se busca la primera detección de cada locación nueva de `test_anelo` y `test_externo` (nunca de `train` ni `val`, que el modelo ya vio) y se compara con `fecha_primer_pozo` (inicio de perforación). Métricas, por zona: % de locaciones detectadas a tiempo, detectadas tarde y nunca detectadas, y **meses de anticipación** (mediana sobre todas las detectadas, incluidas las negativas). Criterios en [EVALUACION.md](EVALUACION.md).

## Bandas e índices

| Banda | Qué mide | Resolución |
|---|---|---|
| B2, B3, B4 | Azul, verde y rojo (color natural) | 10 m |
| B8 | Infrarrojo cercano (NIR): la vegetación refleja mucho | 10 m |
| B11, B12 | Infrarrojo de onda corta (SWIR): distingue suelo seco, húmedo y removido; el agua absorbe casi todo | 20 m, remuestreadas a 10 m |

Índices que se calculan a partir de las bandas, en el Dataset:

- **NDVI** (vegetación) = (NIR − rojo) / (NIR + rojo) = (B8 − B4) / (B8 + B4).
- **NDBI** (superficie construida o suelo compactado) = (SWIR − NIR) / (SWIR + NIR) = (B11 − B8) / (B11 + B8).
- **BSI** (suelo desnudo) = ((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2)).

Para el informe hay que citar las especificaciones oficiales de Sentinel-2 (ESA/Copernicus) y explicar qué aporta cada banda, como pidió el profesor.

## Hipótesis y experimentos

- **H1 (bandas):** agregar información infrarroja a RGB mejora la detección de locaciones nuevas frente a los cambios naturales de la estepa. Experimento: **3 bandas (RGB) contra 6 bandas (RGB + NIR + SWIR B11 y B12, remuestreadas a 10 m) contra 6 bandas + índices (NDVI, NDBI, BSI)**, con la siamesa. Las entradas que no son RGB requieren adaptar la primera capa del encoder.
- **H2 (transfer learning):** fine-tuning > encoder congelado > desde cero, sobre todo con pocos datos. Con la siamesa.
- **H3 (generalización):** el desempeño cae en la zona de test externa (otras operadoras y otro paisaje). La diferencia entre test Añelo y test externo mide la generalización.
- **H4 (arquitectura):** compartir el encoder entre las dos fechas (siamesa) mejora respecto de apilarlas en la entrada (fusión temprana).
- **Extra opcional:** agregar unos pocos pares etiquetados de la zona externa al entrenamiento y medir cuánto se recupera el desempeño (adaptación con pocos datos).

Entrenamientos, en orden de prioridad (si falta tiempo, se recortan de abajo hacia arriba; el 1 y el 2 nunca):

| # | Modelo | Entrada | Modo del encoder | Para |
|---|---|---|---|---|
| 1 | Siamesa | 6 bandas | Fine-tuning | Modelo principal |
| 2 | Fusión temprana | 6 + 6 bandas | Fine-tuning | Línea de base (H4) |
| 3 | Siamesa | RGB | Fine-tuning | H1 |
| 4 | Siamesa | 6 bandas + índices | Fine-tuning | H1 |
| 5 | Siamesa | 6 bandas | Desde cero | H2 |
| 6 | Siamesa | 6 bandas | Congelado | H2 |

## Fuera de alcance (por ahora)

Quedan como trabajo futuro, para no sobredimensionar el TP:
- Segmentar una sola fecha (el enfoque original, ver [ADR-0011](adr/0011-modelo-bitemporal-siames.md)).
- ChangeFormer y otros modelos de detección de cambios con transformers.
- Pérdida robusta a ruido de etiqueta (pensada para etiquetas débiles).
- Preentrenamiento con el Alberta Wells Dataset o con encoders preentrenados sobre Sentinel-2.
- Comparación con el modelo de ESRI ("Well Pad Detection – Permian Basin"), que requiere ArcGIS.
- SAM como línea de base: se usa solo para armar máscaras propuestas.

## Acuerdo entre etiquetadoras

Se eligen 10 pares y cada uno se valida dos veces: la dueña del bloque y otra persona, en rotación (Sofi → Agus → Bian → Sofi). El IoU entre las dos validaciones es el techo realista del modelo.
