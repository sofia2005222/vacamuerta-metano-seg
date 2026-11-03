# Auditando el metano desde el espacio: inventario de infraestructura de Vaca Muerta con Sentinel-2

TP final de **I308 – Visión Artificial** (UdeSA). Tema de la cátedra: **Segmentación**.
*Título de trabajo; ver `docs/POSTER.md` para alternativas.*

## El problema (y por qué es de ahora)

- **Septiembre de 2026:** Neuquén empezó a exigir a las operadoras de Vaca Muerta que **reporten sus emisiones de metano**, con un protocolo alineado al marco internacional OGMP 2.0
  ([LM Neuquén](https://mase.lmneuquen.com/vaca-muerta/el-camino-el-sello-verde-vaca-muerta-arranca-el-reporte-obligatorio-emisiones-metano-n1254232),
  [Examedia](https://examedia.com.ar/portada/post/87901665)).
- **Octubre de 2026:** según la prensa, la Escuela N.º 291 de Aguada Toledo (Loma La Lata–Sierra Barrosa) lleva semanas cerrada por contaminación, a poco más de un kilómetro de unos tanques que el reporte de metano de PNUMA-IMEO (junio de 2026) ubicó entre las fuentes más grandes del mundo; la fuga se habría detectado por satélite entre mayo de 2025 y mayo de 2026
  ([AADEAA](https://aadeaa.org/la-otra-cara-de-vaca-muerta/),
  [Noticias Ambientales](https://noticiasambientales.com/medio-ambiente/neuquen-cierran-una-escuela-rural-luego-de-semanas-sin-clases-por-contaminacion-en-una-zona-clave-de-vaca-muerta/)).
  *Verificar las cifras en la fuente original antes de citarlas en el informe.*
- **La brecha de datos:** para atribuir una fuga o auditar un reporte hay que saber **dónde están** las instalaciones. El padrón oficial de pozos da **un punto por pozo**, no la huella de la locación, y los inventarios públicos de locaciones y tanques tienen huecos incluso a nivel mundial
  (ver [Ramachandran et al., Nature Communications 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11327246), que lo resuelve con deep learning para las cuencas Pérmica y Denver).

## La tarea

**Segmentación semántica binaria de locaciones de producción (pads) en imágenes Sentinel-2 (10 m/píxel)**, entrenada con **etiquetas débiles** (los puntos oficiales de pozos, con un buffer) y evaluada contra un **conjunto de prueba etiquetado a mano**.

Lo que este proyecto **no** hace (y conviene decirlo en el póster):
- No detecta metano. Genera el inventario de fuentes potenciales sobre el que se podrían cruzar detecciones de metano de otros sistemas.
- No distingue tanques individuales: a 10 m un tanque son pocos píxeles. Se segmenta la locación completa; el nivel tanque queda como trabajo futuro con imágenes de mayor resolución.

### Encaje con la consigna

| Pide la consigna | Cómo se cubre |
|---|---|
| Un tema de la lista de la cátedra | **Segmentación** |
| Dataset real | Sentinel-2 L2A (Copernicus) + padrón oficial de pozos (Secretaría de Energía) |
| Al menos una modificación / uso propio | (1) etiquetas débiles desde datos oficiales, (2) pérdida robusta a ruido de etiquetas, (3) conjunto de prueba manual y estudio del ruido de la etiqueta; extensiones opcionales abajo |
| Métricas e imágenes de resultados | IoU, F1, precisión/recall por píxel y por objeto; figura cualitativa en `runs/<corrida>/examples_test.png` |
| Título distintivo en el póster | Ver `docs/POSTER.md` |

## Diseño experimental (qué se compara)

| Corrida | Cambio | Pregunta que responde |
|---|---|---|
| A · baseline | U-Net ResNet34 (ImageNet), pérdida BCE+Dice, 4 bandas | ¿Cuánto se logra con etiquetas débiles "tal cual"? |
| B · robusta | `train.loss=bootstrap_bce_dice` | ¿Ayuda tolerar el ruido de la etiqueta? |
| C · sin preentrenar | `model.init=random` | ¿Cuánto aporta ImageNet? |
| D · solo RGB | `data.channels=[2,1,0]` | ¿Aporta el infrarrojo cercano (NIR)? |
| E · radio del buffer | reconstruir con `--radius-m` 40 / 60 / 90 | ¿Qué tan sensible es el resultado a la suposición del radio? |
| F · (opcional) | `model.init=<ckpt Alberta>` | ¿Sirve preentrenar con pozos de Alberta? Ver `docs/ALBERTA.md` |

Repetir cada corrida con 3 semillas (`train.seed`) y reportar media ± desvío. La referencia clave es **cuánto coincide la etiqueta débil con la manual** (`summary.json → weak_vs_manual`): si el modelo supera ese valor contra la verdad manual, aprendió algo que la etiqueta no decía.

## Inicio rápido

```bash
git clone <URL-de-tu-repo> && cd vacamuerta-metano-seg
pip install -r requirements.txt && pip install -e .
pytest -m "not slow"          # verifica la instalación (~20 s en CPU; sin la marca "not slow" son ~2,5 min)
```

Prueba de humo con datos sintéticos (no necesita Internet):

```bash
python -m vmseg.cli synth --out data/synthetic --size 1024 --n-pads 60
python -m vmseg.cli build-patches --scene data/synthetic/synthetic.tif --wells data/synthetic/synthetic_wells.gpkg \
    --manual data/synthetic/synthetic_manual.gpkg --labeled data/synthetic/synthetic_labeled.gpkg \
    --out data/processed --size 64 --stride 32 --block-px 256
python -m vmseg.cli train --config configs/default.yaml --set model.init=random --set model.encoder=resnet18 --set train.epochs=15 --set train.lr=1e-3
python -m vmseg.cli evaluate --run runs/baseline --split test
```

En **Colab** (la consigna lo recomienda para prototipar) alcanza con una celda: `!git clone ...`, `%cd vacamuerta-metano-seg`, `!pip install -q -r requirements.txt && pip install -q -e .` y activar GPU en *Entorno de ejecución*.

### Pipeline con datos reales

```bash
# 1. Imágenes: compuesto sin nubes de Sentinel-2 sobre un AOI (configs/aois.yaml)
python -m vmseg.cli fetch-s2 --aoi sierra_barrosa --start 2026-01-01 --end 2026-03-31
# 2. Pozos: bajar a mano el "Shapefile de Pozos" de datos.gob.ar (ver docs/DATA.md) y recortarlo al AOI
python -m vmseg.cli prepare-wells --input pozos.zip --aoi sierra_barrosa --out data/interim/wells.gpkg
# 3. Mirar la imagen con los puntos encima (QGIS) y calibrar el radio (docs/LABELING.md). Luego:
python -m vmseg.cli build-patches --scene data/raw/sierra_barrosa_*.tif --wells data/interim/wells.gpkg --radius-m 60
#    -> data/processed/to_label.gpkg = parches de val/test que hay que etiquetar a mano
# 4. Etiquetar ~100-150 parches (docs/LABELING.md) y reconstruir con --manual y --labeled
# 5. Entrenar, evaluar, predecir
python -m vmseg.cli train --config configs/default.yaml --set train.run_name=A_baseline
python -m vmseg.cli evaluate --run runs/A_baseline --split test
python -m vmseg.cli predict --run runs/A_baseline --scene data/raw/sierra_barrosa_*.tif --out runs/A_baseline/pred.gpkg
```

## Estructura

```
configs/        default.yaml (hiperparámetros) · aois.yaml (áreas de interés)
src/vmseg/      geo · labels · tiling · build · dataset · losses · models · train · evaluate · infer · stac · synthetic · cli
tests/          21 pruebas (datos, métricas, compuesto Sentinel-2 local, pipeline completo con datos sintéticos)
docs/           DATA · LABELING · ALBERTA · PLAN · PROPUESTA_PROFESOR · POSTER
```

## Qué está probado y qué no (leer antes de confiar)

**Probado** (21 pruebas automáticas, CPU): tiling y splits sin fuga espacial, rasterización de etiquetas débiles, métricas, armado del compuesto Sentinel-2 con enmascarado de nubes (sobre archivos locales), pipeline completo sintético → parches → entrenamiento → evaluación → polígonos, y que el modelo **aprende** en una escena sintética (la etiqueta débil coincide con la verdad en IoU ≈ 0.46; el modelo llega a ≈ 0.86–0.90 contra la verdad en esa escena de juguete — **no es un resultado del proyecto**, solo una prueba de que el código funciona).

**NO probado todavía:**
- La **consulta** al catálogo STAC real (`search_items` en `src/vmseg/stac.py`): el entorno donde se armó el repo no tenía Internet. El armado del compuesto sí está probado con archivos locales. Probar `fetch-s2` primero en Colab con un AOI chico; si falla por nombres de assets, ver el docstring de `stac.py`.
- El shapefile real de pozos: los nombres de columnas y la cobertura de pozos no convencionales recientes no están verificados. El código solo filtra por caja geográfica.
- Entrenamiento en GPU y con imágenes reales; calibración del radio; el resultado real del modelo.
- Refinado de etiquetas con SAM y preentrenamiento con Alberta: **no implementados** (ideas en `docs/LABELING.md` y `docs/ALBERTA.md`).

## Riesgos conocidos

- **Etiqueta débil ≠ locación:** el punto de un pozo no dice el tamaño del pad; el radio es un supuesto. De ahí la prueba manual y la corrida E.
- **Pads viejos o desmantelados** pueden figurar en el padrón y no verse en la imagen (o al revés: pads nuevos sin registrar). Es ruido de etiqueta, pero también un resultado interesante (¿instalaciones no registradas?). Tratarlo con cuidado en el informe.
- **Resolución:** 10 m limita el tamaño mínimo detectable.
- **Fuga espacial:** por eso la partición es por bloques de 512 px, no por parche.

## Referencias

- Ramachandran et al. (2024), *Deep learning for detecting and characterizing oil and gas well pads in satellite imagery*, Nature Communications.
- Seth et al. (2025), *Alberta Wells Dataset*, ICML. Datos: <https://zenodo.org/records/13743323>.
- Omara et al. (2023), *OGIM*, Earth System Science Data. Datos: <https://zenodo.org/records/13259749>.
- Reed et al. (2015), *Training deep neural networks on noisy labels with bootstrapping* (la pérdida `bootstrap_bce_dice`).

# Alerta temprana de actividad en Vaca Muerta con imágenes satelitales

Trabajo práctico de **I308 – Visión Artificial (Universidad de San Andrés)**, modalidad "propuesta de trabajo por estudiantes".

> **Nota para quien use un asistente de IA (Claude u otro):** este archivo resume todo el proyecto: la idea, las decisiones tomadas y por qué, el dataset ya armado y los próximos pasos. Leelo completo antes de proponer cambios. Las decisiones de la sección 9 ya fueron discutidas; no las reabras salvo que haya un motivo nuevo.

---

## 1. La idea en una frase

Entrenar una red neuronal que **segmente locaciones petroleras y piletas en imágenes satelitales Sentinel-2** de Vaca Muerta y usarla para **detectar locaciones nuevas antes de que empiece la perforación**, midiendo con datos oficiales cuántos meses de anticipación da el satélite.

## 2. El problema (enfoque empresa)

- Vaca Muerta es una formación geológica de Neuquén de la que se extrae petróleo y gas no convencional. Pasó de perforar 10–15 pozos por mes en 2018–2019 a más de 50 por mes en 2026.
- Antes de perforar, la operadora desmonta un rectángulo de tierra (la **locación** o *pad*), que suele alojar varios pozos (en esta zona, la mediana es 4). Es visible desde el satélite.
- **La locación aparece semanas o meses antes de la perforación.** Detectarla temprano es **inteligencia de mercado**: le sirve a cada empresa saber qué está haciendo *la competencia* (cada una ya conoce su propia actividad).
- Quiénes la usarían: empresas de servicios (fractura, perforación, arena, agua, transporte) para anticipar demanda y clientes; proveedores de arena y agua para planificar logística; operadoras para mirar a sus vecinas y valuar bloques; inversores y analistas para anticipar producción (los pozos de hoy impactan en la producción 6–12 meses después); midstream para planificar capacidad de transporte.
- El registro oficial ubica los pozos como **puntos** y con demora; no mide la huella en superficie ni anticipa actividad.

**Referente comercial:** Sourcewater (Houston) vende este servicio en la cuenca Permian (EE. UU.) con su producto *DirtWork Alert*: escanea la cuenca cada 3–5 días y detecta locaciones nuevas. Reportan que las locaciones se desmontan típicamente 1–4 meses antes de que llegue el equipo de perforación, y que en 1 de cada 3 pozos la locación se veía antes de cualquier registro oficial. (Son reportes de la empresa, sin referato: se usan como **motivación**, no como bibliografía.)

**Aplicación secundaria** (mencionar en una línea en la presentación): el mismo modelo sirve para cumplimiento ambiental (inventario de huella, verificar restauración de locaciones abandonadas).

## 3. Requisitos de la cátedra (resumen del PDF de la consigna)

- Problema acotado, con objetivos e hipótesis, implementable en menos del 30% del tiempo. No sobredimensionar.
- Estado del arte **con referato** (no Wikipedia, Medium, blogs; cuidado con arXiv sin publicar).
- Dataset con anotaciones que permitan métricas; split sugerido 70/10/20.
- **Obligatorio implementar una red neuronal** en Python, preferentemente PyTorch.
- Probar: entrenamiento desde cero, extracción de características (encoder congelado) y fine-tuning.
- Curvas de entrenamiento (loss por iteración, métrica en train y val por época).
- Evaluación: matrices de confusión, ejemplos correctos/incorrectos, casos patológicos, hipótesis sobre el comportamiento del modelo.
- Ciclo sugerido: problema → estado del arte → dataset → modelo → evaluación → despliegue.
- El tema debe ser aprobado por el profesor.

## 4. Estado del arte (bibliografía principal)

- **Ramachandran et al. (2024). "Deep learning for detecting and characterizing oil and gas well pads in satellite imagery." *Nature Communications* 15.** DOI: 10.1038/s41467-024-50334-9. Stanford. Detectan locaciones y tanques con deep learning en las cuencas Permian y Denver-Julesburg, arman datasets etiquetados y estudian la generalización a regiones nuevas. Es la referencia metodológica principal. (Su motivación es medir emisiones de metano; nosotras tomamos la metodología y la aplicamos con foco comercial.)
- **Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.** Arquitectura base.
- Pendiente: sumar 1–2 papers con referato de segmentación sobre Sentinel-2 (buscar en Google Scholar).
- Existe un modelo preentrenado de ESRI ("Well Pad Detection – Permian Basin") que funciona con Sentinel-2. Requiere ArcGIS; si fuera accesible, sería una línea de base para comparar.

## 5. Pipeline del proyecto

1. **Imágenes:** Sentinel-2 (gratis, desde 2017, revisita de 5 días, 10 m/píxel en RGB+NIR, 20 m en SWIR). Neuquén es árido: pocas nubes. Fuentes: Copernicus Browser, Google Earth Engine o Microsoft Planetary Computer.
2. **Etiquetado:** máscaras de segmentación dibujadas sobre la imagen con ayuda de **SAM** (clic y corrección). Clases: `locación` y `pileta` (y fondo). **Caminos quedan afuera**: a 10 m/píxel son muy finos. Los puntos oficiales de este dataset indican dónde buscar. Meta inicial: 200–400 recortes de 256×256 px (consultar con el profesor el número).
3. **Modelo:** U-Net (librería `segmentation_models_pytorch`) con encoder preentrenado en ImageNet. Comparar desde cero vs. encoder congelado vs. fine-tuning.
4. **Evaluación:** IoU y F1 por clase, matriz de confusión a nivel píxel, ejemplos buenos/malos, comparación test Añelo vs. test externo.
5. **Despliegue / backtesting:** correr el modelo sobre imágenes mensuales 2019–2023, detectar cuándo aparece cada locación y compararlo con `fecha_primer_pozo` (inicio de perforación). Métricas: % de locaciones nuevas detectadas y **meses de anticipación** (mediana).

## 6. Hipótesis y experimentos

- **H1 (bandas):** agregar las bandas infrarrojas (NIR, SWIR) a RGB mejora la distinción entre locaciones y suelo desnudo natural de la estepa. Experimento: 3 bandas vs. más bandas (requiere adaptar la primera capa del encoder).
- **H2 (transfer learning):** fine-tuning > encoder congelado > desde cero, sobre todo con pocos datos.
- **H3 (generalización):** el desempeño cae en la zona de test externa (otras operadoras y otro paisaje). La diferencia entre test Añelo y test externo mide la generalización.
- **Extra opcional:** agregar unas pocas imágenes etiquetadas de la zona externa al entrenamiento y medir cuánto se recupera el desempeño (adaptación con pocos datos).

## 7. Diseño de la evaluación (división espacial)

Dos niveles de test, ambos se reportan:

| Split | Dónde | Para qué |
|---|---|---|
| `train` | Bloques de Añelo | Entrenar |
| `val` | Bloques de Añelo | Elegir hiperparámetros y época |
| `test_anelo` | Bloques de Añelo nunca vistos | Desempeño en la misma zona (dentro de distribución) |
| `test_externo` | Zona oeste (Fortín de Piedra, Aguada Pichana Este, La Calera) | Generalización a otra zona y otras empresas. **Nunca se usa para entrenar ni para elegir hiperparámetros** |

- La zona de Añelo se divide en una grilla de 4×4 bloques (~7,5 km de lado) y **cada bloque entero** va a un split. Así, recortes vecinos (casi iguales) no quedan repartidos entre train y test, lo que inflaría las métricas.
- Los recortes de imagen que crucen el borde entre bloques de distinto split se descartan.
- Por qué dos tests: Añelo es casi todo YPF (329 de 392 locaciones). Si solo se testea ahí, el modelo podría estar aprendiendo "la forma de construir de YPF". La zona externa es de Tecpetrol, Total y Pluspetrol.

Conteo de locaciones por split:

| Split | Locaciones | Nuevas 2019–2023 |
|---|---|---|
| train | 272 | 81 |
| val | 40 | 20 |
| test_anelo | 80 | 34 |
| test_externo | 150 | 62 |

## 8. Dataset de pozos (este directorio)

**Fuente:** Secretaría de Energía de la Nación, "Producción de petróleo y gas por pozo (Capítulo IV)", datos.gob.ar, descargado en noviembre de 2026. Licencia CC-BY-4.0 (citar la fuente). Archivos de entrada: `capitulo-iv-pozos.csv` y `padrn-de-pozos-de-captulo-iv-con-fecha-de-primera-produccin.csv`.

**Qué hace `armar_dataset.py`:**
1. Filtra pozos de formación Vaca Muerta con recurso no convencional: **3.453 pozos**.
2. Extrae coordenadas (EPSG:4326, boca de pozo).
3. Cruza con la fecha de primera producción por `idpozo`.
4. Agrupa en **locaciones** los pozos a menos de 60 m entre sí (en un pad las bocas están a ~10 m): **1.204 locaciones**.
5. Asigna zonas, bloques y split.

**Archivos:**

| Archivo | Contenido |
|---|---|
| `locaciones_vaca_muerta.csv` / `.geojson` | Una fila por locación. **Archivo principal** |
| `pozos_vaca_muerta.csv` / `.geojson` | Una fila por pozo (se une a locaciones por `locacion_id`) |
| `zonas_estudio.json` | Recuadros de las zonas, grilla y split de cada bloque |
| `armar_dataset.py` | Regenera todo (zonas y semilla configurables al principio) |

Los `.geojson` se abren directo en QGIS o Google Earth Engine.

**Zonas:**
- Añelo (train/val/test): longitud -68.85 a -68.50, latitud -38.42 a -38.12 (~30 × 33 km). Loma Campana, Bandurria Sur, La Amarga Chica, Bajada del Palo Oeste, Cruz de Lorena. Elegida por ser el núcleo con más densidad y actividad 2019–2023.
- Test externo: longitud -69.25 a -68.95, latitud -38.60 a -38.30.

**Columnas de `locaciones_vaca_muerta.csv`:**

| Columna | Significado |
|---|---|
| `locacion_id` | Identificador de la locación |
| `lon`, `lat` | Centro de la locación |
| `n_pozos` | Cantidad de pozos |
| `fecha_primer_pozo` | Inicio de perforación del primer pozo. **Fecha de referencia del backtesting** |
| `fecha_ultimo_pozo` | Inicio de perforación del último pozo |
| `primera_produccion` | Primer mes con producción declarada (ver advertencias) |
| `anio_primer_pozo` | Año del primer pozo |
| `nueva_2019_2023` | `True` si el primer pozo es de 2019–2023 (casos del backtesting) |
| `ampliada_luego` | `True` si se agregaron pozos más de un año después del primero: la locación ya existía en la fecha del pozo agregado |
| `en_zona_anelo`, `en_zona_test_externo` | Pertenencia a cada zona |
| `bloque_anelo` | Bloque de la grilla 4×4 (solo zona Añelo) |
| `split` | `train`, `val`, `test_anelo`, `test_externo` o vacío (fuera de ambas zonas) |
| `empresa`, `area`, `siglas` | Operadora, área de concesión y siglas de los pozos |

**Advertencias:**
- **Usar `fecha_primer_pozo` (inicio de perforación), no `primera_produccion`.** La primera producción del padrón muchas veces aparece antes del fin de la terminación del pozo, lo cual no tiene sentido físico; probablemente sea el primer mes declarado en el sistema.
- Las coordenadas son de boca de pozo; la locación desmontada es más grande (en EE. UU. miden 50–250 m, o sea 5–25 píxeles en Sentinel-2).
- La agrupación por distancia es aproximada: revisar algunos casos a ojo sobre la imagen.
- 35 pozos sin fecha de primera producción y algunos sin fecha de perforación.
- **Los puntos no son las etiquetas del modelo:** las máscaras se dibujan sobre la imagen.

## 9. Decisiones tomadas (y alternativas descartadas)

- **Tema:** se evaluaron opciones de seguridad (parches adversariales, backdoors en GTSRB, anti-spoofing, deepfakes, inyección en VLMs) y otras de Vaca Muerta (flaring, derrames, EPP, corrosión, impacto en suelos). Se eligió segmentación de infraestructura por: datos accesibles, problema real y local, y pipeline completo. **Plan B** si el profesor no lo aprueba: backdoors sobre GTSRB (clasificación de señales de tránsito).
- **Enfoque comercial** (inteligencia de mercado), no ambiental. El ambiental queda como aplicación secundaria.
- **Backtesting sobre 2019–2023** en lugar de tiempo real: hay fechas oficiales de perforación para validar.
- **Sentinel-2** (gratis, histórico) en lugar de imágenes de alta resolución pagas.
- **Sin caminos** como clase.
- **División espacial por bloques + test externo**, en lugar de división aleatoria.

## 10. Riesgos

- **Resolución:** las locaciones chicas ocupan pocos píxeles.
- **Confusión con suelo natural:** la estepa ya es medio pelada (motiva H1).
- **Tiempo de etiquetado:** es la tarea más larga. Medir con 10 recortes antes de planificar.
- **Sobredimensionar:** si falta tiempo, recortar primero el backtesting o la clase `pileta`, nunca la evaluación del modelo.

## 11. Próximos pasos

- [x] Conseguir datos oficiales de pozos con coordenadas y fechas
- [x] Armar dataset de locaciones, zonas y splits
- [ ] Abrir `locaciones_vaca_muerta.geojson` sobre Sentinel-2 y confirmar a ojo que las locaciones se distinguen (comparar 2019 vs. 2023 en Loma Campana)
- [ ] Etiquetar 10 recortes con SAM y medir el tiempo
- [ ] Consultar al profesor el número de imágenes etiquetadas y aprobar el tema
- [ ] Buscar 1–2 papers extra de segmentación en Sentinel-2
- [ ] Descargar imágenes y armar recortes por split
