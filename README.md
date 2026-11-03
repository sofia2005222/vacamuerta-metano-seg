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
