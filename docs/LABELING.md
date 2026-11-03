# Guía de etiquetado manual

Objetivo: construir el **conjunto de prueba** (y validación) con la verdad terreno dibujada a mano. Sin esto, solo se puede medir cuánto se parece el modelo a la etiqueta débil, que es justamente lo que está en duda.

## 0. Antes de nada: calibrar el radio del buffer (1 hora, una persona)

1. Abrir en QGIS la escena (`data/raw/*.tif`, bandas 3-2-1 como RGB) y los puntos (`data/interim/wells.gpkg`).
2. Elegir ~30 locaciones al azar y **medir su ancho** (herramienta *Medir línea*).
3. Tomar la mediana como orden de magnitud del tamaño del pad y fijar `--radius-m` en consecuencia. El valor por defecto (60 m) es un supuesto, no un dato.
4. Anotar la tabla de mediciones: va al informe.

## 1. Convenciones (acordarlas ANTES de dibujar)

Proponemos esta definición; el grupo puede cambiarla, pero tiene que ser la misma para todos:

- **Locación = el playón despejado** (suelo desnudo, rectangular o irregular) con equipos, boca de pozo y estructuras asociadas visibles.
- **Incluye** piletas o tanques que comparten el playón.
- **Excluye** el camino de acceso y las líneas de conducción que salen del playón.
- Si dos playones se tocan, se dibujan dos polígonos.
- Si hay duda (¿pad abandonado? ¿obra en construcción?), **dibujar** y anotar la duda en un campo `nota`.

## 2. Qué parches etiquetar

`build-patches` genera `data/processed/to_label.gpkg`: las huellas de los parches de **val y test**. No hay que etiquetar los de train (esos usan la etiqueta débil).

1. Cargar `to_label.gpkg` y la escena en QGIS.
2. Crear una capa nueva `manual.gpkg` (polígonos, CRS de la escena o EPSG:4326).
3. Para **cada huella** que se decida etiquetar: dibujar **todos** los polígonos de locaciones que aparezcan en ella. Una huella sin ningún polígono es un **negativo verdadero**, así que no se puede dejar "a medias".
4. Copiar `to_label.gpkg` a `labeled.gpkg` y **borrar las huellas que no se terminaron**. Lo que queda en `labeled.gpkg` es la zona etiquetada exhaustivamente.
5. Reconstruir: `build-patches ... --manual manual.gpkg --labeled labeled.gpkg`.

Se necesitan, como mínimo, ~100 parches con manual en test, de los cuales una buena parte (al menos un tercio) con locaciones adentro. Repartir el trabajo entre las personas del grupo.

## 3. Control de calidad (se reporta en el informe)

- **Acuerdo entre etiquetadoras:** dos personas etiquetan los mismos 10–15 parches por separado y se calcula el IoU entre ambas. Ese número es el **techo realista** del modelo: no se puede esperar más acuerdo con la verdad que el que tienen dos humanos entre sí.
- **Ruido de la etiqueta débil:** `summary.json → weak_vs_manual` (IoU y coberturas de la etiqueta débil contra la manual).

## 4. Extensión opcional (no implementada): refinar etiquetas débiles con SAM

Idea: usar cada punto de pozo como *prompt* de un modelo de segmentación general (*Segment Anything*) sobre la imagen, y quedarse con la máscara resultante en lugar del disco. Se espera una etiqueta más fiel a la forma del playón.

- No está en el repo. Si se implementa, comparar tres versiones de etiqueta (disco, SAM, manual) y medir cada una contra el conjunto manual de test.
- Cuidado: SAM está entrenado con fotos comunes; a 10 m/píxel puede segmentar mal. Probar primero en 20 pads y mirar.
