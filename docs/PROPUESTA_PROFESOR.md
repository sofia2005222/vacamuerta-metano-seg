# Borrador de mail al profesor

*Completar lo que está entre corchetes. Ajustar el tono a como le escriben normalmente.*

**Asunto:** Propuesta de tema para el TP final – Segmentación con imágenes satelitales (Vaca Muerta)

Hola [Nombre]:

Somos [nombres] y queremos proponerles el tema de nuestro TP final, dentro de la categoría **Segmentación**.

**Problema.** En septiembre Neuquén empezó a exigir a las operadoras de Vaca Muerta que reporten sus emisiones de metano. Para auditar esos reportes y atribuir fugas hay que saber dónde están las instalaciones, pero el padrón oficial de pozos solo da un punto por pozo, no la huella de la locación.

**Qué proponemos.** Segmentar locaciones de producción en imágenes Sentinel-2 (10 m) con una U-Net con encoder preentrenado. Entrenamos con **etiquetas débiles** (los puntos oficiales de pozos con un buffer) y evaluamos contra un conjunto de test que **etiquetamos nosotras a mano** (~100–150 parches). Comparamos variantes: pérdida robusta a ruido de etiquetas, bandas RGB vs RGB+NIR, tamaño del buffer y, si hay tiempo, preentrenamiento con un dataset de pozos de Alberta.

**Datos.** Sentinel-2 L2A (Copernicus, abierto) y el shapefile de pozos de la Secretaría de Energía (datos.gob.ar). No dependemos de datos privados.

**Aporte propio.** El uso de datos oficiales ruidosos como supervisión y la medición de cuánto ruido hay (comparando la etiqueta débil con la manual), más el análisis de sensibilidad.

**Consultas.**
1. ¿Les parece que encaja como Segmentación y que el alcance es razonable para [3/4] personas?
2. ¿Es válido que parte del dataset (el conjunto de test) sea de etiquetado propio?
3. ¿Hay alguna fecha tope para confirmar el tema?

Tenemos un repositorio con el pipeline armado y probado con datos sintéticos; si les interesa se lo compartimos.

Gracias,
[Nombres]
