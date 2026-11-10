---
status: accepted
---

# Modelo bitemporal siamés con máscaras de cambio

El modelo recibe un **par**: un encuadre en dos imágenes mensuales, una antes (T1) y otra después (T2) de que se construya la locación, y predice una máscara solo de lo que **apareció** entre las dos fechas (locación nueva y pileta nueva). Todo lo que ya estaba en T1 es fondo. El modelo principal es una **red siamesa**: un mismo encoder preentrenado procesa cada fecha por separado y el decoder trabaja con la diferencia de sus representaciones (estilo FC-Siam-diff, Daudt et al., 2018). La línea de base es una **U-Net con fusión temprana**: las dos fechas apiladas como 12 canales de entrada. Las dos usan los mismos pares y las mismas máscaras.

Lo elegimos por tres motivos. Responde directo a la pregunta del proyecto, que es dónde aparece una locación nueva y no qué hay en una imagen. Descarta por construcción los falsos positivos estáticos (locaciones viejas, canteras, el pueblo), incluidas las **locaciones preexistentes** que el padrón marca como nuevas: en la revisión manual de noviembre de 2026, la locación 882 ya estaba desmontada en 2019 aunque su primer pozo es de 2023. Y lo sugirió el profesor al revisar esa revisión. La siamesa, a diferencia de la fusión temprana, mantiene el encoder preentrenado con sus 6 bandas por fecha, así que H2 conserva el sentido.

## Opciones consideradas

- **Segmentar una sola fecha** (el plan original) y detectar el cambio recién en el backtesting, comparando las predicciones mes a mes. Necesita etiquetar todas las locaciones de cada recorte, y las locaciones preexistentes hay que filtrarlas aparte. El profesor aclaró que con uno de los dos enfoques alcanza para el trabajo final.
- **Hacer los dos enfoques completos** para compararlos. Duplica el etiquetado, porque las máscaras de una fecha y las de cambio son distintas, y no entra en las semanas que quedan.
- **ChangeFormer u otros modelos con transformers.** Más complejos que lo visto en la materia y con más riesgo de pelear con dependencias.

## Consecuencias

- Modifica a [ADR-0008](0008-mascaras-dibujadas-sobre-imagen.md): la máscara es de un par y marca solo lo que apareció entre T1 y T2. SAM sigue armando la máscara propuesta, ahora sobre T2 (la prueba de la issue #8 dio 20 de 20 con la imagen entera), y una persona la valida en todos los splits. La resta de NDVI entre T1 y T2 ayuda a quien valida a ver qué cambió. Si la resta no muestra cambio en una locación nueva del padrón, el par se revisa a mano y no se descarta: puede ser un positivo con poco contraste o una locación preexistente.
- [ADR-0005](0005-clases-sin-caminos.md) sigue valiendo: las clases son fondo, locación y pileta, sin caminos; ahora locación y pileta quieren decir "nueva entre T1 y T2".
- La validación se hace en QGIS, con T1, T2 y la resta de NDVI como capas.
- Se mantienen las decisiones de la sesión de `/grill-with-docs` (PR #30): la grilla de encuadres, la locación nueva desde julio de 2019 (que deja los 6 meses de imágenes que necesita T1), el backtesting solo sobre los dos test y la primera capa entrenable en el modo congelado.
- Los pares se arman sobre los encuadres a partir de las fechas del padrón, más pares sin cambio de estaciones distintas para que el modelo no confunda humedad o luz con una locación (ver [METODO.md](../METODO.md)).
- El backtesting corre el modelo sobre pares de meses de la serie mensual (ver [EVALUACION.md](../EVALUACION.md)).
- Hay pocos positivos (las locaciones nuevas del padrón), y se compensa con varios pares por locación, aumento de datos y el encoder preentrenado.
