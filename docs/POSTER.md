# Guía para el póster y la presentación oral

Requisitos de la consigna: **A0 o A1 vertical**, "Título" y "Autores" arriba (con institución y emails), ≤ 3 colores, poco texto, viñetas, fondo sobrio. Tamaños sugeridos: título 72–100 pt, autores 40–70 pt, secciones 36–48 pt, texto 24–36 pt, leyendas 18–24 pt.

## Mensaje clave (una sola idea)

> *Para auditar el metano de Vaca Muerta primero hay que saber dónde están las instalaciones. Los datos oficiales dan puntos, no huellas; mostramos que un modelo entrenado con esos puntos ruidosos puede segmentar las locaciones en imágenes Sentinel-2, y medimos cuánto.*

El título tiene que reflejar la extensión (no "Segmentación de pozos"). Opciones:

1. *Auditando el metano desde el espacio: segmentación de infraestructura de Vaca Muerta con etiquetas débiles y Sentinel-2*
2. *¿Dónde están las fuentes? Inventario satelital de locaciones petroleras para auditar el reporte de metano en Neuquén*
3. *De puntos a polígonos: aprender a segmentar locaciones de Vaca Muerta con datos oficiales ruidosos*

## Estructura sugerida (de arriba a abajo)

| Bloque | Contenido | Figura |
|---|---|---|
| Contexto | Reporte obligatorio de metano (sept. 2026) y la brecha de datos; 3 viñetas | Mapa de la cuenca con el AOI |
| Problema y datos | Punto de pozo → huella de la locación; Sentinel-2 10 m + padrón oficial | Imagen con los puntos encima y el disco de buffer |
| Método | U-Net ResNet34; etiquetas débiles; pérdida robusta; prueba manual | Diagrama del pipeline (4 cajas) |
| Resultados | Tabla A–E (IoU, F1, recall/precisión por objeto, media ± desvío) | Tabla + curva de ablación del radio |
| Caso de estudio | Sierra Barrosa: la predicción sobre la imagen | Imagen · predicción · polígonos |
| Errores | Falsos positivos y negativos típicos (pads viejos, caminos, suelo desnudo) | 4–6 recortes |
| Conclusiones | 3 viñetas con números | — |
| Trabajo futuro | "Con 6 meses más": tanques individuales con mayor resolución; cruce con detecciones de metano; cambio temporal para detectar pads nuevos | — |
| Referencias | Ramachandran 2024, Seth 2025, Omara 2023 y fuentes de prensa | — |

## Presentación oral (~5 minutos)

1. **Gancho (30 s):** la escuela cerrada y el reporte obligatorio — *decirlo como "según la prensa"*.
2. **El problema (45 s):** puntos vs huellas.
3. **Qué hicimos (90 s):** etiquetas débiles, prueba manual, qué comparamos.
4. **Qué encontramos (90 s):** los dos o tres números que importan y el caso Sierra Barrosa.
5. **Límites y cierre (45 s):** qué no hace (no detecta metano ni tanques), qué sigue.

Preguntas que seguramente harán (tenerlas ensayadas):
- ¿Por qué no etiquetar todo a mano? → costo; mostramos cuánto se gana con datos oficiales.
- ¿Cómo saben que no hay fuga entre train y test? → partición por bloques espaciales.
- ¿Qué pasa si el padrón está desactualizado? → es ruido de etiqueta; lo medimos en la prueba manual.
- ¿Qué aporta la pérdida robusta? → mostrar la comparación A vs B con desvíos, sin sobre-interpretar si la diferencia es chica.
- ¿Y el metano? → este modelo no lo mide; produce el inventario sobre el cual cruzarlo.
