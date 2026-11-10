# Plan de trabajo (4 semanas)

Entrega: primeros días de diciembre de 2026. Cada tarea es un issue de GitHub con su hito (`Semana 1` a `Semana 4`, `Entrega`). El plan se ajusta sobre la marcha.

| Semana | Sofi (`sofia2005222`) | Agus (`agusbarrena`) | Bian (`biancapaz`) |
|---|---|---|---|
| **1** · 3–9 nov · bases | Cobertura de Cloud Score+; bucket y accesos; script GEE de mediana mensual; exportar imágenes | Esqueleto del repo; **formato de recorte**; U-Net con recortes falsos | Prueba de SAM con 20 locaciones (hecha); generador de máscaras propuestas con SAM sobre T2 y proyecto de QGIS con T1, T2 y la resta de NDVI |
| **2** · 10–16 nov · etiquetado | Armar los pares por split; validar ~100 pares; correcciones del dataset (incluye locaciones preexistentes) | Formato de par; siamesa y fusión temprana; entradas RGB, 6 bandas y 6 + índices; los 3 modos de H2; curvas; validar ~100 pares | Medir el tiempo con 10 pares; validar ~100 pares; métricas por píxel y por objeto; locación capturada y primera detección sobre pares |
| **3** · 17–23 nov · experimentos | Rotación del acuerdo entre etiquetadoras; revisión de 20 detecciones sin registro; informe (problema, datos, estado del arte) | Los 6 entrenamientos de [METODO.md](METODO.md) | Evaluación en `test_anelo` y `test_externo`, matrices, ejemplos, cálculo del acuerdo |
| **4** · 24–30 nov · cierre | Póster | Análisis de errores y casos patológicos | Backtesting con el mejor modelo |
| **1–4 dic** | Informe final: cada una escribe su etapa | | |

## Si nos atrasamos: recorte en este orden

1. Backtesting solo en la Zona Añelo y con imágenes trimestrales en lugar de mensuales.
2. Los entrenamientos de H1 y H2 (del 6 al 3 de la tabla de [METODO.md](METODO.md), de abajo hacia arriba).
3. La clase pileta (el problema queda locación / fondo).

**Nunca** se recortan la evaluación del modelo ni los entrenamientos 1 (siamesa) y 2 (fusión temprana).
