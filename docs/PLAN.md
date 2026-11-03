# Plan de trabajo (4 semanas)

Entrega: primeros días de diciembre de 2026. Cada tarea es un issue de GitHub con su hito (`Semana 1` a `Semana 4`, `Entrega`). El plan se ajusta sobre la marcha.

| Semana | Sofi (`sofia2005222`) | Agus (`agusbarrena`) | Bian (`biancapaz`) |
|---|---|---|---|
| **1** · 3–9 nov · bases | Cobertura de Cloud Score+; bucket y accesos; script GEE de mediana mensual; exportar imágenes | Esqueleto del repo; **formato de recorte**; U-Net con recortes falsos | Prueba de SAM con 20 locaciones; generador de máscaras propuestas; medir tiempo con 10 recortes (decide el plan de respaldo de ADR-0008) |
| **2** · 10–16 nov · etiquetado | Validar ~100 recortes; recortes por split; correcciones del dataset | Validar ~100 recortes; entrada de 6 bandas; los 3 modos de H2; curvas | Validar ~100 recortes; métricas por píxel y por objeto; locación capturada y primera detección |
| **3** · 17–23 nov · experimentos | Rotación del acuerdo entre etiquetadoras; revisión de 20 detecciones sin registro; informe (problema, datos, estado del arte) | Los 6 entrenamientos (3 vs. 6 bandas × 3 modos) | Evaluación en `test_anelo` y `test_externo`, matrices, ejemplos, cálculo del acuerdo |
| **4** · 24–30 nov · cierre | Póster | Análisis de errores y casos patológicos | Backtesting con el mejor modelo |
| **1–4 dic** | Informe final: cada una escribe su etapa | | |

## Si nos atrasamos: recorte en este orden

1. Backtesting solo en la Zona Añelo y con imágenes trimestrales en lugar de mensuales.
2. La clase pileta (el problema queda locación / fondo).
3. La variante intermedia de bandas (RGB + NIR), que ya era opcional.

**Nunca** se recorta la evaluación del modelo.
