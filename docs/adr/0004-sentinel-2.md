---
status: accepted
---

# Sentinel-2 en lugar de imágenes pagas de alta resolución

Usamos Sentinel-2: es gratis, tiene archivo desde 2017, revisita cada 5 días y da 10 m/píxel en RGB+NIR y 20 m en SWIR. Como Neuquén es árido, hay pocas nubes. El backtesting necesita un histórico mensual largo, y con alta resolución paga (Planet, Maxar) eso no está al alcance de un TP.

## Consecuencias

- La resolución limita el tamaño mínimo detectable: una locación de 50–250 m ocupa 5–25 píxeles.
- No se pueden distinguir tanques individuales.
