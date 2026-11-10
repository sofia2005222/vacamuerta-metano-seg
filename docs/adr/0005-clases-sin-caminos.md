---
status: accepted
---

# Clases: fondo, locación y pileta; los caminos quedan afuera

El modelo segmenta tres clases: fondo, locación y pileta. Los caminos de acceso no son una clase porque a 10 m/píxel son demasiado finos para segmentarlos de forma confiable.

## Consecuencias

- Si falta tiempo, la clase pileta se puede recortar (y el problema queda binario). En ese caso las piletas ya validadas en las máscaras pasan a ser **locación**, sin volver a etiquetar nada, así el modelo no aprende a ignorar las superficies excavadas.
