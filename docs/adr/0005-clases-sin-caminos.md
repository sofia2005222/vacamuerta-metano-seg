---
status: accepted
---

# Clases: fondo, locación y pileta; los caminos quedan afuera

El modelo segmenta tres clases: fondo, locación y pileta. Los caminos de acceso no son una clase porque a 10 m/píxel son demasiado finos para segmentarlos de forma confiable.

## Consecuencias

- Si falta tiempo, la clase pileta se puede recortar (y el problema queda binario).
