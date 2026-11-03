---
status: accepted
---

# Reescribir el código de cero en lugar de adaptar la versión 1

El código de la versión 1 (enfoque metano: segmentación binaria con etiquetas débiles, splits al azar dentro de una escena, un solo compuesto de imágenes) se borró del repo. El proyecto se reescribe de cero según los ADR vigentes, y no se escribe código hasta tener todo el trabajo desglosado en issues. Adaptar la v1 hubiera arrastrado supuestos que cambiaron (clases, etiquetas, splits, dimensión temporal). La v1 queda en el tag `v1-metano`: cualquier pieza se recupera con `git checkout v1-metano -- <ruta>`. Por ejemplo, la descarga de Sentinel-2 vía STAC está en `src/vmseg/stac.py`.
