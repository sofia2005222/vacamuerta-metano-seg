---
status: accepted
---

# Imágenes desde Google Earth Engine

Las imágenes Sentinel-2 se descargan y procesan en Google Earth Engine, no en Microsoft Planetary Computer. Sofi ya trabaja con GEE, el equipo tiene créditos de Google Cloud, y la mediana mensual con nubes enmascaradas por Cloud Score+ es un procedimiento estándar de su catálogo. GEE exige una cuenta con un proyecto de Cloud registrado. Solo la necesita quien genera las imágenes: el resto del equipo trabaja con los archivos exportados.

## Opciones consideradas

- **Planetary Computer (STAC):** sin cuenta, pero la mediana se calcula en la máquina propia. Además el código de la v1 (tag `v1-metano`) nunca se probó contra la red.
- **Copernicus Browser:** descarga manual, inviable para 60 meses × 2 zonas.
