# Riesgos y próximos pasos

## Riesgos

- **Resolución:** las locaciones chicas ocupan pocos píxeles.
- **Confusión con suelo natural:** la estepa ya es medio pelada (motiva H1).
- **Tiempo de etiquetado:** es la tarea más larga. Medir con 10 recortes antes de planificar.
- **Sobredimensionar:** si falta tiempo, recortar en el orden de [PLAN.md](PLAN.md), nunca la evaluación del modelo.

## Próximos pasos

- [x] Conseguir datos oficiales de pozos con coordenadas y fechas
- [x] Armar dataset de locaciones, zonas y splits
- [ ] Abrir `locaciones_vaca_muerta.geojson` sobre Sentinel-2 y confirmar a ojo que las locaciones se distinguen (comparar 2019 vs. 2023 en Loma Campana)
- [ ] Etiquetar 10 recortes con SAM y medir el tiempo
- [x] Mandar el mail al profesor (aprobación del tema y número de recortes etiquetados): enviado, falta la respuesta
- [ ] Buscar 1–2 papers extra de segmentación en Sentinel-2
- [ ] Descargar imágenes y armar recortes por split
