# Riesgos y próximos pasos

## Riesgos

- **Resolución:** las locaciones chicas ocupan pocos píxeles. La revisión manual de noviembre de 2026 mostró que las locaciones nuevas miden 150–200 m de lado (15–20 píxeles) y se distinguen bien; el profesor considera que la resolución alcanza.
- **Pocos positivos:** hay 172 locaciones nuevas en las dos zonas, y solo 89 en `train` y `val`. Se compensa con aumento de datos, el encoder preentrenado y, si hace falta, varios pares por locación.
- **Cambios que no son locaciones:** el contraste cambia con la estación y con el suelo húmedo (en invierno las locaciones se ven más tenues). Se cubre con pares negativos de estaciones distintas.
- **Anticipación corta:** en la revisión manual, la locación apareció entre 5 días y 4,5 meses antes del inicio de perforación. Con meses enteros, los casos de pocos días dan anticipación 0.
- **Locaciones preexistentes:** el padrón marca como nuevas algunas locaciones que ya existían (por ejemplo, la 882). Hay que sacarlas antes del backtesting.
- **Confusión con suelo natural:** la estepa ya es medio pelada (motiva H1).
- **Tiempo de etiquetado:** es la tarea más larga. Medir con 10 pares antes de planificar.
- **Sobredimensionar:** si falta tiempo, recortar en el orden de [PLAN.md](PLAN.md), nunca la evaluación del modelo.

## Próximos pasos

- [x] Conseguir datos oficiales de pozos con coordenadas y fechas
- [x] Armar dataset de locaciones, zonas y splits
- [x] Confirmar a ojo que las locaciones se distinguen en Sentinel-2: revisadas a mano 4 locaciones nuevas en Copernicus Browser (809, 667, 738 y 882), con su fecha de aparición contra el inicio de perforación
- [ ] Validar 10 pares en QGIS y medir el tiempo
- [x] Mandar el mail al profesor: aprobó el tema, la resolución le parece suficiente y sugirió el enfoque bitemporal ([ADR-0011](adr/0011-modelo-bitemporal-siames.md))
- [ ] Buscar 1–2 papers extra de segmentación en Sentinel-2
- [ ] Descargar imágenes y armar los pares por split
