# Preguntas abiertas (insumo para `/grill-with-docs`)

Lo que falta decidir. Cuando una pregunta se resuelve, se borra de acá: el término va a `GLOSSARY.md`, la decisión a un ADR si cumple los criterios, y el detalle al doc temático que corresponda.

## Del enfoque bitemporal ([ADR-0011](adr/0011-modelo-bitemporal-siames.md))

- **Separación de los pares del backtesting:** hoy (*m* − 6, *m*). ¿Conviene una separación más corta para detectar antes, o eso agrega cambios estacionales? Depende de lo que salga de los primeros entrenamientos.
- **Fechas de los pares de entrenamiento:** se arranca con T1 = inicio de perforación − 6 meses y T2 = inicio de perforación + 1. Si fallan seguido en los primeros 20 pares, se pasa a la serie mensual de NDVI ([METODO.md](METODO.md)). ¿Se suman pares extra por locación (otros T1 y T2) para tener más positivos, reusando la máscara validada?
- **Cantidad de pares:** meta provisoria de 300 (los 172 positivos y unos 130 negativos). Se ajusta con la medición de tiempo de validación.
