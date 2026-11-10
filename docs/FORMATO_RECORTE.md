# Formato de un par

Contrato entre las tres etapas. Desde [ADR-0011](adr/0011-modelo-bitemporal-siames.md), la unidad es un **par**: un encuadre en dos imágenes mensuales (T1 antes, T2 después) y una máscara de lo que apareció entre los dos. El código está en `src/vmseg/par.py` (`guardar_par`, `leer_par`, `leer_indice_pares` y `generar_pares_falsos`).

## Archivos

Una carpeta de pares tiene, por cada par, tres GeoTIFF con la misma grilla, más un índice:

```
pares/
  indice.csv
  <id>_t1.tif       recorte antes (T1)
  <id>_t2.tif       recorte después (T2)
  <id>_mask.tif     máscara
```

## Recortes (`<id>_t1.tif` y `<id>_t2.tif`)

Los dos tienen exactamente el mismo formato y la misma grilla:

| | |
|---|---|
| Tamaño | 256 × 256 px (2,56 km de lado) |
| Resolución | 10 m/píxel. B11 y B12 (20 m) se remuestrean a 10 m |
| Bandas | 6, en este orden: B2, B3, B4, B8, B11, B12 |
| Tipo | `float32` |
| Valores | Reflectancia en [0, 1] (valor de Sentinel-2 ÷ 10000, recortado a [0, 1]) |
| Sin dato | `NaN` (por ejemplo, un píxel nublado todo el mes) |
| CRS | EPSG:32719 (UTM 19S) |

En el archivo va la reflectancia **sin normalizar**. La normalización para el modelo (media y desvío por banda calculados sobre `train`) y los índices (NDVI, NDBI, BSI) los calcula el Dataset al leer. Así, si cambia la normalización o se agrega un índice, no hay que regenerar los recortes.

## Máscara (`<id>_mask.tif`)

Marca solo lo que **apareció entre T1 y T2**. Lo que ya estaba en T1 es fondo.

| Valor | Clase |
|---|---|
| 0 | fondo (incluye locaciones y piletas que ya estaban en T1) |
| 1 | locación nueva |
| 2 | pileta nueva |
| 255 | sin etiqueta (no cuenta en la loss ni en las métricas) |

Tipo `uint8`, una banda, misma grilla que los recortes. En los pares negativos es todo 0.

## Índice (`indice.csv`)

Una fila por par.

| Columna | Ejemplo | Descripción |
|---|---|---|
| `id` | `anelo_b07_e4_738` | Nombre de los archivos, sin sufijo ni extensión |
| `encuadre` | `anelo_b07_e4` | Encuadre del par (bloque y posición en la grilla de 3×3) |
| `locacion_id` | `738` | Locación nueva del padrón que originó el par; `-1` en los negativos |
| `zona` | `anelo` / `externa` | Zona de estudio |
| `bloque` | `7` | Bloque de la grilla 4×4 (0–15); `-1` en la Zona externa |
| `split` | `train` | `train`, `val`, `test_anelo` o `test_externo` |
| `mes_t1` | `2021-04` | Mes de la mediana mensual de T1 |
| `mes_t2` | `2021-11` | Mes de la mediana mensual de T2 |
| `x_min`, `y_max` | `330000`, `5770000` | Esquina superior izquierda, en metros del CRS |
| `crs` | `EPSG:32719` | CRS de las coordenadas |
| `validada` | `True` | Si una persona ya validó la máscara (si no, es una máscara propuesta) |
| `nota` | | Texto libre |

## Pares falsos

`generar_recortes_falsos` tiene que pasar a generar pares con este formato: T1 con suelo y ruido, T2 igual pero con locaciones (rectángulos claros) y piletas (círculos oscuros) nuevas, y algunas locaciones que ya están en las dos fechas, para probar que el modelo no las marque. Sirven para desarrollar y probar el modelo y la evaluación sin esperar las imágenes reales.

## Para revisar

- ¿GEE exporta en EPSG:32719, o hay que reproyectar?
- ¿`NaN` para sin dato les sirve, o prefieren una banda de máscara de nubes?
- `float32` ocupa unos 1,4 MB por recorte (unos 840 MB para 300 pares). Si molesta, se puede pasar a `uint16` con el valor original de Sentinel-2.
