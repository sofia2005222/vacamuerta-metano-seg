# Formato de un recorte

El código está en `src/vmseg/recorte.py`.

## Archivos

Una carpeta de recortes tiene, por cada recorte, dos GeoTIFF con la misma grilla, más un índice:

```
recortes/
  indice.csv
  <id>.tif          imagen
  <id>_mask.tif     máscara
```

## Imagen (`<id>.tif`)

| | |
|---|---|
| Tamaño | 256 × 256 px (2,56 km de lado) |
| Resolución | 10 m/píxel. B11 y B12 (20 m) se remuestrean a 10 m |
| Bandas | 6, en este orden: B2, B3, B4, B8, B11, B12 |
| Tipo | `float32` |
| Valores | Reflectancia en [0, 1] (valor de Sentinel-2 ÷ 10000, recortado a [0, 1]) |
| Sin dato | `NaN` (por ejemplo, un píxel nublado todo el mes) |
| CRS | EPSG:32719 (UTM 19S) |

En el archivo va la reflectancia **sin normalizar**. La normalización para el modelo (media y desvío por banda calculados sobre `train`) la hace el Dataset al leer. Así, si cambia la normalización, no hay que regenerar los recortes.

## Máscara (`<id>_mask.tif`)

| Valor | Clase |
|---|---|
| 0 | fondo |
| 1 | locación |
| 2 | pileta |
| 255 | sin etiqueta (no cuenta en la loss ni en las métricas) |

Tipo `uint8`, una banda, misma grilla que la imagen.

## Índice (`indice.csv`)

Una fila por recorte.

| Columna | Ejemplo | Descripción |
|---|---|---|
| `id` | `anelo_b07_0012` | Nombre de los archivos, sin extensión |
| `zona` | `anelo` / `externa` | Zona de estudio |
| `bloque` | `7` | Bloque de la grilla 4×4 (0–15); `-1` en la Zona externa |
| `split` | `train` | `train`, `val`, `test_anelo` o `test_externo` |
| `mes` | `2021-03` | Mes de la mediana mensual |
| `x_min`, `y_max` | `330000`, `5770000` | Esquina superior izquierda, en metros del CRS |
| `crs` | `EPSG:32719` | CRS de las coordenadas |
| `validada` | `True` | Si una persona ya validó la máscara (si no, es una máscara propuesta) |
| `nota` | | Texto libre |

## Recortes falsos

`generar_recortes_falsos(carpeta, n)` escribe recortes con este mismo formato: suelo con ruido, locaciones como rectángulos claros y piletas como círculos oscuros. Sirven para desarrollar y probar el modelo y la evaluación sin esperar las imágenes reales.

## Para revisar  

- ¿GEE exporta en EPSG:32719, o hay que reproyectar?
- ¿`NaN` para sin dato les sirve, o prefieren una banda de máscara de nubes?
- `float32` ocupa unos 1,4 MB por recorte (unos 420 MB para 300 recortes). Si molesta, se puede pasar a `uint16` con el valor original de Sentinel-2.
