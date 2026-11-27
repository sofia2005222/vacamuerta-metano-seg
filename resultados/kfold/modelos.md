# Comparación de los 6 modelos (#20)

Corrida del 2026-11-27, con el dataset cerrado (294 pares validados). Los 6 modelos usan la misma validación cruzada por bloques: train + val (159 pares, 91 con locación nueva) repartidos en 5 folds de 31 a 33 pares, con cada bloque entero en un solo fold (`asignar_folds`). Todos se entrenan igual: CE + Dice (elegida en `perdidas.md`), 50 épocas, AdamW con lr `1e-3`, `ReduceLROnPlateau` (factor 0,5, paciencia 5), batch 8, giros y rotaciones de 90°. Los dos test no se usan. Notebook: `notebooks/entrenar_kfold.ipynb`.

Métrica: IoU de locación en el fold de validación, en la mejor época y como promedio de las últimas 10 épocas (más estable). Se reporta media ± desvío de los 5 folds.

## Resultados

| Modelo | Arquitectura | Entrada | Encoder | IoU val (mejor época) | IoU val (promedio últimas 10) | Mejores épocas por fold |
|---|---|---|---|---|---|---|
| **1** | **Siamesa** | **6 bandas** | **ImageNet, fine-tuning** | **0,775 ± 0,052** | **0,747 ± 0,049** | 48, 42, 31, 42, 40 |
| 2 | Fusión temprana | 6 bandas | ImageNet, fine-tuning | 0,761 ± 0,051 | 0,736 ± 0,047 | 40, 46, 30, 21, 34 |
| 3 | Siamesa | RGB | ImageNet, fine-tuning | 0,763 ± 0,060 | 0,715 ± 0,074 | 33, 40, 6, 50, 27 |
| 4 | Siamesa | 6 bandas + NDVI, NDBI, BSI | ImageNet, fine-tuning | 0,759 ± 0,069 | 0,736 ± 0,072 | 16, 34, 40, 36, 45 |
| 5 | Siamesa | 6 bandas | Desde cero | 0,759 ± 0,052 | 0,731 ± 0,052 | 32, 44, 37, 49, 39 |
| 6 | Siamesa | 6 bandas | ImageNet, congelado | 0,760 ± 0,063 | 0,734 ± 0,056 | 33, 44, 30, 47, 25 |

IoU val (mejor época) por fold. Cada fold es el mismo conjunto de pares en los seis modelos, así que se pueden comparar fila por fila:

| Fold | Bloques | Pares val (con locación) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 31 (24) | 0,685 | 0,688 | 0,687 | 0,648 | 0,683 | 0,652 |
| 1 | 0, 4, 5 | 33 (14) | 0,783 | 0,744 | 0,820 | 0,739 | 0,729 | 0,757 |
| 2 | 1, 9 | 31 (19) | 0,790 | 0,783 | 0,711 | 0,784 | 0,780 | 0,792 |
| 3 | 11, 13 | 31 (17) | 0,804 | 0,765 | 0,798 | 0,803 | 0,801 | 0,800 |
| 4 | 2, 7, 8 | 33 (17) | 0,816 | 0,825 | 0,803 | 0,823 | 0,803 | 0,801 |

Los folds no son los mismos que en `perdidas.md`: con los pares nuevos cambió la repartición de bloques. Por eso el modelo 1 da 0,775 acá y 0,787 allá.

## Conclusión

- **El modelo 1 (siamesa, 6 bandas, fine-tuning) tiene la mejor media**, pero las diferencias con los otros cinco son chicas: de 0,012 a 0,016 de IoU, menos que el desvío entre folds (~0,05). Comparando fold por fold, el modelo 1 le gana a cada uno en 3 a 5 de los 5 folds, pero ninguna diferencia es significativa (t pareado con 4 grados de libertad entre 0,6 y 2,3; hace falta 2,78 para p < 0,05). **Con 159 pares no podemos separar a los 6 modelos.**
- **H4 (siamesa contra fusión temprana):** la siamesa queda arriba (+0,015, gana en 3 de 5 folds), pero dentro del ruido.
- **H1 (bandas):** RGB llega a una media parecida, pero es el menos estable: tiene el peor promedio de las últimas 10 épocas (0,715 ± 0,074) y en el fold 2 su mejor época es la 6 y después cae a ~0,61. Agregar índices (modelo 4) ya no empeora como en la corrida preliminar (0,541 con val fijo y CE con pesos): con CE + Dice y más datos queda a la par de los demás. Lo que se veía antes era un problema de entrenamiento, no de los índices.
- **H2 (transfer de ImageNet):** fine-tuning, desde cero y congelado dan casi lo mismo (0,775, 0,759, 0,760). Es coherente con lo que dijo el profe: ImageNet es otro dominio y aporta poco con imágenes multiespectrales.
- **El fold 0 (bloque 6) es el más difícil para todos** (0,65 a 0,69): es el fold con más locaciones (24 de 31 pares).
- **Para el modelo final elegimos el modelo 1:** mejor media, mejor promedio de las últimas 10 y es la configuración base del proyecto. Se reentrena con los 159 pares durante 42 épocas (mediana de sus mejores épocas) y se evalúa en los test en el #21.

## Archivos

- `<modelo>/resultado.json`: configuración, media y desvío, y el resumen de cada fold.
- `<modelo>/fold<i>/curvas.png`: loss por iteración, loss por época e IoU de locación por época.
