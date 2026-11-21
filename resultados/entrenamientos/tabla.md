# Entrenamientos del #20 (preliminares)

Corrida del 2026-11-21, con los pares validados a esa fecha: **113 de train y 29 de val**. Faltan los pares regenerados de Agus (4) y de Bian (11). Cuando se cierre el dataset, se vuelven a correr los 6 y esta tabla se reemplaza.

Receta común: siamesa o fusión temprana sobre U-Net con ResNet-34, 2 clases (fondo y locación; la pileta cuenta como locación), cross entropy con pesos `[1, 10]`, AdamW con lr `1e-3`, `ReduceLROnPlateau` (factor 0,5, paciencia 5), 50 épocas, batch 8, giros y espejados como aumento de datos. Notebook: `notebooks/entrenar_pares.ipynb`. Los `best.pt` están en `gs://vmseg-imagenes/modelos/<nombre>/`.

| # | Modelo | Para | IoU locación val (mejor época) | IoU locación val (promedio últimas 10) |
|---|---|---|---|---|
| 1 | Siamesa, 6 bandas, fine-tuning | Principal | **0,745** (21) | **0,686** |
| 2 | Fusión temprana, 6 + 6 bandas, fine-tuning | H4 | 0,680 (46) | 0,615 |
| 3 | Siamesa, RGB, fine-tuning | H1 | 0,658 (39) | 0,640 |
| 4 | Siamesa, 6 bandas + índices, fine-tuning | H1 | 0,541 (35) | 0,324 |
| 5 | Siamesa, 6 bandas, desde cero | H2 | 0,729 (47) | 0,669 |
| 6 | Siamesa, 6 bandas, encoder congelado | H2 | 0,734 (34) | 0,672 |

**Cómo leer los números.** El IoU es solo de la clase locación (con 2 clases, `miou_objetos` devuelve eso), acumulado sobre todos los píxeles de val. La "mejor época" es la que guarda `best.pt`; con 29 pares de val la curva salta mucho y ese máximo favorece a los modelos más ruidosos, así que el promedio de las últimas 10 épocas es la comparación más estable. La misma configuración del modelo 1 dio 0,705 y 0,745 en dos corridas: **diferencias menores a ~0,04 son ruido**.

## Lo que dicen las hipótesis

- **H4 (arquitectura): se cumple.** La siamesa supera a la fusión temprana por ~0,07 en las dos columnas.
- **H1 (bandas): se cumple en parte.** 6 bandas le gana a RGB (0,05 en el promedio, en el límite del ruido). Agregar NDVI, NDBI y BSI empeora mucho: train llega a 0,70 como los demás, pero val salta entre 0,1 y 0,5 durante todo el entrenamiento, es decir que no generaliza. Hipótesis a revisar en el #25: los índices amplifican los cambios estacionales de la estepa entre T1 y T2.
- **H2 (transfer learning): no se distingue.** Fine-tuning, congelado y desde cero quedan entre 0,669 y 0,686, dentro del ruido. Posibles razones: ImageNet es RGB y NIR/SWIR arrancan sin preentrenamiento igual; desde cero tuvo su mejor época en la 47, así que quizás le faltan épocas.

**Mejor modelo: el 1** (siamesa, 6 bandas, fine-tuning).

## Antes de esta receta

Con cross entropy sin pesos, el modelo 1 se quedó 30 épocas prediciendo todo fondo (IoU 0) y recién después saltó a ~0,7. En los pares validados de Agus, la locación es el 0,32 % de los píxeles; ninguna validadora marcó piletas, por eso se sacó esa clase. Con pesos `[1, 10]` el IoU sube desde la primera época.

Las curvas (`<nombre>/curvas.png`) tienen el título viejo "mIoU (locación y pileta)" en el tercer gráfico, pero es el IoU de locación.
