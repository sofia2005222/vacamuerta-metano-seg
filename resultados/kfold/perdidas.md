# Elección de la pérdida (#20)

Corrida del 2026-11-24 y 2026-11-25: comparar la cross entropy con pesos que usábamos contra combinaciones de CE con Dice y Lovász (`src/vmseg/perdidas.py`, con `smp.losses`).

**Cómo se comparó.** Mismo modelo en los cuatro experimentos (modelo 1: siamesa, 6 bandas, fine-tuning) y validación cruzada por bloques: train + val (148 pares validados, 11 bloques de Añelo) repartidos en 5 folds de 28 a 31 pares, con cada bloque entero en un solo fold (`asignar_folds`). Cada fold se entrena con los otros cuatro: 50 épocas, AdamW con lr `1e-3`, `ReduceLROnPlateau` (factor 0,5, paciencia 5), batch 8, giros y rotaciones de 90°. Los dos test no se usan. Notebook: `notebooks/entrenar_kfold.ipynb`.

Métrica: IoU de locación en el fold de validación, en la mejor época y como promedio de las últimas 10 épocas (más estable). Se reporta media ± desvío de los 5 folds.

## Resultados

| Pérdida | IoU val (mejor época) | IoU val (promedio últimas 10) | Mejores épocas por fold |
|---|---|---|---|
| CE con pesos `[1, 10]` (la anterior) | 0,720 ± 0,030 | 0,659 ± 0,055 | 34, 26, 43, 8, 27 |
| **CE + Dice** | **0,787 ± 0,038** | **0,760 ± 0,055** | 43, 37, 48, 39, 33 |
| CE + Lovász | 0,788 ± 0,034 | 0,722 ± 0,031 | 38, 29, 43, 30, 50 |
| CE + Dice + Lovász | 0,789 ± 0,033 | 0,760 ± 0,039 | 44, 48, 50, 34, 49 |

IoU val (mejor época) por fold. Cada fold es el mismo conjunto de pares en los cuatro experimentos, así que se pueden comparar fila por fila:

| Fold | Bloques | Pares val (positivos) | CE con pesos | CE + Dice | CE + Lovász | CE + Dice + Lovász |
|---|---|---|---|---|---|---|
| 0 | 6, 8 | 31 (20) | 0,708 | 0,749 | 0,778 | 0,762 |
| 1 | 0, 4, 5 | 30 (11) | 0,687 | 0,763 | 0,734 | 0,755 |
| 2 | 9, 13 | 30 (18) | 0,750 | 0,815 | 0,797 | 0,809 |
| 3 | 1, 11 | 29 (15) | 0,702 | 0,770 | 0,814 | 0,784 |
| 4 | 2, 7 | 28 (16) | 0,755 | 0,838 | 0,816 | 0,835 |

El fold 3 son los bloques 1 y 11, que eran el val fijo de la primera corrida (`resultados/entrenamientos/tabla.md`).

## Conclusión

- **Las tres combinaciones con Dice o Lovász superan a la CE con pesos** por ~0,07 de IoU. CE + Dice le gana en los 5 folds (de +0,04 a +0,08), y en el promedio de las últimas 10 épocas la diferencia es mayor (+0,10): además de llegar más alto, el entrenamiento es más estable. Con CE con pesos el modelo llega a su mejor época antes y después empeora (en el fold 3, en la época 8).
- **Entre las tres combinaciones no hay diferencia** en la mejor época (0,787 a 0,789, dentro del desvío; por fold, CE + Dice y CE + Dice + Lovász difieren en menos de 0,015). CE + Lovász es menos estable al final (0,722 en las últimas 10).
- **Elegida: CE + Dice.** Rinde igual que las otras, es la más simple, es la del tutorial 11 de la materia y llega a su mejor época antes (33 a 48; CE + Dice + Lovász llega a 48–50 en tres folds, lo que sugiere que con 50 épocas no termina de converger).

Los modelos 2 a 6 del #20 se entrenan con CE + Dice. Si entran pares nuevos antes del corte del dataset, el modelo 1 se vuelve a correr con CE + Dice junto con los demás.

## Archivos

- `<experimento>/resultado.json`: configuración, media y desvío, y el resumen de cada fold.
- `<experimento>/fold<i>/curvas.png`: loss por iteración, loss por época e IoU de locación por época.
- Los checkpoints de cada fold no se guardaron: el K-fold solo mide; el modelo final se entrena después con los 148 pares.
