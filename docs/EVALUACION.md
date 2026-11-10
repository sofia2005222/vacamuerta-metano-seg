# Diseño de la evaluación (división espacial)

Dos niveles de test, ambos se reportan:

| Split | Dónde | Para qué |
|---|---|---|
| `train` | Bloques de Añelo | Entrenar |
| `val` | Bloques de Añelo | Elegir hiperparámetros y época |
| `test_anelo` | Bloques de Añelo nunca vistos | Desempeño en la misma zona (dentro de distribución) |
| `test_externo` | Zona externa (Fortín de Piedra, Aguada Pichana Este, La Calera) | Generalización a otra zona y otras empresas. **Nunca se usa para entrenar ni para elegir hiperparámetros** |

- La zona de Añelo se divide en una grilla de 4×4 bloques (~7,5 km de lado) y **cada bloque entero** va a un split. Así, recortes vecinos (casi iguales) no quedan repartidos entre train y test, lo que inflaría las métricas.
- Los encuadres se arman adentro de cada bloque (grilla de 3×3, ver [METODO.md](METODO.md)), así que ningún recorte cruza el borde entre bloques de distinto split.
- La Zona externa también se divide en bloques, pero solo para repartir el etiquetado: todos son `test_externo`.
- El bloque 0 no tiene ninguna locación del padrón (por eso el script no le asignó split): sus pares van a `train` como negativos.
- Cada locación nueva da un par positivo en el split de su bloque, así que la columna "Nuevas" es el número de positivos por split ([METODO.md](METODO.md)).
- Por qué dos tests: Añelo es casi todo YPF (329 de 392 locaciones). Si solo se testea ahí, el modelo podría estar aprendiendo "la forma de construir de YPF". La zona externa es de Tecpetrol, Total y Pluspetrol.

Conteo de locaciones por split:

| Split | Locaciones | Nuevas (jul 2019 – dic 2023) |
|---|---|---|
| train | 272 | 71 |
| val | 40 | 18 |
| test_anelo | 80 | 32 (casos del backtesting) |
| test_externo | 150 | 51 (casos del backtesting) |


## Métricas por objeto (contra las máscaras)

Las métricas se calculan sobre la máscara de cada par, que marca lo que apareció entre T1 y T2 ([ADR-0011](adr/0011-modelo-bitemporal-siames.md)).

Siguiendo a Ramachandran et al. (2024): cada componente conexo de la predicción es un objeto. Un objeto predicho coincide con uno de la máscara si su IoU es ≥ 0,3, con emparejamiento greedy. Los pares que coinciden son verdaderos positivos, los objetos predichos sin pareja son falsos positivos y los de la máscara sin pareja son falsos negativos. Con eso se calculan precisión y recall por objeto.

## Criterios del backtesting (contra el padrón)

Siguen a Ramachandran et al. (2024), que para comparar contra registros oficiales agregan un margen de 50 m alrededor de cada detección y cuentan como verificadas las que contienen un punto del registro.

- **Casos:** solo las locaciones nuevas (primer pozo entre julio de 2019 y diciembre de 2023) de `test_anelo` y `test_externo`. Las de `train` y `val` quedan afuera porque el modelo ya vio esos lugares con máscara. Arrancar en julio de 2019 deja al menos 6 meses de imágenes antes de cada perforación, para que la anticipación no quede censurada por el comienzo del archivo. Los resultados se informan por separado para cada zona.
- **Par del mes *m*:** el modelo se corre sobre el par (*m* − 6, *m*) de las imágenes mensuales, y su predicción es la del mes *m*. La separación de 6 meses está a confirmar ([PREGUNTAS_ABIERTAS.md](PREGUNTAS_ABIERTAS.md)).
- **Locaciones preexistentes:** las que ya se ven en enero de 2019 salen del backtesting; son errores del padrón, no locaciones nuevas.
- **Objeto predicho:** componente conexo de los píxeles predichos como locación **o** pileta, tomadas juntas. Así una pileta pintada sobre el centro de una locación no impide capturarla. Las métricas por objeto contra las máscaras siguen siendo por clase.
- **Locación capturada:** el centro de la locación del padrón queda adentro de un objeto predicho al que se le agregó un margen de 50 m.
- **Primera detección:** el primero de **2 meses seguidos con imagen válida** en que la locación queda capturada. Los meses sin imagen válida (por ejemplo, nublados) se saltean sin cortar la racha, salvo que el hueco sea de más de 2 meses. Como análisis de sensibilidad se informa también el resultado con 1 mes.
- **Anticipación:** meses entre la primera detección y el inicio de perforación del primer pozo (`fecha_primer_pozo`). Es negativa si la detección llega después.
- **Resultados:** % de locaciones detectadas a tiempo (anticipación ≥ 0), detectadas tarde y nunca detectadas; mediana de anticipación sobre todas las detectadas, incluidas las negativas.
- **Locaciones ampliadas:** cuenta solo el primer pozo; agregar pozos después no genera un caso nuevo.
- **Detecciones sin registro:** objetos predichos que no contienen ninguna boca de pozo del **padrón completo** (todas las formaciones y todas las fechas, incluidos los pozos de 2024 en adelante: una detección en 2023 de una locación perforada en 2024 es una alerta temprana, no un error). Cuentan como falsos positivos en las métricas. Aparte se revisa a ojo una muestra (unas 20) y se informa qué fracción eran locaciones sin registrar (el paper hace lo mismo con 5.000 detecciones).
