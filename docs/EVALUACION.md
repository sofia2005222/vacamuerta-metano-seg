# Diseño de la evaluación (división espacial)

Dos niveles de test, ambos se reportan:

| Split | Dónde | Para qué |
|---|---|---|
| `train` | Bloques de Añelo | Entrenar |
| `val` | Bloques de Añelo | Elegir hiperparámetros y época |
| `test_anelo` | Bloques de Añelo nunca vistos | Desempeño en la misma zona (dentro de distribución) |
| `test_externo` | Zona oeste (Fortín de Piedra, Aguada Pichana Este, La Calera) | Generalización a otra zona y otras empresas. **Nunca se usa para entrenar ni para elegir hiperparámetros** |

- La zona de Añelo se divide en una grilla de 4×4 bloques (~7,5 km de lado) y **cada bloque entero** va a un split. Así, recortes vecinos (casi iguales) no quedan repartidos entre train y test, lo que inflaría las métricas.
- Los recortes de imagen que crucen el borde entre bloques de distinto split se descartan.
- El bloque 0 no tiene ninguna locación del padrón (por eso el script no le asignó split): sus recortes van a `train` como negativos.
- Por qué dos tests: Añelo es casi todo YPF (329 de 392 locaciones). Si solo se testea ahí, el modelo podría estar aprendiendo "la forma de construir de YPF". La zona externa es de Tecpetrol, Total y Pluspetrol.

Conteo de locaciones por split:

| Split | Locaciones | Nuevas 2019–2023 |
|---|---|---|
| train | 272 | 81 |
| val | 40 | 20 |
| test_anelo | 80 | 34 |
| test_externo | 150 | 62 |


## Métricas por objeto (contra las máscaras)

Siguiendo a Ramachandran et al. (2024): cada componente conexo de la predicción es un objeto. Un objeto predicho coincide con uno de la máscara si su IoU es ≥ 0,3, con emparejamiento greedy. Los pares que coinciden son verdaderos positivos, los objetos predichos sin pareja son falsos positivos y los de la máscara sin pareja son falsos negativos. Con eso se calculan precisión y recall por objeto.

## Criterios del backtesting (contra el padrón)

Siguen a Ramachandran et al. (2024), que para comparar contra registros oficiales agregan un margen de 50 m alrededor de cada detección y cuentan como verificadas las que contienen un punto del registro.

- **Locación capturada:** el centro de la locación del padrón queda adentro de un objeto predicho de clase locación al que se le agregó un margen de 50 m.
- **Primera detección:** el primero de **2 meses seguidos** en que la locación queda capturada. Como análisis de sensibilidad se informa también el resultado con 1 mes.
- **Anticipación:** meses entre la primera detección y el inicio de perforación del primer pozo (`fecha_primer_pozo`).
- **Locaciones ampliadas:** cuenta solo el primer pozo; agregar pozos después no genera un caso nuevo.
- **Detecciones sin registro:** cuentan como falsos positivos en las métricas. Aparte se revisa a ojo una muestra (unas 20) y se informa qué fracción eran locaciones sin registrar (el paper hace lo mismo con 5.000 detecciones).
