---
status: accepted
---

# División espacial por bloques y una zona de test externa

Los splits no se arman al azar. La Zona Añelo se divide en una grilla de 4×4 bloques (de unos 7,5 km de lado) y **cada bloque entero** va a `train`, `val` o `test_anelo`. Así, recortes vecinos (casi iguales) no quedan repartidos entre train y test, lo que inflaría las métricas. Los recortes que cruzan el borde entre bloques de distinto split se descartan. Además hay una Zona externa (`test_externo`) que **nunca** se usa para entrenar ni para elegir hiperparámetros. Hace falta porque la Zona Añelo es casi toda de YPF (329 de 392 locaciones), y si solo se testeara ahí, el modelo podría estar aprendiendo "la forma de construir de YPF". La Zona externa es de Tecpetrol, Total y Pluspetrol.

## Consecuencias

- Se reportan dos tests. La diferencia entre `test_anelo` y `test_externo` mide la generalización (hipótesis H3).
- La asignación de bloques a splits depende de una semilla elegida para que quede cerca de 70/10/20 en cantidad de locaciones (`data/zonas_estudio.json`).
