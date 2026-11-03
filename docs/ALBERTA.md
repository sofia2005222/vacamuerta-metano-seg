# Preentrenar con el Alberta Wells Dataset (opcional, no implementado)

Es la "modificación F" del README. Sirve como ejercicio de **fine-tuning**: preentrenar en Alberta y luego ajustar en Vaca Muerta.

## Qué es

- ~188.000 parches de Planet, unos 94.000 con pozos (más de 213.000 pozos), con máscara de segmentación y cajas. Zenodo: <https://zenodo.org/records/13743323>. Artículo: <https://arxiv.org/abs/2410.09032>.

## Diferencias que hay que resolver (por eso es opcional)

| Alberta | Este proyecto | Qué hacer |
|---|---|---|
| ≈ 3 m/píxel (Planet) | 10 m/píxel (Sentinel-2) | Remuestrear Alberta a 10 m antes de entrenar, o aceptar la brecha y medirla |
| Etiqueta = **pozo** (sitio puntual) | Etiqueta = **locación** (playón) | La semántica no es idéntica: puede que la transferencia ayude poco. Es una hipótesis a medir |
| Bandas y orden de Planet | azul, verde, rojo, NIR de Sentinel-2 | Verificar que coincidan los 4 canales y su orden |
| Formato desconocido | `.npz` con `image`, `weak`, `manual` | Escribir un conversor (ver abajo) |

❓ No se inspeccionó el formato del dataset. Paso 1: descargar una porción y mirar cómo vienen los archivos.

## Cómo enchufarlo al código

1. Convertir los parches de Alberta (remuestreados) al formato de `build-patches`: un `index.csv` (columnas de `vmseg.tiling.INDEX_COLUMNS`) y un `.npz` por parche con `image` [4,H,W] float16 en [0,1], `weak` y `manual` [H,W] uint8. Se puede usar la máscara de Alberta como `weak`.
2. Entrenar en esos parches: `train --set data.root=<carpeta> --set train.run_name=alberta --set model.init=imagenet`.
3. Ajustar en Vaca Muerta partiendo de ese checkpoint: `train --set model.init=runs/alberta/best.pt --set train.run_name=F_alberta_ft --set train.lr=1e-4`.
4. Comparar contra A y C en el mismo conjunto de test manual. **Un resultado nulo también es publicable** en el informe, si se explica por las diferencias de arriba.
