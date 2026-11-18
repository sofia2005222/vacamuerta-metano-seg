# Cómo validar los pares

Guía para validar las máscaras de los pares asignados (issues #12, #13 y #14). Cada una valida los pares de sus bloques: la columna `responsable` de `data/indice_pares.csv` dice de quién es cada uno (unos 100 cada una). Con el circuito del #9 se tarda unos 2 minutos por par (#10).

## 0. Una sola vez

- El repo actualizado (`git pull`) y el paquete instalado (`pip install -e .`).
- QGIS (probado con 4.2).

## 1. Bajar tus pares

Con la cuenta que tiene acceso al bucket, abrí en el navegador:

- Sofi: https://storage.cloud.google.com/vmseg-imagenes/pares/paquetes/sofi.zip
- Agus: https://storage.cloud.google.com/vmseg-imagenes/pares/paquetes/agus.zip
- Bian: https://storage.cloud.google.com/vmseg-imagenes/pares/paquetes/bian.zip

Descomprimilo **adentro de la carpeta del repo**: queda `vacamuerta-metano-seg/pares_<persona>/`. Así QGIS encuentra solo el padrón y los pozos de `data/`. La carpeta no se sube al repo.

Por cada par hay cinco archivos:

| Archivo | Qué es |
|---|---|
| `<id>_t1.tif`, `<id>_t2.tif` | Imagen antes (T1) y después (T2): mediana mensual de Sentinel-2 |
| `<id>_dndvi.tif` | Caída de NDVI entre T1 y T2: blanco = se perdió vegetación |
| `<id>_propuesta.gpkg` | Máscara propuesta por SAM, para corregir |
| `<id>_mask.tif` | Máscara final; hasta validarla está toda en 255 (sin etiqueta) |

Y un `indice.csv` con tus pares. Mirá la columna `nota` antes de abrir cada uno.

## 2. Validar cada par en QGIS

1. En QGIS: Plugins → Python Console → Show Editor → abrir `scripts/qgis_abrir_par.py` → Run Script.
2. Elegí cualquier archivo del par (por ejemplo `<id>_t2.tif`). Se arma el proyecto con estas capas:
   - **pozos de otras formaciones** (triángulos naranjas): mes del pozo y formación;
   - **padrón** (puntos blancos): locaciones de Vaca Muerta, con el mes del primer pozo;
   - **máscara (editar)**: rojo = locación nueva, azul = pileta nueva;
   - **caída de NDVI** (apagada), **T2** y **T1** (apagada).
3. Prendé y apagá T1, T2 y la caída de NDVI para ver qué apareció entre las dos fechas.
4. Corregí la capa "máscara (editar)": lápiz (Toggle Editing) → Vertex Tool para mover bordes, Add Polygon Feature para dibujar (clase 1 = locación nueva, 2 = pileta nueva), Delete Selected para borrar. Guardá con Save Layer Edits.

**Qué se marca** (ver `GLOSSARY.md`, `docs/METODO.md` y ADR-0011):

- Solo lo que **apareció entre T1 y T2**. Lo que ya estaba en T1 es fondo: si SAM lo marcó, se borra.
- **Locación**: todo desmonte nuevo con forma de locación, terminado o en obra, esté o no en el padrón.
- **Pileta**: cualquier fosa o cuerpo de agua excavado nuevo. Sobre una locación, sus píxeles son pileta.
- Los **caminos** son fondo.
- La caída de NDVI ayuda, pero da falsos cambios en el agua: confirmá siempre con T1 y T2.
- Si hay duda, se pinta igual y se escribe en el campo `nota` del polígono.

**Qué revisar en cada par:**

| Ves | Qué hacer |
|---|---|
| Polígono con `¿se ve?` | Es una locación que se perfora poco después de T2: dejalo solo si ya se ve en T2 |
| Polígono con nota "pozo de otra formación" | SAM hizo clic en un pozo convencional o de otra formación: dejalo solo si es una locación nueva |
| Una locación nueva sin polígono (en el #10 pasó en 4 de 10 pares) | Dibujala, o fijate si tiene un triángulo naranja cerca |
| Un par sin polígonos (los que terminan en `_neg` son negativos) | Confirmá con la caída de NDVI que no apareció nada; si apareció algo, marcalo |
| Nota del índice "pocas observaciones despejadas" | Ese mes puede tener restos de nubes, sombras o nieve: no los marques |
| Nota del índice "también la locación X" | El par tiene dos locaciones del padrón: que estén las dos |
| La locación del padrón que originó el par **ya se ve en T1** | No lo valides: agregá una fila a `data/correcciones_pares.csv` (ver abajo) |
| Dos puntos del padrón que son una sola locación (por ejemplo, la 720 y la 721) | Marcala igual y escribilo en `nota`, para el #15 |

## 3. Pasar a máscara final

En una terminal, en la carpeta del repo:

```
python -m vmseg.mascaras_propuestas validar pares_<persona> <id>
```

Sin `<id>`, hace todos los pares de la carpeta. Escribe `<id>_mask.tif` y marca `validada=True` en tu `indice.csv`.

## 4. Devolverlos

Subí, de cada par validado, `<id>_mask.tif` y `<id>_propuesta.gpkg` a la carpeta `pares/` del bucket (https://console.cloud.google.com/storage/browser/vmseg-imagenes/pares, botón Upload files; reemplaza los anteriores). Conviene hacerlo de a tandas, al final de cada sesión. Sofi corre `python -m vmseg.generar_pares estado` para actualizar `pares/indice.csv` del bucket y ver cuántos lleva cada una.

## Cuando la locación ya se ve en T1

Agregá una fila a `data/correcciones_pares.csv` (por PR o pasándosela a Sofi):

```
id,accion,nota
anelo_b06_e13_810,t1_menos_12,se ve desmontada en T1
```

- `t1_menos_12`: T1 pasa de 6 a 12 meses antes del inicio de perforación (no antes de enero de 2019).
- `preexistente`: la locación ya estaba en enero de 2019. El par sale del dataset y se anota para el #15.

Sofi regenera ese par (ver abajo) y lo vuelve a subir con el mismo `id`; se baja de nuevo de `pares/` y se valida.

## Para Sofi: generar y regenerar los pares

Todo se guarda en `data/pares/`; usa las credenciales de Earth Engine para leer y escribir el bucket.

```
python -m vmseg.armar_pares                   # data/indice_pares.csv (aplica las correcciones)
python -m vmseg.generar_pares recortar        # T1, T2 y máscara sin etiquetar, desde las imágenes mensuales
python -m vmseg.generar_pares subir           # pares cortados a gs://vmseg-imagenes/pares/
# SAM: notebooks/proponer_pares_colab.ipynb en Colab con GPU (baja, propone y sube; minutos)
python -m vmseg.generar_pares bajar           # trae las propuestas y la caída de NDVI
python -m vmseg.generar_pares empaquetar      # un zip por responsable
python -m vmseg.generar_pares subir           # sube los zips (no pisa máscaras ni propuestas ya subidas)
python -m vmseg.generar_pares estado          # baja las máscaras y cuenta las validadas
```

Para regenerar pares corregidos: `recortar <id>`, `proponer <id>` (en la compu tarda ~2 min por par sin GPU) y `subir <id>`.

SAM necesita `segment-anything` y el checkpoint `sam_vit_h_4b8939.pth` en `~/.cache/sam/` (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
