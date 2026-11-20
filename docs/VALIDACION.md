# Cómo validar los pares

Guía para validar las máscaras de los pares asignados (issues #12, #13 y #14). Cada una valida los pares de sus bloques: la columna `responsable` de `data/indice_pares.csv` dice de quién es cada uno (unos 100 cada una). Primero una **revisión rápida** de todos (unos segundos por par) y después QGIS solo para los que hay que corregir (unos 2 minutos cada uno, #10).

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

## 2. Revisión rápida

En una terminal, en la carpeta del repo:

```
python -m vmseg.revisar pares_<persona>
```

Se abre una ventana con T1, T2 y la caída de NDVI de cada par, con los polígonos de SAM (rojo = locación, celeste = pileta), el padrón (puntos blancos con el mes del primer pozo) y los pozos de otras formaciones (triángulos naranjas). Arriba están las notas del par. Con la lupa de la barra de abajo se hace zoom (los tres paneles se mueven juntos; la casita vuelve al par entero).

Mirá el par y apretá una tecla:

| Tecla | Cuándo | Qué hace |
|---|---|---|
| **A** | La propuesta está bien (o el negativo no tiene nada nuevo) | Escribe `<id>_mask.tif` y marca el par validado |
| **C** | Hay que corregir algo: borrar, agregar o mover bordes | Lo deja para QGIS |
| **T** | La locación del padrón que originó el par **ya se ve en T1** | Agrega `t1_menos_12` a `data/correcciones_pares.csv` |
| **P** | Esa locación ya estaba en enero de 2019 | Agrega `preexistente` a `data/correcciones_pares.csv` |
| **←** / **→** | Volver al anterior / saltear | Si volvés y apretás otra tecla, se deshace la decisión anterior |
| **Q** | Salir | Lo decidido queda guardado en `pares_<persona>/revision.csv`; al volver a abrir sigue donde quedaste |

Aceptá con A solo si **no hay nada que tocar**. Ante la duda, C.

## 3. Corregir en QGIS los marcados con C

1. En QGIS: Plugins → Python Console → Show Editor → abrir `scripts/qgis_abrir_par.py` → Run Script.
2. Elegí `pares_<persona>/revision.csv`: abre el primer par marcado con C. Para el siguiente, escribí `siguiente()` en la consola y Enter. (También podés elegir un archivo de un par para abrir ese.)
3. Capas del proyecto:
   - **pozos de otras formaciones** (triángulos naranjas): mes del pozo y formación;
   - **padrón** (puntos blancos): locaciones de Vaca Muerta, con el mes del primer pozo;
   - **máscara (editar)**: rojo = locación nueva, azul = pileta nueva;
   - **caída de NDVI** (apagada), **T2** y **T1** (apagada).
4. Prendé y apagá T1, T2 y la caída de NDVI para ver qué apareció entre las dos fechas.
5. Corregí la capa "máscara (editar)": lápiz (Toggle Editing) → Vertex Tool para mover bordes, Add Polygon Feature para dibujar (clase 1 = locación nueva, 2 = pileta nueva), Delete Selected para borrar. Guardá con Save Layer Edits.
6. Cuando termines con todos, volvé a correr `python -m vmseg.revisar pares_<persona>`: los corregidos aparecen al final, ya con tus cambios. Aceptalos con **A** (eso escribe la máscara final).

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
| La locación del padrón que originó el par **ya se ve en T1** | En la revisión rápida, **T** (no lo corrijas en QGIS) |
| Dos puntos del padrón que son una sola locación (por ejemplo, la 720 y la 721) | Marcala igual y escribilo en `nota`, para el #15 |

## 4. Devolverlos

Subí, de cada par validado, `<id>_mask.tif` y `<id>_propuesta.gpkg`, y también `revision.csv` **renombrado a `revision_<persona>.csv`** (si no, las tres se pisan), a la carpeta `pares/` del bucket (https://console.cloud.google.com/storage/browser/vmseg-imagenes/pares, botón Upload files; reemplaza los anteriores). Conviene hacerlo de a tandas, al final de cada sesión. Sofi corre `python -m vmseg.generar_pares estado` para actualizar `pares/indice.csv` del bucket y ver cuántos lleva cada una.

## Cuando la locación ya se ve en T1

La revisión rápida agrega la fila sola con **T** o **P** en `data/correcciones_pares.csv`. Pasásela a Sofi (por PR o por mensaje). A mano es así:

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

Si cambia el generador de propuestas, `proponer --todos` y `subir --todos` (en el notebook, `TODOS = "--todos"`) rehacen y reemplazan todas las propuestas sin tocar las máscaras; antes de `bajar`, borrá las `_propuesta.gpkg` y `_dndvi.tif` locales para traer las nuevas.

Para regenerar pares corregidos: `recortar <id>`, `proponer <id>` (en la compu tarda ~2 min por par sin GPU) y `subir <id>`.

SAM necesita `segment-anything` y el checkpoint `sam_vit_h_4b8939.pth` en `~/.cache/sam/` (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
