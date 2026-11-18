# Dataset de pozos y locaciones (`data/`)

**Fuente:** Secretaría de Energía de la Nación, "Producción de petróleo y gas por pozo (Capítulo IV)", datos.gob.ar, descargado en noviembre de 2026. Licencia CC-BY-4.0 (citar la fuente). Archivos de entrada: `capitulo-iv-pozos.csv` y `padrn-de-pozos-de-captulo-iv-con-fecha-de-primera-produccin.csv`.

**Qué hace `armar_dataset.py`:**
1. Filtra pozos de formación Vaca Muerta con recurso no convencional: **3.453 pozos**.
2. Extrae coordenadas (EPSG:4326, boca de pozo).
3. Cruza con la fecha de primera producción por `idpozo`.
4. Agrupa en **locaciones** los pozos a menos de 60 m entre sí (en un pad las bocas están a ~10 m): **1.204 locaciones**.
5. Asigna zonas, bloques y split.

**Archivos:**

| Archivo | Contenido |
|---|---|
| `locaciones_vaca_muerta.csv` / `.geojson` | Una fila por locación. **Archivo principal** |
| `pozos_vaca_muerta.csv` / `.geojson` | Una fila por pozo (se une a locaciones por `locacion_id`) |
| `zonas_estudio.json` | Recuadros de las zonas, grilla y split de cada bloque |
| `pozos_todos.csv` | Todos los pozos del padrón (cualquier formación y fecha) en las dos zonas, con ~1 km de margen: **2.482 pozos**. Para elegir negativos y clasificar detecciones sin registro |
| `armar_dataset.py` | Regenera todo (zonas y semilla configurables al principio) |
| `armar_pozos_todos.py` | Regenera `pozos_todos.csv`; solo necesita `capitulo-iv-pozos.csv` (va aparte para no regenerar las locaciones con otra descarga) |

Los `.geojson` se abren directo en QGIS o Google Earth Engine.

**Zonas:**
- Añelo (train/val/test): longitud -68.85 a -68.50, latitud -38.42 a -38.12 (~30 × 33 km). Loma Campana, Bandurria Sur, La Amarga Chica, Bajada del Palo Oeste, Cruz de Lorena. Elegida por ser el núcleo con más densidad y actividad 2019–2023.
- Test externo: longitud -69.25 a -68.95, latitud -38.60 a -38.30.

**Columnas de `locaciones_vaca_muerta.csv`:**

| Columna | Significado |
|---|---|
| `locacion_id` | Identificador de la locación |
| `lon`, `lat` | Centro de la locación |
| `n_pozos` | Cantidad de pozos |
| `fecha_primer_pozo` | Inicio de perforación del primer pozo. **Fecha de referencia del backtesting** |
| `fecha_ultimo_pozo` | Inicio de perforación del último pozo |
| `primera_produccion` | Primer mes con producción declarada (ver advertencias) |
| `anio_primer_pozo` | Año del primer pozo |
| `nueva_2019_2023` | `True` si el primer pozo es de 2019–2023. **Pendiente:** la locación nueva ahora arranca en julio de 2019 (ver `GLOSSARY.md`); hay que ajustar la columna en `armar_dataset.py` |
| `ampliada_luego` | `True` si se agregaron pozos más de un año después del primero: la locación ya existía en la fecha del pozo agregado |
| `en_zona_anelo`, `en_zona_test_externo` | Pertenencia a cada zona |
| `bloque_anelo` | Bloque de la grilla 4×4 (solo zona Añelo) |
| `split` | `train`, `val`, `test_anelo`, `test_externo` o vacío (fuera de ambas zonas) |
| `empresa`, `area`, `siglas` | Operadora, área de concesión y siglas de los pozos |

**Advertencias:**
- **Usar `fecha_primer_pozo` (inicio de perforación), no `primera_produccion`.** La primera producción del padrón muchas veces aparece antes del fin de la terminación del pozo, lo cual no tiene sentido físico; probablemente sea el primer mes declarado en el sistema.
- Las coordenadas son de boca de pozo; la locación desmontada es más grande (en EE. UU. miden 50–250 m, o sea 5–25 píxeles en Sentinel-2).
- La agrupación por distancia es aproximada: una locación con varias filas de pozos puede quedar partida en dos. Cuando eso se ve al validar máscaras, se anota en `data/correcciones_locaciones.csv` ("la locación X y la Y son la misma") y `armar_dataset.py` aplica las correcciones al regenerar.
- 35 pozos sin fecha de primera producción y algunos sin fecha de perforación.
- **Los puntos no son las etiquetas del modelo:** las máscaras se dibujan sobre la imagen.
- **Solo Vaca Muerta no convencional:** las locaciones de pozos convencionales o de otras formaciones no están en estos archivos, pero se ven igual desde el satélite y van en las máscaras. Para eso está `pozos_todos.csv`, con todas las bocas de pozo del padrón en las zonas (descargado el 18 de noviembre de 2026).
- **`pozos_todos.csv`:** un mismo pozo puede aparecer en varias filas, una por formación (misma `sigla` y coordenadas, distinto `idpozo`). La columna `vaca_muerta` usa el mismo criterio que `armar_dataset.py`. **157 pozos de la Zona externa no tienen fecha de inicio de perforación** (todos de la formación Mulichinco, casi todos activos): como no se puede saber si se perforaron durante 2019–2023, los encuadres que tienen alguno (con 150 m de margen, 54 de 192) no se usan para pares negativos.

