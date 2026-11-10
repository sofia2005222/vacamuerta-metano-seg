# Alerta temprana de actividad en Vaca Muerta

Detectar en imágenes satelitales las locaciones petroleras nuevas de Vaca Muerta antes de que empiece la perforación, y medir con datos oficiales cuánta anticipación da el satélite.

## Infraestructura

**Pozo**:
Perforación individual registrada en el padrón de pozos, ubicada por un único punto (su boca de pozo).
_Avoid_: sondeo, well

**Boca de pozo**:
Punto en superficie donde está el pozo; es la coordenada que da el padrón.
_Avoid_: cabeza de pozo, ubicación del pozo

**Locación**:
Superficie de tierra desmontada por la operadora para alojar uno o más pozos, terminada o todavía en obra, de cualquier formación y esté o no en nuestro dataset; es lo que se ve desde el satélite y una de las clases a segmentar.
_Avoid_: pad, well pad, playón, locación de producción, sitio

**Pileta**:
Cualquier fosa o cuerpo de agua excavado por la actividad petrolera, esté dentro o fuera de una locación; donde hay pileta, la clase es pileta y no locación.
_Avoid_: pit, pond, tanque

**Operadora**:
Empresa titular de la concesión, que perfora y opera los pozos.
_Avoid_: empresa, compañía, cliente

**Área de concesión**:
Superficie otorgada a una operadora para explotar hidrocarburos.
_Avoid_: bloque (en este proyecto "bloque" es otra cosa), yacimiento

## Tiempo y backtesting

**Inicio de perforación**:
Fecha en que empieza a perforarse un pozo; el inicio de perforación del primer pozo de una locación es la fecha de referencia del backtesting.
_Avoid_: spud date, fecha del pozo

**Primera producción**:
Primer mes con producción declarada de un pozo en el padrón; no se usa como referencia temporal.
_Avoid_: puesta en producción, fecha de producción

**Locación nueva**:
Locación de Vaca Muerta cuyo primer pozo empezó a perforarse entre julio de 2019 y diciembre de 2023; las de `test_anelo` y `test_externo` son los casos del backtesting. Empieza en julio para que haya al menos 6 meses de imágenes antes de la perforación.
_Avoid_: locación reciente, pad nuevo

**Locación preexistente**:
Locación que el padrón marca como nueva pero que ya estaba desmontada en enero de 2019, el primer mes de las imágenes; sale del dataset y del backtesting.
_Avoid_: locación vieja, falsa nueva

**Locación ampliada**:
Locación a la que se agregaron pozos más de un año después del primero; sigue siendo una sola locación y su fecha de referencia es la de su primer pozo.
_Avoid_: reampliación, re-entry

**Objeto predicho**:
En el backtesting, cada componente conexo de los píxeles que el modelo predice como locación o pileta (nuevas en el par), tomadas juntas.
_Avoid_: detección, blob

**Locación capturada**:
Locación del padrón cuyo centro queda adentro de un objeto predicho, contando un margen de 50 m alrededor del objeto.
_Avoid_: acierto, match, locación encontrada

**Detección sin registro**:
Objeto predicho que no contiene ninguna boca de pozo del padrón completo (cualquier formación, cualquier fecha, incluidos los pozos posteriores a 2023), contando el mismo margen de 50 m.
_Avoid_: detección nueva (se confunde con locación nueva), falsa alarma

**Locación sin registrar**:
Detección sin registro que la revisión manual confirma como una locación real.
_Avoid_: locación no declarada, locación ilegal

**Primera detección**:
El primero de dos meses seguidos con imagen válida en que una locación queda capturada; los meses sin imagen válida no cuentan ni cortan la racha.
_Avoid_: aparición, alerta

**Anticipación**:
Meses entre la primera detección de una locación y el inicio de perforación de su primer pozo; es negativa si la detección llega después.
_Avoid_: lead time, adelanto, ventaja

**Locación detectada a tiempo**:
Locación nueva cuya primera detección cae en el mes del inicio de perforación o antes (anticipación ≥ 0).
_Avoid_: detección temprana, alerta exitosa

**Backtesting**:
Correr el modelo sobre imágenes mensuales históricas (2019–2023) para medir, sobre las locaciones nuevas de los dos test, cuántas se detectan y con cuánta anticipación.
_Avoid_: validación histórica, simulación

## Datos y evaluación

**Padrón de pozos**:
Registro oficial de pozos de la Secretaría de Energía (Capítulo IV), con coordenadas y fechas, de todas las formaciones; las locaciones de Vaca Muerta son un subconjunto.
_Avoid_: registro oficial, shapefile de pozos, Capítulo IV

**Imagen mensual**:
Mediana de los píxeles despejados de Sentinel-2 de un mes sobre una zona de estudio.
_Avoid_: compuesto, mosaico, escena

**Encuadre**:
Cuadrado fijo de 256×256 px (2,56 km) sobre el terreno, contenido entero en un bloque; no tiene fecha.
_Avoid_: ventana, lugar, celda

**Recorte**:
Un encuadre en una imagen mensual.
_Avoid_: parche, patch, tile, chip

**Par**:
Un encuadre en dos imágenes mensuales, una antes (T1) y otra después (T2); es la unidad de etiquetado y de entrada al modelo.
_Avoid_: pareja, imagen bitemporal, par de imágenes

**Máscara**:
Etiqueta validada por una persona que asigna a cada píxel de un par una clase según lo que apareció entre T1 y T2: fondo, locación o pileta; lo que ya estaba en T1 es fondo.
_Avoid_: anotación, ground truth, etiqueta débil

**Máscara propuesta**:
Máscara generada automáticamente (con SAM sobre T2) que todavía no validó ninguna persona.
_Avoid_: pre-etiqueta, máscara automática, borrador

**Zona Añelo**:
Zona de estudio alrededor de Añelo donde se entrena, valida y hace el test dentro de la misma zona.
_Avoid_: AOI, zona de entrenamiento, núcleo

**Zona externa**:
Zona de estudio al oeste (Fortín de Piedra, Aguada Pichana Este, La Calera), con otras operadoras y otro paisaje, reservada para medir generalización.
_Avoid_: zona oeste, zona de test

**Bloque**:
Celda de la grilla en que se divide una zona de estudio (4×4 en la Zona Añelo); cada bloque entero pertenece a un solo split y a una sola persona que lo etiqueta. En la Zona externa todos los bloques son `test_externo`.
_Avoid_: celda, área (y no confundir con área de concesión)

**Split**:
Conjunto al que pertenece un par o una locación: `train`, `val`, `test_anelo` o `test_externo`.
_Avoid_: partición, fold
