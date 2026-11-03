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
Superficie de tierra desmontada por la operadora para alojar uno o más pozos, terminada o todavía en obra; es lo que se ve desde el satélite y una de las clases a segmentar.
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
Locación cuyo primer pozo empezó a perforarse entre 2019 y 2023; son los casos del backtesting.
_Avoid_: locación reciente, pad nuevo

**Locación ampliada**:
Locación a la que se agregaron pozos más de un año después del primero; sigue siendo una sola locación y su fecha de referencia es la de su primer pozo.
_Avoid_: reampliación, re-entry

**Locación capturada**:
Locación del padrón que queda adentro de un objeto predicho de clase locación, contando un margen de 50 m alrededor del objeto.
_Avoid_: acierto, match, locación encontrada

**Detección sin registro**:
Objeto predicho de clase locación que no captura ninguna locación del padrón.
_Avoid_: detección nueva (se confunde con locación nueva), falsa alarma

**Locación sin registrar**:
Detección sin registro que la revisión manual confirma como una locación real.
_Avoid_: locación no declarada, locación ilegal

**Primera detección**:
El primero de dos meses seguidos en que una locación queda capturada.
_Avoid_: aparición, alerta

**Anticipación**:
Meses entre la primera detección de una locación y el inicio de perforación de su primer pozo.
_Avoid_: lead time, adelanto, ventaja

**Backtesting**:
Correr el modelo sobre imágenes históricas (2019–2023) para medir sobre locaciones nuevas cuántas se detectan y con cuánta anticipación.
_Avoid_: validación histórica, simulación

## Datos y evaluación

**Padrón de pozos**:
Registro oficial de pozos de la Secretaría de Energía (Capítulo IV), con coordenadas y fechas.
_Avoid_: registro oficial, shapefile de pozos, Capítulo IV

**Recorte**:
Porción cuadrada de una imagen satelital que es la unidad de etiquetado y de entrada al modelo.
_Avoid_: parche, patch, tile, chip

**Máscara**:
Etiqueta validada por una persona que asigna a cada píxel de un recorte una clase (fondo, locación o pileta).
_Avoid_: anotación, ground truth, etiqueta débil

**Máscara propuesta**:
Máscara generada automáticamente que todavía no validó ninguna persona.
_Avoid_: pre-etiqueta, máscara automática, borrador

**Zona Añelo**:
Zona de estudio alrededor de Añelo donde se entrena, valida y hace el test dentro de la misma zona.
_Avoid_: AOI, zona de entrenamiento, núcleo

**Zona externa**:
Zona de estudio al oeste (Fortín de Piedra, Aguada Pichana Este, La Calera), con otras operadoras y otro paisaje, reservada para medir generalización.
_Avoid_: zona oeste, zona de test

**Bloque**:
Celda de la grilla de 4×4 en que se divide la Zona Añelo; cada bloque entero pertenece a un solo split.
_Avoid_: celda, área (y no confundir con área de concesión)

**Split**:
Conjunto al que pertenece un recorte o una locación: `train`, `val`, `test_anelo` o `test_externo`.
_Avoid_: partición, fold
