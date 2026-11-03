# Alerta temprana de actividad en Vaca Muerta con imágenes satelitales

Trabajo práctico de **I308 – Visión Artificial (Universidad de San Andrés)**, modalidad "propuesta de trabajo por estudiantes".

> **Nota para quien use un asistente de IA (Claude u otro):** antes de proponer cambios, leé `GLOSSARY.md` (vocabulario del proyecto), `docs/adr/` (decisiones ya tomadas y por qué) y `docs/PREGUNTAS_ABIERTAS.md` (lo que falta decidir). Las decisiones de los ADR ya se discutieron: no las reabras salvo que haya un motivo nuevo, y si las contradecís, decilo explícitamente. Usá los términos del glosario y no sus sinónimos marcados como _Avoid_.

## La idea en una frase

Entrenar una red neuronal que **segmente locaciones petroleras y piletas en imágenes satelitales Sentinel-2** de Vaca Muerta y usarla para **detectar locaciones nuevas antes de que empiece la perforación**, midiendo con datos oficiales cuántos meses de anticipación da el satélite.

## Documentación

| Documento | Contenido |
|---|---|
| [`GLOSSARY.md`](GLOSSARY.md) | Vocabulario del proyecto (locación, pileta, anticipación, bloque, split…) |
| [`docs/adr/`](docs/adr/) | Decisiones tomadas, una por archivo (tema, enfoque comercial, Sentinel-2, clases, splits, fecha de referencia, etiquetado, reescritura, Earth Engine) |
| [`docs/PREGUNTAS_ABIERTAS.md`](docs/PREGUNTAS_ABIERTAS.md) | Contradicciones y huecos pendientes; es el insumo para `/grill-with-docs` |
| [`docs/PROBLEMA.md`](docs/PROBLEMA.md) | El problema con enfoque empresa, quiénes lo usarían y el referente comercial |
| [`docs/CONSIGNA.md`](docs/CONSIGNA.md) | Requisitos de la cátedra |
| [`docs/ESTADO_DEL_ARTE.md`](docs/ESTADO_DEL_ARTE.md) | Bibliografía principal |
| [`docs/METODO.md`](docs/METODO.md) | Pipeline (imágenes → etiquetado → modelo → evaluación → backtesting) e hipótesis H1–H3 |
| [`docs/EVALUACION.md`](docs/EVALUACION.md) | División espacial por bloques, test externo y conteo por split |
| [`docs/DATASET.md`](docs/DATASET.md) | Dataset de pozos y locaciones de `data/`: fuente, columnas y advertencias |
| [`docs/RIESGOS_Y_PASOS.md`](docs/RIESGOS_Y_PASOS.md) | Riesgos y checklist de próximos pasos |
| [`docs/PLAN.md`](docs/PLAN.md) | Plan de 4 semanas por persona y orden de recorte si hay atraso |
| [`docs/EQUIPO.md`](docs/EQUIPO.md) | Reparto entre Sofi, Agus y Bian, herramientas y cómputo |

## Estado

**Todavía no hay código.** El código de la versión 1 (enfoque metano) se borró y el proyecto se reescribe de cero ([ADR-0009](docs/adr/0009-reescribir-de-cero.md)). Antes de programar, todo el trabajo se desglosa en issues de GitHub repartidos entre el equipo. La v1 se puede consultar en el tag `v1-metano`.

El dataset de pozos y locaciones ya está armado en `data/`. Para regenerarlo, ver `data/armar_dataset.py` y [`docs/DATASET.md`](docs/DATASET.md).

## Estructura

```
GLOSSARY.md     vocabulario del proyecto
docs/           adr/ · documentos temáticos
data/           pozos y locaciones de Vaca Muerta, zonas de estudio y script que los genera
```
