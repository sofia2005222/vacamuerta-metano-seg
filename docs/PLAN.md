# Plan de trabajo (5 semanas, grupo de 3 o 4)

Las fechas reales dependen de la entrega y de la jornada de pósters: completar acá cuando la cátedra las anuncie.

| Semana | Objetivo | Entregable de la semana |
|---|---|---|
| 1 | Aprobación del tema y datos en mano | Mail al profesor enviado (`PROPUESTA_PROFESOR.md`); `pytest` verde; AOI descargado; shapefile de pozos recortado; radio calibrado |
| 2 | Baseline + empezar el etiquetado manual | `build-patches` corrido; corrida A entrenada; 50 % de los parches manuales |
| 3 | Etiquetado terminado + ablaciones | 100–150 parches manuales; corridas A–E con 3 semillas; acuerdo entre etiquetadoras |
| 4 | Análisis y caso de estudio | Tabla de resultados; análisis de errores (falsos positivos / negativos con imágenes); inferencia sobre Sierra Barrosa y Añelo; mapas |
| 5 | Informe y póster | Informe IEEE ≤ 8 páginas; póster A0/A1 vertical; ensayo de la presentación oral de ~5 min |

## Reparto sugerido

| Rol | Tareas |
|---|---|
| Datos | `fetch-s2`, `prepare-wells`, calibración del radio, `build-patches`, `summary.json` |
| Modelo | Entrenamientos y ablaciones (Colab/Kaggle), semillas, curvas de entrenamiento |
| Evaluación | Coordinar el etiquetado manual (QGIS), acuerdo entre etiquetadoras, métricas, análisis de errores |
| Comunicación | Informe, póster, caso de estudio, ensayo oral |

Todas colaboran en el etiquetado manual.

## Qué decidir juntas cuanto antes

1. ¿Qué AOIs usar además de `sierra_barrosa`? (más variedad = mejor generalización, más trabajo de etiquetado).
2. Convenciones de etiquetado (`LABELING.md`, sección 1).
3. Si se intenta una extensión (SAM o Alberta) o se profundiza en lo que ya está. **Recomendación:** hacer bien A–E antes de pensar en F.

## Si falta tiempo: recorte en orden

1. Dejar F (Alberta) y SAM afuera.
2. Reducir a un solo AOI.
3. Bajar de 3 a 2 semillas.
4. Mantener siempre: conjunto manual de test, corridas A y B, caso Sierra Barrosa.
