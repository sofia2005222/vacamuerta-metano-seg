# Equipo y forma de trabajo

## Reparto

Cada persona es dueña de una etapa del pipeline. El etiquetado se reparte en partes iguales **por bloques**, así nadie toca los recortes de otra. Informe y póster: cada una escribe la parte de su etapa.

| Persona | GitHub | Etapa propia |
|---|---|---|
| Sofi | `sofia2005222` | Imágenes y recortes: descarga desde GEE, mediana mensual, recortes por split, correcciones del dataset |
| Agus | `agusbarrena` | Modelo: U-Net, 3 vs. 6 bandas (H1), desde cero / encoder congelado / fine-tuning (H2), curvas de entrenamiento |
| Bian | `biancapaz` | Evaluación y backtesting: métricas por píxel y por objeto, matriz de confusión, test externo (H3), backtesting, revisión de detecciones sin registro |

La carga tiene que quedar pareja. Si una etapa queda más liviana, Sofi toma lo que sobre.

El primer issue de código define el **formato de un recorte** (archivo, bandas, máscara). Con ese formato fijo, Agus y Bian pueden arrancar con recortes de prueba sin esperar las imágenes reales.

## Herramientas

- **Código:** archivos `.py` en el repo, con pruebas, editados en VS Code. Todo pasa por GitHub (issues y pull requests).
- **Notebooks:** solo para correr y mostrar. Clonan el repo y llaman a las funciones de los `.py`.
- **Cómputo:** lo que no necesita GPU (datos, etiquetado, evaluación) corre en una compu local (alcanzan 16 GB de RAM). El entrenamiento corre en Colab con GPU: cada una usa su cuenta gratis y, si el límite de horas molesta, se pasa a Colab Enterprise pagado con créditos.
- **Almacenamiento:** las imágenes mensuales (~14 GB) van a un bucket de Google Cloud Storage en el proyecto de Sofi; GEE exporta directo ahí y las tres tienen acceso de lectura.
- **Créditos:** USD 50 de Google Cloud por persona (USD 150 en total). El bucket cuesta centavos por mes; el resto queda para Colab Enterprise si hace falta.
