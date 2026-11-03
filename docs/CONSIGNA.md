# Requisitos de la cátedra (resumen del PDF de la consigna)

- Problema acotado, con objetivos e hipótesis, implementable en menos del 30% del tiempo. No sobredimensionar.
- Estado del arte **con referato** (no Wikipedia, Medium, blogs; cuidado con arXiv sin publicar).
- Dataset con anotaciones que permitan métricas; split sugerido 70/10/20.
- **Obligatorio implementar una red neuronal** en Python, preferentemente PyTorch.
- Probar: entrenamiento desde cero, extracción de características (encoder congelado) y fine-tuning.
- Curvas de entrenamiento (loss por iteración, métrica en train y val por época).
- Evaluación: matrices de confusión, ejemplos correctos/incorrectos, casos patológicos, hipótesis sobre el comportamiento del modelo.
- Ciclo sugerido: problema → estado del arte → dataset → modelo → evaluación → despliegue.
- El tema debe ser aprobado por el profesor.
