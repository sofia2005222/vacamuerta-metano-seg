---
status: accepted
---

# Las etiquetas son máscaras sobre la imagen, propuestas por SAM y validadas a mano

Las etiquetas del modelo son máscaras sobre la imagen, en **todos** los splits. Las arma SAM y las personas solo las validan y corrigen: para cada locación del padrón, su centro sirve como clic de SAM, que propone el contorno (máscara propuesta). Los puntos del padrón solo indican **dónde buscar**, nunca son la etiqueta. Es un cambio respecto de la versión 1, que entrenaba con etiquetas débiles (un disco alrededor de cada boca de pozo). Con etiquetas débiles no se puede aprender la clase pileta, porque el padrón no tiene piletas. Meta inicial: 200–400 recortes de 256×256 px; hay que consultar el número con el profesor.

## Opciones consideradas

- **Plan de respaldo:** si validar una máscara tarda en promedio más de 10 minutos por recorte (medido con 10 recortes de prueba), en `train` se usan etiquetas débiles y las máscaras quedan solo para `val` y los dos test. En ese caso se da de baja la clase pileta, porque el padrón no tiene piletas para entrenarla.

## Consecuencias

- El tiempo de etiquetado es el riesgo principal. Medirlo con 10 recortes antes de planificar.
