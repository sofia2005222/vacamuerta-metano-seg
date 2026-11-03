---
status: accepted
---

# La fecha de referencia es el inicio de perforación, no la primera producción

El backtesting compara la primera detección con el **inicio de perforación** del primer pozo de la locación (`fecha_primer_pozo`), no con la **primera producción** del padrón. La primera producción muchas veces aparece antes de que termine la terminación del pozo, lo cual no tiene sentido físico: probablemente es el primer mes declarado en el sistema. A quien lea el dataset le parecería natural usar la producción, por eso queda registrado.
