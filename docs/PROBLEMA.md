# El problema (enfoque comercial)

> Entrenar una red neuronal que **segmente locaciones petroleras y piletas en imágenes satelitales Sentinel-2** de Vaca Muerta y usarla para **detectar locaciones nuevas antes de que empiece la perforación**, midiendo con datos oficiales cuántos meses de anticipación da el satélite.


- Vaca Muerta es una formación geológica de Neuquén de la que se extrae petróleo y gas no convencional. Pasó de perforar 10–15 pozos por mes en 2018–2019 a más de 50 por mes en 2026.
- Antes de perforar, la operadora desmonta un rectángulo de tierra (la **locación** o *pad*), que suele alojar varios pozos (en esta zona, la mediana es 4). Es visible desde el satélite.
- **La locación aparece semanas o meses antes de la perforación.** Detectarla temprano es **inteligencia de mercado**: le sirve a cada empresa saber qué está haciendo *la competencia* (cada una ya conoce su propia actividad).
- Quiénes la usarían: empresas de servicios (fractura, perforación, arena, agua, transporte) para anticipar demanda y clientes; proveedores de arena y agua para planificar logística; operadoras para mirar a sus vecinas y valuar bloques; inversores y analistas para anticipar producción (los pozos de hoy impactan en la producción 6–12 meses después); midstream para planificar capacidad de transporte.
- El registro oficial ubica los pozos como **puntos** y con demora; no mide la huella en superficie ni anticipa actividad.

**Referente comercial:** Sourcewater (Houston) vende este servicio en la cuenca Permian (EE. UU.) con su producto *DirtWork Alert*: escanea la cuenca cada 3–5 días y detecta locaciones nuevas. Reportan que las locaciones se desmontan típicamente 1–4 meses antes de que llegue el equipo de perforación, y que en 1 de cada 3 pozos la locación se veía antes de cualquier registro oficial. (Son reportes de la empresa, sin referato: se usan como **motivación**, no como bibliografía.)

**Aplicación secundaria** (mencionar en una línea en la presentación): el mismo modelo sirve para cumplimiento ambiental (inventario de huella, verificar restauración de locaciones abandonadas).

