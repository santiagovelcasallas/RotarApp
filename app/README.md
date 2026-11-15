# app/

La interfaz vive en **Lovable** (repositorio propio vía sincronización con GitHub). La app no ejecuta la simulación: solo lee resultados precalculados en `data/processed/`.

## Contrato de datos (borrador, por definir con el equipo)

Por cada rotación y píxel (o lote de ejemplo):

- `rotacion_id`, `nombre`
- `puntajes`: `agua`, `erosion`, `rentabilidad`, `salud_suelo`, cada uno con `p10`, `p50`, `p90`
- `riego_estimado_mm`
- `explicacion`: `por_que`, `evidencia`, `beneficios`, `vacios`, `riesgos`
