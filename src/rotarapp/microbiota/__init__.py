"""5. Índice Biológico del Suelo (IBS): estimación de condiciones para la vida microbiana.

- `indice`: núcleo de dominio (solo numpy). Calcula los componentes C1-C4 desde series
  observadas o simuladas, la referencia por grupo de suelo y el puntaje 0-100.
- `exportar`: lleva el IBS observado de una zona a la malla de RotarApp (requiere `[geo]`).

Uso desde el simulador: `ibs_de_rotacion(ndvi_sim, ndti_sim, fechas, lluvia_3d, grupo, ref)`
con `ref = Referencia.desde_json(...)` de `data/processed/microbiota/<zona>/referencia_suelos.json`.
"""
from rotarapp.microbiota.indice import (COMPONENTES, NDTI_TIPICO, UMBRALES_IOWA, UMBRALES_VALLE,
                                        Componentes, Puntaje, Referencia, Umbrales,
                                        componentes_desde_serie, confianza, ibs_de_rotacion,
                                        lluvia_3_dias, puntuar, referencia_por_suelo,
                                        resumen_escenarios)

__all__ = ["COMPONENTES", "NDTI_TIPICO", "UMBRALES_IOWA", "UMBRALES_VALLE", "Componentes",
           "Puntaje", "Referencia", "Umbrales", "componentes_desde_serie", "confianza",
           "ibs_de_rotacion", "lluvia_3_dias", "puntuar", "referencia_por_suelo",
           "resumen_escenarios"]
