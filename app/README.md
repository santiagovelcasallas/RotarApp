# app/

La interfaz vive en **Lovable**. La app no simula: envía la finca al modelo y muestra el resultado.

## Entrada (lo que envía la app)

```json
{
  "nombre": "Finca La Esperanza",
  "municipio": "Palmira",
  "poligono": {"type": "Polygon", "coordinates": [[[lon, lat], ...]]},
  "cultivo_anterior": "cana_azucar",
  "plagas_anteriores": ["Barrenadores del tallo"],
  "prioridades": {"ingreso": 0.25, "agua": 0.25, "suelo": 0.25, "erosion": 0.25},
  "precios_cop_kg": {"cana_azucar": 150}
}
```

`cultivo_anterior`, `plagas_anteriores` y `precios_cop_kg` son opcionales. Los ids de cultivo y los
nombres de plaga son los de `data/referencia/`. Ejemplo: `data/ejemplos/finca_demo.json`.

```bash
python scripts/recomendar_finca.py data/ejemplos/finca_demo.json
```

## Salida (`data/processed/modelo/fincas/<nombre>/resultado.json`)

| Campo | Contenido |
|---|---|
| `area_ha`, `n_pixeles`, `clases_igac` | Finca rasterizada a 30 m |
| `estrategias.A/B/C` | Actual, mejor plan único, por zonas: `planes`, `nombres`, `metricas` (p10/p50/p90 de ingreso COP/ha/año, riego mm/año, ibs, salud, erosión t/ha/año, utilidad) y `explicacion` (`por_que`, `evidencia`, `beneficios`, `vacios`, `riesgos`) |
| `estrategias.C.p_mejor_que_B` | Probabilidad de que la rotación por zonas supere al mejor plan único |
| `zonas` | Plan, área y probabilidad media de cada zona |
| `pixeles` | `lon`, `lat`, `plan` asignado y `p_mejor` de los 5 planes principales: **el mapa de calor** |
| `ranking_planes` | Los 5 planes más probables con sus medianas y rendimientos |
| `planes_sin_precio` | Planes que no compiten por ingreso (p. ej. renovación de caña) con sus métricas de agua, suelo y erosión |
| `modelo` | Muestras, escenarios, métricas de validación y el aviso de que no reemplaza al agrónomo |
