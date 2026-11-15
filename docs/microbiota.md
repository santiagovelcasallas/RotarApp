# Módulo 5: Índice Biológico del Suelo (IBS)

Código: `src/rotarapp/microbiota/` · Pruebas: `tests/test_microbiota_*.py`

## Qué estima

El satélite no ve la microbiota. Ve los factores de manejo de los que depende: alimento
(residuo y carbono), perturbación (labranza, quema) y continuidad (raíz viva). El IBS los
resume en un número de 0 a 100 **relativo a la misma unidad de suelo** (50 = típico de ese
suelo), siguiendo a Bünemann et al. (2018, no hay valores objetivo universales) y el índice
de multifuncionalidad por promedio de z de Delgado-Baquerizo et al. (2016).

| Componente | Señal | Para la microbiota | Escala mínima |
|---|---|---|---|
| C1 Carbono | SOCI = azul/(rojo·verde), suelo expuesto (NDVI<0.25, NDTI<0.10) y seco | reserva de alimento | 0.3 |
| C2 Residuo | NDTI = (SWIR1−SWIR2)/(SWIR1+SWIR2) sin cultivo verde (NDVI<0.30) | sustrato fresco | 0.01 |
| C3 Perturbación | media anual del NDTI mínimo (Zheng et al. 2013) | labranza, quema | 0.01 |
| C4 Raíz viva | fracción del tiempo con NDVI>0.30 | exudados, continuidad | 0.05 |

z = (valor − mediana del suelo) / max(MAD, escala mínima) · puntaje = logística(1.5·z) ·
IBS = 100 × promedio de puntajes. C1 es opcional; C2–C4 son obligatorios.

## API (`rotarapp.microbiota`)

```python
from rotarapp.microbiota import Referencia, ibs_de_rotacion, resumen_escenarios, NDTI_TIPICO

ref = Referencia.desde_json(open("data/processed/microbiota/palmira_candelaria/referencia_suelos.json",
                                 encoding="utf-8").read())
# Por escenario climático: series simuladas a paso fijo sobre TODA la rotación
# (en caña, los 5-6 cortes), forma (T, ...). ndti_sim con NDTI_TIPICO según el estado del lote.
p = ibs_de_rotacion(ndvi_sim, ndti_sim, fechas, lluvia_3d_mm, grupo_suelo, ref)
# p.ibs (0-100), p.puntaje["C2"..."C4"], p.z[...], p.ibs_sin_c4
salud_suelo = resumen_escenarios(np.stack(ibs_por_escenario))  # {"p10", "p50", "p90"}
```

- `grupo_suelo`: id del grupo de suelo (clave de `referencia_suelos.json`; el nombre IGAC está en `nombre`).
- `NDTI_TIPICO`: suelo labrado 0.06, quema 0.05, rastrojo de soya 0.09, de maíz 0.15, hojarasca de caña 0.14, abono verde incorporado 0.08. Los de caña y suelo expuesto salen de las observaciones; los demás son supuestos a calibrar.

## Exportar el IBS observado de una zona

Hoy los datos se descargan y preparan con el prototipo `NASA_IBS` (scripts `s01`–`s02`); este módulo los lee:

```bash
pip install -e ".[geo,dev]"
python -m rotarapp.microbiota.exportar --prototipo ../NASA_IBS/data/valle --zona palmira_candelaria
```

| Salida | Contenido |
|---|---|
| `data/processed/microbiota/<zona>/ibs_9377_100m.tif` | Malla de RotarApp (EPSG:9377, 100 m). Bandas: IBS, IBS_sin_C4, s_C1…s_C4, confianza |
| `data/processed/microbiota/<zona>/referencia_suelos.json` | Por grupo de suelo: nombre IGAC, píxeles, mediana y escala de cada componente. **Es lo que usa el simulador** |
| `data/processed/microbiota/<zona>/resumen.json` | Periodo, escenas, cobertura, umbrales y advertencias |
| `data/interim/microbiota/<zona>/ibs_30m.tif` | Igual en la malla nativa UTM 30 m (no se versiona) |

Al pasar a 100 m se promedian los píxeles de 30 m con dato; por eso la cobertura a 100 m (86 %) es mayor que a 30 m (51 %).

## Resultados del prototipo

| Zona | Escenas | IBS calculable en cultivo | Validación |
|---|---|---|---|
| Iowa, condado Howard (Landsat 2016–2021) | 89 | 99 % | Ordena las rotaciones: pradera 98 > extendida 61 > maíz-soya 46 > maíz continuo 40 (1 070 lotes, comparados dentro del mismo suelo). Sin C4, extendida ≈ maíz-soya. SOCI vs carbono SSURGO: Spearman 0.80 por unidad de suelo |
| Valle, Palmira–Candelaria (NASA HLS 2020–2025) | 340 | 51 % | Sin validar: no hay datos públicos de quema/verde, renovación ni rotaciones |

El módulo reproduce exactamente el prototipo en ambas zonas (`tests/test_microbiota_equivalencia.py`).

## Vacíos y riesgos para revisión local

- **No mide microbiota** y no está calibrado con laboratorio: los pesos son iguales.
- **C1 no aplica en caña:** el suelo casi nunca se ve (mediana de 2 observaciones sin dosel en 6 años, y cubierto de hojarasca). En el Valle el carbono debe venir del estudio de suelos.
- **Nubes:** en el 49 % del cultivo de Palmira–Candelaria el satélite nunca vio el suelo sin dosel.
- **Evaluar la rotación en su ciclo completo:** la renovación de caña con leguminosa deja el suelo expuesto unos meses y el índice la castiga si se mira como foto.
- **50 no es "bueno":** es lo típico de ese suelo en la zona. No sirve para comparar suelos distintos entre sí.
