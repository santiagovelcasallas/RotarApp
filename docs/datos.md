# Datos

## NASA

| Dataset | Variable | Uso en RotarApp |
|---|---|---|
| NASA POWER | Temperatura, radiación, humedad, viento | Clima diario histórico para el simulador |
| GPM IMERG | Precipitación | Lluvia del simulador, erosividad R (RUSLE) |
| HLS (Landsat + Sentinel-2) | NDVI | Curvas de vigor, factor C, calibración de rendimiento |
| MODIS | NDVI, evapotranspiración | Respaldo con más pasadas para series largas |
| ECOSTRESS | Evapotranspiración, temperatura de superficie | Balance hídrico y estrés por calor |
| SMAP | Humedad del suelo | Contexto regional del balance hídrico |
| SRTM / NASADEM | Elevación | Pendiente, factor LS, filtro de restricciones |
| GEOS-S2S | Pronóstico estacional | Perspectiva de la próxima temporada |

## Otros

| Fuente | Uso |
|---|---|
| Capacidad de uso de las tierras (IGAC 2006, 1:100.000) | Filtro de cultivos viables por finca (`data/referencia/igac/`) |
| Mapa de suelos (IGAC / SoilGrids) | Textura, drenaje, factor K, entradas de microbiota |
| Índice ENSO (ONI) | Agrupar años climáticos para el Monte Carlo |
| Evaluaciones agropecuarias municipales | Rendimientos para calibrar NDVI → rendimiento |
| Precios locales de mercado | Rentabilidad |
| Papers de microbiota del suelo | Relaciones para el índice de salud biológica |
| WeatherNext (Google DeepMind) | Pronóstico de corto plazo para ventana de siembra (opcional) |
| Prioridades del agricultor | Pesos del comparador, prácticas (factor P), restricciones |

## Reglas

- Lo crudo va en `data/raw/` y lo intermedio en `data/interim/`; ninguno se versiona.
- Cada descarga queda documentada en `src/rotarapp/datos/` (fuente, fechas, resolución).
- Todo se lleva a la rejilla común definida en `config/zona.toml`.
