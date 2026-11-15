# Arquitectura

## Principios

- **Una sola pregunta:** qué rotación conviene, sin agroquímicos y con el menor riego posible.
- **Una sola zona:** zona plana del Valle del Cauca (ver `config/zona.toml`).
- **Explicable antes que perfecto:** reglas simples por píxel y un modelo de IA solo donde aporta (NDVI → rendimiento).
- **La incertidumbre es un resultado:** cada puntaje es una distribución sobre muchos años climáticos, no un número único.

## Rejilla común

Todos los datos se llevan a una rejilla de **100 m** (configurable). Los datasets gruesos (SMAP, NASA POWER, GPM) se interpolan o se usan como contexto; los finos (HLS a 30 m) se agregan.

## Módulos

### 1. Filtro de restricciones (`filtro/`)
Por píxel: pendiente (SRTM), drenaje y textura (mapa de suelos), grados-día, rango de temperatura y lluvia. Resultado: qué cultivos son viables en cada píxel.

### 2. Simulador por píxel (`simulador/`)
Un autómata celular con paso de ~10 días. Estado de cada celda:

| Variable | Sube con | Baja con |
|---|---|---|
| Agua del suelo | Lluvia, riego | Evapotranspiración, drenaje |
| Vigor (NDVI) | Curva esperada del cultivo y su etapa | Estrés hídrico, enfermedad |
| Inóculo por patógeno | Cultivo hospedero + clima favorable + vecinos | Cultivo no hospedero, tiempo |
| Salud del suelo | Diversidad de la rotación, cobertura, residuos | Suelo desnudo, monocultivo |

La única interacción espacial es la **difusión del inóculo a los vecinos**. Así se ve por qué la rotación rompe ciclos de plagas.

**Clima:** se remuestrean años históricos (NASA POWER, GPM) agrupados por fase ENSO, para obtener muchos escenarios. WeatherNext se usa como complemento operativo de corto plazo (ventana de siembra), no para la simulación de temporadas.

### 3. Erosión (`erosion/`)
RUSLE: `A = R · K · LS · C · P`.
- R: erosividad de la lluvia (GPM IMERG).
- K: erodabilidad del suelo (mapa de suelos).
- LS: pendiente y longitud (SRTM / NASADEM).
- C: cobertura, derivada del NDVI simulado. Aquí la rotación cambia el resultado.
- P: prácticas de conservación (entrada del agricultor).

En la zona plana el factor LS es bajo; el resultado se reporta así, sin inflarlo.

### 4. Rentabilidad (`economia/`)
- Modelo de IA: curva de NDVI de la temporada → rendimiento estimado, calibrado con rendimientos municipales conocidos.
- `rendimiento = potencial × factor_estrés_hídrico × (1 − pérdida_por_enfermedad)`
- `ganancia = rendimiento × precio_local − costos`

### 5. Índice de microbiota (`microbiota/`)
Estima indicadores biológicos medibles (por ejemplo, carbono de biomasa microbiana, respiración, relación hongos/bacterias, micorrizas) a partir de proxies: suelo, clima, humedad, historial de rotación y vegetación. Debe responder a la rotación, porque esa es la base de la comparación. Se presenta siempre como **estimación con incertidumbre** e indica dónde conviene tomar una muestra de laboratorio.

### 6. Comparador (`comparador/`)
Combina los 4 puntajes con pesos que define el agricultor (agua, ingreso, suelo, erosión). Genera para cada rotación las 5 explicaciones que exige el reto: por qué se sugiere, evidencia, beneficios, vacíos de información y riesgos.

## Salida hacia la app

Resultados precalculados por píxel y por rotación (GeoTIFF / JSON). La app de Lovable solo lee y muestra; no ejecuta la simulación. El contrato está en `app/README.md`.

## Vacíos conocidos

- Nubosidad alta en el valle: huecos en series ópticas (mitigado con HLS e interpolación).
- SMAP (9–36 km) es demasiado grueso para el valle: se usa como contexto.
- Pocos datos de campo de microbiota y enfermedades en Colombia para calibrar.
- Los rendimientos históricos provienen de manejo convencional: posible sesgo al estimar manejo agroecológico.
