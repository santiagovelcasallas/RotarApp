# RotarApp

Herramienta de apoyo a decisiones para **rotación de cultivos agroecológica** (cero agroquímicos, mínimo riego artificial) en la **zona plana del Valle del Cauca**, usando observaciones de la Tierra de NASA, información local de suelos, características de cultivos y prioridades del agricultor.

Reto NASA Space Apps: *Crop Rotation Decision-Support Tool*.

> RotarApp **apoya, no reemplaza**, el criterio del agricultor ni la asesoría agronómica local. Cada resultado indica su incertidumbre y qué debe validarse en campo.

## Qué hace

Compara **al menos 3 rotaciones, incluida la actual**, en cada píxel del área de estudio. Cada rotación recibe 4 puntajes:

| Puntaje | Pregunta que responde |
|---|---|
| Agua y clima | ¿Cuánto déficit hídrico enfrenta y cuánto riego necesitaría? |
| Erosión | ¿Cuánto suelo se pierde (RUSLE)? |
| Rentabilidad | ¿Qué ganancia se espera, en un año típico y en uno malo? |
| Salud biológica del suelo | ¿Cómo evoluciona la microbiota del suelo (índice estimado)? |

Para cada estrategia, la app explica: **por qué se sugiere, qué evidencia la respalda, qué beneficios ofrece, dónde hay vacíos de información y qué riesgos requieren revisión local.**

## Cómo funciona (pipeline)

```
Datos NASA + locales ──► 1. Filtro de restricciones por píxel (pendiente, drenaje, clima, suelo)
                              │  → cultivos viables por píxel
                              ▼
                         2. Simulador por píxel (autómata celular, paso de ~10 días)
                              estado: agua del suelo · vigor (NDVI) · inóculo de enfermedades · salud del suelo
                              corrido sobre muchos años climáticos (Monte Carlo por fase ENSO)
                              │
                              ├─► 3. Erosión (RUSLE)
                              ├─► 4. Rentabilidad (NDVI → rendimiento × precio − costos)
                              └─► 5. Índice de microbiota del suelo
                              ▼
                         6. Comparador de rotaciones, ponderado por las prioridades del agricultor
                              ▼
                         Resultados precalculados (JSON / GeoTIFF) ──► App (Lovable)
```

Detalle en [docs/arquitectura.md](docs/arquitectura.md). Datasets en [docs/datos.md](docs/datos.md). Rotaciones candidatas en [docs/rotaciones.md](docs/rotaciones.md).

## Estructura del repo

```
RotarApp/
├── config/              # Área de estudio y parámetros de cultivos
├── data/                # Datos (raw/ e interim/ no se versionan)
├── docs/                # Arquitectura, datasets, rotaciones
├── notebooks/           # Exploración y prototipos
├── src/rotarapp/
│   ├── datos/           # Descarga y armonización a rejilla común
│   ├── filtro/          # 1. Restricciones por píxel
│   ├── simulador/       # 2. Autómata celular (agua, vigor, enfermedad, suelo)
│   ├── erosion/         # 3. RUSLE
│   ├── economia/        # 4. NDVI → rendimiento → ganancia
│   ├── microbiota/      # 5. Índice de salud biológica del suelo
│   └── comparador/      # 6. Puntajes, ponderación y explicaciones
├── app/                 # Contrato de datos con la app de Lovable
└── tests/
```

## Equipo

| Nombre | Rol |
|---|---|
| _por completar_ | |
| _por completar_ | |
| _por completar_ | |
| _por completar_ | |
| _por completar_ | |
| _por completar_ | |
