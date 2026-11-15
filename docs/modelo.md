# Modelo de RotarApp

Modelo **bayesiano híbrido**: un simulador por píxel con ecuaciones diferenciales (agronomía),
calibrado con imágenes NASA HLS, y una decisión bayesiana que entrega, por píxel, la
probabilidad de que cada plan de rotación sea el mejor.

## Por qué no un transformer ni un random forest

En Palmira–Candelaria (389.000 píxeles de cultivo, ~114 imágenes sin nube por píxel en 2020–2025)
**ningún píxel muestra más de 1,5 ciclos por año**: el satélite solo ve caña. No hay ejemplos de
soya, maíz, abonos verdes ni hortalizas, ni etiquetas de "la mejor rotación". Un clasificador
supervisado aprendería caña y no podría extrapolar a rotaciones que nunca vio. El simulador sí
puede, porque su estructura viene de la agronomía y sus parámetros de la tabla de cultivos.

## La ventana de suelo desnudo

El IBS se mide **cuando el lote está sin dosel**: después de la cosecha y antes de que el cultivo
siguiente cubra el suelo. El simulador la representa así:

- **Latencia** tras la siembra o la cosecha: décadas sin dosel antes de emerger o rebrotar. En caña
  se calibra con HLS (posterior p10–p90: 8–17 días hasta el rebrote de la soca); en cultivos
  sembrados, 5–15 días de emergencia (prior).
- **Barbecho** entre cosecha y siembra en las rotaciones: 40 días por defecto (`dias_barbecho`,
  ajustable). En caña, el lote pasa una mediana de 50 días (30–80) con baja cobertura tras la cosecha.
- **Estado del suelo en la ventana:** labrado, rastrojo del cultivo anterior u hojarasca/cobertura,
  según el manejo (NDTI). Ese estado es el que puntúa el IBS.
- **IBS sin C4:** solo C2 (residuo) y C3 (perturbación) medidos en las ventanas sin dosel, frente a
  lo normal del mismo suelo. La fracción del tiempo con planta verde (C4) no entra.

## Capas

| Capa | Código | Qué hace |
|---|---|---|
| Clima | `datos/power.py` | NASA POWER diario 2001–2025, corregido por altitud (celda de 0,5° a 2.112 m; lotes a 975 m), ET0 Hargreaves |
| Filtro | `filtro/aptitud.py` | Clase IGAC-CVC 2023 (3 plena, 4 restringida, 6+ no), temperatura, pH, altitud y textura de la tabla |
| Simulador | `simulador/dinamica.py` | Agua del suelo (FAO-56), latencia sin dosel y dosel logístico con f(T) y estrés hídrico, paso de 10 días |
| Calibración | `simulador/calibracion.py` | Prior uniforme → hipercubo latino (2.000) → verosimilitud atemperada → pesos de importancia |
| Evaluación | `simulador/evaluacion.py` | Clasificación del suelo sin dosel: AUC-ROC, AUC-PR, precisión, recall, F1, Brier |
| Rendimiento | `economia/rendimiento.py` | Potencial × biomasa relativa × pH × textura × nitrógeno (sin fertilizante) × (1 − plagas) |
| Erosión | `erosion/rusle.py` | RUSLE; C = dosel (NDVI) × subfactor de cobertura del suelo exp(−0,05 × % con residuo) |
| Suelo | `comparador/decision.py` | IBS en las ventanas de suelo desnudo |
| Decisión | `comparador/decision.py` | Utilidad con las prioridades del agricultor → P(plan es el mejor) por píxel → zonas → estrategias A, B, C |
| Explicaciones | `comparador/explicaciones.py` | Por qué, evidencia, beneficios, vacíos y riesgos de cada estrategia |

## Entrenamiento, validación y prueba

13.434 curvas de rebrote de caña (cosecha → cosecha) en 6.000 píxeles muestreados de HLS.

| Conjunto | Datos | Segmentos | Uso |
|---|---|---|---|
| Entrenamiento | 70 % de los bloques de ~2 km, inicio 2020–2023 | 7.252 | Posterior de parámetros |
| Validación | mismos bloques, 2024 | 2.395 | Ancho del intervalo y umbrales de clasificación |
| Prueba temporal | mismos bloques, 2025 | 1.337 | Evaluación final |
| Prueba espacial | 30 % de bloques nunca vistos | 4.515 | Evaluación final |
| **Prueba externa** | **Valle norte** (Roldanillo–La Unión–Toro), otra zona | 8.862 | Nunca usada para ajustar |

Verosimilitud atemperada a un dato efectivo por par bloque × año (224).

### Regresión del NDVI (RMSE; cobertura del intervalo del 80 %)

| Conjunto | Simulador | Climatología | Gradient boosting | Cobertura 80 % |
|---|---|---|---|---|
| Validación | 0,101 | 0,100 | 0,098 | 81 % |
| Prueba temporal | 0,114 | 0,112 | 0,111 | 75 % |
| Prueba espacial | 0,113 | 0,116 | 0,111 | 79 % |
| **Externa (Valle norte)** | **0,156** | 0,163 | 0,161 | 64 % |

### Clasificación: ¿lote con baja cobertura (NDVI < 0,45)?

Umbral de probabilidad elegido en validación (máximo F1) y aplicado igual en prueba.

| Conjunto | Prevalencia | AUC-ROC | AUC-PR | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|
| Validación | 15,6 % | 0,944 | 0,834 | 0,80 | 0,76 | 0,78 |
| Prueba temporal | 24,3 % | 0,929 | 0,846 | 0,84 | 0,76 | 0,80 |
| Prueba espacial | 17,3 % | 0,922 | 0,810 | 0,83 | 0,69 | 0,75 |
| Externa (Valle norte) | 19,2 % | 0,804 | 0,647 | 0,73 | 0,52 | 0,61 |

Sin dosel estricto (NDVI < 0,30, 3–6 % de las observaciones porque la hojarasca de caña mantiene
el NDVI cerca de 0,33): AUC-ROC 0,87–0,88 en prueba, precisión 0,23–0,28, recall 0,58–0,69, F1 0,33–0,40.

**Lectura:**
- En la zona de entrenamiento, simulador, climatología y gradient boosting quedan a la par: en
  caña, el momento del suelo desnudo depende casi solo de la edad del rebrote.
- En otra zona todos empeoran, pero el simulador generaliza mejor que el gradient boosting
  (NDVI 0,156 vs 0,161; F1 0,61 vs 0,53).
- Su intervalo del 80 % solo cubre el 64 % en Valle norte: fuera de Palmira es sobreconfiado y
  hay que recalibrar por zona.

La decisión "mejor plan" **no tiene métricas de clasificación**: nadie ha observado cuál era la
mejor rotación en esos lotes. Lo que se valida es el simulador que la alimenta.

## Decisión

- **Muestras:** 23 escenarios climáticos de 2 años (2001–2023, siembra el 21 de noviembre) × 4 muestras
  del posterior = 92.
- **Planes:** leguminosa de descanso → cultivo de venta con precio, sin descanso, y, si el lote viene
  de caña, renovación con leguminosa → caña.
- **Manejo:** con leguminosa de cobertura, siembra directa sobre la cobertura (NDTI 0,14); sin
  cobertura hay que labrar para controlar arvenses sin herbicidas.
- **Utilidad:** pesos del agricultor sobre ingreso bruto, ahorro de agua (800 mm = 0), IBS y erosión
  (10 t/ha/año = 0). En clase IGAC 4: × (1 − 0,3 × fracción del tiempo en cultivo de venta).
- **Salida:** P(plan es el mejor) por píxel, máximo 3 zonas de ≥ 0,5 ha, estrategias A (actual),
  B (mejor plan único) y C (por zonas), con P(C supera a B).

## Supuestos que requieren revisión local

| Supuesto | Valor |
|---|---|
| Barbecho entre cosecha y siembra | 40 días |
| Mineralización de N | 60 kg/ha en 120 días con IBS 50 |
| N que toma el cultivo siguiente de la leguminosa | 50 % del aporte × biomasa lograda |
| Pérdida máxima por plagas | 30 %; 15 % de presión si no hay datos |
| NDVI de suelo labrado | 0,18 (el calibrado en caña, 0,34, es lote con hojarasca) |
| Abono verde como cobertura | NDTI 0,14 |
| Residuo → % de suelo cubierto | NDTI 0,05 → 0 %, 0,15 → 80 % |

## Vacíos conocidos

- **Sin costos de producción:** el ingreso es bruto y favorece cultivos de alto valor (hortalizas).
- Sin precio para caña, sorgo, algodón, soya, maní, caupí y guandul (la app acepta `precios_cop_kg`).
- El residuo de la cobertura no depende de la especie (mucuna produce más biomasa que crotalaria).
- La renovación de caña se evalúa en 2 años; en su ciclo real (5–6 cortes) pesa menos.
- Humedad relativa de POWER corregida (53 %) menor que en estaciones (~72 %): solo se usa como anomalía.
- Pendiente NASADEM ruidosa en terreno plano: se suaviza 5×5 píxeles.
