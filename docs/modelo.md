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

## Capas

| Capa | Código | Qué hace |
|---|---|---|
| Clima | `datos/power.py` | NASA POWER diario 2001–2025, corregido por altitud (la celda de 0,5° está a 2.112 m; los lotes a 975 m: −7,4 °C), ET0 Hargreaves, décadas |
| Filtro | `filtro/aptitud.py` | Clase IGAC-CVC 2023 (3 plena, 4 restringida, 6+ no), temperatura, pH, altitud y textura de la tabla |
| Simulador | `simulador/dinamica.py` | Agua del suelo (FAO-56) y dosel logístico con f(T) y estrés hídrico, paso de 10 días |
| Calibración | `simulador/calibracion.py` | Prior uniforme → hipercubo latino (2.000) → verosimilitud atemperada → pesos de importancia |
| Rendimiento | `economia/rendimiento.py` | Potencial × biomasa relativa × pH × textura × nitrógeno (sin fertilizante) × (1 − plagas) |
| Erosión | `erosion/rusle.py` | RUSLE: R (lluvia), K (arcilla, carbono), LS (pendiente del DEM suavizado), C (NDVI simulado) |
| Suelo | `microbiota/` + `comparador/decision.py` | IBS oficial y un índice comparativo con sus mismos componentes en escala absoluta |
| Decisión | `comparador/decision.py` | Utilidad con las prioridades del agricultor → P(plan es el mejor) por píxel → zonas → estrategias A, B, C |
| Explicaciones | `comparador/explicaciones.py` | Por qué, evidencia, beneficios, vacíos y riesgos de cada estrategia |

## Entrenamiento, validación y prueba

Datos: 13.434 curvas de rebrote de caña (cosecha → cosecha) detectadas en la serie HLS de 6.000
píxeles muestreados. Partición sin fuga:

| Conjunto | Datos | Segmentos | Uso |
|---|---|---|---|
| Entrenamiento | 70 % de los bloques de ~2 km, inicio 2020–2023 | 7.252 | Posterior de parámetros |
| Validación | mismos bloques, 2024 | 2.395 | Ajustar el ancho del intervalo; decisiones de modelado |
| Prueba temporal | mismos bloques, 2025 | 1.337 | Una sola vez, al final |
| Prueba espacial | 30 % de bloques nunca vistos | 4.515 | Una sola vez, al final |
| Prueba externa | Valle norte (Roldanillo–La Unión–Toro) | — | Pendiente: descarga en curso, nunca mirada |

Verosimilitud atemperada a un dato efectivo por par bloque × año (224): los segmentos de un mismo
bloque y año comparten clima y manejo. Sin esto el posterior colapsa en una sola muestra.

### Resultados (RMSE del NDVI; cobertura del intervalo del 80 %)

| Conjunto | Simulador bayesiano | Climatología | Gradient boosting | Cobertura 80 % |
|---|---|---|---|---|
| Entrenamiento | 0,111 | 0,114 | 0,093 | 80 % |
| Validación | 0,100 | 0,100 | 0,098 | 82 % |
| Prueba temporal (2025) | 0,114 | 0,112 | 0,111 | 76 % |
| Prueba espacial | **0,114** | 0,116 | 0,111 | 78 % |

Los tres quedan a menos de 3 % entre sí. El gradient boosting es algo mejor en caña, pero solo
sirve para caña; el simulador queda a la par, con incertidumbre bien calibrada, y es el único que
puede simular los otros cultivos.

Nota de transparencia: la prueba se evaluó una vez antes de dos correcciones (un error de mi
código que reiniciaba el dosel al final de la ventana y la senescencia de la caña madura). Ambas
se justificaron con entrenamiento y validación, no con la prueba. La prueba externa (Valle norte)
sigue intacta.

Posterior (p10–p90): k_mult 1,04–1,51 · NDVI de suelo con hojarasca 0,18–0,30 · NDVI máximo
0,78–0,81 · p (agotamiento sin estrés) 0,32–0,68 · fracción del déficit regada 0,06–0,85 · v0 0,15–0,27.

## Decisión

- **Muestras:** 23 escenarios climáticos de 2 años (2001–2023, siembra el 21 de noviembre) × 4 muestras
  del posterior = 92. Los cultivos sin datos usan el prior (k_mult 0,7–1,3, p 0,4–0,6, v0 0,03–0,10)
  y los parámetros de sensor calibrados en caña.
- **Planes:** leguminosa de descanso → cultivo de venta con precio (maíz, arroz, hortalizas, ají,
  fríjol), sin descanso, y, si el lote viene de caña, renovación con leguminosa → caña.
- **Utilidad:** pesos del agricultor sobre ingreso bruto (normalizado al mejor plan), ahorro de agua
  (riego para no tener estrés, 800 mm = 0), salud del suelo (índice comparativo) y erosión
  (10 t/ha/año = 0). En clase IGAC 4 la utilidad se multiplica por 1 − 0,3 × fracción del tiempo
  en cultivo de venta (la clase admite transitorios solo restringidos; la cobertura no).
- **Salida:** P(plan es el mejor) por píxel = el mapa de calor. Zonas: plan más probable por
  píxel, máximo 3 zonas de al menos 0,5 ha. Estrategias: **A** actual, **B** mejor plan único,
  **C** por zonas, con P(C supera a B).

## Supuestos que requieren revisión local

| Supuesto | Valor | Dónde |
|---|---|---|
| Mineralización de N | 60 kg/ha en 120 días con IBS 50 | `economia/rendimiento.py` |
| N que toma el cultivo siguiente de la leguminosa | 50 % del aporte × biomasa lograda | ídem |
| Pérdida máxima por plagas | 30 %; 15 % de presión si no hay datos | ídem |
| NDVI de suelo labrado | 0,18 | `comparador/decision.py` |
| Abono verde | se corta y se deja como cobertura (NDTI 0,14), siembra directa | ídem |
| Sistemas con leguminosa | siembra directa; sin cobertura hay que labrar (sin herbicidas) | ídem |
| Textura preferida, ciclo, raíz, potencial | `data/referencia/cultivos/cultivos_modelo.csv` | tabla |

## Vacíos conocidos

- **Sin costos de producción:** el ingreso es bruto y favorece cultivos de alto valor (hortalizas).
- Sin precio para caña, sorgo, algodón, soya, maní, caupí y guandul; la app puede recibir
  `precios_cop_kg` del agricultor.
- El IBS oficial compara con lo típico del suelo (aquí, caña verde todo el año) y satura en 0 para
  rotaciones con semanas sin cobertura; por eso la utilidad usa el índice comparativo.
- En caña, la renovación se evalúa en 2 años; en su ciclo real (5–6 cortes) pesa menos.
- La humedad relativa de POWER corregida por altitud (53 %) es menor que la de estaciones (~72 %);
  solo se usa como anomalía para el riesgo de plagas.
- Pendiente NASADEM ruidosa en terreno plano: se suaviza 5×5 píxeles.
