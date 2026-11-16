# app/

La interfaz es la app **Expo (React Native)** del equipo (`rotarapp`, rama `conexion-modelo`). La
app conserva su diseño; cada dato de ejemplo se reemplazó por la salida del modelo.

## Cómo correrlo

1. API del modelo (necesita los datos del prototipo `NASA_IBS/data`):

   ```bash
   python scripts/api.py --datos C:/Users/santi/NASA_IBS/data --puerto 8000
   ```

   Para un celular en la misma red: `--host 0.0.0.0` y en la app `EXPO_PUBLIC_API_URL=http://<ip>:8000`.

2. App (en la carpeta de la app): `npx expo start --web` o `npx expo start` para Expo Go.

3. Datos fijos de la app (catálogos, zonas piloto y el resultado del lote de ejemplo, para que la
   demo funcione sin conexión):

   ```bash
   python scripts/exportar_app.py <carpeta de la app>
   ```

## API

| Ruta | Qué hace |
|---|---|
| `GET /salud` | Comprueba que el modelo está arriba |
| `GET /catalogo` | Cultivos, plagas (con hospedantes), prioridades y zonas piloto |
| `POST /recomendacion` | Calcula el lote (3–20 s) |

Entrada de `POST /recomendacion` (lo que envía la pantalla de preguntas):

```json
{
  "poligono": [{"latitude": 3.56, "longitude": -76.31}, ...],
  "cultivos_anteriores": ["cana_azucar"],
  "plagas": ["Barrenadores del tallo"],
  "prioridades": ["agua"],
  "precios_cop_kg": {"cana_azucar": 150}
}
```

## Qué alimenta cada pantalla

La salida la arma `src/rotarapp/comparador/app.py` (tipos en `src/data/modelo.ts` de la app):

| Pantalla | Campo | Contenido |
|---|---|---|
| Preguntas | `GET /catalogo` (o `catalogo.json`) | Cultivos y plagas del modelo; las plagas se filtran por el cultivo anterior; nueva pregunta de prioridades |
| Planes | `planes[]` | 3 planes: recomendado (por zonas), alternativa y lo que hace hoy. Campañas, ejes Suelo/Agua/Sanidad/Erosión/Dinero frente a hoy, confianza = P(mejor plan), alertas |
| Dónde va cada cultivo | `celdas`, `planes[i].campanas[j]` | Celdas de 30 m: color = cultivo, intensidad = probabilidad de que el plan sea el mejor ahí (mapa de calor); reparto, dónde va y por qué; escenarios seco/normal/húmedo; vacíos y riesgos |
| Dinero | `planes[i].dinero` | Ingreso bruto por campaña (SIPSA 2026); abonos verdes marcados como "no se vende" |
| Calendario | `planes[i].calendario` | Fechas de siembra y cosecha simuladas; siembra directa en sistemas con cobertura; riego por cultivo |
| Mi suelo | `suelo.salud`, `suelo.erosion` | IBS por celda (huecos de nube rellenados con vecinos a 150 m) y erosión RUSLE con el manejo de hoy |

Zonas con datos: Palmira – Candelaria y Roldanillo – La Unión – Toro. Fuera de ellas la app avisa.

## Publicación (en producción)

| Pieza | Dónde | Cómo actualizar |
|---|---|---|
| API del modelo | AWS cuenta personal 809411919665, us-east-1: Lambda `rotarapp-api` (Python 3.12, 3008 MB, 60 s) detrás de API Gateway HTTP `stt69lu023` → https://stt69lu023.execute-api.us-east-1.amazonaws.com | `python deploy/empaquetar_lambda.py` → `aws s3 cp deploy/build/rotarapp-api.zip s3://rotarapp-deploy-809411919665/` → `aws lambda update-function-code --function-name rotarapp-api --s3-bucket rotarapp-deploy-809411919665 --s3-key rotarapp-api.zip` |
| App web | Cloudflare Workers (archivos estáticos) `rotarapp` → https://rotarapp.rotarapp.workers.dev | En la app: `EXPO_PUBLIC_API_URL=<url API> npx expo export --platform web` y `npx wrangler deploy` (config en `wrangler.jsonc`) |

Lotes de hasta ~108 ha de cultivo (1.200 celdas); las muestras se ajustan al tamaño para responder en menos de 30 s (límite de API Gateway).
