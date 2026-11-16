"""AWS Lambda (API Gateway HTTP, payload 2.0) para la API del modelo RotarApp.

Mismas rutas que scripts/api.py: GET /salud, GET /catalogo, POST /recomendacion.
El servicio se carga una vez por contenedor (arranque en frío ~2-4 s).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from rotarapp.servicio import Servicio  # noqa: E402

SERVICIO = Servicio(Path(os.environ.get("ROTARAPP_DATOS", str(RAIZ / "nasa_ibs"))))
CATALOGO = SERVICIO.catalogo()
CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}


def _resp(codigo: int, cuerpo: dict) -> dict:
    return {"statusCode": codigo, "headers": {"Content-Type": "application/json; charset=utf-8", **CORS},
            "body": json.dumps(cuerpo, ensure_ascii=False)}


def handler(event: dict, context: object) -> dict:
    http = event.get("requestContext", {}).get("http", {})
    metodo, ruta = http.get("method", "GET"), event.get("rawPath", "/")
    if metodo == "OPTIONS":
        return {"statusCode": 204, "headers": CORS, "body": ""}
    if metodo == "GET" and ruta == "/salud":
        return _resp(200, {"ok": True, "zonas": [z["nombre"] for z in CATALOGO["zonas"]]})
    if metodo == "GET" and ruta == "/catalogo":
        return _resp(200, CATALOGO)
    if metodo == "POST" and ruta == "/recomendacion":
        try:
            cuerpo = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                cuerpo = base64.b64decode(cuerpo).decode("utf-8")
            req = json.loads(cuerpo)
            if len(req.get("poligono", [])) < 3:
                return _resp(422, {"error": "El lote necesita al menos 3 esquinas."})
            return _resp(200, SERVICIO.recomendar(req))
        except ValueError as e:
            return _resp(422, {"error": str(e)})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return _resp(500, {"error": "El modelo falló al calcular este lote."})
    return _resp(404, {"error": "Ruta no encontrada"})
