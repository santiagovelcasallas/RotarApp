"""API local del modelo para la app (solo biblioteca estándar).

    python scripts/api.py [--datos ruta/NASA_IBS/data] [--host 127.0.0.1] [--puerto 8000]

GET  /salud            → {"ok": true}
GET  /catalogo         → cultivos, plagas, prioridades y zonas piloto
POST /recomendacion    → resultado para las pantallas de la app (ver app/README.md)

Para probar desde un celular en la misma red use --host 0.0.0.0 y en la app
EXPO_PUBLIC_API_URL=http://<ip-del-computador>:8000.
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from rotarapp.servicio import Servicio  # noqa: E402


def arg(nombre: str, defecto: str) -> str:
    return sys.argv[sys.argv.index(nombre) + 1] if nombre in sys.argv else defecto


SERVICIO = Servicio(Path(arg("--datos", str(Path.home() / "NASA_IBS/data"))))
CATALOGO = SERVICIO.catalogo()


class Manejador(BaseHTTPRequestHandler):
    def _responder(self, codigo: int, cuerpo: dict) -> None:
        datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(datos)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._responder(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/salud":
            return self._responder(200, {"ok": True, "zonas": [z["nombre"] for z in CATALOGO["zonas"]]})
        if self.path == "/catalogo":
            return self._responder(200, CATALOGO)
        self._responder(404, {"error": "Ruta no encontrada"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/recomendacion":
            return self._responder(404, {"error": "Ruta no encontrada"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            if len(req.get("poligono", [])) < 3:
                return self._responder(422, {"error": "El lote necesita al menos 3 esquinas."})
            self._responder(200, SERVICIO.recomendar(req))
        except ValueError as e:
            self._responder(422, {"error": str(e)})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._responder(500, {"error": "El modelo falló al calcular este lote."})


if __name__ == "__main__":
    host, puerto = arg("--host", "127.0.0.1"), int(arg("--puerto", "8000"))
    print(f"RotarApp API en http://{host}:{puerto}  (zonas: {', '.join(z['nombre'] for z in CATALOGO['zonas'])})", flush=True)
    ThreadingHTTPServer((host, puerto), Manejador).serve_forever()
