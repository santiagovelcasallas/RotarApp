"""Escribe en la app los datos fijos que salen del modelo:

- src/data/catalogo.json: cultivos, plagas (con sus hospedantes) y prioridades del modelo.
- src/data/zonas-habilitadas.json: las zonas piloto con datos (se conservan municipios y vista).
- src/data/demo-resultado.json: resultado del lote de ejemplo, para mostrar sin conexión.

    python scripts/exportar_app.py <carpeta de la app> [ruta NASA_IBS/data]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from rotarapp.servicio import Servicio  # noqa: E402

LOTE_EJEMPLO = [  # el DEFAULT_LOT de la app (src/data/geo.ts), Palmira
    {"latitude": 3.56096, "longitude": -76.31904}, {"latitude": 3.56141, "longitude": -76.31809},
    {"latitude": 3.56058, "longitude": -76.31732}, {"latitude": 3.55975, "longitude": -76.31777},
    {"latitude": 3.56, "longitude": -76.31898},
]


def main() -> None:
    app = Path(sys.argv[1])
    datos = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "NASA_IBS/data"
    s = Servicio(datos)
    cat = s.catalogo()
    d = app / "src/data"
    (d / "catalogo.json").write_text(json.dumps({k: cat[k] for k in ("cultivos", "plagas", "prioridades")},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    zonas = json.loads((d / "zonas-habilitadas.json").read_text(encoding="utf-8"))
    zonas["actualizado"] = date.today().isoformat()
    zonas["nota"] = ("Zonas piloto con datos del modelo (cajas de 20 × 20 km del prototipo NASA_IBS). "
                     "Generado por RotarApp/scripts/exportar_app.py.")
    zonas["mensajeFueraDeZona"] = ("Por ahora RotarApp tiene datos en " + " y ".join(z["nombre"] for z in cat["zonas"])
                                   + ". Pronto llegaremos a todo el Valle del Cauca.")
    zonas["zonas"]["features"] = [{
        "type": "Feature", "id": z["id"],
        "properties": {"nombre": z["nombre"], "tipo": "zona piloto", "habilitada": True, "color": "#1F6B4A"},
        "geometry": {"type": "Polygon", "coordinates": [z["anillo"]]},
    } for z in cat["zonas"]]
    (d / "zonas-habilitadas.json").write_text(json.dumps(zonas, ensure_ascii=False, indent=2), encoding="utf-8")
    demo = s.recomendar({"poligono": LOTE_EJEMPLO, "cultivos_anteriores": ["cana_azucar"],
                         "plagas": ["Barrenadores del tallo"], "prioridades": ["suelo"]})
    (d / "demo-resultado.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
    print("catalogo:", len(cat["cultivos"]), "cultivos,", len(cat["plagas"]), "plagas | zonas:",
          [z["nombre"] for z in cat["zonas"]], "| demo:", demo["n_celdas"], "celdas,", [p["tag"] for p in demo["planes"]])


if __name__ == "__main__":
    main()
