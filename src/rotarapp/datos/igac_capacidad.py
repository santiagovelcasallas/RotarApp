"""Descarga los polígonos de Capacidad de Uso de las Tierras del Valle del Cauca (IGAC, 2006, 1:100.000).

Servicio público del IGAC (ArcGIS MapServer, capa 0). Guarda un GeoJSON en EPSG:4326 con los
campos CLASE, SUBCLASE, GRUPO_MANEJO, UCS, PRINCIPALES_LIMITANES, USOS_RECOMENDADOS,
PRACTICAS_MANEJO y AREA (ha).

Uso:
    python -m rotarapp.datos.igac_capacidad data/referencia/igac/capacidad_uso_valle.geojson
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SERVICIO = (
    "https://mapas.igac.gov.co/server/rest/services/agrologia/"
    "mapageneraldecapacidaddeusodelastierrasdepartamentodevalledelcauca/MapServer/0/query"
)
CAMPOS = [
    "OBJECTID", "UCS_CP", "CLASE", "SUBCLASE", "GRUPO_MANEJO", "UCS",
    "PRINCIPALES_LIMITANES", "USOS_RECOMENDADOS", "PRACTICAS_MANEJO", "AREA",
]
PAGINA = 250  # consultas grandes devuelven HTTP 500
# A 1:100.000 la precisión posicional es de decenas de metros: simplificar a ~11 m y
# redondear a ~1 m no pierde información y reduce el archivo a una fracción.
SIMPLIFICAR_GRADOS = 0.0001
DECIMALES = 5


def _consulta(params: dict[str, str]) -> dict[str, Any]:
    url = SERVICIO + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _redondear(coords: Any) -> Any:
    if isinstance(coords, (int, float)):
        return round(coords, DECIMALES)
    return [_redondear(c) for c in coords]


def descargar() -> dict[str, Any]:
    ids: list[int] = sorted(_consulta({"where": "1=1", "returnIdsOnly": "true", "f": "json"})["objectIds"])
    features: list[dict[str, Any]] = []
    for i in range(0, len(ids), PAGINA):
        lote = _consulta({
            "objectIds": ",".join(map(str, ids[i:i + PAGINA])),
            "outFields": ",".join(CAMPOS),
            "outSR": "4326",
            "maxAllowableOffset": str(SIMPLIFICAR_GRADOS),
            "f": "geojson",
        })["features"]
        for f in lote:
            if f.get("geometry"):
                f["geometry"]["coordinates"] = _redondear(f["geometry"]["coordinates"])
        features.extend(lote)
        print(f"  {len(features)}/{len(ids)}", file=sys.stderr)
    if len(features) != len(ids):
        raise RuntimeError(f"Se esperaban {len(ids)} polígonos y llegaron {len(features)}")
    return {"type": "FeatureCollection", "name": "capacidad_uso_valle_igac_2006", "features": features}


def main(argv: list[str]) -> int:
    salida = Path(argv[1] if len(argv) > 1 else "data/referencia/igac/capacidad_uso_valle.geojson")
    fc = descargar()
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(fc['features'])} polígonos -> {salida} ({salida.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
