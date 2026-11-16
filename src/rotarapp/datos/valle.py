"""Datos generales para cualquier lote del Valle del Cauca, fuera de las zonas piloto con HLS.

Capas (scripts/preparar_valle.py): elevación y pendiente (Terrarium SRTM/NASADEM, ~76 m), suelo
(SoilGrids 250 m, 0-30 cm), capacidad de uso IGAC 2006 (1:100.000) y clima NASA POWER en una
rejilla de 0,5° corregido a la altitud del lote. No hay IBS observado por satélite: el suelo se
compara con "lo normal" general del Valle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos import power
from rotarapp.microbiota import Referencia

F = NDArray[np.float64]
CRS_UTM = "EPSG:32618"
CELDA_M = 30.0
GRUPO_GENERAL = 0
NO_CULTIVO = {"CA", "ZU", "REVISAR", "", "None"}


@dataclass(frozen=True)
class CeldaClima:
    lon: float
    lat: float
    ruta: Path


class DatosValle:
    def __init__(self, dir_: Path):
        d = np.load(dir_ / "dem.npz")
        self.elev, self.pend10 = d["elev"], d["pendiente_x10"]
        self.zoom, self.x0, self.y0 = int(d["zoom"]), int(d["x0"]), int(d["y0"])
        s = np.load(dir_ / "suelo.npz")
        self.clay, self.ph, self.soc = s["clay"], s["phh2o"], s["soc"]
        self.s_tr = s["transform"]
        g = np.load(dir_ / "igac.npz")
        self.ig_coords, self.ig_inicio, self.ig_bbox, self.ig_clase = g["coords"], g["inicio"], g["bbox"], g["clase"]
        self.celdas_clima = []
        for p in sorted((dir_ / "clima").glob("*.npz")):
            z = np.load(p)
            self.celdas_clima.append(CeldaClima(float(z["lon"]), float(z["lat"]), p))
        self.ref = Referencia.desde_json((dir_ / "referencia_general.json").read_text(encoding="utf-8"))
        self._clima_cache: dict[tuple[int, int], power.ClimaDecadal] = {}

    # ---------------------------------------------------------------- muestreo de capas
    def _px_dem(self, lon: F, lat: F) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        n = 2 ** self.zoom
        x = (lon + 180) / 360 * n
        lr = np.radians(lat)
        y = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n
        c = np.clip(((x - self.x0) * 256).astype(np.int64), 0, self.elev.shape[1] - 1)
        f = np.clip(((y - self.y0) * 256).astype(np.int64), 0, self.elev.shape[0] - 1)
        return f, c

    def elevacion(self, lon: F, lat: F) -> F:
        f, c = self._px_dem(lon, lat)
        return self.elev[f, c].astype(np.float64)

    def pendiente(self, lon: F, lat: F) -> F:
        f, c = self._px_dem(lon, lat)
        return self.pend10[f, c].astype(np.float64) / 10.0

    def suelo(self, lon: F, lat: F) -> tuple[F, F, F]:
        """Arcilla (%), pH y carbono orgánico (%) — mismas unidades que las zonas piloto."""
        a, _, c, _, e, f = self.s_tr
        col = np.clip(((lon - c) / a).astype(np.int64), 0, self.clay.shape[1] - 1)
        fil = np.clip(((lat - f) / e).astype(np.int64), 0, self.clay.shape[0] - 1)
        return self.clay[fil, col] / 10.0, self.ph[fil, col] / 10.0, self.soc[fil, col] / 100.0

    def clase(self, lon: F, lat: F) -> NDArray[np.object_]:
        """Clase de capacidad de uso IGAC ('2'…'8', 'CA', 'ZU') por punto; '' fuera del Valle."""
        out = np.full(lon.shape, "", dtype=object)
        bb = self.ig_bbox
        cand = np.nonzero((bb[:, 0] <= lon.max()) & (bb[:, 2] >= lon.min()) & (bb[:, 1] <= lat.max()) & (bb[:, 3] >= lat.min()))[0]
        for k in cand:
            anillo = self.ig_coords[self.ig_inicio[k]:self.ig_inicio[k + 1]]
            x1, y1 = anillo[:-1, 0][:, None], anillo[:-1, 1][:, None]
            x2, y2 = anillo[1:, 0][:, None], anillo[1:, 1][:, None]
            cruza = ((y1 > lat) != (y2 > lat)) & (lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1)
            dentro = (cruza.sum(axis=0) % 2) == 1
            out[dentro & (out == "")] = str(self.ig_clase[k])
        return out

    def en_valle(self, lon: float, lat: float) -> bool:
        return self.clase(np.array([lon]), np.array([lat]))[0] != ""

    # ---------------------------------------------------------------- clima
    def clima(self, lon: float, lat: float, elev_m: float) -> power.ClimaDecadal:
        """POWER de la celda más cercana, corregido a la altitud del lote (en pasos de 50 m)."""
        k = int(np.argmin([(c.lon - lon) ** 2 + (c.lat - lat) ** 2 for c in self.celdas_clima]))
        clave = (k, int(round(elev_m / 50.0)))
        if clave not in self._clima_cache:
            z = np.load(self.celdas_clima[k].ruta)
            fechas = [date.fromordinal(int(o)) for o in z["fechas"]]
            dia = power.ClimaDiario(fechas, *(z[v].astype(np.float64) for v in ("t2m", "tmax", "tmin", "hr", "lluvia", "rad", "toa")),
                                    float(z["elev"]))
            self._clima_cache[clave] = power.a_decadas(dia.a_altitud(clave[1] * 50.0))
        return self._clima_cache[clave]

    def lluvia_anual(self, clima: power.ClimaDecadal) -> float:
        return float(np.nanmean(clima.lluvia.sum(axis=1)))


def malla_utm(poligono_lonlat: list[tuple[float, float]]) -> tuple[F, F, F, F, list[list[list[float]]]]:
    """Celdas de 30 m (UTM 18N, alineadas a múltiplos de 30 m) con centro dentro del polígono.
    Devuelve lon, lat de los centros, x, y UTM y los anillos [lng, lat] de cada celda."""
    from pyproj import Transformer
    a_utm = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
    a_ll = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)
    anillo = [a_utm.transform(lon, lat) for lon, lat in poligono_lonlat]
    xs, ys = np.array([p[0] for p in anillo]), np.array([p[1] for p in anillo])
    gx = np.arange(np.floor(xs.min() / CELDA_M) * CELDA_M + CELDA_M / 2, xs.max(), CELDA_M)
    gy = np.arange(np.floor(ys.min() / CELDA_M) * CELDA_M + CELDA_M / 2, ys.max(), CELDA_M)
    X, Y = np.meshgrid(gx, gy)
    X, Y = X.ravel(), Y.ravel()
    dentro = np.zeros(X.shape, bool)
    n = len(anillo)
    for i in range(n):
        x1, y1 = anillo[i]; x2, y2 = anillo[(i + 1) % n]
        dentro ^= ((y1 > Y) != (y2 > Y)) & (X < (x2 - x1) * (Y - y1) / (y2 - y1 + 1e-12) + x1)
    X, Y = X[dentro], Y[dentro]
    lon, lat = a_ll.transform(X, Y)
    h = CELDA_M / 2
    anillos = []
    for xi, yi in zip(X, Y):
        pts = [a_ll.transform(xi + dx, yi + dy) for dx, dy in ((-h, h), (h, h), (h, -h), (-h, -h))]
        anillos.append([[round(p[0], 6), round(p[1], 6)] for p in pts])
    return np.asarray(lon), np.asarray(lat), X, Y, anillos


def cargar(dir_: Path) -> DatosValle | None:
    return DatosValle(dir_) if (dir_ / "dem.npz").exists() else None


def _probar() -> None:  # pragma: no cover - utilidad manual
    v = DatosValle(Path(__file__).resolve().parents[3] / "data/referencia/valle")
    print(json.dumps({"celdas_clima": len(v.celdas_clima)}))
