"""Clima diario de NASA POWER (punto) y su agregación a décadas (pasos de 10 días).

NASA POWER es una malla de ~0,5°: el mismo clima aplica a toda la finca. ET0 se calcula con
Hargreaves-Samani usando la radiación extraterrestre de POWER (TOA_SW_DWN).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
VARIABLES = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "TOA_SW_DWN")
PASO_DIAS = 10  # décadas: 36 por año (la última del mes absorbe los días sobrantes)

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ClimaDiario:
    fechas: list[date]
    t2m: F64
    tmax: F64
    tmin: F64
    hr: F64
    lluvia: F64
    rad: F64
    toa: F64
    elev_m: float  # elevación de la celda POWER

    def a_altitud(self, elev_destino_m: float, gradiente_c_km: float = 6.5) -> ClimaDiario:
        """Corrige temperatura por altitud (la celda de 0,5° incluye montaña) y ajusta la
        humedad relativa conservando la presión de vapor."""
        dT = gradiente_c_km * (self.elev_m - elev_destino_m) / 1000.0
        t_nueva = self.t2m + dT
        hr = np.clip(self.hr * _es(self.t2m) / _es(t_nueva), 1.0, 100.0)
        return ClimaDiario(self.fechas, t_nueva, self.tmax + dT, self.tmin + dT, hr,
                           self.lluvia, self.rad, self.toa, elev_destino_m)

    def et0(self) -> F64:
        """ET0 Hargreaves-Samani (mm/día). Ra en MJ/m²/día → mm/día con 0,408."""
        dt = np.clip(self.tmax - self.tmin, 0.0, None)
        return np.clip(0.0023 * (self.t2m + 17.8) * np.sqrt(dt) * 0.408 * self.toa, 0.0, None)


@dataclass(frozen=True)
class ClimaDecadal:
    """Clima por década. Índices (año, década 0..35)."""
    anios: list[int]
    t_media: F64   # °C
    hr: F64        # %
    lluvia: F64    # mm en la década
    et0: F64       # mm en la década
    dias: F64      # días de la década

    def anio(self, a: int) -> int:
        return self.anios.index(a)


def descargar(lat: float, lon: float, inicio: date, fin: date, cache: Path) -> ClimaDiario:
    """Descarga (o lee de caché) la serie diaria de POWER para un punto."""
    if not cache.exists():
        params = {
            "parameters": ",".join(VARIABLES), "community": "AG",
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "start": inicio.strftime("%Y%m%d"), "end": fin.strftime("%Y%m%d"), "format": "JSON",
        }
        with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(params), timeout=300) as r:
            datos = json.loads(r.read().decode("utf-8"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(datos), encoding="utf-8")
    return desde_json(json.loads(cache.read_text(encoding="utf-8")))


def desde_json(datos: dict) -> ClimaDiario:
    p = datos["properties"]["parameter"]
    claves = sorted(p["T2M"])
    fechas = [date(int(k[:4]), int(k[4:6]), int(k[6:])) for k in claves]

    def serie(v: str) -> F64:
        x = np.array([p[v][k] for k in claves], dtype=np.float64)
        x[x <= -990] = np.nan  # POWER marca faltantes con -999
        return _rellenar(x)

    return ClimaDiario(fechas, serie("T2M"), serie("T2M_MAX"), serie("T2M_MIN"), serie("RH2M"),
                       np.clip(serie("PRECTOTCORR"), 0.0, None), serie("ALLSKY_SFC_SW_DWN"),
                       serie("TOA_SW_DWN"), float(datos["geometry"]["coordinates"][2]))


def _es(t: F64) -> F64:
    """Presión de vapor de saturación (kPa), FAO-56."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _rellenar(x: F64) -> F64:
    """Interpola faltantes aislados; si todo falta, deja NaN."""
    m = np.isfinite(x)
    if m.all() or not m.any():
        return x
    i = np.arange(x.size)
    return np.interp(i, i[m], x[m])


def decada_del_anio(d: date) -> int:
    return (d.month - 1) * 3 + min((d.day - 1) // PASO_DIAS, 2)


def a_decadas(c: ClimaDiario) -> ClimaDecadal:
    """Agrega a décadas solo los años completos."""
    anios = sorted({d.year for d in c.fechas})
    completos = [a for a in anios if sum(1 for d in c.fechas if d.year == a) >= 365]
    idx = {a: i for i, a in enumerate(completos)}
    shape = (len(completos), 36)
    suma = {k: np.zeros(shape) for k in ("t", "hr", "ll", "et")}
    n = np.zeros(shape)
    et0 = c.et0()
    for j, d in enumerate(c.fechas):
        if d.year not in idx:
            continue
        a, k = idx[d.year], decada_del_anio(d)
        suma["t"][a, k] += c.t2m[j]; suma["hr"][a, k] += c.hr[j]
        suma["ll"][a, k] += c.lluvia[j]; suma["et"][a, k] += et0[j]; n[a, k] += 1
    return ClimaDecadal(completos, suma["t"] / n, suma["hr"] / n, suma["ll"], suma["et"], n)


def fecha_decada(anio: int, k: int) -> date:
    """Fecha central de la década k (0..35) del año."""
    return date(anio, k // 3 + 1, 1) + timedelta(days=(k % 3) * PASO_DIAS + 4)
