"""Erosión hídrica con RUSLE: A = R · K · LS · C · P (t/ha/año)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float32]


def factor_r(lluvia_anual_mm: NDArray[np.floating]) -> F:
    """Erosividad (MJ·mm/ha/h/año), Renard y Freimund (1994) con la lluvia anual."""
    p = np.asarray(lluvia_anual_mm, dtype=np.float64)
    r = np.where(p <= 850, 0.0483 * p**1.61, 587.8 - 1.219 * p + 0.004105 * p**2)
    return r.astype(np.float32)


def factor_k(arcilla_pct: NDArray[np.floating], soc_pct: NDArray[np.floating]) -> F:
    """Erodabilidad (t·h/MJ/mm) aproximada por arcilla, con corrección por carbono orgánico
    (factor de materia orgánica de EPIC). Simplificación: no hay arena/limo por píxel."""
    c = np.nan_to_num(arcilla_pct, nan=35.0)
    base = np.interp(c, [10, 20, 35, 50, 70], [0.040, 0.035, 0.030, 0.025, 0.020])
    s = np.nan_to_num(soc_pct, nan=2.0)
    fom = 1 - 0.25 * s / (s + np.exp(3.72 - 2.95 * s))
    return (base * fom).astype(np.float32)


def factor_ls(pendiente_pct: NDArray[np.floating], largo_m: float = 30.0) -> F:
    th = np.arctan(np.nan_to_num(pendiente_pct, nan=0.0) / 100.0)
    s = 65.41 * np.sin(th) ** 2 + 4.56 * np.sin(th) + 0.065
    pend = np.nan_to_num(pendiente_pct, nan=0.0)
    m = np.select([pend < 1, pend < 3, pend < 4.5], [0.2, 0.3, 0.4], 0.5)
    return ((largo_m / 22.13) ** m * s).astype(np.float32)


def factor_c(ndvi: NDArray[np.floating]) -> F:
    """Cobertura desde NDVI (van der Knijff et al., 2000), promediado en el tiempo (eje 0)."""
    n = np.clip(ndvi, 0.0, 0.95)
    return np.clip(np.exp(-2.0 * n / (1.0 - n)), 0.0, 1.0).mean(axis=0).astype(np.float32)


def erosion(lluvia_anual_mm: NDArray[np.floating], arcilla_pct: NDArray[np.floating],
            soc_pct: NDArray[np.floating], pendiente_pct: NDArray[np.floating],
            ndvi_serie: NDArray[np.floating], p_practicas: float = 1.0) -> F:
    return (factor_r(lluvia_anual_mm) * factor_k(arcilla_pct, soc_pct) * factor_ls(pendiente_pct)
            * factor_c(ndvi_serie) * p_practicas).astype(np.float32)
