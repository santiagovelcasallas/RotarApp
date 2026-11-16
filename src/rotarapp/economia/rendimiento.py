"""Rendimiento, nitrógeno sin fertilizante, pérdida por plagas e ingreso.

rendimiento = potencial × biomasa_relativa (simulador) × f_pH × f_N × (1 − pérdida_plagas)

Sin agroquímicos el nitrógeno viene de la mineralización del suelo (escalada por el IBS
observado) y del aporte de la leguminosa anterior. Todos los coeficientes son supuestos
agronómicos declarados aquí para revisión local.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.cultivos import Cultivo, Plagas

F = NDArray[np.float32]

N_MINERALIZACION_120D = 60.0   # kg N/ha en 120 días con IBS típico (50)
RECUPERACION_LEGUMINOSA = 0.5  # fracción del aporte de la leguminosa que toma el cultivo siguiente
PERDIDA_MAX_SIN_N = 0.6        # pérdida de rendimiento con cero N disponible
PERDIDA_MAX_PLAGAS = 0.30
PERDIDA_TOPE = 0.60
PRESION_SIN_DATOS = 0.15       # prior cuando el JSON no tiene plagas del cultivo
PRESION_AMBIENTE = 0.20        # plagas y esporas que llegan de fuera del lote aunque se rote
FRAC_HUMEDA_SIN_DATOS = 0.40   # fracción de enfermedades por humedad si el JSON no tiene el cultivo
EXTRA_PLAGA_REPORTADA = 0.5


def n_disponible(c: Cultivo, ibs_obs: NDArray[np.floating], anterior: Cultivo | None,
                 biomasa_anterior: NDArray[np.floating] | float = 1.0) -> F:
    """N para el cultivo: mineralización (escalada por el IBS observado) + aporte de la
    leguminosa anterior proporcional a la biomasa que realmente logró (0..1)."""
    esc = np.clip(np.nan_to_num(ibs_obs, nan=50.0) / 50.0, 0.5, 1.5)
    n = N_MINERALIZACION_120D * (c.ciclo_dias / 120.0) * esc
    if anterior is not None and anterior.fija_n:
        n = n + RECUPERACION_LEGUMINOSA * anterior.n_aporte * np.nan_to_num(biomasa_anterior, nan=0.0)
    return np.asarray(n, np.float32)


def f_nitrogeno(c: Cultivo, n_disp: F) -> F:
    if c.n_demanda <= 0:
        return np.ones_like(n_disp)
    return (1 - PERDIDA_MAX_SIN_N * np.clip(1 - n_disp / c.n_demanda, 0, 1)).astype(np.float32)


def presion_plagas(c: Cultivo, anterior: Cultivo | None, plagas: Plagas,
                   reportadas_hospedan: bool) -> tuple[float, bool]:
    """(presión 0..1, hay_datos). `reportadas_hospedan`: alguna plaga que el agricultor reportó
    ataca también a este cultivo."""
    p = None if anterior is None else plagas.presion(anterior.id, c.id)
    con_datos = p is not None
    p = PRESION_SIN_DATOS if p is None else p
    if reportadas_hospedan:
        p = min(1.0, p + EXTRA_PLAGA_REPORTADA)
    return p, con_datos


def perdida_plagas(presion: float, frac_humeda: float | None, mojado_ciclo: NDArray[np.floating],
                   mojado_medio: float, hr_ciclo: NDArray[np.floating], hr_media: float) -> F:
    """Pérdida por plagas y enfermedades.

    presión total = ambiente + arrastre del cultivo anterior. La parte de hongos, bacterias y
    oomicetos (según el JSON) escala con la frecuencia de días con lluvia del ciclo (hoja mojada)
    y con la anomalía de humedad relativa; la de insectos no.
    """
    p = PRESION_AMBIENTE + (1 - PRESION_AMBIENTE) * presion
    fh = FRAC_HUMEDA_SIN_DATOS if frac_humeda is None else frac_humeda
    f_moj = np.clip(np.asarray(mojado_ciclo) / max(mojado_medio, 1e-3), 0.5, 2.0)
    f_hr = np.clip(1 + (np.asarray(hr_ciclo) - hr_media) / 20.0, 0.7, 1.3)
    return np.minimum(PERDIDA_MAX_PLAGAS * p * ((1 - fh) + fh * f_moj * f_hr), PERDIDA_TOPE).astype(np.float32)
