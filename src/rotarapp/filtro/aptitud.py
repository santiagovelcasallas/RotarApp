"""Filtro 1: qué cultivos son viables en cada píxel (vocación IGAC, clima, pH, altitud)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.cultivos import Cultivo
from rotarapp.simulador.dinamica import f_temperatura

# Capacidad de uso IGAC → qué admite la clase para una rotación de cultivos transitorios
ROTACION_POR_CLASE: dict[str, str] = {
    "1": "plena", "2": "plena", "3": "plena",
    "4": "restringida",       # transitorios solo con prácticas de conservación
    "5": "no", "6": "no", "7": "no", "8": "no",   # permanentes, agroforestería o protección
}
FACTOR_RESTRINGIDA = 0.85     # penaliza la utilidad de transitorios en clase 4


def f_ph(ph: NDArray[np.floating], c: Cultivo, margen: float = 0.5) -> NDArray[np.float32]:
    lo, o0, o1, hi = c.ph
    return np.interp(np.nan_to_num(ph, nan=(o0 + o1) / 2), [lo - margen, lo, o0, o1, hi, hi + margen],
                     [0.0, 0.5, 1.0, 1.0, 0.5, 0.0]).astype(np.float32)


def f_textura(arcilla_pct: NDArray[np.floating], c: Cultivo) -> NDArray[np.float32]:
    """Penaliza suelos más arcillosos que la textura preferida del cultivo (columna 'Textura del
    suelo' del Excel): 1 hasta arcilla_opt, 0,5 en arcilla_lim, 0,2 quince puntos después."""
    o, l = c.arcilla
    return np.interp(np.nan_to_num(arcilla_pct, nan=o), [o, l, l + 15], [1.0, 0.5, 0.2]).astype(np.float32)


def rotacion_permitida(clase: NDArray[np.object_]) -> NDArray[np.str_]:
    return np.array([ROTACION_POR_CLASE.get(str(k), "no") for k in clase], dtype=str)


def aptitud(c: Cultivo, t_media: float, ph: NDArray[np.floating], alt: NDArray[np.floating],
            clase: NDArray[np.object_]) -> NDArray[np.float32]:
    """0..1 por píxel: temperatura media anual × pH × altitud × clase IGAC."""
    ft = float(f_temperatura(np.array([t_media]), c)[0])
    fa = ((alt >= c.alt[0]) & (alt <= c.alt[1])).astype(np.float32)
    rot = rotacion_permitida(clase)
    fc = np.where(rot == "plena", 1.0, np.where(rot == "restringida", FACTOR_RESTRINGIDA, 0.0))
    return (ft * f_ph(ph, c) * fa * fc).astype(np.float32)
