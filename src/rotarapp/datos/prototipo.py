"""Lectura de las capas del prototipo NASA_IBS (malla UTM 30 m de 666×666 píxeles).

Capas fijas (`base.npz`), IBS observado (`ibs.npz`) y la serie HLS de NDVI (`cube_ndvi.npy`,
340 escenas 2020-2025). Las genera la cadena s01–s03 del prototipo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CLASE_CULTIVO = 5  # 'CULT cultivo estable' (Impact Observatory 2017-2023)
BLOQUE_PX = 67     # ~2 km: unidad de partición espacial


@dataclass(frozen=True)
class Zona:
    dir: Path
    crs: str
    transform: tuple[float, ...]       # (a, b, c, d, e, f) afín
    alto: int
    ancho: int
    cls: NDArray[np.uint8]
    peer: NDArray[np.int32]            # grupo de suelo (IBS)
    clay_pct: NDArray[np.float32]
    ph: NDArray[np.float32]
    soc_pct: NDArray[np.float32]
    pendiente_pct: NDArray[np.float32]
    dem: NDArray[np.float32]
    cap_clase: NDArray[np.object_]     # clase de capacidad de uso IGAC-CVC 2023 ('2', '3', ..., 'CA')
    ibs: NDArray[np.float32]           # IBS observado 2020-2025
    fechas: list[date]

    def ndvi(self) -> NDArray[np.float16]:
        """Cubo (escenas, alto, ancho) en memoria mapeada."""
        return np.load(self.dir / "cube_ndvi.npy", mmap_mode="r")

    def bloque(self) -> NDArray[np.int16]:
        f = np.minimum(np.arange(self.alto) // BLOQUE_PX, self.alto // BLOQUE_PX - 1)
        c = np.minimum(np.arange(self.ancho) // BLOQUE_PX, self.ancho // BLOQUE_PX - 1)
        return (f[:, None] * (self.ancho // BLOQUE_PX) + c[None, :]).astype(np.int16)

    def xy(self, fila: NDArray[np.integer], col: NDArray[np.integer]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        a, b, c, d, e, f = self.transform
        return c + (col + 0.5) * a, f + (fila + 0.5) * e


def _suavizar(x: NDArray[np.float32], r: int) -> NDArray[np.float32]:
    """Media móvil (2r+1)² con bordes replicados."""
    p = np.pad(x, r, mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    s = np.pad(s, ((1, 0), (1, 0)))
    n = 2 * r + 1
    out = s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]
    return (out / n**2).astype(np.float32)


def pendiente_desde_dem(dem: NDArray[np.float32], celda_m: float, radio_px: int = 2) -> NDArray[np.float32]:
    """Pendiente (%) del DEM suavizado: en terreno plano el ruido de NASADEM (dosel de caña,
    surcos) domina la pendiente píxel a píxel."""
    z = _suavizar(dem, radio_px)
    gy, gx = np.gradient(z, celda_m)
    return (100 * np.hypot(gx, gy)).astype(np.float32)


def cargar_zona(dir_region: Path) -> Zona:
    b = np.load(dir_region / "base.npz")
    i = np.load(dir_region / "ibs.npz")
    malla = json.loads(str(b["malla"]))
    vals = json.loads(str(b["igac_vals"]))["CLASE"]
    lut = np.array([""] + vals, dtype=object)          # código 0 = sin dato
    nombres = json.load(open(dir_region / "cube_index.json"))
    fechas = [date(int(n[:4]), int(n[4:6]), int(n[6:8])) for n in nombres]
    tr = tuple(malla["transform"])
    return Zona(
        dir=dir_region, crs=malla["crs"], transform=tr, alto=malla["height"], ancho=malla["width"],
        cls=b["cls"], peer=b["peer"], clay_pct=b["clay"] / 10.0, ph=b["ph"] / 10.0,
        soc_pct=b["soc"] / 100.0, pendiente_pct=pendiente_desde_dem(b["dem"], abs(tr[0])),
        dem=b["dem"], cap_clase=lut[b["cap_clase"]], ibs=i["IBS"], fechas=fechas,
    )


def awc_mm_por_m(clay_pct: NDArray[np.floating]) -> NDArray[np.float32]:
    """Agua disponible (mm por m de suelo) aproximada por arcilla: francos ~150, arcillas
    pesadas ~130 (pedotransferencia simplificada, supuesto)."""
    c = np.nan_to_num(clay_pct, nan=35.0)
    return np.clip(110 + 1.6 * c - 0.022 * c**2, 90, 180).astype(np.float32)
