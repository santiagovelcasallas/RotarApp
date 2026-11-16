"""Datos de entrenamiento, validación y prueba: curvas de rebrote de caña observadas con HLS.

Cada *segmento* es un píxel desde una cosecha detectada hasta la siguiente (máx. 40 décadas).
El simulador debe reproducir esa curva con el clima real; es la única dinámica de cultivo que
el satélite muestra en Palmira–Candelaria (ningún píxel tiene más de 1,5 ciclos por año).

Partición (sin fuga entre conjuntos):
  bloques espaciales de ~2 km: 70 % entrenamiento, 30 % prueba espacial (semilla fija)
  por año de inicio del segmento: ≤2023 entrenamiento, 2024 validación, 2025 prueba temporal
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.power import ClimaDecadal, decada_del_anio
from rotarapp.datos.prototipo import CLASE_CULTIVO, Zona, awc_mm_por_m

VENTANA = 40               # décadas simuladas por segmento (~13 meses)
NDVI_COSECHA = 0.45        # NDVI bajo este valor tras dosel alto = cosecha
NDVI_ALTO = 0.65
MIN_ENTRE_COSECHAS = 24    # décadas (8 meses)
MIN_OBS = 6


@dataclass(frozen=True)
class Segmentos:
    obs: NDArray[np.float32]        # (S, VENTANA) NDVI observado, NaN sin dato o fuera del segmento
    inicio_g: NDArray[np.int32]     # década global de inicio (índice en el clima decadal)
    anio: NDArray[np.int16]
    bloque: NDArray[np.int16]
    fila: NDArray[np.int32]
    col: NDArray[np.int32]
    awc: NDArray[np.float32]
    rasgos: dict[str, NDArray[np.float32]]   # suelo y terreno por segmento
    conjunto: NDArray[np.str_]      # 'train', 'val', 'test_t', 'test_e'

    def sub(self, nombre: str) -> Segmentos:
        m = self.conjunto == nombre
        return Segmentos(self.obs[m], self.inicio_g[m], self.anio[m], self.bloque[m], self.fila[m],
                         self.col[m], self.awc[m], {k: v[m] for k, v in self.rasgos.items()},
                         self.conjunto[m])

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def particion_bloques(n_bloques: int, frac_prueba: float = 0.3, semilla: int = 42) -> NDArray[np.bool_]:
    """True = bloque de prueba espacial."""
    rng = np.random.default_rng(semilla)
    prueba = np.zeros(n_bloques, bool)
    prueba[rng.permutation(n_bloques)[: int(round(frac_prueba * n_bloques))]] = True
    return prueba


def compuesto_decadal(serie: NDArray[np.floating], fechas: list[date], anio0: int, n_anios: int) -> NDArray[np.float32]:
    """Máximo NDVI por década (reduce nube residual). serie: (escenas, N) → (n_anios·36, N)."""
    out = np.full((n_anios * 36, serie.shape[1]), np.nan, np.float32)
    for j, d in enumerate(fechas):
        g = (d.year - anio0) * 36 + decada_del_anio(d)
        if 0 <= g < out.shape[0]:
            out[g] = np.fmax(out[g], serie[j].astype(np.float32))
    return out


def detectar_cosechas(x: NDArray[np.float32]) -> list[int]:
    """Índices de cosecha en una serie decadal de un píxel."""
    cos: list[int] = []
    for t in range(9, len(x)):
        if not np.isfinite(x[t]) or x[t] >= NDVI_COSECHA:
            continue
        previo = x[t - 9:t]
        if np.nanmax(previo, initial=-1) >= NDVI_ALTO and (not cos or t - cos[-1] >= MIN_ENTRE_COSECHAS):
            cos.append(t)
    return cos


def construir(z: Zona, clima: ClimaDecadal, px_por_bloque: int = 60, semilla: int = 0) -> Segmentos:
    rng = np.random.default_rng(semilla)
    bloque = z.bloque()
    n_bloques = int(bloque.max()) + 1
    prueba_e = particion_bloques(n_bloques)
    ok = (z.cls == CLASE_CULTIVO) & np.isfinite(z.clay_pct)
    filas, cols = [], []
    for b in range(n_bloques):
        f, c = np.nonzero(ok & (bloque == b))
        if f.size:
            k = rng.choice(f.size, min(px_por_bloque, f.size), replace=False)
            filas.append(f[k]); cols.append(c[k])
    fi, co = np.concatenate(filas), np.concatenate(cols)

    cubo = z.ndvi()
    serie = np.stack([cubo[:, f, c] for f, c in zip(fi, co)], axis=1)   # (escenas, N)
    anio0 = min(d.year for d in z.fechas)
    n_anios = max(d.year for d in z.fechas) - anio0 + 1
    dec = compuesto_decadal(serie, z.fechas, anio0, n_anios)
    g0 = clima.anio(anio0) * 36

    rows: dict[str, list] = {k: [] for k in ("obs", "g", "anio", "px")}
    for n in range(fi.size):
        x = dec[:, n]
        cos = detectar_cosechas(x)
        for a, t in enumerate(cos):
            fin = cos[a + 1] if a + 1 < len(cos) else len(x)
            o = np.full(VENTANA, np.nan, np.float32)
            seg = x[t:min(fin, t + VENTANA)]
            o[:seg.size] = seg
            if np.isfinite(o).sum() < MIN_OBS:
                continue
            rows["obs"].append(o); rows["g"].append(g0 + t)
            rows["anio"].append(anio0 + t // 36); rows["px"].append(n)

    px = np.array(rows["px"])
    f, c = fi[px], co[px]
    anio = np.array(rows["anio"], np.int16)
    bl = bloque[f, c]
    conj = np.where(prueba_e[bl], "test_e",
                    np.where(anio <= 2023, "train", np.where(anio == 2024, "val", "test_t")))
    rasgos = {"arcilla": z.clay_pct[f, c], "ph": z.ph[f, c], "pendiente": z.pendiente_pct[f, c],
              "dem": z.dem[f, c]}
    return Segmentos(np.stack(rows["obs"]), np.array(rows["g"], np.int32), anio, bl,
                     f.astype(np.int32), c.astype(np.int32), awc_mm_por_m(z.clay_pct[f, c]),
                     {k: v.astype(np.float32) for k, v in rasgos.items()}, conj.astype(str))


def clima_de_segmentos(s: Segmentos, clima: ClimaDecadal) -> dict[str, NDArray[np.float32]]:
    """Clima real de cada segmento: (VENTANA, 1, S). Después del último año de POWER se rellena
    con la climatología de la década (ahí tampoco hay observaciones HLS)."""
    idx = s.inicio_g[None, :] + np.arange(VENTANA)[:, None]
    def g(a: NDArray[np.float64]) -> NDArray[np.float32]:
        ext = np.concatenate([a.reshape(-1), np.tile(np.nanmean(a, axis=0), 2)])
        return ext[idx][:, None, :].astype(np.float32)
    return {"lluvia": g(clima.lluvia), "et0": g(clima.et0), "temp": g(clima.t_media),
            "calor": {u: g(v) for u, v in clima.frac_calor.items()}, "mojado": g(clima.frac_lluvia)}
