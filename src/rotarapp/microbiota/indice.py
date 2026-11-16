"""Índice Biológico del Suelo (IBS): núcleo de dominio, solo numpy.

El satélite no ve la microbiota; ve los factores de manejo de los que depende:

    C1 Carbono       SOCI = azul / (rojo · verde) en suelo expuesto y seco       más es mejor
    C2 Residuo       NDTI = (SWIR1 − SWIR2) / (SWIR1 + SWIR2) sin cultivo verde    más es mejor
    C3 Perturbación  media anual del NDTI mínimo (minNDTI, Zheng et al. 2013)      más alto = menos labranza/quema
    C4 Raíz viva     fracción del tiempo con NDVI > 0.30                          más es mejor

Cada componente se compara con lo normal en su mismo grupo de suelo (Bünemann et al. 2018:
no hay valores objetivo universales) con un z robusto (mediana, MAD con piso). El puntaje
de cada componente es logística(1.5 · z) y el IBS = 100 × promedio de puntajes, como el
índice de multifuncionalidad por promedio de z de Delgado-Baquerizo et al. (2016).
50 = típico de ese suelo. C1 es opcional (en caña o siembra directa el suelo casi nunca
se ve); C2, C3 y C4 son obligatorios.

La misma función `componentes_desde_serie` sirve para:
- series observadas (NASA HLS / Landsat, con huecos NaN por nubes), y
- series simuladas por rotación (autómata celular), sin SOCI y con el NDTI del residuo
  tomado de `NDTI_TIPICO` según el estado del lote.
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.floating]
COMPONENTES: tuple[str, ...] = ("C1", "C2", "C3", "C4")

# Piso de la escala robusta: por debajo, las diferencias no son agronómicamente relevantes.
# Sin él, un componente saturado (C4 en caña: verde 95-100 % del tiempo) convierte ruido en
# desviaciones enormes.
PISO_ESCALA: dict[str, float] = {"C1": 0.3, "C2": 0.01, "C3": 0.01, "C4": 0.05}

# NDTI típico del suelo según su estado, para series simuladas. Valores iniciales tomados de
# las observaciones del prototipo (Iowa 2016-2021, Palmira-Candelaria 2020-2025): suelo
# expuesto < 0.10; hojarasca de caña cosechada en verde ~0.14. Los de quema y rastrojos de
# soya/maíz son supuestos a calibrar en campo.
NDTI_TIPICO: dict[str, float] = {
    "suelo_labrado": 0.06,
    "quema": 0.05,
    "rastrojo_soya": 0.09,
    "rastrojo_maiz": 0.15,
    "hojarasca_cana": 0.14,
    "abono_verde_incorporado": 0.08,
}


@dataclass(frozen=True)
class Umbrales:
    ndvi_desnudo: float = 0.25        # C1: suelo sin vegetación
    ndvi_verde: float = 0.30          # C2/C3: sin cultivo verde; C4: con raíz viva
    ndti_expuesto: float = 0.10       # C1: poco residuo encima (si no, SOCI lee el rastrojo)
    lluvia_seca_mm: float = 5.0       # lluvia del día + 2 previos para considerar suelo seco
    min_obs_desnudo: int = 2          # C1, C2
    min_anios: int = 2                # C3
    min_obs: int = 3                  # C4
    ndti_solo_seco: bool = False      # exigir suelo seco también para C2
    meses_labranza: tuple[int, ...] = tuple(range(1, 13))  # C3


# Trópico (caña, cosecha todo el año, mucha nube): umbrales laxos, NDTI sin filtro de lluvia.
UMBRALES_VALLE = Umbrales()
# Templado (Iowa): ventanas de primavera para la labranza, más observaciones disponibles.
UMBRALES_IOWA = Umbrales(min_obs_desnudo=3, min_anios=3, ndti_solo_seco=True,
                         meses_labranza=(3, 4, 5, 6))


@dataclass(frozen=True)
class Componentes:
    """Componentes crudos por píxel y cuántas observaciones respaldan cada uno."""
    valores: dict[str, Arr]
    n_obs: dict[str, NDArray[np.integer]]


@dataclass(frozen=True)
class Referencia:
    """Lo normal en cada grupo de suelo: mediana y escala (MAD con piso) por componente."""
    mediana: dict[int, dict[str, float]]
    escala: dict[int, dict[str, float]]
    n_pixeles: dict[int, int]
    nombres: dict[int, str] = field(default_factory=dict)

    def a_json(self) -> str:
        return json.dumps({str(g): {"nombre": self.nombres.get(g, ""),
                                    "n_pixeles": self.n_pixeles[g],
                                    "mediana": self.mediana[g], "escala": self.escala[g]}
                           for g in sorted(self.mediana)}, ensure_ascii=False, indent=1)

    @staticmethod
    def desde_json(texto: str) -> Referencia:
        d = json.loads(texto)
        return Referencia(mediana={int(g): v["mediana"] for g, v in d.items()},
                          escala={int(g): v["escala"] for g, v in d.items()},
                          n_pixeles={int(g): int(v["n_pixeles"]) for g, v in d.items()},
                          nombres={int(g): v["nombre"] for g, v in d.items()})


@dataclass(frozen=True)
class Puntaje:
    z: dict[str, Arr]
    puntaje: dict[str, Arr]   # 0..1 por componente
    ibs: Arr                  # 0..100
    ibs_sin_c4: Arr           # prueba de circularidad: sin cobertura viva


def lluvia_3_dias(fechas: Sequence[date], lluvia_diaria_mm: dict[date, float]) -> NDArray[np.float64]:
    """Lluvia del día de la observación más los 2 días previos."""
    from datetime import timedelta
    return np.array([sum(max(lluvia_diaria_mm.get(d - timedelta(days=k), 0.0), 0.0)
                         for k in range(3)) for d in fechas], dtype=np.float64)


def _mediana_mascara(x: Arr, mascara: NDArray[np.bool_], minimo: int) -> tuple[Arr, NDArray[np.integer]]:
    y = np.where(mascara, x, np.nan)
    n = np.isfinite(y).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(y, axis=0)
    return np.where(n >= minimo, med, np.nan), n


def componentes_desde_serie(ndvi: Arr, ndti: Arr, soci: Arr | None, fechas: Sequence[date],
                            lluvia_3d_mm: NDArray[np.floating],
                            umbrales: Umbrales = UMBRALES_VALLE) -> Componentes:
    """Calcula C1-C4 a partir de series temporales.

    ndvi, ndti, soci: forma (T, ...) con NaN donde no hay observación (nubes). `soci=None`
    para series simuladas (C1 queda NaN). fechas: T fechas. lluvia_3d_mm: (T,).
    """
    u = umbrales
    ndvi = np.asarray(ndvi, dtype=np.float32)
    ndti = np.asarray(ndti, dtype=np.float32)
    if ndvi.shape != ndti.shape or ndvi.shape[0] != len(fechas) or len(lluvia_3d_mm) != len(fechas):
        raise ValueError("ndvi, ndti, fechas y lluvia deben tener la misma longitud temporal")
    extra = (1,) * (ndvi.ndim - 1)
    seco = (np.asarray(lluvia_3d_mm) < u.lluvia_seca_mm).reshape(-1, *extra)
    sin_verde = ndvi < u.ndvi_verde

    if soci is None:
        c1 = np.full(ndvi.shape[1:], np.nan, dtype=np.float32)
        n1 = np.zeros(ndvi.shape[1:], dtype=np.int64)
    else:
        c1, n1 = _mediana_mascara(np.asarray(soci, dtype=np.float32),
                                  seco & (ndvi < u.ndvi_desnudo) & (ndti < u.ndti_expuesto),
                                  u.min_obs_desnudo)
    c2, n2 = _mediana_mascara(ndti, (seco if u.ndti_solo_seco else True) & sin_verde,
                              u.min_obs_desnudo)

    anios = np.array([d.year for d in fechas])
    meses = np.array([d.month for d in fechas])
    minimos = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in np.unique(anios):
            sel = (anios == a) & np.isin(meses, u.meses_labranza)
            if sel.any():
                minimos.append(np.nanmin(np.where(sin_verde[sel], ndti[sel], np.nan), axis=0))
        anual = np.stack(minimos) if minimos else np.full((1, *ndvi.shape[1:]), np.nan)
        n3 = np.isfinite(anual).sum(axis=0)
        c3 = np.where(n3 >= u.min_anios, np.nanmean(anual, axis=0), np.nan)

    validas = np.isfinite(ndvi)
    n4 = validas.sum(axis=0)
    c4 = np.where(n4 >= u.min_obs, (ndvi > u.ndvi_verde).sum(axis=0) / np.maximum(n4, 1), np.nan)
    return Componentes(
        valores={"C1": c1.astype(np.float32), "C2": c2.astype(np.float32),
                 "C3": c3.astype(np.float32), "C4": c4.astype(np.float32)},
        n_obs={"C1": n1, "C2": n2, "C3": n3, "C4": n4})


def referencia_por_suelo(comp: Componentes, grupo: NDArray[np.integer],
                         mascara_ref: NDArray[np.bool_], min_pixeles: int = 50,
                         nombres: dict[int, str] | None = None) -> Referencia:
    """Mediana y escala robusta de cada componente en los píxeles de referencia de cada suelo.

    Un grupo sin suficientes píxeles válidos en un componente queda sin ese componente.
    """
    mediana: dict[int, dict[str, float]] = {}
    escala: dict[int, dict[str, float]] = {}
    n_px: dict[int, int] = {}
    for g in np.unique(grupo[mascara_ref & (grupo > 0)]):
        sel = (grupo == g) & mascara_ref
        med_g: dict[str, float] = {}
        esc_g: dict[str, float] = {}
        for k in COMPONENTES:
            v = comp.valores[k][sel]
            v = v[np.isfinite(v)]
            if v.size < min_pixeles:
                continue
            m = float(np.median(v))
            med_g[k] = m
            esc_g[k] = max(float(1.4826 * np.median(np.abs(v - m))), PISO_ESCALA[k])
        if med_g:
            gi = int(g)
            mediana[gi], escala[gi], n_px[gi] = med_g, esc_g, int(sel.sum())
    return Referencia(mediana, escala, n_px, dict(nombres or {}))


def puntuar(comp: Componentes, grupo: NDArray[np.integer] | int, ref: Referencia) -> Puntaje:
    """z, puntaje por componente e IBS. `grupo` puede ser un raster o un solo grupo de suelo."""
    forma = comp.valores["C4"].shape
    grupo_arr = np.broadcast_to(np.asarray(grupo), forma)
    z: dict[str, Arr] = {}
    s: dict[str, Arr] = {}
    for k in COMPONENTES:
        med = np.full(forma, np.nan, dtype=np.float32)
        esc = np.full(forma, np.nan, dtype=np.float32)
        for g in np.unique(grupo_arr):
            if int(g) in ref.mediana and k in ref.mediana[int(g)]:
                sel = grupo_arr == g
                med[sel] = ref.mediana[int(g)][k]
                esc[sel] = ref.escala[int(g)][k]
        z[k] = ((comp.valores[k] - med) / esc).astype(np.float32)
        s[k] = (1.0 / (1.0 + np.exp(-1.5 * np.clip(z[k], -6, 6)))).astype(np.float32)
    pila = np.stack([s[k] for k in COMPONENTES])
    ok = np.isfinite(pila[1:]).all(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ibs = np.where(ok, 100 * np.nanmean(pila, axis=0), np.nan).astype(np.float32)
        ibs_sin_c4 = np.where(ok, 100 * np.nanmean(pila[:3], axis=0), np.nan).astype(np.float32)
    return Puntaje(z, s, ibs, ibs_sin_c4)


def confianza(comp: Componentes, umbrales: Umbrales = UMBRALES_VALLE) -> NDArray[np.uint8]:
    """3 alta, 2 media, 1 baja, 0 sin carbono medido (según observaciones de suelo usadas)."""
    n_eff = np.minimum(np.where(np.isfinite(comp.valores["C1"]), comp.n_obs["C1"], 0),
                       comp.n_obs["C2"])
    return np.select([n_eff >= 10, n_eff >= 5, n_eff >= umbrales.min_obs_desnudo],
                     [3, 2, 1], 0).astype(np.uint8)


def ibs_de_rotacion(ndvi_sim: Arr, ndti_sim: Arr, fechas: Sequence[date],
                    lluvia_3d_mm: NDArray[np.floating], grupo: NDArray[np.integer] | int,
                    ref: Referencia, umbrales: Umbrales = UMBRALES_VALLE) -> Puntaje:
    """IBS de una rotación simulada sobre su ciclo completo (no una foto).

    ndvi_sim, ndti_sim: forma (T, ...) a paso fijo (p. ej. 10 días) cubriendo toda la
    rotación (en caña, los 5-6 cortes). ndti_sim solo importa donde NDVI < 0.30; llénelo con
    `NDTI_TIPICO` según el estado del lote. C1 no se simula.
    """
    comp = componentes_desde_serie(ndvi_sim, ndti_sim, None, fechas, lluvia_3d_mm, umbrales)
    return puntuar(comp, grupo, ref)


def resumen_escenarios(ibs_por_escenario: Arr) -> dict[str, Arr]:
    """p10, p50 y p90 del IBS sobre el eje 0 (escenarios climáticos)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.nanpercentile(ibs_por_escenario, [10, 50, 90], axis=0)
    return {"p10": p[0], "p50": p[1], "p90": p[2]}
