"""Calibración bayesiana del simulador con curvas de rebrote de caña (HLS).

Prior uniforme por parámetro (rangos agronómicos) → muestras por hipercubo latino →
verosimilitud gaussiana con σ desconocido (integrado: L ∝ SSE^(−n/2)) → pesos por muestreo de
importancia. Las observaciones de un mismo segmento están autocorrelacionadas, así que la
verosimilitud se atempera a **un dato efectivo por segmento** (n_ef = número de segmentos):
sin esto el posterior colapsa en una sola muestra y subestima la incertidumbre.

La predictiva posterior suma la incertidumbre de parámetros (muestras del posterior) y un error
de modelo gaussiano σ·c, donde c se ajusta en validación para que el intervalo 80 % cubra ~80 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.cultivos import Cultivo
from rotarapp.simulador.dinamica import Parametros, Siembra, simular
from rotarapp.simulador.observaciones import VENTANA, Segmentos

F = NDArray[np.float32]

PRIOR: dict[str, tuple[float, float]] = {
    "k_mult": (0.4, 2.5),
    "ndvi_suelo": (0.15, 0.45),
    "ndvi_max": (0.70, 0.95),
    "p_agot": (0.30, 0.70),
    "riego": (0.0, 1.0),
    "v0": (0.02, 0.30),
}


def muestrear_prior(n: int, semilla: int = 0) -> dict[str, F]:
    """Hipercubo latino sobre el prior uniforme."""
    rng = np.random.default_rng(semilla)
    out: dict[str, F] = {}
    for k, (lo, hi) in PRIOR.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        out[k] = (lo + (hi - lo) * u).astype(np.float32)
    return out


def a_parametros(m: dict[str, F], idx: NDArray[np.integer] | None = None) -> Parametros:
    sel = (lambda v: v if idx is None else v[idx])
    return Parametros(**{k: sel(v).reshape(-1, 1).astype(np.float32) for k, v in m.items()})


def simular_segmentos(seg: Segmentos, clima: dict[str, F], th: Parametros, cana: Cultivo) -> F:
    """NDVI simulado (VENTANA, D, S): rebrote de caña desde la cosecha detectada."""
    r = simular([Siembra(cana, 0, VENTANA)], clima["lluvia"], clima["et0"], clima["temp"],
                seg.awc[None, :], th, w0_frac=0.5)
    return r.ndvi


def sse_por_muestra(seg: Segmentos, clima: dict[str, F], m: dict[str, F], cana: Cultivo,
                    bloque_d: int = 50) -> tuple[F, int]:
    obs = seg.obs.T[:, None, :]                      # (T, 1, S)
    ok = np.isfinite(obs)
    n = int(ok.sum())
    M = next(iter(m.values())).size
    sse = np.empty(M, np.float64)
    for a in range(0, M, bloque_d):
        idx = np.arange(a, min(a + bloque_d, M))
        nd = simular_segmentos(seg, clima, a_parametros(m, idx), cana)
        r = np.where(ok, nd - obs, 0.0)
        sse[idx] = (r.astype(np.float64) ** 2).sum(axis=(0, 2))
    return sse.astype(np.float32), n


@dataclass(frozen=True)
class Posterior:
    muestras: dict[str, F]
    pesos: F
    sigma: float          # error de modelo (desv. estándar del residuo con la media posterior)
    inflacion: float      # c: factor para que el 80 % cubra ~80 % en validación

    @property
    def ess(self) -> float:
        return float(1.0 / np.sum(self.pesos.astype(np.float64) ** 2))

    def remuestrear(self, k: int, semilla: int = 1) -> NDArray[np.int64]:
        rng = np.random.default_rng(semilla)
        return rng.choice(self.pesos.size, size=k, p=self.pesos / self.pesos.sum())

    def resumen(self) -> dict[str, dict[str, float]]:
        w = self.pesos / self.pesos.sum()
        out = {}
        for k, v in self.muestras.items():
            o = np.argsort(v); c = np.cumsum(w[o])
            q = lambda p: float(v[o][np.searchsorted(c, p)])
            out[k] = {"prior_min": PRIOR[k][0], "prior_max": PRIOR[k][1], "p10": q(0.1),
                      "p50": q(0.5), "p90": q(0.9), "media": float((w * v).sum())}
        return out


def pesos_posterior(sse: F, n_obs: int, n_ef: int) -> F:
    ll = -(n_ef / 2.0) * np.log(sse.astype(np.float64) / n_obs)
    w = np.exp(ll - ll.max())
    return (w / w.sum()).astype(np.float32)


def predictiva(seg: Segmentos, clima: dict[str, F], post: Posterior, cana: Cultivo,
               k: int = 40, semilla: int = 1) -> tuple[F, F]:
    """Media y muestras (k, T, S) del NDVI de la posterior (sin el error de modelo)."""
    idx = post.remuestrear(k, semilla)
    nd = simular_segmentos(seg, clima, a_parametros(post.muestras, idx), cana)   # (T, k, S)
    return nd.mean(axis=1), np.moveaxis(nd, 1, 0)


def cuantiles(muestras: F, sigma_total: float, q: tuple[float, float] = (0.1, 0.9),
              n_ruido: int = 5, semilla: int = 2) -> tuple[F, F]:
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0, sigma_total, (n_ruido, *muestras.shape)).astype(np.float32)
    x = (muestras[None] + ruido).reshape(-1, *muestras.shape[1:])
    lo, hi = np.quantile(x, q, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def metricas(pred: F, obs: F, lo: F | None = None, hi: F | None = None) -> dict[str, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    e = (pred - obs)[ok]
    o = obs[ok]
    out = {"n_obs": int(ok.sum()), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
           "sesgo": float(np.mean(e)), "r2": float(1 - np.sum(e**2) / np.sum((o - o.mean()) ** 2))}
    if lo is not None and hi is not None:
        out["cobertura_80"] = float(np.mean((obs[ok] >= lo[ok]) & (obs[ok] <= hi[ok])))
        out["ancho_80"] = float(np.mean((hi - lo)[ok]))
    return out


def ajustar_inflacion(muestras: F, obs: F, sigma: float, objetivo: float = 0.8) -> float:
    """c tal que el intervalo 80 % (posterior + N(0, σc)) cubra `objetivo` en validación."""
    ok = np.isfinite(obs)
    mejor, err = 1.0, 1.0
    for c in np.linspace(0.5, 3.0, 26):
        lo, hi = cuantiles(muestras, sigma * c)
        cob = float(np.mean((obs[ok] >= lo[ok]) & (obs[ok] <= hi[ok])))
        if abs(cob - objetivo) < err:
            mejor, err = float(c), abs(cob - objetivo)
    return mejor
