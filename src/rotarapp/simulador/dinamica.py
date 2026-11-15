"""Simulador por píxel: ecuaciones diferenciales de agua del suelo y dosel, paso de 10 días.

Cada siembra (o rebrote de caña) empieza con suelo sin dosel: el cultivo emerge tras una
latencia (décadas). Esa ventana de suelo desnudo es la que mide el IBS.

Estado por píxel y por muestra (escenario climático × parámetros del posterior):
  W  agua disponible en la zona de raíces (mm), entre 0 y TAW
  V  fracción de dosel (0..1); NDVI = a + (NDVI_max − a)·V

Ecuaciones (Euler explícito, dt = 1 década):
  dV/dt = k·f_T(T)·K_s(W)·V·(1 − V)            crecimiento logístico hasta la senescencia
  dV/dt = −k_sen·V                              senescencia, hasta la cosecha
  dW/dt = lluvia + riego − K_s·Kc(V)·ET0        el exceso sobre TAW drena
  K_s   = clip(W / ((1 − p)·TAW), 0, 1)         FAO-56: sin estrés hasta agotar p·TAW
  f_T   = trapecio que vale 1 en el óptimo del Excel, 0,5 en sus límites y 0 a 3 °C fuera

Todas las entradas se pasan con forma (T, D, P) o broadcastable: D = muestras, P = píxeles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.cultivos import Cultivo

F = NDArray[np.float32]
V_COSECHA = 0.02     # dosel tras la cosecha (arvenses, rebrote)
V_LLENO = 0.9        # dosel "cerrado"
KC_SUELO = 0.30      # evaporación de suelo desnudo
KC_INICIAL = 0.35
MARGEN_T = 3.0       # °C fuera del rango del Excel donde f_T llega a 0


@dataclass(frozen=True)
class Parametros:
    """Parámetros del simulador por muestra; cada campo con forma (D, 1)."""
    k_mult: F       # multiplicador de la tasa de crecimiento
    ndvi_suelo: F   # NDVI con dosel 0 (suelo + residuo)
    ndvi_max: F     # NDVI con dosel 1
    p_agot: F       # fracción de TAW agotable sin estrés
    riego: F        # fracción del déficit que se repone con riego (0 = secano)
    v0: F           # dosel al emerger o rebrotar
    latencia: F | None = None   # décadas sin dosel tras la siembra o la cosecha (None = 0)

    def lat(self) -> F:
        return np.zeros_like(self.k_mult) if self.latencia is None else self.latencia

    @property
    def n(self) -> int:
        return int(self.k_mult.shape[0])


@dataclass(frozen=True)
class Siembra:
    cultivo: Cultivo
    inicio: int   # década (índice del horizonte) de siembra o rebrote
    fin: int      # década de cosecha (exclusiva)


@dataclass(frozen=True)
class Resultado:
    V: F                 # (T, D, P) dosel
    ndvi: F              # (T, D, P)
    ks: F                # (T, D, P) estrés hídrico
    deficit_mm: F        # (T, D, P) ETc potencial − ETc real: riego que evitaría el estrés
    activo: NDArray[np.int16]   # (T,) índice de la siembra activa, −1 si no hay cultivo
    rend_relativo: list[F]      # por siembra, (D, P): biomasa real / biomasa sin estrés


def f_temperatura(t: NDArray[np.floating], c: Cultivo) -> NDArray[np.floating]:
    tmin, to0, to1, tmax = c.t
    x = [tmin - MARGEN_T, tmin, to0, to1, tmax, tmax + MARGEN_T]
    y = [0.0, 0.5, 1.0, 1.0, 0.5, 0.0]
    return np.interp(t, x, y).astype(np.float32)


def tasa_base(c: Cultivo, v0: F, decadas: float) -> F:
    """k tal que, sin estrés, el dosel pasa de v0 a V_LLENO en frac_crecimiento del ciclo."""
    num = np.log(V_LLENO / (1 - V_LLENO)) + np.log((1 - v0) / v0)
    return (num / max(c.frac_crecimiento * decadas, 1.0)).astype(np.float32)


def simular(siembras: list[Siembra], lluvia: F, et0: F, temp: F, awc_mm_m: F,
            th: Parametros, w0_frac: float = 0.7) -> Resultado:
    """Corre una secuencia de siembras sobre T décadas.

    lluvia, et0, temp: (T, Dc, Pc) broadcastable a (T, D, P). awc_mm_m: (1|D, P).
    """
    T = lluvia.shape[0]
    forma = np.broadcast_shapes(lluvia.shape[1:], th.k_mult.shape, awc_mm_m.shape)
    activo = np.full(T, -1, dtype=np.int16)
    for i, s in enumerate(siembras):
        activo[max(s.inicio, 0):min(s.fin, T)] = i

    V = np.full(forma, V_COSECHA, dtype=np.float32)
    Vp = V.copy()                                  # dosel potencial (sin estrés ni frío/calor)
    raiz0 = siembras[0].cultivo.raiz_m if siembras else 0.5
    W = (w0_frac * awc_mm_m * raiz0 * np.ones(forma)).astype(np.float32)
    out_V, out_ks, out_def = (np.empty((T, *forma), np.float32) for _ in range(3))
    real = [np.zeros(forma, np.float32) for _ in siembras]
    pot = [np.zeros(forma, np.float32) for _ in siembras]
    k_cache: dict[int, tuple[F, F, int, int]] = {}

    for t in range(T):
        i = int(activo[t])
        if i >= 0:
            s = siembras[i]
            c = s.cultivo
            L = max(s.fin - s.inicio, 1)
            if i not in k_cache:
                kb = tasa_base(c, th.v0, L) * th.k_mult
                sen = max(int(round(c.frac_senescencia * L)), 1)
                ksen = (np.float32(np.log(V_LLENO / min(c.dosel_final, V_LLENO)) / max(L - sen, 1))
                        if sen < L else np.float32(0))
                k_cache[i] = (kb, np.full_like(kb, ksen), sen, L)
            kb, ksen, sen, L = k_cache[i]
            edad = t - s.inicio
            if edad == 0:
                V = np.full(forma, V_COSECHA, np.float32)
                Vp = V.copy()
                emergio = np.zeros(forma, bool)
            taw = (awc_mm_m * c.raiz_m).astype(np.float32)
            kc_mid = c.kc
            fT = f_temperatura(temp[t], c)
        else:
            taw = (awc_mm_m * 0.5).astype(np.float32)
            kc_mid, fT, edad, sen = KC_SUELO, None, -1, 0

        W = np.minimum(W, taw)
        if i >= 0:
            kc = KC_INICIAL + (kc_mid - KC_INICIAL) * np.minimum(V / 0.8, 1.0)
        else:
            kc = np.full(forma, KC_SUELO, np.float32)
        etc = kc * et0[t]
        umbral = (1 - th.p_agot) * taw
        # Riego por agotamiento (FAO-56): al bajar del umbral se repone una fracción hasta TAW
        W = W + th.riego * np.where(W < umbral, taw - W, 0.0)
        ks = np.clip(W / np.maximum(umbral, 1e-3), 0.0, 1.0)
        eta = ks * etc
        deficit = etc - eta
        W = np.clip(W + lluvia[t] - eta, 0.0, taw)
        # Se reporta el estado al inicio de la década (lo que ve el satélite en ese paso)
        out_V[t] = V
        out_ks[t], out_def[t] = ks, deficit

        if i >= 0:
            nuevo = ~emergio & (edad >= th.lat())
            V = np.where(nuevo, th.v0, V)
            Vp = np.where(nuevo, th.v0, Vp)
            emergio = emergio | nuevo
            if edad < sen:
                V = V + kb * fT * ks * V * (1 - V)
                Vp = Vp + kb * Vp * (1 - Vp)
            else:
                V = V * np.exp(-ksen)
                Vp = Vp * np.exp(-ksen)
            V = np.where(emergio, np.clip(V, 0.0, 1.0), V_COSECHA).astype(np.float32)
            Vp = np.where(emergio, np.clip(Vp, 0.0, 1.0), V_COSECHA).astype(np.float32)
            real[i] += np.where(emergio, V * fT * ks, 0.0)
            pot[i] += np.where(emergio, Vp, 0.0)
            if t + 1 == s.fin or t + 1 == T:
                V = np.full(forma, V_COSECHA, np.float32)
                Vp = V.copy()
        else:
            V = np.full(forma, V_COSECHA, np.float32)

    ndvi = (th.ndvi_suelo + (th.ndvi_max - th.ndvi_suelo) * out_V).astype(np.float32)
    rel = [np.where(p > 0, r / np.maximum(p, 1e-6), np.nan).astype(np.float32) for r, p in zip(real, pot)]
    return Resultado(out_V, ndvi, out_ks, out_def, activo, rel)
