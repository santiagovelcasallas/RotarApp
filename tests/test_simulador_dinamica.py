"""Comportamiento físico básico del simulador por píxel."""

from __future__ import annotations

import numpy as np

from rotarapp.datos.cultivos import cargar_cultivos
from rotarapp.simulador.dinamica import V_LLENO, Parametros, Siembra, f_temperatura, simular

C = cargar_cultivos()


def _th(n: int = 1, riego: float = 0.0) -> Parametros:
    one = np.ones((n, 1), np.float32)
    return Parametros(k_mult=one, ndvi_suelo=0.2 * one, ndvi_max=0.85 * one, p_agot=0.5 * one,
                      riego=riego * one, v0=0.05 * one)


def _clima(T: int, lluvia: float, temp: float = 25.0, et0: float = 40.0):
    f = lambda v: np.full((T, 1, 1), v, np.float32)
    return f(lluvia), f(et0), f(temp)


def _maiz(T: int = 12) -> list[Siembra]:
    return [Siembra(C["maiz"], 0, T)]


def test_sin_estres_cierra_dosel_y_rinde_potencial() -> None:
    ll, et, tt = _clima(12, lluvia=80)
    r = simular(_maiz(), ll, et, tt, np.full((1, 1), 150, np.float32), _th())
    llega = 0.40 * 12  # frac_crecimiento del maíz
    # V se reporta al inicio de cada década: el dosel cerrado aparece un paso después
    assert r.V[int(llega) + 1].item() >= V_LLENO - 0.15
    assert r.rend_relativo[0].item() > 0.95


def test_sequia_reduce_rendimiento_y_deja_deficit() -> None:
    awc = np.full((1, 1), 150, np.float32)
    seco = simular(_maiz(), *_clima(12, lluvia=0), awc, _th())
    humedo = simular(_maiz(), *_clima(12, lluvia=80), awc, _th())
    assert seco.rend_relativo[0].item() < 0.6 * humedo.rend_relativo[0].item()
    assert seco.deficit_mm.sum() > humedo.deficit_mm.sum() + 100


def test_riego_completo_elimina_estres() -> None:
    r = simular(_maiz(), *_clima(12, lluvia=0), np.full((1, 1), 150, np.float32), _th(riego=1.0))
    assert r.rend_relativo[0].item() > 0.9
    assert r.ks[3:].min() > 0.95


def test_temperatura_fuera_de_rango() -> None:
    assert f_temperatura(np.array([26.0]), C["maiz"])[0] == 1.0
    assert f_temperatura(np.array([C["maiz"].t[0]]), C["maiz"])[0] == 0.5
    assert f_temperatura(np.array([10.0]), C["maiz"])[0] == 0.0
    frio = simular(_maiz(), *_clima(12, lluvia=80, temp=12.0), np.full((1, 1), 150, np.float32), _th())
    assert frio.rend_relativo[0].item() < 0.1


def test_secuencia_y_broadcast_muestras_por_pixeles() -> None:
    D, P, T = 4, 3, 30
    s = [Siembra(C["crotalaria"], 0, 8), Siembra(C["maiz"], 9, 21)]
    ll = np.random.default_rng(0).uniform(0, 80, (T, D, 1)).astype(np.float32)
    r = simular(s, ll, np.full((T, 1, 1), 40, np.float32), np.full((T, 1, 1), 25, np.float32),
                np.full((1, P), 150, np.float32), _th(D))
    assert r.ndvi.shape == (T, D, P)
    assert list(r.activo[:10]) == [0] * 8 + [-1, 1]
    assert all(x.shape == (D, P) for x in r.rend_relativo)
