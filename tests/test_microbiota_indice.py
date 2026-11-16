"""Pruebas del núcleo del IBS con series sintéticas."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from rotarapp.microbiota import (NDTI_TIPICO, Referencia, UMBRALES_VALLE, componentes_desde_serie,
                                 ibs_de_rotacion, puntuar, referencia_por_suelo,
                                 resumen_escenarios)
from rotarapp.microbiota.indice import PISO_ESCALA, Componentes


def fechas_cada(dias: int, anios: int = 3) -> list[date]:
    inicio = date(2021, 1, 1)
    return [inicio + timedelta(days=d) for d in range(0, 365 * anios, dias)]


def serie_rotacion(fechas: list[date], meses_desnudo: set[int], estado: str) -> tuple[np.ndarray, np.ndarray]:
    """NDVI verde salvo en los meses indicados, donde el suelo queda con el residuo `estado`."""
    desnudo = np.array([f.month in meses_desnudo for f in fechas])
    ndvi = np.where(desnudo, 0.15, 0.75).astype(np.float32)
    ndti = np.where(desnudo, NDTI_TIPICO[estado], 0.0).astype(np.float32)
    return ndvi, ndti


def test_c4_es_fraccion_del_tiempo_verde() -> None:
    f = fechas_cada(10)
    ndvi, ndti = serie_rotacion(f, {1, 2, 3}, "suelo_labrado")
    c = componentes_desde_serie(ndvi[:, None], ndti[:, None], None, f, np.zeros(len(f)))
    esperado = (ndvi > 0.30).mean()
    assert c.valores["C4"][0] == pytest.approx(esperado, abs=1e-6)
    assert np.isnan(c.valores["C1"][0])  # sin SOCI no hay carbono


def test_nubes_nan_no_cuentan() -> None:
    f = fechas_cada(10)
    ndvi, ndti = serie_rotacion(f, {1, 2, 3}, "suelo_labrado")
    ndvi_n = ndvi.copy()
    ndvi_n[::2] = np.nan
    c = componentes_desde_serie(ndvi_n[:, None], ndti[:, None], None, f, np.zeros(len(f)))
    assert c.n_obs["C4"][0] == np.isfinite(ndvi_n).sum()


def test_labranza_baja_c3_y_c2_frente_a_hojarasca() -> None:
    f = fechas_cada(10)
    a = serie_rotacion(f, {1, 2}, "hojarasca_cana")
    b = serie_rotacion(f, {1, 2}, "suelo_labrado")
    ndvi = np.stack([a[0], b[0]], axis=1)
    ndti = np.stack([a[1], b[1]], axis=1)
    c = componentes_desde_serie(ndvi, ndti, None, f, np.zeros(len(f)))
    assert c.valores["C3"][0] > c.valores["C3"][1]
    assert c.valores["C2"][0] > c.valores["C2"][1]


def test_c1_solo_en_suelo_expuesto_y_seco() -> None:
    f = fechas_cada(10, anios=1)
    ndvi = np.full(len(f), 0.1, np.float32)
    ndti = np.full(len(f), 0.05, np.float32)
    soci = np.full(len(f), 6.0, np.float32)
    lluvia = np.full(len(f), 20.0)  # todo mojado
    c = componentes_desde_serie(ndvi[:, None], ndti[:, None], soci[:, None], f, lluvia)
    assert np.isnan(c.valores["C1"][0])
    c = componentes_desde_serie(ndvi[:, None], ndti[:, None], soci[:, None], f, np.zeros(len(f)))
    assert c.valores["C1"][0] == pytest.approx(6.0)


def test_valores_en_la_mediana_dan_50() -> None:
    rng = np.random.default_rng(0)
    vals = {"C1": rng.normal(6, 1, 200), "C2": rng.normal(0.14, 0.02, 200),
            "C3": rng.normal(0.12, 0.02, 200), "C4": rng.normal(0.8, 0.1, 200)}
    comp = Componentes({k: v.astype(np.float32) for k, v in vals.items()},
                       {k: np.full(200, 10) for k in vals})
    grupo = np.ones(200, np.int32)
    ref = referencia_por_suelo(comp, grupo, np.ones(200, bool))
    centro = Componentes({k: np.array([ref.mediana[1][k]], np.float32) for k in vals},
                         {k: np.array([10]) for k in vals})
    p = puntuar(centro, 1, ref)
    assert p.ibs[0] == pytest.approx(50.0, abs=1e-4)


def test_piso_de_escala_en_componente_saturado() -> None:
    n = 100
    comp = Componentes({"C1": np.full(n, np.nan, np.float32), "C2": np.full(n, 0.14, np.float32),
                        "C3": np.full(n, 0.12, np.float32),
                        "C4": np.linspace(0.97, 0.99, n).astype(np.float32)},
                       {k: np.full(n, 10) for k in ("C1", "C2", "C3", "C4")})
    ref = referencia_por_suelo(comp, np.ones(n, np.int32), np.ones(n, bool))
    assert ref.escala[1]["C4"] == PISO_ESCALA["C4"]
    assert "C1" not in ref.mediana[1]  # sin datos de carbono: el componente se omite


def test_rotacion_con_cobertura_supera_a_barbecho_desnudo() -> None:
    rng = np.random.default_rng(1)
    f = fechas_cada(10, anios=6)
    # referencia: lotes de caña con hojarasca y distinto número de meses sin dosel
    ndvi_r, ndti_r = [], []
    for _ in range(120):
        meses = set(rng.choice(range(1, 13), size=rng.integers(1, 4), replace=False).tolist())
        a, b = serie_rotacion(f, meses, "hojarasca_cana")
        ndvi_r.append(a + rng.normal(0, 0.02, a.size).astype(np.float32))
        ndti_r.append(b + rng.normal(0, 0.01, b.size).astype(np.float32))
    comp = componentes_desde_serie(np.stack(ndvi_r, 1), np.stack(ndti_r, 1), None, f,
                                   np.zeros(len(f)), UMBRALES_VALLE)
    ref = referencia_por_suelo(comp, np.ones(120, np.int32), np.ones(120, bool), min_pixeles=50)

    abono_verde = serie_rotacion(f, {6}, "hojarasca_cana")
    barbecho = serie_rotacion(f, {5, 6, 7, 8}, "suelo_labrado")
    ndvi = np.stack([abono_verde[0], barbecho[0]], 1)
    ndti = np.stack([abono_verde[1], barbecho[1]], 1)
    p = ibs_de_rotacion(ndvi, ndti, f, np.zeros(len(f)), 1, ref)
    assert p.ibs[0] > 50 > p.ibs[1]


def test_referencia_json_ida_y_vuelta() -> None:
    ref = Referencia({7: {"C2": 0.14, "C4": 0.9}}, {7: {"C2": 0.02, "C4": 0.05}}, {7: 321},
                     {7: "PWARA-Ea"})
    otra = Referencia.desde_json(ref.a_json())
    assert otra == ref


def test_resumen_escenarios_percentiles() -> None:
    ibs = np.arange(1, 101, dtype=np.float32)[:, None]
    r = resumen_escenarios(ibs)
    assert r["p50"][0] == pytest.approx(50.5)
    assert r["p10"][0] < r["p50"][0] < r["p90"][0]


def test_longitudes_inconsistentes_fallan() -> None:
    f = fechas_cada(10, anios=1)
    with pytest.raises(ValueError):
        componentes_desde_serie(np.zeros((len(f), 1)), np.zeros((len(f) - 1, 1)), None, f,
                                np.zeros(len(f)))
