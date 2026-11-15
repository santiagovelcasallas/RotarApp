"""Coherencia agronómica de la capa de decisión con datos sintéticos (sin el prototipo)."""

from __future__ import annotations

import numpy as np
import pytest

from rotarapp.comparador import decision as dc
from rotarapp.datos.cultivos import cargar_cultivos, cargar_plagas
from rotarapp.datos.power import ClimaDecadal
from rotarapp.microbiota import Referencia

C = cargar_cultivos()
PL = cargar_plagas()
P = 4


def _clima(anios: int = 6, lluvia_dec: float = 35.0) -> ClimaDecadal:
    sh = (anios, 36)
    rng = np.random.default_rng(0)
    return ClimaDecadal(list(range(2001, 2001 + anios)), np.full(sh, 24.0), np.full(sh, 75.0),
                        rng.uniform(0.3, 1.7, sh) * lluvia_dec, np.full(sh, 42.0), np.full(sh, 10.0))


def _finca(clase: str = "3") -> dc.Finca:
    f = lambda v: np.full(P, v, np.float32)
    return dc.Finca(np.arange(P, dtype=np.int32), np.zeros(P, np.int32), np.array([clase] * P, dtype=object),
                    f(6.5), f(1000), f(30), f(2.5), f(1.5), f(150), np.ones(P, np.int32), f(50))


def _ref() -> Referencia:
    return Referencia({1: {"C2": 0.12, "C3": 0.08, "C4": 0.7}}, {1: {"C2": 0.03, "C3": 0.03, "C4": 0.1}}, {1: 100})


def _esc(clima: ClimaDecadal) -> dc.Escenarios:
    m = {"k_mult": np.array([1.1], np.float32), "ndvi_suelo": np.array([0.25], np.float32),
         "ndvi_max": np.array([0.85], np.float32), "p_agot": np.array([0.5], np.float32),
         "riego": np.array([0.0], np.float32), "v0": np.array([0.1], np.float32)}
    return dc.escenarios(clima, m, np.array([1.0], np.float32), s0=32, k_post=2)


@pytest.fixture(scope="module")
def base():
    clima = _clima()
    return clima, _esc(clima), dc.Entrada(cultivo_anterior="maiz")


def _eval(plan: dc.Plan, base, finca: dc.Finca | None = None) -> dc.Evaluacion:
    clima, esc, ent = base
    return dc.evaluar(plan, finca or _finca(), esc, C, PL, _ref(), ent, 24.0)


def test_leguminosa_antes_de_maiz_sube_su_rendimiento(base) -> None:
    solo = _eval(dc.Plan(None, "maiz"), base).rendimientos["maiz"]
    con = _eval(dc.Plan("mucuna", "maiz"), base).rendimientos["maiz"]
    assert np.nanmedian(con) > 1.1 * np.nanmedian(solo)


def test_repetir_cultivo_tiene_mas_presion_de_plagas() -> None:
    assert PL.presion("maiz", "maiz") == 1.0
    assert PL.presion("frijol", "maiz") < 1.0
    assert PL.presion("crotalaria", "maiz") is None   # sin datos ≠ cero plagas


def test_clase4_penaliza_mas_al_que_pasa_mas_tiempo_en_cultivo_de_venta(base) -> None:
    f4 = _finca("4")
    sin = _eval(dc.Plan(None, "maiz"), base, f4).factor_clase[0]
    con = _eval(dc.Plan("mucuna", "maiz"), base, f4).factor_clase[0]
    assert sin < con < 1.0
    assert _eval(dc.Plan(None, "maiz"), base).factor_clase[0] == 1.0


def test_clase_sin_vocacion_no_es_factible(base) -> None:
    assert not _eval(dc.Plan(None, "maiz"), base, _finca("7")).factible.any()


def test_probabilidades_validas_y_tres_estrategias(base) -> None:
    clima, esc, ent = base
    planes = [dc.Plan(None, "maiz", actual=True), dc.Plan(None, "maiz"), dc.Plan("mucuna", "maiz"),
              dc.Plan("crotalaria", "frijol")]
    evals = [dc.evaluar(p, _finca(), esc, C, PL, _ref(), ent, 24.0) for p in planes]
    d = dc.decidir(planes, evals, ent, min_px_zona=1)
    assert d.p_mejor.shape == (len(planes), P)
    assert np.all(d.p_mejor >= 0) and np.all(d.p_mejor.sum(axis=0) <= 1 + 1e-6)
    assert d.p_mejor[0].sum() == 0          # el plan actual es la línea base, no compite
    assert set(d.estrategias) == {"A", "B", "C"}


def test_secuencia_renovacion_de_cana() -> None:
    sec = dc.secuencia(dc.Plan("crotalaria", "cana_azucar"), C)
    assert sec[0].cultivo.id == "crotalaria" and sec[1].cultivo.id == "cana_azucar"
    assert sec[1].inicio == sec[0].fin + 1
    assert all(s.fin <= dc.HORIZONTE for s in sec)


def test_salud_premia_cobertura_continua(base) -> None:
    abono = _eval(dc.Plan("mucuna", "maiz"), base).salud
    solo = _eval(dc.Plan(None, "maiz"), base).salud
    assert np.nanmedian(abono) > np.nanmedian(solo)
