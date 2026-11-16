"""El módulo de RotarApp reproduce el IBS del prototipo NASA_IBS (mismas entradas, mismo resultado).

Se salta si los datos del prototipo no están en esta máquina. Ruta configurable con la
variable de entorno NASA_IBS_DATA (por defecto ../NASA_IBS/data, junto a RotarApp).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
from rotarapp.microbiota.exportar import cargar_entrada, componentes_por_franjas  # noqa: E402
from rotarapp.microbiota.indice import (UMBRALES_IOWA, UMBRALES_VALLE, puntuar,  # noqa: E402
                                        referencia_por_suelo)

DATOS = Path(os.environ.get("NASA_IBS_DATA",
                            Path(__file__).resolve().parents[2] / "NASA_IBS" / "data"))
CASOS = [("valle", UMBRALES_VALLE, (5,)), ("iowa", UMBRALES_IOWA, (1, 2, 3))]


@pytest.mark.parametrize("zona,umbrales,clases", CASOS, ids=[c[0] for c in CASOS])
def test_reproduce_prototipo(zona: str, umbrales, clases: tuple[int, ...]) -> None:
    carpeta = DATOS / zona
    if not (carpeta / "ibs.npz").exists() or not (carpeta / "cube_index.json").exists():
        pytest.skip(f"sin datos del prototipo en {carpeta}")
    base, fechas, lluvia3, cubo = cargar_entrada(carpeta)
    comp = componentes_por_franjas(fechas, lluvia3, cubo, umbrales)
    ref = referencia_por_suelo(comp, base["peer"], np.isin(base["cls"], clases))
    p = puntuar(comp, base["peer"], ref)
    esperado = np.load(carpeta / "ibs.npz")
    for k in ("C1", "C2", "C3", "C4"):
        np.testing.assert_allclose(comp.valores[k], esperado[k], rtol=1e-5, atol=1e-6,
                                   equal_nan=True, err_msg=k)
    np.testing.assert_allclose(p.ibs, esperado["IBS"], rtol=1e-4, atol=1e-3, equal_nan=True)
