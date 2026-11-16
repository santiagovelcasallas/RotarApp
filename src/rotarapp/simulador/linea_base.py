"""Líneas base que el simulador debe superar (o con las que compararse) en validación y prueba.

1. Climatología: mediana del NDVI de entrenamiento según la edad del rebrote (décadas).
2. Gradient boosting (scikit-learn): emulador puramente estadístico con edad, clima acumulado
   desde la cosecha y suelo. Solo sirve donde hay datos (caña); no puede simular otros cultivos.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rotarapp.simulador.observaciones import VENTANA, Segmentos

F = NDArray[np.float32]


def climatologia(train: Segmentos) -> F:
    return np.nanmedian(train.obs, axis=0).astype(np.float32)          # (VENTANA,)


def rasgos(seg: Segmentos, clima: dict[str, F]) -> F:
    """Filas (S·T, n_rasgos) en el mismo orden que seg.obs.reshape(-1)."""
    ll, et, tt = (clima[k][:, 0, :].T for k in ("lluvia", "et0", "temp"))   # (S, T)
    edad = np.broadcast_to(np.arange(VENTANA, dtype=np.float32), ll.shape)
    gdd = np.cumsum(np.clip(tt - 10.0, 0, None) * 10.0, axis=1)
    ll_c, et_c = np.cumsum(ll, axis=1), np.cumsum(et, axis=1)
    ll_3 = np.stack([np.convolve(r, np.ones(3), mode="full")[:VENTANA] for r in ll])
    bal_3 = ll_3 - np.stack([np.convolve(r, np.ones(3), mode="full")[:VENTANA] for r in et])
    S = len(seg)
    col = lambda v: np.broadcast_to(v[:, None], (S, VENTANA))
    X = np.stack([edad, gdd, ll_c, et_c, ll_3, bal_3, tt,
                  col(((seg.inicio_g % 36) // 3).astype(np.float32)),
                  col(seg.rasgos["arcilla"]), col(seg.rasgos["ph"]),
                  col(seg.rasgos["pendiente"]), col(seg.rasgos["dem"])], axis=-1)
    return X.reshape(-1, X.shape[-1]).astype(np.float32)


NOMBRES_RASGOS = ["edad", "gdd_acum", "lluvia_acum", "et0_acum", "lluvia_3dec", "balance_3dec",
                  "temp", "mes_inicio", "arcilla", "ph", "pendiente", "dem"]


def entrenar_gbm(train: Segmentos, clima: dict[str, F], semilla: int = 0):  # -> HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = rasgos(train, clima)
    y = train.obs.reshape(-1)
    ok = np.isfinite(y)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                      early_stopping=True, validation_fraction=0.15,
                                      random_state=semilla)
    return m.fit(X[ok], y[ok])


def predecir_gbm(modelo, seg: Segmentos, clima: dict[str, F]) -> F:
    return modelo.predict(rasgos(seg, clima)).reshape(len(seg), VENTANA).astype(np.float32)
