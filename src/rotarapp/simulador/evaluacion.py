"""Métricas de clasificación del evento que mide el IBS: ¿el lote está sin dosel en esa década?

El simulador da una probabilidad P(NDVI < umbral) a partir de la predictiva posterior
(muestras de parámetros + error de modelo gaussiano). El umbral de decisión se elige en
validación (máximo F1) y se aplica sin cambios a prueba. Evento raro (3–6 % en caña), así que
se reporta AUC-PR además de AUC-ROC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

F = NDArray[np.float32]
UMBRALES_NDVI = {"sin_dosel": 0.30, "baja_cobertura": 0.45}


def prob_bajo(muestras: F, umbral: float, sigma: float) -> F:
    """P(NDVI < umbral): muestras (K, T, S) de la posterior + N(0, σ)."""
    return norm.cdf((umbral - muestras) / sigma).mean(axis=0).astype(np.float32)


def prob_bajo_punto(pred: F, umbral: float, sigma: float) -> F:
    return norm.cdf((umbral - pred) / sigma).astype(np.float32)


def prob_climatologia(obs_train: F, umbral: float) -> F:
    """P(NDVI < umbral | edad del rebrote) empírica en entrenamiento. obs_train: (T, S)."""
    return np.array([np.mean(o[np.isfinite(o)] < umbral) for o in obs_train], np.float32)


def etiquetas(obs: F, umbral: float) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    ok = np.isfinite(obs)
    return (obs[ok] < umbral).astype(int), ok


def umbral_max_f1(y: NDArray[np.int_], p: F) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return float(th[int(np.argmax(f[:-1]))])


def metricas(y: NDArray[np.int_], p: F, umbral: float) -> dict[str, float]:
    yh = (p >= umbral).astype(int)
    return {"n": int(y.size), "prevalencia": float(y.mean()), "auc_roc": float(roc_auc_score(y, p)),
            "auc_pr": float(average_precision_score(y, p)), "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh)), "f1": float(f1_score(y, yh)), "brier": float(brier_score_loss(y, p)),
            "umbral_prob": umbral}
