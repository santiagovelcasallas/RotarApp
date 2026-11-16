"""Entrena (calibra), valida y prueba el simulador con las curvas de rebrote de caña.

Uso:
    python scripts/calibrar.py [ruta NASA_IBS/data/valle] [--prueba] [--externa ruta/valle_norte]

Sin --prueba solo reporta entrenamiento y validación (para ajustar el modelo sin mirar la
prueba). La prueba se corre una vez, al final, con --prueba. --externa evalúa además en otra
zona que nunca se usó para ajustar nada.

Reporta la regresión del NDVI (RMSE, R², cobertura del intervalo 80 %) y la clasificación del
evento que mide el IBS: lote sin dosel (NDVI < 0,30) y con baja cobertura (NDVI < 0,45), con
AUC-ROC, AUC-PR, precisión, recall, F1 y Brier; umbrales de decisión elegidos en validación.

Salidas en data/processed/modelo/calibracion/: posterior.json y metricas.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from rotarapp.datos import power, prototipo  # noqa: E402
from rotarapp.datos.cultivos import cargar_cultivos  # noqa: E402
from rotarapp.simulador import calibracion as cal  # noqa: E402
from rotarapp.simulador import evaluacion as ev  # noqa: E402
from rotarapp.simulador import linea_base as lb  # noqa: E402
from rotarapp.simulador import observaciones as ob  # noqa: E402

M_PRIOR = 2000


def clima_zona(z: prototipo.Zona) -> power.ClimaDecadal:
    """Clima POWER del centro de la zona, corregido a la altitud mediana de sus cultivos."""
    a, _, c, _, e, f = z.transform
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True).transform(
        c + a * z.ancho / 2, f + e * z.alto / 2)
    elev = float(np.nanmedian(z.dem[z.cls == prototipo.CLASE_CULTIVO]))
    diario = power.descargar(round(lat, 2), round(lon, 2), date(2001, 1, 1), date(2025, 12, 31),
                             RAIZ / f"data/raw/power/{z.dir.name}.json")
    return power.a_decadas(diario.a_altitud(elev))


def evaluar(s: ob.Segmentos, clima_s: dict, post: cal.Posterior, cana, clim: np.ndarray, gbm, sigma_gbm: float,
            p_clim: dict[str, np.ndarray], umbrales: dict[str, float]) -> dict:
    """Regresión del NDVI y clasificación del suelo sin dosel para un conjunto."""
    media, mu = cal.predictiva(s, clima_s, post, cana)
    sig = post.sigma * post.inflacion
    lo, hi = cal.cuantiles(mu, sig)
    obs = s.obs.T
    pred_gbm = lb.predecir_gbm(gbm, s, clima_s).T
    out: dict = {"n_segmentos": len(s), "regresion_ndvi": {
        "simulador_bayesiano": cal.metricas(media, obs, lo, hi),
        "climatologia": cal.metricas(np.broadcast_to(clim[:, None], obs.shape), obs),
        "gradient_boosting": cal.metricas(pred_gbm, obs)}, "clasificacion": {}}
    for nombre, u in ev.UMBRALES_NDVI.items():
        y, ok = ev.etiquetas(obs, u)
        probs = {"simulador_bayesiano": ev.prob_bajo(mu, u, sig)[ok],
                 "climatologia": np.broadcast_to(p_clim[nombre][:, None], obs.shape)[ok],
                 "gradient_boosting": ev.prob_bajo_punto(pred_gbm, u, sigma_gbm)[ok]}
        out["clasificacion"][nombre] = {m: ev.metricas(y, p, umbrales[f"{nombre}:{m}"]) for m, p in probs.items()}
    return out


def imprimir(k: str, r: dict) -> None:
    print(f"\n== {k} ({r['n_segmentos']} segmentos)")
    for mod, mm in r["regresion_ndvi"].items():
        extra = f" cob80={mm['cobertura_80']:.2f}" if "cobertura_80" in mm else ""
        print(f"  NDVI           {mod:20s} RMSE={mm['rmse']:.4f} R2={mm['r2']:.3f}{extra}")
    for ev_n, d in r["clasificacion"].items():
        for mod, mm in d.items():
            print(f"  {ev_n:14s} {mod:20s} prev={mm['prevalencia']:.3f} AUC-ROC={mm['auc_roc']:.3f} "
                  f"AUC-PR={mm['auc_pr']:.3f} P={mm['precision']:.3f} R={mm['recall']:.3f} "
                  f"F1={mm['f1']:.3f} Brier={mm['brier']:.4f}")


def main() -> None:
    t0 = time.time()
    argv = sys.argv[1:]
    externa = Path(argv[argv.index("--externa") + 1]) if "--externa" in argv else None
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--externa")]
    con_prueba = "--prueba" in argv
    dir_region = Path(args[0]) if args else Path.home() / "NASA_IBS/data/valle"
    z = prototipo.cargar_zona(dir_region)
    clima = clima_zona(z)
    cana = cargar_cultivos()["cana_azucar"]
    seg = ob.construir(z, clima)
    conj = {k: seg.sub(k) for k in ("train", "val", "test_t", "test_e")}
    cl = {k: ob.clima_de_segmentos(s, clima) for k, s in conj.items()}
    print({k: len(s) for k, s in conj.items()}, f"{time.time() - t0:.0f}s")

    # --- Entrenamiento: posterior ---
    tr = conj["train"]
    prior = cal.muestrear_prior(M_PRIOR)
    sse, n = cal.sse_por_muestra(tr, cl["train"], prior, cana)
    n_ef = len({(int(b), int(a)) for b, a in zip(tr.bloque, tr.anio)})  # unidades independientes
    w = cal.pesos_posterior(sse, n, n_ef=n_ef)
    post0 = cal.Posterior(prior, w, sigma=0.0, inflacion=1.0)
    media_tr, _ = cal.predictiva(tr, cl["train"], post0, cana)
    sigma = float(np.nanstd(media_tr.T - tr.obs))
    print(f"posterior: n_ef={n_ef} ESS={post0.ess:.1f} de {M_PRIOR}, sigma={sigma:.3f}  {time.time() - t0:.0f}s")

    # --- Validación: ancho del intervalo y umbrales de decisión ---
    _, mu_v = cal.predictiva(conj["val"], cl["val"], post0, cana)
    c = cal.ajustar_inflacion(mu_v, conj["val"].obs.T, sigma)
    post = cal.Posterior(prior, w, sigma=sigma, inflacion=c)
    clim = lb.climatologia(tr)
    gbm = lb.entrenar_gbm(tr, cl["train"])
    sigma_gbm = float(np.nanstd(lb.predecir_gbm(gbm, tr, cl["train"]) - tr.obs))
    p_clim = {k: ev.prob_climatologia(tr.obs.T, u) for k, u in ev.UMBRALES_NDVI.items()}
    _, mu_v = cal.predictiva(conj["val"], cl["val"], post, cana)
    obs_v = conj["val"].obs.T
    pg_v = lb.predecir_gbm(gbm, conj["val"], cl["val"]).T
    umbrales: dict[str, float] = {}
    for nombre, u in ev.UMBRALES_NDVI.items():
        y, ok = ev.etiquetas(obs_v, u)
        candidatos = {"simulador_bayesiano": ev.prob_bajo(mu_v, u, sigma * c)[ok],
                      "climatologia": np.broadcast_to(p_clim[nombre][:, None], obs_v.shape)[ok],
                      "gradient_boosting": ev.prob_bajo_punto(pg_v, u, sigma_gbm)[ok]}
        for m, p in candidatos.items():
            umbrales[f"{nombre}:{m}"] = ev.umbral_max_f1(y, p)

    metr: dict[str, dict] = {}
    for k in ("train", "val", "test_t", "test_e") if con_prueba else ("train", "val"):
        if len(conj[k]):
            metr[k] = evaluar(conj[k], cl[k], post, cana, clim, gbm, sigma_gbm, p_clim, umbrales)
    if externa is not None:
        ze = prototipo.cargar_zona(externa)
        ce = clima_zona(ze)
        se = ob.construir(ze, ce)
        metr[f"externa_{externa.name}"] = evaluar(se, ob.clima_de_segmentos(se, ce), post, cana, clim, gbm,
                                                  sigma_gbm, p_clim, umbrales)

    salida = RAIZ / "data/processed/modelo/calibracion"
    salida.mkdir(parents=True, exist_ok=True)
    top = np.argsort(-w)[:200]
    (salida / "posterior.json").write_text(json.dumps({
        "cultivo": "cana_azucar", "n_ef": n_ef, "prior": cal.PRIOR, "ess": post.ess, "sigma": sigma, "inflacion": c,
        "resumen": post.resumen(),
        "muestras": {k: v[top].round(4).tolist() for k, v in prior.items()},
        "pesos": (w[top] / w[top].sum()).round(6).tolist(),
    }, indent=1), encoding="utf-8")
    (salida / "metricas.json").write_text(json.dumps({
        "particion": {"bloques_px": prototipo.BLOQUE_PX, "frac_prueba_espacial": 0.3, "semilla": 42,
                      "train": "bloques de entrenamiento, segmentos que inician 2020-2023",
                      "val": "bloques de entrenamiento, 2024", "test_t": "bloques de entrenamiento, 2025",
                      "test_e": "bloques de prueba espacial, 2020-2025",
                      "externa": "otra zona (Valle norte), todos los segmentos; nunca usada para ajustar"},
        "umbrales_clasificacion_elegidos_en_val": umbrales,
        "metricas": metr}, indent=1, ensure_ascii=False), encoding="utf-8")

    print({k: (round(v["p10"], 3), round(v["p50"], 3), round(v["p90"], 3)) for k, v in post.resumen().items()})
    for k, r in metr.items():
        imprimir(k, r)
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
