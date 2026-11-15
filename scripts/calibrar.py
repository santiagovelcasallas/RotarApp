"""Entrena (calibra), valida y prueba el simulador con las curvas de rebrote de caña.

Uso:
    python scripts/calibrar.py [ruta NASA_IBS/data/valle] [--prueba]

Sin --prueba solo reporta entrenamiento y validación (para ajustar el modelo sin mirar la
prueba). La prueba se corre una vez, al final, con --prueba.

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
from rotarapp.simulador import linea_base as lb  # noqa: E402
from rotarapp.simulador import observaciones as ob  # noqa: E402

M_PRIOR = 2000


def clima_zona(z: prototipo.Zona) -> power.ClimaDecadal:
    a, _, c, _, e, f = z.transform
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True).transform(
        c + a * z.ancho / 2, f + e * z.alto / 2)
    elev = float(np.nanmedian(z.dem[z.cls == prototipo.CLASE_CULTIVO]))
    diario = power.descargar(round(lat, 2), round(lon, 2), date(2001, 1, 1), date(2025, 12, 31),
                             RAIZ / "data/raw/power/palmira_candelaria.json")
    return power.a_decadas(diario.a_altitud(elev))


def main() -> None:
    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    con_prueba = "--prueba" in sys.argv
    dir_region = Path(args[0]) if args else Path.home() / "NASA_IBS/data/valle"
    z = prototipo.cargar_zona(dir_region)
    clima = clima_zona(z)
    cana = cargar_cultivos()["cana_azucar"]
    seg = ob.construir(z, clima)
    conj = {k: seg.sub(k) for k in ("train", "val", "test_t", "test_e")}
    cl = {k: ob.clima_de_segmentos(s, clima) for k, s in conj.items()}
    print({k: len(s) for k, s in conj.items()}, f"{time.time() - t0:.0f}s")

    # --- Entrenamiento: posterior ---
    prior = cal.muestrear_prior(M_PRIOR)
    sse, n = cal.sse_por_muestra(conj["train"], cl["train"], prior, cana)
    tr = conj["train"]
    n_ef = len({(int(b), int(a)) for b, a in zip(tr.bloque, tr.anio)})  # unidades independientes
    w = cal.pesos_posterior(sse, n, n_ef=n_ef)
    post0 = cal.Posterior(prior, w, sigma=0.0, inflacion=1.0)
    media_tr, _ = cal.predictiva(conj["train"], cl["train"], post0, cana)
    sigma = float(np.nanstd(media_tr.T - conj["train"].obs))
    print(f"posterior: n_ef={n_ef} ESS={post0.ess:.1f} de {M_PRIOR}, σ={sigma:.3f}  {time.time() - t0:.0f}s")

    # --- Validación: inflación del intervalo ---
    media_v, mu_v = cal.predictiva(conj["val"], cl["val"], post0, cana)
    obs_v = conj["val"].obs.T[None]
    c = cal.ajustar_inflacion(mu_v, np.broadcast_to(obs_v, mu_v.shape)[0], sigma)
    post = cal.Posterior(prior, w, sigma=sigma, inflacion=c)

    # --- Líneas base ---
    clim = lb.climatologia(conj["train"])
    gbm = lb.entrenar_gbm(conj["train"], cl["train"])

    metr: dict[str, dict[str, dict[str, float]]] = {}
    for k in ("train", "val", "test_t", "test_e") if con_prueba else ("train", "val"):
        s = conj[k]
        if len(s) == 0:
            continue
        media, mu = cal.predictiva(s, cl[k], post, cana)
        lo, hi = cal.cuantiles(mu, sigma * c)
        obs = s.obs.T
        metr[k] = {
            "simulador_bayesiano": cal.metricas(media, obs, lo, hi),
            "climatologia": cal.metricas(np.broadcast_to(clim[:, None], obs.shape), obs),
            "gradient_boosting": cal.metricas(lb.predecir_gbm(gbm, s, cl[k]).T, obs),
            "n_segmentos": {"n": len(s)},
        }

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
                      "test_e": "bloques de prueba espacial, 2020-2025"},
        "metricas": metr}, indent=1, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(post.resumen(), indent=1))
    for k, v in metr.items():
        print(f"\n== {k} ({v['n_segmentos']['n']} segmentos)")
        for mod in ("simulador_bayesiano", "climatologia", "gradient_boosting"):
            mm = v[mod]
            extra = f" cob80={mm['cobertura_80']:.2f} ancho={mm['ancho_80']:.3f}" if "cobertura_80" in mm else ""
            print(f"  {mod:20s} RMSE={mm['rmse']:.4f} MAE={mm['mae']:.4f} sesgo={mm['sesgo']:+.4f} R²={mm['r2']:.3f}{extra}")
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
