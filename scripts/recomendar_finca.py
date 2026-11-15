"""Corre el modelo completo para una finca dibujada por el agricultor.

Uso:
    python scripts/recomendar_finca.py [entrada.json] [ruta NASA_IBS/data/valle]

entrada.json (lo que envía la app):
    {"nombre", "municipio", "poligono": GeoJSON Polygon en lon/lat, "cultivo_anterior",
     "plagas_anteriores": [...], "prioridades": {"ingreso", "agua", "suelo", "erosion"}}

Salida: data/processed/modelo/fincas/<nombre>/resultado.json y mapa.png
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from calibrar import clima_zona  # noqa: E402
from rotarapp.comparador import decision as dc  # noqa: E402
from rotarapp.comparador.explicaciones import explicar, nombre_plan  # noqa: E402
from rotarapp.datos import prototipo  # noqa: E402
from rotarapp.datos.cultivos import cargar_cultivos, cargar_plagas  # noqa: E402
from rotarapp.microbiota import Referencia  # noqa: E402


def dentro(px: np.ndarray, py: np.ndarray, poli: list[tuple[float, float]]) -> np.ndarray:
    """Punto en polígono (ray casting), vectorizado."""
    x, y = np.asarray(px), np.asarray(py)
    ins = np.zeros(x.shape, bool)
    n = len(poli)
    for i in range(n):
        x1, y1 = poli[i]; x2, y2 = poli[(i + 1) % n]
        cruza = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        ins ^= cruza
    return ins


def finca_desde_poligono(z: prototipo.Zona, poligono: dict) -> dc.Finca:
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", z.crs, always_xy=True)
    anillo = [t.transform(lon, lat) for lon, lat in poligono["coordinates"][0]]
    f, c = np.mgrid[0:z.alto, 0:z.ancho]
    x, y = z.xy(f, c)
    m = dentro(x, y, anillo) & (z.cls == prototipo.CLASE_CULTIVO)
    fi, co = np.nonzero(m)
    if fi.size == 0:
        raise ValueError("El polígono no cae sobre cultivo dentro de la zona con datos")
    return dc.Finca(fi.astype(np.int32), co.astype(np.int32), z.cap_clase[fi, co], z.ph[fi, co],
                    z.dem[fi, co], z.clay_pct[fi, co], z.soc_pct[fi, co], z.pendiente_pct[fi, co],
                    prototipo.awc_mm_por_m(z.clay_pct[fi, co]), z.peer[fi, co], z.ibs[fi, co])


def mapa(dec: dc.Decision, finca: dc.Finca, cultivos, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0, c0 = finca.fila.min(), finca.col.min()
    H, W = finca.fila.max() - f0 + 1, finca.col.max() - c0 + 1
    def grid(v):
        g = np.full((H, W), np.nan); g[finca.fila - f0, finca.col - c0] = v; return g
    keep = sorted(set(dec.asignacion.tolist()))
    top = [int(k) for k in np.argsort(-dec.p_mejor.sum(axis=1))[:3]]
    fig, ax = plt.subplots(1, 3 + len(top[:2]), figsize=(4 * (3 + len(top[:2])), 4))
    zonas = grid(np.array([keep.index(k) for k in dec.asignacion]))
    ax[0].imshow(zonas, cmap="Set2", vmin=0, vmax=7); ax[0].set_title("Zonas (estrategia C)")
    for j, k in enumerate(keep):
        ax[0].plot([], [], "s", color=plt.cm.Set2(j / 7), label=nombre_plan(dec.planes[k], cultivos))
    fig.legend(loc="lower left", fontsize=8, ncol=len(keep), frameon=False)
    im = ax[1].imshow(grid(finca.ibs_obs), cmap="YlGn", vmin=20, vmax=80); ax[1].set_title("IBS observado hoy")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(grid(np.array([str(k) for k in finca.clase]) == "4"), cmap="Oranges", vmin=0, vmax=1.5)
    ax[2].set_title("Clase IGAC 4 (naranja)")
    for j, k in enumerate(top[:2]):
        im = ax[3 + j].imshow(grid(dec.p_mejor[k]), cmap="magma", vmin=0, vmax=1)
        ax[3 + j].set_title(f"P(mejor): {nombre_plan(dec.planes[k], cultivos)}", fontsize=8)
        plt.colorbar(im, ax=ax[3 + j], fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(ruta, dpi=120); plt.close(fig)


def main() -> None:
    t0 = time.time()
    args = sys.argv[1:]
    entrada_json = json.loads(Path(args[0] if args else RAIZ / "data/ejemplos/finca_demo.json").read_text(encoding="utf-8"))
    dir_region = Path(args[1]) if len(args) > 1 else Path.home() / "NASA_IBS/data/valle"

    z = prototipo.cargar_zona(dir_region)
    clima = clima_zona(z)
    cultivos, plagas = cargar_cultivos(), cargar_plagas()
    finca = finca_desde_poligono(z, entrada_json["poligono"])
    calib = json.loads((RAIZ / "data/processed/modelo/calibracion/posterior.json").read_text(encoding="utf-8"))
    metr = json.loads((RAIZ / "data/processed/modelo/calibracion/metricas.json").read_text(encoding="utf-8"))
    metr["n_train"] = metr["metricas"]["train"]["n_segmentos"]
    ref = Referencia.desde_json((RAIZ / "data/processed/microbiota/palmira_candelaria/referencia_suelos.json").read_text(encoding="utf-8"))

    entrada = dc.Entrada(entrada_json.get("cultivo_anterior"), entrada_json.get("plagas_anteriores", []),
                         entrada_json.get("prioridades") or dc.Entrada().pesos,
                         precios=entrada_json.get("precios_cop_kg") or {})
    esc = dc.escenarios(clima, {k: np.array(v, np.float32) for k, v in calib["muestras"].items()},
                        np.array(calib["pesos"], np.float32), entrada.decada_siembra)
    t_media = float(np.nanmean(clima.t_media))
    planes = dc.planes_candidatos(cultivos, entrada)
    evals = [dc.evaluar(p, finca, esc, cultivos, plagas, ref, entrada, t_media) for p in planes]
    dec = dc.decidir(planes, evals, entrada)
    expl = explicar(dec, cultivos, plagas, entrada, metr, finca)

    from pyproj import Transformer
    tr = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(*z.xy(finca.fila, finca.col))
    orden = np.argsort(-dec.p_mejor.sum(axis=1))[:5]
    ranking = [{"plan": dec.planes[k].id, "nombre": nombre_plan(dec.planes[k], cultivos),
                "p_mejor_finca": round(float(dec.p_mejor[k].mean()), 3),
                "ingreso_p50": float(np.nanmedian(evals[k].ingreso)),
                "riego_p50": float(np.nanmedian(evals[k].riego)), "ibs_p50": float(np.nanmedian(evals[k].ibs)),
                "erosion_p50": float(np.nanmedian(evals[k].erosion)),
                "rendimiento_p50_kg_ha": {c: float(np.nanmedian(v)) for c, v in evals[k].rendimientos.items()}}
               for k in orden]
    sin_precio = [{"plan": e.plan.id, "nombre": nombre_plan(e.plan, cultivos),
                   "nota": "ingreso sin dato: falta el precio de " + ", ".join(
                       cultivos[c].nombre.lower() for c in e.rendimientos if entrada.precio(cultivos[c]) is None),
                   "riego_p50": float(np.nanmedian(e.riego)), "ibs_p50": float(np.nanmedian(e.ibs)),
                   "erosion_p50": float(np.nanmedian(e.erosion)),
                   "rendimiento_p50_kg_ha": {c: float(np.nanmedian(v)) for c, v in e.rendimientos.items()}}
                  for e in evals if not np.isfinite(e.ingreso).any()]
    resultado = {
        "finca": {k: entrada_json.get(k) for k in ("nombre", "municipio", "cultivo_anterior", "plagas_anteriores", "prioridades")},
        "area_ha": round(finca.n * dc.HA_POR_PIXEL, 2), "n_pixeles": finca.n,
        "clases_igac": {str(k): int(v) for k, v in zip(*np.unique(finca.clase.astype(str), return_counts=True))},
        "estrategias": {k: {**v, "nombres": [nombre_plan(dec.planes[next(i for i, p in enumerate(dec.planes) if p.id == pid)], cultivos) for pid in v["planes"]],
                            "explicacion": expl[k]} for k, v in dec.estrategias.items()},
        "zonas": [{**zz, "nombre": nombre_plan(dec.planes[next(i for i, p in enumerate(dec.planes) if p.id == zz["plan"])], cultivos)} for zz in dec.zonas],
        "ranking_planes": ranking,
        "planes_sin_precio": sin_precio,
        "pixeles": [{"lon": round(float(a), 6), "lat": round(float(b), 6), "plan": dec.planes[dec.asignacion[i]].id,
                     "p_mejor": {dec.planes[k].id: round(float(dec.p_mejor[k, i]), 3) for k in orden if dec.p_mejor[k, i] > 0}}
                    for i, (a, b) in enumerate(zip(lon, lat))],
        "modelo": {"muestras": int(esc.lluvia.shape[1]), "escenarios_climaticos": int(esc.lluvia.shape[1] // 4),
                   "planes_evaluados": len(planes), "calibracion_val": metr["metricas"]["val"],
                   "aviso": "Herramienta de apoyo: no reemplaza el criterio del agricultor ni la asesoría agronómica local."},
    }
    nombre = re.sub(r"[^a-z0-9]+", "_", (entrada_json.get("nombre") or "finca").lower()).strip("_")[:40]
    salida = RAIZ / "data/processed/modelo/fincas" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    mapa(dec, finca, cultivos, salida / "mapa.png")

    print(f"{finca.n} píxeles ({resultado['area_ha']} ha), clases {resultado['clases_igac']}, {len(planes)} planes, {time.time() - t0:.0f}s")
    for r in ranking:
        print(f"  {r['nombre']:45s} P={r['p_mejor_finca']:.2f} ingreso={r['ingreso_p50']/1e6:6.1f}M riego={r['riego_p50']:5.0f} IBS={r['ibs_p50']:4.1f} eros={r['erosion_p50']:.2f} rend={ {k: round(v) for k, v in r['rendimiento_p50_kg_ha'].items()} }")
    for r in sin_precio[:8]:
        print(f"  (sin precio) {r['nombre']:45s} riego={r['riego_p50']:5.0f} IBS={r['ibs_p50']:4.1f} eros={r['erosion_p50']:.2f}")
    for k, v in dec.estrategias.items():
        mm = v["metricas"]
        print(f"[{k}] {v['nombre']}: {resultado['estrategias'][k]['nombres']}")
        print("     " + " | ".join(f"{m}: {mm[m]['p50']:.4g} ({mm[m]['p10']:.3g}–{mm[m]['p90']:.3g})" for m in mm))
    print("zonas:", [(zz["nombre"], zz["area_ha"]) for zz in resultado["zonas"]], "P(C>B)=", dec.estrategias["C"].get("p_mejor_que_B"))
    print(salida)


if __name__ == "__main__":
    main()
