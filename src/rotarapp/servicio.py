"""Orquesta el modelo para una finca dibujada en la app: zona piloto → malla de 30 m → planes →
decisión → datos de pantalla. Lo usan la API (`scripts/api.py`) y el script de línea de comandos."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from rotarapp.comparador import decision as dc
from rotarapp.comparador.app import Constructor, Malla
from rotarapp.datos import power, prototipo
from rotarapp.datos.cultivos import Cultivo, Plagas, cargar_cultivos, cargar_plagas
from rotarapp.microbiota import Referencia

RAIZ = Path(__file__).resolve().parents[2]
PRIORIDADES = {"ingreso": "Ganar más", "agua": "Gastar menos agua", "suelo": "Cuidar el suelo", "erosion": "Evitar la erosión"}


@dataclass
class ZonaModelo:
    id: str
    nombre: str
    zona: prototipo.Zona
    clima: power.ClimaDecadal
    ref: Referencia
    anillo_lonlat: list[list[float]]

    def contiene(self, lon: float, lat: float) -> bool:
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", self.zona.crs, always_xy=True).transform(lon, lat)
        a, _, c, _, e, f = self.zona.transform
        return c <= x <= c + a * self.zona.ancho and f + e * self.zona.alto <= y <= f


def clima_zona(z: prototipo.Zona) -> power.ClimaDecadal:
    """Clima POWER del centro de la zona, corregido a la altitud mediana de sus cultivos."""
    from datetime import date as d_
    from pyproj import Transformer
    a, _, c, _, e, f = z.transform
    lon, lat = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True).transform(c + a * z.ancho / 2, f + e * z.alto / 2)
    elev = float(np.nanmedian(z.dem[z.cls == prototipo.CLASE_CULTIVO]))
    diario = power.descargar(round(lat, 2), round(lon, 2), d_(2001, 1, 1), d_(2025, 12, 31),
                             RAIZ / f"data/raw/power/{z.dir.name}.json")
    return power.a_decadas(diario.a_altitud(elev))


def cargar_zona_modelo(id_: str, nombre: str, dir_region: Path, ref_json: Path) -> ZonaModelo:
    from pyproj import Transformer
    z = prototipo.cargar_zona(dir_region)
    a, _, c, _, e, f = z.transform
    t = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True)
    esquinas = [(c, f), (c + a * z.ancho, f), (c + a * z.ancho, f + e * z.alto), (c, f + e * z.alto), (c, f)]
    anillo = [list(map(lambda v: round(v, 5), t.transform(x, y))) for x, y in esquinas]
    return ZonaModelo(id_, nombre, z, clima_zona(z), Referencia.desde_json(ref_json.read_text(encoding="utf-8")), anillo)


def _dentro(px: np.ndarray, py: np.ndarray, anillo: list[tuple[float, float]]) -> np.ndarray:
    ins = np.zeros(px.shape, bool)
    n = len(anillo)
    for i in range(n):
        x1, y1 = anillo[i]; x2, y2 = anillo[(i + 1) % n]
        ins ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
    return ins


def finca_y_malla(z: prototipo.Zona, poligono_lonlat: list[tuple[float, float]]) -> tuple[dc.Finca, Malla]:
    from pyproj import Transformer
    a, _, c, _, e, f = z.transform
    a_utm = Transformer.from_crs("EPSG:4326", z.crs, always_xy=True)
    anillo = [a_utm.transform(lon, lat) for lon, lat in poligono_lonlat]
    xs, ys = zip(*anillo)
    c0 = max(int((min(xs) - c) // a) - 1, 0); c1 = min(int((max(xs) - c) // a) + 2, z.ancho)
    f0 = max(int((max(ys) - f) // e) - 1, 0); f1 = min(int((min(ys) - f) // e) + 2, z.alto)
    if c0 >= c1 or f0 >= f1:
        raise ValueError("El lote está fuera de la zona con datos del modelo.")
    ff, cc = np.mgrid[f0:f1, c0:c1]
    x, y = z.xy(ff, cc)
    m = _dentro(x, y, anillo)
    if not m.any():
        raise ValueError("El lote es muy pequeño: dibuje un lote de al menos 30 × 30 m.")
    fi, co = ff[m], cc[m]
    cultivo = z.cls[fi, co] == prototipo.CLASE_CULTIVO
    if not cultivo.any():
        raise ValueError("Según la cobertura 2017–2023 este lote no es de cultivo (bosque, pasto o construcción).")
    a_ll = Transformer.from_crs(z.crs, "EPSG:4326", always_xy=True)
    xc, yc = z.xy(fi, co)
    h = abs(a) / 2
    anillos = []
    for xi, yi in zip(xc, yc):
        pts = [a_ll.transform(xi + dx, yi + dy) for dx, dy in ((-h, h), (h, h), (h, -h), (-h, -h))]
        anillos.append([[round(lon, 6), round(lat, 6)] for lon, lat in pts])
    lon_c, lat_c = a_ll.transform(xc, yc)
    ibs = ibs_relleno(z.ibs, fi, co)
    fc, cc_ = fi[cultivo], co[cultivo]
    finca = dc.Finca(fc.astype(np.int32), cc_.astype(np.int32), z.cap_clase[fc, cc_], z.ph[fc, cc_], z.dem[fc, cc_],
                     z.clay_pct[fc, cc_], z.soc_pct[fc, cc_], z.pendiente_pct[fc, cc_],
                     prototipo.awc_mm_por_m(z.clay_pct[fc, cc_]), z.peer[fc, cc_], ibs[cultivo])
    idx = np.full(fi.size, -1, np.int64)
    idx[cultivo] = np.arange(int(cultivo.sum()))
    return finca, Malla(anillos, np.column_stack([lon_c, lat_c]), cultivo, idx, ibs.astype(np.float32))


def ibs_relleno(ibs: np.ndarray, fi: np.ndarray, co: np.ndarray, radio: int = 2) -> np.ndarray:
    """IBS por celda; donde las nubes no dejaron dato a 30 m se usa el promedio de los vecinos a
    150 m (5 × 5 celdas). Si tampoco hay vecinos con dato, queda NaN."""
    out = ibs[fi, co].astype(np.float64)
    for k in np.nonzero(~np.isfinite(out))[0]:
        v = ibs[max(fi[k] - radio, 0):fi[k] + radio + 1, max(co[k] - radio, 0):co[k] + radio + 1]
        if np.isfinite(v).any():
            out[k] = float(np.nanmean(v))
    return out


def proxima_siembra(hoy: date) -> tuple[date, int]:
    """Próximo 21 de noviembre (segunda temporada de lluvias) y su década."""
    d = date(hoy.year, 11, 21)
    if d < hoy:
        d = date(hoy.year + 1, 11, 21)
    return d, power.decada_del_anio(d)


class Servicio:
    def __init__(self, dir_nasa_ibs: Path):
        self.cultivos: dict[str, Cultivo] = cargar_cultivos()
        self.plagas: Plagas = cargar_plagas()
        cal = RAIZ / "data/processed/modelo/calibracion"
        self.post = json.loads((cal / "posterior.json").read_text(encoding="utf-8"))
        self.metr = json.loads((cal / "metricas.json").read_text(encoding="utf-8"))
        self.metr["n_train"] = self.metr["metricas"]["train"]["n_segmentos"]
        mic = RAIZ / "data/processed/microbiota"
        self.zonas = [
            cargar_zona_modelo("palmira_candelaria", "Palmira – Candelaria", dir_nasa_ibs / "valle", mic / "palmira_candelaria/referencia_suelos.json"),
            cargar_zona_modelo("valle_norte", "Roldanillo – La Unión – Toro", dir_nasa_ibs / "valle_norte", mic / "valle_norte/referencia_suelos.json"),
        ]

    def catalogo(self) -> dict:
        todos = json.loads((RAIZ / "data/referencia/plagas/plagas_valle_del_cauca.json").read_text(encoding="utf-8"))
        nombres = todos["cultivos"]
        import csv
        filas = list(csv.DictReader(open(RAIZ / "data/referencia/cultivos/cultivos_atributos.csv", encoding="utf-8")))
        cultivos = [{"id": r["id"], "nombre": r["Cultivo"].split(" (")[0], "rotable": r["id"] in self.cultivos} for r in filas]
        plagas: dict[str, set[str]] = {}
        for p in todos["plagas"]:
            plagas.setdefault(p["plaga"], set()).update({p["cultivo_id"], *(p.get("otros_hospedantes") or [])})
        return {"cultivos": cultivos, "plagas": [{"nombre": k, "cultivos": sorted(v)} for k, v in sorted(plagas.items())],
                "prioridades": [{"id": k, "nombre": v} for k, v in PRIORIDADES.items()],
                "zonas": [{"id": z.id, "nombre": z.nombre, "anillo": z.anillo_lonlat} for z in self.zonas],
                "nombres_cultivos_json": nombres}

    def recomendar(self, req: dict, hoy: date | None = None) -> dict:
        t0 = time.time()
        pts = [(float(p["longitude"]), float(p["latitude"])) for p in req["poligono"]]
        lon_c, lat_c = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
        zm = next((z for z in self.zonas if z.contiene(lon_c, lat_c)), None)
        if zm is None:
            raise ValueError("Por ahora el modelo solo tiene datos para "
                             + " y ".join(z.nombre for z in self.zonas) + ". Dibuje su lote dentro de esas zonas.")
        finca, malla = finca_y_malla(zm.zona, pts)
        avisos: list[str] = []
        anteriores = [c for c in req.get("cultivos_anteriores", []) if c]
        anterior = next((c for c in anteriores if c in self.cultivos), None)
        if len(anteriores) > 1:
            avisos.append(f"Marcó varios cultivos anteriores; el modelo usa {self.cultivos[anterior].nombre.lower() if anterior else 'ninguno'} como el actual.")
        if anteriores and anterior is None:
            avisos.append("Su cultivo anterior es perenne o no está en la tabla: se comparan los planes sin la opción 'lo que hace hoy'.")
        prio = set(req.get("prioridades", []))
        pesos = {k: (2.0 if k in prio else 1.0) for k in PRIORIDADES}
        fecha, dec_ = proxima_siembra(hoy or date.today())
        entrada = dc.Entrada(anterior, list(req.get("plagas", [])), pesos, decada_siembra=dec_,
                             precios={k: float(v) for k, v in (req.get("precios_cop_kg") or {}).items()})
        esc = dc.escenarios(zm.clima, {k: np.array(v, np.float32) for k, v in self.post["muestras"].items()},
                            np.array(self.post["pesos"], np.float32), entrada.decada_siembra)
        t_media = float(np.nanmean(zm.clima.t_media))
        planes = dc.planes_candidatos(self.cultivos, entrada)
        evals = [dc.evaluar(p, finca, esc, self.cultivos, self.plagas, zm.ref, entrada, t_media) for p in planes]
        dec = dc.decidir(planes, evals, entrada)
        area = float(len(malla.modelada) * dc.HA_POR_PIXEL)
        datos = Constructor(dec, finca, malla, esc, self.cultivos, self.plagas, entrada, self.metr, fecha,
                            float(np.nanmean(zm.clima.lluvia.sum(axis=1)))).construir(area)
        if (~malla.modelada).any():
            avisos.append(f"{int((~malla.modelada).sum())} de {len(malla.modelada)} celdas no son de cultivo según la cobertura "
                          "2017–2023 (construcción, árboles o pasto); se muestran en gris.")
        avisos.append("El ingreso es bruto: todavía no descuenta costos de producción.")
        if anterior == "cana_azucar" and "cana_azucar" not in entrada.precios:
            avisos.append("Sin precio de la caña: el plan de hoy no tiene ingreso comparable.")
        val = self.metr["metricas"]["val"]
        return {
            "zona": zm.nombre, "area_ha": round(area, 2), "n_celdas": len(malla.modelada),
            "celdas": [{"ring": [{"latitude": la, "longitude": lo} for lo, la in r]} for r in malla.anillos],
            "modelada": malla.modelada.tolist(),
            "planes": datos["planes"], "suelo": datos["suelo"], "avisos": avisos,
            "entrada": {"cultivo_anterior": anterior, "plagas": entrada.plagas_anteriores, "pesos": pesos,
                        "siembra": fecha.isoformat()},
            "modelo": {"segundos": round(time.time() - t0, 1), "muestras": int(esc.lluvia.shape[1]),
                       "planes_evaluados": len(planes),
                       "validacion": {"rmse_ndvi": round(val["regresion_ndvi"]["simulador_bayesiano"]["rmse"], 3),
                                      "f1_baja_cobertura": round(val["clasificacion"]["baja_cobertura"]["simulador_bayesiano"]["f1"], 2)},
                       "aviso": "Herramienta de apoyo: no reemplaza el criterio del agricultor ni la asesoría agronómica local."},
        }
