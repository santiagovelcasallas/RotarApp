"""Orquesta el modelo para una finca dibujada en la app: zona piloto → malla de 30 m → planes →
decisión → datos de pantalla. Lo usan la API (`scripts/api.py`) y el script de línea de comandos."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from rotarapp.comparador import decision as dc
from rotarapp.comparador.app import Constructor, Malla
from rotarapp.datos import power, prototipo, valle
from rotarapp.filtro.aptitud import rotacion_permitida
from rotarapp.datos.cultivos import Cultivo, Plagas, cargar_cultivos, cargar_plagas
from rotarapp.microbiota import Referencia

RAIZ = Path(__file__).resolve().parents[2]
MAX_CELDAS = 1200   # ~108 ha de cultivo a 30 m
MIN_FRAC_APTA = 0.25  # si menos de un cuarto del lote admite transitorios, no se recomienda rotación
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
    cultivo = cultivo & _con_vocacion(z.cap_clase[fi, co])
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


def _con_vocacion(clase: np.ndarray) -> np.ndarray:
    """Celdas cuya clase IGAC admite cultivos transitorios (plena o restringida). Si ninguna la
    admite, explica por qué en vez de recomendar una rotación."""
    ok = np.isin(rotacion_permitida(clase), ["plena", "restringida"])
    if ok.mean() < MIN_FRAC_APTA:
        # Clases IGAC 5-8 (montaña y occidente): mensaje definido por el equipo
        raise ValueError("Estas zonas aún están en proceso de identificación.")
    return ok


def finca_y_malla_valle(v: valle.DatosValle, poligono_lonlat: list[tuple[float, float]],
                        ibs_estimado: "Callable[[np.ndarray], np.ndarray]") -> tuple[dc.Finca, Malla, float]:
    """Finca para cualquier lote del Valle con las capas generales; el IBS se estima con la
    muestra de la zona piloto más cercana (ver Servicio.ibs_por_clase)."""
    lon, lat, _, _, anillos = valle.malla_utm(poligono_lonlat)
    if lon.size == 0:
        raise ValueError("El lote es muy pequeño: dibuje un lote de al menos 30 × 30 m.")
    clase = v.clase(lon, lat)
    if (clase == "").all():
        raise ValueError("El lote queda fuera del Valle del Cauca.")
    cultivo = ~np.isin(clase.astype(str), list(valle.NO_CULTIVO))
    if not cultivo.any():
        raise ValueError("Según el mapa del IGAC el lote es cuerpo de agua o zona urbana.")
    cultivo = cultivo & _con_vocacion(clase)
    elev = v.elevacion(lon, lat)
    clay, ph, soc = v.suelo(lon, lat)
    pend = v.pendiente(lon, lat)
    m = cultivo
    n = int(m.sum())
    ibs = ibs_estimado(clase).astype(np.float32)
    finca = dc.Finca(np.arange(n, dtype=np.int32), np.zeros(n, np.int32), clase[m], ph[m].astype(np.float32),
                     elev[m].astype(np.float32), clay[m].astype(np.float32), soc[m].astype(np.float32),
                     pend[m].astype(np.float32), prototipo.awc_mm_por_m(clay[m]),
                     np.full(n, valle.GRUPO_GENERAL, np.int32), ibs[m])
    idx = np.full(lon.size, -1, np.int64)
    idx[m] = np.arange(n)
    malla = Malla(anillos, np.column_stack([lon, lat]), m, idx, np.where(m, ibs, np.nan).astype(np.float32))
    return finca, malla, float(np.nanmedian(elev[m]))


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
        self.valle = valle.cargar(RAIZ / "data/referencia/valle")
        self.ibs_muestra = {z.id: self._ibs_por_clase(z) for z in self.zonas}

    @staticmethod
    def _ibs_por_clase(zm: ZonaModelo) -> tuple[tuple[float, float], dict[str, float], float]:
        """Centro (lon, lat) y mediana del IBS observado en cultivo por clase IGAC de una zona piloto."""
        z = zm.zona
        ok = (z.cls == prototipo.CLASE_CULTIVO) & np.isfinite(z.ibs)
        clases = z.cap_clase[ok].astype(str)
        vals = z.ibs[ok]
        por_clase = {c: float(np.median(vals[clases == c])) for c in np.unique(clases) if (clases == c).sum() >= 200}
        lons = [p[0] for p in zm.anillo_lonlat]; lats = [p[1] for p in zm.anillo_lonlat]
        return (float(np.mean(lons)), float(np.mean(lats))), por_clase, float(np.median(vals))

    def ibs_estimado(self, lon: float, lat: float) -> tuple[str, "Callable[[np.ndarray], np.ndarray]"]:
        """IBS típico por clase IGAC en la zona piloto más cercana: no es el IBS de ese lote, es una
        estimación con una muestra de la misma región del Valle."""
        zid = min(self.ibs_muestra, key=lambda k: (self.ibs_muestra[k][0][0] - lon) ** 2 + (self.ibs_muestra[k][0][1] - lat) ** 2)
        _, por_clase, general = self.ibs_muestra[zid]
        nombre = next(z.nombre for z in self.zonas if z.id == zid)
        return nombre, lambda clase: np.array([por_clase.get(str(c), general) for c in clase], np.float64)

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
        avisos: list[str] = []
        if zm is not None:
            finca, malla = finca_y_malla(zm.zona, pts)
            clima, ref, nombre_zona = zm.clima, zm.ref, zm.nombre
            excluidas = "no son de cultivo según la cobertura 2017–2023 o su clase IGAC no admite transitorios"
        elif self.valle is not None and self.valle.en_valle(lon_c, lat_c):
            zona_ibs, f_ibs = self.ibs_estimado(lon_c, lat_c)
            finca, malla, elev = finca_y_malla_valle(self.valle, pts, f_ibs)
            clima, ref, nombre_zona = self.valle.clima(lon_c, lat_c, elev), self.valle.ref, "Valle del Cauca"
            excluidas = "son agua, zona urbana o de una clase IGAC que no admite transitorios"
            avisos.append(f"Fuera de las zonas piloto: suelo de SoilGrids (250 m), relieve SRTM y clima NASA POWER corregido a "
                          f"{elev:.0f} m. La salud del suelo (IBS) se estima con la muestra de {zona_ibs}, la zona piloto más cercana, según la clase IGAC de cada celda.")
        else:
            raise ValueError("Por ahora RotarApp solo funciona en el Valle del Cauca. Dibuje su lote dentro del departamento.")
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
        if finca.n > MAX_CELDAS:
            raise ValueError(f"El lote es muy grande para el piloto (máximo ~{MAX_CELDAS * dc.HA_POR_PIXEL:.0f} ha de cultivo). "
                             "Divídalo en lotes más pequeños.")
        # Muestras del posterior por escenario climático según el tamaño, para responder en < 30 s
        k_post = 4 if finca.n <= 150 else 2 if finca.n <= 400 else 1
        esc = dc.escenarios(clima, {k: np.array(v, np.float32) for k, v in self.post["muestras"].items()},
                            np.array(self.post["pesos"], np.float32), entrada.decada_siembra, k_post=k_post)
        t_media = float(np.nanmean(clima.t_media))
        planes = dc.planes_candidatos(self.cultivos, entrada)
        evals = [dc.evaluar(p, finca, esc, self.cultivos, self.plagas, ref, entrada, t_media) for p in planes]
        dec = dc.decidir(planes, evals, entrada)
        area = float(len(malla.modelada) * dc.HA_POR_PIXEL)
        datos = Constructor(dec, finca, malla, esc, self.cultivos, self.plagas, entrada, self.metr, fecha,
                            float(np.nanmean(clima.lluvia.sum(axis=1)))).construir(area)
        if (~malla.modelada).any():
            avisos.append(f"{int((~malla.modelada).sum())} de {len(malla.modelada)} celdas {excluidas}; se muestran en gris.")
        avisos.append("El ingreso es bruto: todavía no descuenta costos de producción.")
        if anterior == "cana_azucar" and "cana_azucar" not in entrada.precios:
            avisos.append("Sin precio de la caña: el plan de hoy no tiene ingreso comparable.")
        val = self.metr["metricas"]["val"]
        return {
            "zona": nombre_zona, "area_ha": round(area, 2), "n_celdas": len(malla.modelada),
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
