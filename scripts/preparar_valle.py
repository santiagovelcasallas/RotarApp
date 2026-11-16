"""Datos generales para todo el Valle del Cauca (fuera de las zonas piloto con HLS).

    python scripts/preparar_valle.py

Escribe data/referencia/valle/:
  dem.npz      elevación (m) y pendiente (%) de las teselas Terrarium (SRTM/NASADEM, ~76 m, zoom 11)
  suelo.npz    arcilla (g/kg), pH×10 y carbono orgánico (dg/kg) de SoilGrids 250 m, promedio 0-30 cm
  igac.npz     polígonos de capacidad de uso IGAC 2006 (1:100.000) en formato compacto
  clima/*.npz  NASA POWER diario 2001-2025 en una rejilla de 0,5°
  referencia_general.json  "lo normal" del IBS para suelos sin grupo (mediana de las zonas piloto)
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from rotarapp.datos import power  # noqa: E402

OUT = RAIZ / "data/referencia/valle"
LON0, LON1, LAT0, LAT1 = -77.6, -75.6, 2.9, 5.1
Z = 11
UA = {"User-Agent": "Mozilla/5.0 RotarApp-preparacion"}
ROMANOS = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8"}


def get(url: str, intentos: int = 4) -> bytes:
    for i in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == intentos - 1:
                raise
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def tile_xy(lon: float, lat: float) -> tuple[float, float]:
    n = 2 ** Z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def dem() -> None:
    from PIL import Image
    x0, y0 = (int(v) for v in tile_xy(LON0, LAT1))
    x1, y1 = (int(v) for v in tile_xy(LON1, LAT0))
    W, H = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    z = np.zeros((H, W), np.float32)
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            img = np.asarray(Image.open(io.BytesIO(get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{tx}/{ty}.png"))).convert("RGB"), np.float32)
            z[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = img[..., 0] * 256 + img[..., 1] + img[..., 2] / 256 - 32768
    lat_c = (LAT0 + LAT1) / 2
    px_m = 156543.03392 * math.cos(math.radians(lat_c)) / 2 ** Z
    gy, gx = np.gradient(z, px_m)
    pend = 100 * np.hypot(gx, gy)
    np.savez_compressed(OUT / "dem.npz", elev=np.round(z).astype(np.int16), pendiente_x10=np.clip(pend * 10, 0, 32000).astype(np.int16),
                        zoom=Z, x0=x0, y0=y0, px_m=px_m)
    print(f"dem: {W}×{H} px, {px_m:.0f} m/px, elev {z.min():.0f}–{z.max():.0f} m")


def suelo() -> None:
    import rasterio
    capas: dict[str, np.ndarray] = {}
    meta = None
    for prop in ("clay", "phh2o", "soc"):
        acc, pesos = None, 0
        for prof, w in (("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)):
            url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
                   f"&COVERAGEID={prop}_{prof}_mean&FORMAT=image/tiff&SUBSET=long({LON0},{LON1})&SUBSET=lat({LAT0},{LAT1})"
                   "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")
            with rasterio.MemoryFile(get(url)) as mf, mf.open() as ds:
                a = ds.read(1).astype(np.float32)
                a[a <= 0] = np.nan
                meta = (ds.transform, a.shape)
            acc = a * w if acc is None else acc + a * w
            pesos += w
        capas[prop] = (acc / pesos).astype(np.float32)
        print(f"suelo {prop}: mediana {np.nanmedian(capas[prop]):.1f}")
    tr, shape = meta
    np.savez_compressed(OUT / "suelo.npz", clay=capas["clay"], phh2o=capas["phh2o"], soc=capas["soc"],
                        transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]), shape=np.array(shape))


def igac() -> None:
    fc = json.loads((RAIZ / "data/referencia/igac/capacidad_uso_valle.geojson").read_text(encoding="utf-8"))
    coords, inicio, bbox, clase = [], [0], [], []
    for f in fc["features"]:
        anillo = np.array(f["geometry"]["coordinates"][0], np.float32)
        coords.append(anillo); inicio.append(inicio[-1] + len(anillo))
        bbox.append([anillo[:, 0].min(), anillo[:, 1].min(), anillo[:, 0].max(), anillo[:, 1].max()])
        c = str(f["properties"].get("CLASE") or "")
        clase.append(ROMANOS.get(c, c))
    np.savez_compressed(OUT / "igac.npz", coords=np.concatenate(coords), inicio=np.array(inicio, np.int64),
                        bbox=np.array(bbox, np.float32), clase=np.array(clase))
    print(f"igac: {len(clase)} polígonos")


def clima() -> None:
    d = OUT / "clima"
    d.mkdir(exist_ok=True)
    vistos = set()
    for lat in np.arange(3.0, 5.01, 0.5):
        for lon in np.arange(-77.5, -75.6, 0.625):
            cache = RAIZ / f"data/raw/power/grid/{lat:.2f}_{lon:.3f}.json"
            c = power.descargar(float(lat), float(lon), date(2001, 1, 1), date(2025, 12, 31), cache)
            datos = json.loads(cache.read_text(encoding="utf-8"))
            clon, clat, elev = datos["geometry"]["coordinates"]
            clave = (round(clon, 3), round(clat, 3))
            if clave in vistos:
                continue
            vistos.add(clave)
            np.savez_compressed(d / f"{clat:.3f}_{clon:.3f}.npz", lon=clon, lat=clat, elev=elev,
                                fechas=np.array([x.toordinal() for x in c.fechas], np.int32),
                                **{k: getattr(c, k).astype(np.float32) for k in ("t2m", "tmax", "tmin", "hr", "lluvia", "rad", "toa")})
            print(f"clima: celda {clat:.2f}, {clon:.3f} ({elev:.0f} m)", flush=True)


def referencia() -> None:
    comps: dict[str, list[tuple[float, float]]] = {}
    for z in ("palmira_candelaria", "valle_norte"):
        ref = json.loads((RAIZ / f"data/processed/microbiota/{z}/referencia_suelos.json").read_text(encoding="utf-8"))
        for g in ref.values():
            if g["nombre"] in ("CA", "HAC", "ZU", "TA", "Q1"):
                continue
            for k, v in g["mediana"].items():
                comps.setdefault(k, []).append((v, g["escala"][k]))
    out = {"0": {"nombre": "Valle del Cauca (general)", "n_pixeles": 0,
                 "mediana": {k: float(np.median([a for a, _ in v])) for k, v in comps.items()},
                 "escala": {k: float(np.median([b for _, b in v])) for k, v in comps.items()}}}
    (OUT / "referencia_general.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("referencia general:", out["0"]["mediana"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pasos = sys.argv[1:] or ["igac", "referencia", "dem", "suelo", "clima"]
    for p in pasos:
        t = time.time()
        globals()[p]()
        print(f"  {p}: {time.time() - t:.0f}s", flush=True)
