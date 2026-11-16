"""Exporta el IBS observado de una zona a la malla de RotarApp (infraestructura: E/S y rasters).

Entrada: carpeta de datos del prototipo NASA_IBS de una zona (`data/<zona>/`), con
  base.npz            cls, peer (grupo de suelo), malla (CRS/transform), peer_nombres
  cube_{ndvi,ndti,soci}.npy + cube_index.json   series HLS/Landsat en la malla nativa
  power_lluvia.json   lluvia diaria NASA POWER (YYYYMMDD -> mm)

Salidas:
  data/interim/microbiota/<zona>/ibs_30m.tif           malla nativa UTM 30 m (no se versiona)
  data/processed/microbiota/<zona>/ibs_9377_100m.tif   EPSG:9377, 100 m (malla de RotarApp)
  data/processed/microbiota/<zona>/referencia_suelos.json   lo normal por grupo de suelo:
        es lo que necesita el simulador para puntuar rotaciones (`indice.ibs_de_rotacion`)
  data/processed/microbiota/<zona>/resumen.json

Bandas de los GeoTIFF: IBS, IBS_sin_C4, s_C1..s_C4 (0-1), confianza (0-3).

Uso:
  python -m rotarapp.microbiota.exportar --prototipo ../NASA_IBS/data/valle --zona valle
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

from rotarapp.microbiota.indice import (COMPONENTES, UMBRALES_IOWA, UMBRALES_VALLE,
                                        Componentes, Puntaje, Referencia, Umbrales, confianza,
                                        componentes_desde_serie, lluvia_3_dias, puntuar,
                                        referencia_por_suelo)

RAIZ = Path(__file__).resolve().parents[3]
CRS_ROTARAPP = "EPSG:9377"
RES_ROTARAPP = 100.0
BANDAS = ("IBS", "IBS_sin_C4", "s_C1", "s_C2", "s_C3", "s_C4", "confianza")
FRANJA = 48


def cargar_entrada(carpeta: Path) -> tuple[dict, list[date], np.ndarray, dict[str, np.ndarray]]:
    base = np.load(carpeta / "base.npz")
    nombres_escenas = json.loads((carpeta / "cube_index.json").read_text())
    fechas = [date(int(n[:4]), int(n[4:6]), int(n[6:8])) for n in nombres_escenas]
    lluvia_txt = json.loads((carpeta / "power_lluvia.json").read_text())
    lluvia = {date(int(k[:4]), int(k[4:6]), int(k[6:8])): float(v) for k, v in lluvia_txt.items()}
    cubo = {v: np.load(carpeta / f"cube_{v}.npy", mmap_mode="r") for v in ("ndvi", "ndti", "soci")}
    if cubo["ndvi"].shape[0] != len(fechas):
        raise ValueError("cube_index.json no coincide con el cubo")
    return dict(base), fechas, lluvia_3_dias(fechas, lluvia), cubo


def componentes_por_franjas(fechas: list[date], lluvia3: np.ndarray,
                            cubo: dict[str, np.ndarray], umbrales: Umbrales) -> Componentes:
    """Corre `componentes_desde_serie` por franjas de filas (el cubo no cabe en RAM)."""
    _, alto, ancho = cubo["ndvi"].shape
    valores = {k: np.full((alto, ancho), np.nan, np.float32) for k in COMPONENTES}
    n_obs = {k: np.zeros((alto, ancho), np.int32) for k in COMPONENTES}
    for f0 in range(0, alto, FRANJA):
        f1 = min(f0 + FRANJA, alto)
        parte = componentes_desde_serie(cubo["ndvi"][:, f0:f1].astype(np.float32),
                                        cubo["ndti"][:, f0:f1].astype(np.float32),
                                        np.asarray(cubo["soci"][:, f0:f1], dtype=np.float32),
                                        fechas, lluvia3, umbrales)
        for k in COMPONENTES:
            valores[k][f0:f1] = parte.valores[k]
            n_obs[k][f0:f1] = parte.n_obs[k]
    return Componentes(valores, n_obs)


def pila_bandas(p: Puntaje, conf: np.ndarray) -> np.ndarray:
    return np.stack([p.ibs, p.ibs_sin_c4, *(p.puntaje[k] for k in COMPONENTES),
                     conf.astype(np.float32)]).astype(np.float32)


def escribir_tif(ruta: Path, datos: np.ndarray, crs: str, transform: Affine) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ruta, "w", driver="GTiff", height=datos.shape[1], width=datos.shape[2],
                       count=datos.shape[0], dtype="float32", crs=crs, transform=transform,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(datos)
        for i, nombre in enumerate(BANDAS, 1):
            dst.set_band_description(i, nombre)


def a_malla_rotarapp(datos: np.ndarray, crs: str, transform: Affine) -> tuple[np.ndarray, Affine]:
    """Remuestrea a EPSG:9377, 100 m: promedio para puntajes, moda para la confianza."""
    alto, ancho = datos.shape[1:]
    izq, arriba = transform * (0, 0)
    der, abajo = transform * (ancho, alto)
    t_dst, w, h = calculate_default_transform(crs, CRS_ROTARAPP, ancho, alto,
                                              left=izq, bottom=abajo, right=der, top=arriba,
                                              resolution=RES_ROTARAPP)
    salida = np.full((datos.shape[0], h, w), np.nan, np.float32)
    for i, nombre in enumerate(BANDAS):
        reproject(datos[i], salida[i], src_transform=transform, src_crs=crs, src_nodata=np.nan,
                  dst_transform=t_dst, dst_crs=CRS_ROTARAPP, dst_nodata=np.nan,
                  resampling=Resampling.mode if nombre == "confianza" else Resampling.average)
    return salida, t_dst


def exportar(carpeta: Path, zona: str, umbrales: Umbrales,
             clases_referencia: tuple[int, ...]) -> dict:
    base, fechas, lluvia3, cubo = cargar_entrada(carpeta)
    malla = json.loads(str(base["malla"]))
    transform = Affine(*malla["transform"])
    nombres = ({int(k): v for k, v in json.loads(str(base["peer_nombres"])).items()}
               if "peer_nombres" in base else {})
    comp = componentes_por_franjas(fechas, lluvia3, cubo, umbrales)
    ref_mask = np.isin(base["cls"], clases_referencia)
    ref = referencia_por_suelo(comp, base["peer"], ref_mask, nombres=nombres)
    p = puntuar(comp, base["peer"], ref)
    conf = confianza(comp, umbrales)
    datos = pila_bandas(p, conf)

    salida_proc = RAIZ / "data" / "processed" / "microbiota" / zona
    escribir_tif(RAIZ / "data" / "interim" / "microbiota" / zona / "ibs_30m.tif",
                 datos, malla["crs"], transform)
    datos_100, t100 = a_malla_rotarapp(datos, malla["crs"], transform)
    escribir_tif(salida_proc / "ibs_9377_100m.tif", datos_100, CRS_ROTARAPP, t100)
    (salida_proc / "referencia_suelos.json").write_text(ref.a_json(), encoding="utf-8")

    valido = np.isfinite(p.ibs[ref_mask])
    resumen = {
        "zona": zona,
        "periodo": [str(min(fechas)), str(max(fechas))],
        "escenas": len(fechas),
        "grupos_de_suelo": len(ref.mediana),
        "cobertura_ibs_en_cultivo": round(float(valido.mean()), 3),
        "cobertura_c1_en_cultivo": round(float(np.isfinite(p.z["C1"][ref_mask]).mean()), 3),
        "ibs_mediana_cultivo": round(float(np.nanmedian(p.ibs[ref_mask])), 1),
        "umbrales": umbrales.__dict__ | {"meses_labranza": list(umbrales.meses_labranza)},
        "bandas": list(BANDAS),
        "advertencias": [
            "IBS estima condiciones para la vida microbiana (residuo, perturbación, raíz viva, "
            "carbono); no mide microbiota. Sin calibración con datos de laboratorio.",
            "50 = típico de ese grupo de suelo; no es comparable entre suelos distintos.",
            "C1 (carbono por satélite) solo existe donde el suelo se vio expuesto y seco.",
        ],
    }
    (salida_proc / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prototipo", type=Path, required=True,
                    help="carpeta data/<zona> del prototipo NASA_IBS")
    ap.add_argument("--zona", required=True)
    ap.add_argument("--umbrales", choices=("valle", "iowa"), default="valle")
    ap.add_argument("--clases-referencia", default="5",
                    help="clases de cultivo de referencia, separadas por coma (Iowa: 1,2,3)")
    a = ap.parse_args()
    umbrales = UMBRALES_VALLE if a.umbrales == "valle" else UMBRALES_IOWA
    clases = tuple(int(c) for c in a.clases_referencia.split(","))
    print(json.dumps(exportar(a.prototipo, a.zona, umbrales, clases), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
