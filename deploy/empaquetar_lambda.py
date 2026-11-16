"""Arma el zip de la Lambda: handler + código + datos mínimos + numpy/pyproj para Linux.

    python deploy/empaquetar_lambda.py [ruta NASA_IBS/data] [salida.zip]

Solo copia lo que usa el servicio: base.npz, ibs.npz y cube_index.json de cada zona (no los
cubos HLS de 1,5 GB), las tablas de referencia, la calibración y el clima POWER en caché.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY = "3.12"


def main() -> None:
    nasa = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "NASA_IBS/data"
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "deploy/build/rotarapp-api.zip"
    tmp = salida.parent / "paquete"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(tmp), "--platform", "manylinux2014_x86_64",
                    "--implementation", "cp", "--python-version", PY, "--only-binary=:all:", "numpy<2.3", "pyproj"], check=True)
    shutil.copy(RAIZ / "deploy/lambda_handler.py", tmp / "lambda_handler.py")
    shutil.copytree(RAIZ / "src/rotarapp", tmp / "src/rotarapp", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ["data/referencia/cultivos", "data/referencia/plagas", "data/processed/modelo/calibracion", "data/referencia/valle"]:
        shutil.copytree(RAIZ / rel, tmp / rel)
    for z in ("palmira_candelaria", "valle_norte"):
        (tmp / f"data/processed/microbiota/{z}").mkdir(parents=True)
        shutil.copy(RAIZ / f"data/processed/microbiota/{z}/referencia_suelos.json", tmp / f"data/processed/microbiota/{z}/")
    (tmp / "data/raw/power").mkdir(parents=True)
    for region in ("valle", "valle_norte"):
        shutil.copy(RAIZ / f"data/raw/power/{region}.json", tmp / "data/raw/power/")
        (tmp / f"nasa_ibs/{region}").mkdir(parents=True)
        for f in ("base.npz", "ibs.npz", "cube_index.json"):
            shutil.copy(nasa / region / f, tmp / f"nasa_ibs/{region}/{f}")
    if salida.exists():
        salida.unlink()
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(tmp).as_posix())
    tam = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    print(f"{salida} — {salida.stat().st_size / 1e6:.1f} MB comprimido, {tam / 1e6:.0f} MB descomprimido")


if __name__ == "__main__":
    main()
