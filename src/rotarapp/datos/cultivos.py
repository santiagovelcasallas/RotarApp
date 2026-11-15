"""Carga la tabla de cultivos (data/referencia/cultivos) en objetos que usa el simulador."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3] / "data" / "referencia"


@dataclass(frozen=True)
class Cultivo:
    id: str
    nombre: str
    tipo: str            # Transitorio, Semiperenne, Permanente, Descanso
    rol: str             # Producción, Descanso, Ambos
    familia: str
    fija_n: bool
    t: tuple[float, float, float, float]       # T_min, T_opt_min, T_opt_max, T_max
    ph: tuple[float, float, float, float]
    alt: tuple[float, float]                   # alt_min, alt_max
    kc: float
    ciclo_dias: int
    frac_crecimiento: float
    frac_senescencia: float
    dosel_final: float     # dosel al cosechar (grano seco 0,2; caña madura 0,75)
    raiz_m: float
    arcilla: tuple[float, float]  # % arcilla: hasta [0] sin efecto, en [1] rinde la mitad (textura del Excel)
    rend_pot_kg_ha: float
    n_demanda: float
    n_aporte: float
    residuo: str
    precio_cop_kg: float | None                # mediana SIPSA 2026 (mayorista)
    producto_precio: str

    @property
    def es_leguminosa(self) -> bool:
        return self.fija_n

    @property
    def solo_descanso(self) -> bool:
        return self.rol == "Descanso"


@dataclass(frozen=True)
class Plagas:
    """Plagas compartidas por género del agente y plagas por cultivo."""
    compartidas: dict[str, dict[str, int]]
    n_por_cultivo: dict[str, int]
    hospedantes_por_plaga: dict[str, set[str]]   # nombre de plaga (minúsculas) → cultivos

    def presion(self, anterior: str, siguiente: str) -> float | None:
        """Fracción de las plagas del cultivo siguiente que el anterior deja en el lote.
        None si el cultivo siguiente no tiene datos de plagas."""
        n = self.n_por_cultivo.get(siguiente, 0)
        if n == 0 or (anterior not in self.n_por_cultivo and anterior not in self.compartidas):
            return None  # sin datos de plagas para alguno de los dos: no es "cero plagas"
        if anterior == siguiente:
            return 1.0
        return min(1.0, self.compartidas.get(anterior, {}).get(siguiente, 0) / n)


def _f(x: str) -> float:
    return float(x.replace(",", "."))


def cargar_cultivos(raiz: Path = RAIZ) -> dict[str, Cultivo]:
    c = raiz / "cultivos"
    par = {r["id"]: r for r in csv.DictReader(open(c / "cultivos_parametros.csv", encoding="utf-8"))}
    atr = {r["id"]: r for r in csv.DictReader(open(c / "cultivos_atributos.csv", encoding="utf-8"))}
    mod = {r["id"]: r for r in csv.DictReader(open(c / "cultivos_modelo.csv", encoding="utf-8"))}
    pre = {r["id"]: r for r in csv.DictReader(open(c / "precios_sipsa_2026_resumen.csv", encoding="utf-8"))}
    out: dict[str, Cultivo] = {}
    for cid, m in mod.items():
        if m["rotable"] != "si":
            continue
        p, a = par[cid], atr[cid]
        precio = pre.get(cid, {}).get("Mediana COP/kg") or ""
        out[cid] = Cultivo(
            id=cid, nombre=p["Cultivo"], tipo=a["Tipo"], rol=a["Rol en rotación"], familia=a["Familia"],
            fija_n=p["fija_N"] == "True",
            t=(_f(p["T_min"]), _f(p["T_opt_min"]), _f(p["T_opt_max"]), _f(p["T_max"])),
            ph=(_f(p["pH_min"]), _f(p["pH_opt_min"]), _f(p["pH_opt_max"]), _f(p["pH_max"])),
            alt=(_f(p["alt_min_m"]), _f(p["alt_max_m"])), kc=_f(p["Kc_medio"]),
            ciclo_dias=int(m["ciclo_dias"]), frac_crecimiento=_f(m["frac_crecimiento"]),
            frac_senescencia=_f(m["frac_senescencia"]), dosel_final=_f(m["dosel_final"]), raiz_m=_f(m["raiz_m"]),
            arcilla=(_f(m["arcilla_opt_pct"]), _f(m["arcilla_lim_pct"])),
            rend_pot_kg_ha=_f(m["rend_pot_kg_ha"]), n_demanda=_f(m["n_demanda_kg_ha"]),
            n_aporte=_f(m["n_aporte_kg_ha"]), residuo=m["residuo"],
            precio_cop_kg=float(precio) if precio else None, producto_precio=m["producto_precio"],
        )
    return out


def _genero(agente: str) -> str:
    t = re.split(r"[\s,(/]+", agente.strip())[0].lower()
    return agente.strip().lower() if t in ("por", "virus", "género", "sugarcane", "pokkah") else t


def cargar_plagas(raiz: Path = RAIZ) -> Plagas:
    pj = json.loads((raiz / "plagas" / "plagas_valle_del_cauca.json").read_text(encoding="utf-8"))
    comp: dict[str, dict[str, set[str]]] = {}
    n: dict[str, int] = {}
    hosp: dict[str, set[str]] = {}

    def add(a: str, b: str, g: str) -> None:
        comp.setdefault(a, {}).setdefault(b, set()).add(g)

    for p in pj["plagas"]:
        a, g = p["cultivo_id"], _genero(p["agente"])
        n[a] = n.get(a, 0) + 1
        hs = {a, *(p.get("otros_hospedantes") or [])}
        hosp.setdefault(p["plaga"].lower(), set()).update(hs)
        for b in hs - {a}:
            add(a, b, g); add(b, a, g)
    for m in pj["matriz_hospedantes"]:
        g, cs = _genero(m["plaga"]), list(m["cultivos"])
        hosp.setdefault(m["plaga"].lower(), set()).update(cs)
        for a in cs:
            for b in cs:
                if a != b:
                    add(a, b, g)
    return Plagas({a: {b: len(s) for b, s in d.items()} for a, d in comp.items()}, n, hosp)
