"""Convierte la decisión del modelo en los datos exactos que usa cada pantalla de la app.

Planes (3, uno es lo que el agricultor hace hoy) → campañas → celdas de 30 m con su cultivo y
la probabilidad de que ese plan sea el mejor ahí (mapa de calor); ejes frente a hoy, escenarios
de clima, ganancia por campaña, calendario con fechas y capas de suelo (IBS y erosión).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from rotarapp.comparador import decision as dc
from rotarapp.comparador.explicaciones import explicar
from rotarapp.datos.cultivos import Cultivo, Plagas

F = NDArray[np.float32]
EJES = ["Suelo", "Agua", "Sanidad", "Erosión", "Dinero"]


@dataclass(frozen=True)
class Malla:
    """Todas las celdas de 30 m dentro del polígono (también las que no son cultivo)."""
    anillos: list[list[list[float]]]     # por celda: 4 esquinas [lng, lat]
    centros: NDArray[np.float64]         # (N, 2) lng, lat
    modelada: NDArray[np.bool_]          # (N,) celda de cultivo que el modelo simula
    idx_finca: NDArray[np.int64]         # (N,) índice en la Finca, −1 si no se modela
    ibs_obs: F                           # (N,) IBS observado (NaN sin dato)


def _q(x: NDArray[np.floating], q: float = 50) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = float(np.nanpercentile(x, q)) if np.isfinite(x).any() else float("nan")
    return v


def _med_d(x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Mediana sobre las muestras (eje 0) por píxel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=0)


def _nombre(c: Cultivo) -> str:
    return c.nombre.split(" (")[0]


def _fecha(inicio: date, decada: int) -> date:
    return inicio + timedelta(days=10 * decada)


def _fmt_fecha(d: date) -> str:
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    return f"{d.day} {meses[d.month - 1]} {d.year}"


def _mill(x: float) -> float | None:
    return None if not np.isfinite(x) else round(x / 1e6, 1)


class Constructor:
    def __init__(self, dec: dc.Decision, finca: dc.Finca, malla: Malla, esc: dc.Escenarios,
                 cultivos: dict[str, Cultivo], plagas: Plagas, entrada: dc.Entrada, calib: dict,
                 fecha_siembra: date, lluvia_anual_mm: float):
        self.d, self.f, self.m, self.esc = dec, finca, malla, esc
        self.C, self.pl, self.ent, self.calib = cultivos, plagas, entrada, calib
        self.t0, self.lluvia = fecha_siembra, lluvia_anual_mm
        self.P = finca.n
        self.ref_ing = max((_q(e.ingreso) for e, p in zip(dec.evals, dec.planes)
                            if not p.actual and np.isfinite(e.ingreso).any()), default=1.0)
        self.A = next((i for i, p in enumerate(dec.planes) if p.actual), None)

    # ------------------------------------------------------------ utilidades por asignación
    def _sel(self, g: NDArray[np.integer], campo: str) -> F:
        """(D, P) del campo para la asignación g (plan por píxel)."""
        return np.stack([getattr(self.d.evals[g[p]], campo)[:, p] for p in range(self.P)], axis=1)

    def _util(self, g: NDArray[np.integer]) -> F:
        u = np.stack([dc.utilidad(self.d.evals[g[p]], self.ent.pesos, self.ref_ing)[:, p] for p in range(self.P)], axis=1)
        return u.mean(axis=1)                                                    # (D,)

    def _metricas(self, g: NDArray[np.integer]) -> dict[str, float]:
        m = {k: float(np.nanmean(_med_d(self._sel(g, k)))) if np.isfinite(self._sel(g, k)).any() else float("nan")
             for k in ("ingreso", "riego", "ibs", "erosion", "perdida")}
        m["riego_p90"] = float(np.nanmean(np.nanpercentile(self._sel(g, "riego"), 90, axis=0)))
        return m

    # ------------------------------------------------------------ planes que muestra la app
    def asignaciones(self) -> list[tuple[str, NDArray[np.int32], bool]]:
        g1 = self.d.asignacion.astype(np.int32)
        usados = set(np.unique(g1).tolist())
        orden = np.argsort(-(self.d.p_mejor.mean(axis=1) + 1e-6 * np.nan_to_num(
            np.array([_q(e.ingreso) for e in self.d.evals]), nan=0) / 1e9))
        out = [("1 · Recomendado", g1, True)]
        alt = next((int(k) for k in orden if int(k) not in usados and not self.d.planes[k].actual), None)
        if alt is not None:
            out.append(("2 · Alternativa", np.full(self.P, alt, np.int32), False))
        if self.A is not None:
            out.append(("3 · Lo que hace hoy", np.full(self.P, self.A, np.int32), False))
        else:
            otro = next((int(k) for k in orden if int(k) not in usados and int(k) != alt), None)
            if otro is not None:
                out.append(("3", np.full(self.P, otro, np.int32), False))
        return out

    def campanas(self, g: NDArray[np.int32], n: int = 3) -> list[dict]:
        secs = {k: self.d.evals[k].siembras for k in np.unique(g)}
        prob = np.array([self.d.p_mejor[g[p], p] for p in range(self.P)])
        es_actual = all(self.d.planes[k].actual for k in secs)
        out = []
        for j in range(n):
            crops: list[str] = []
            por_pix = np.full(self.P, -1, np.int32)
            for p in range(self.P):
                s = secs[g[p]]
                if j < len(s):
                    cid = s[j].cultivo
                    if cid not in crops:
                        crops.append(cid)
                    por_pix[p] = crops.index(cid)
            if not crops:
                break
            # orden por área para que el color 0 sea el cultivo principal
            area = [int((por_pix == i).sum()) for i in range(len(crops))]
            orden = list(np.argsort(-np.array(area)))
            remap = {old: new for new, old in enumerate(orden)}
            crops = [crops[i] for i in orden]
            por_pix = np.array([remap.get(int(x), -1) for x in por_pix], np.int32)
            split = []
            total = 0.0
            ini, fin = 10**6, -1
            for ci, cid in enumerate(crops):
                pix = np.nonzero(por_pix == ci)[0]
                gan = 0.0
                for p in pix:
                    sj = secs[g[p]][j]
                    ini, fin = min(ini, sj.inicio), max(fin, sj.fin)
                    gan += float(_med_d(sj.ingreso[:, p:p + 1])[0]) * dc.HA_POR_PIXEL
                total += gan if np.isfinite(gan) else 0.0
                split.append({"id": cid, "name": _nombre(self.C[cid]), "share": round(100 * len(pix) / self.P),
                              "gain": _mill(gan) if self.C[cid].rend_pot_kg_ha > 0 else 0.0,
                              "where": self._donde(pix, len(crops)), "position": self._centro(pix),
                              "present": bool(len(pix)), "abono_verde": self.C[cid].solo_descanso})
            celda_cultivo = np.full(len(self.m.modelada), -1, np.int32)
            celda_prob = np.zeros(len(self.m.modelada), np.float32)
            for k, ip in enumerate(self.m.idx_finca):
                if ip >= 0:
                    celda_cultivo[k] = por_pix[ip]
                    celda_prob[k] = 0.85 if es_actual else float(prob[ip])
            out.append({"crops": [_nombre(self.C[c]) for c in crops], "ids": crops,
                        "celda_cultivo": celda_cultivo.tolist(), "celda_prob": np.round(celda_prob, 2).tolist(),
                        "split": split, "total": round(total / 1e6, 1),
                        "siembra": _fmt_fecha(_fecha(self.t0, ini)), "cosecha": _fmt_fecha(_fecha(self.t0, fin))})
        return out

    def _centro(self, pix: NDArray[np.integer]) -> dict[str, float]:
        if not len(pix):
            return {"latitude": 0.0, "longitude": 0.0}
        celdas = [k for k, ip in enumerate(self.m.idx_finca) if ip in set(pix.tolist())]
        c = self.m.centros[celdas]
        mid = c.mean(axis=0)
        k = celdas[int(np.argmin(((c - mid) ** 2).sum(axis=1)))]
        return {"latitude": float(self.m.centros[k, 1]), "longitude": float(self.m.centros[k, 0])}

    def _donde(self, pix: NDArray[np.integer], n_cultivos: int) -> str:
        if n_cultivos == 1 or len(pix) == self.P:
            return "Va en todo el lote."
        if not len(pix):
            return "No le tocó una zona propia."
        rasgos = []
        c4 = np.mean([str(x) == "4" for x in self.f.clase[pix]])
        c4_t = np.mean([str(x) == "4" for x in self.f.clase])
        if c4 > c4_t + 0.15:
            rasgos.append("la parte de clase IGAC 4 (con restricciones)")
        elif c4 < c4_t - 0.15:
            rasgos.append("la parte de clase IGAC 3")
        ibs, ibs_t = np.nanmean(self.f.ibs_obs[pix]), np.nanmean(self.f.ibs_obs)
        if np.isfinite(ibs) and ibs < ibs_t - 5:
            rasgos.append("el suelo con menor salud biológica")
        elif np.isfinite(ibs) and ibs > ibs_t + 5:
            rasgos.append("el suelo con mejor salud biológica")
        arc, arc_t = np.nanmean(self.f.arcilla[pix]), np.nanmean(self.f.arcilla)
        if arc > arc_t + 1.5:
            rasgos.append("lo más arcilloso")
        elif arc < arc_t - 1.5:
            rasgos.append("lo menos arcilloso")
        return ("Va en " + " y ".join(rasgos) + ".") if rasgos else "Va en la zona donde más gana frente a las otras opciones."

    # ------------------------------------------------------------ ejes, escenarios y textos
    def ejes(self, m: dict[str, float], base: dict[str, float] | None, ing_max: float) -> list[float]:
        def cl(x: float) -> float:
            return round(float(np.clip(x, -2, 2)) * 2) / 2 if np.isfinite(x) else 0.0
        if base is None:
            return [cl((m["ibs"] - 50) / 20), cl((300 - m["riego"]) / 200), cl((0.12 - m["perdida"]) / 0.05),
                    cl((2 - m["erosion"]) / 1.0), cl(2 * m["ingreso"] / max(ing_max, 1) if np.isfinite(m["ingreso"]) else 0)]
        din = ((m["ingreso"] - base["ingreso"]) / max(0.25 * ing_max, 1) if np.isfinite(base["ingreso"])
               else (2 * m["ingreso"] / max(ing_max, 1) if np.isfinite(m["ingreso"]) else float("nan")))
        return [cl((m["ibs"] - base["ibs"]) / 20), cl((base["riego"] - m["riego"]) / 200),
                cl((base["perdida"] - m["perdida"]) / 0.05), cl((base["erosion"] - m["erosion"]) / 1.0), cl(din)]

    def escenarios(self, u: F) -> list[dict]:
        ll = self.esc.lluvia_anual[:, 0]
        lo, hi = np.percentile(ll, [25, 75])
        grupos = [("Seco", ll <= lo, "25 %"), ("Normal", (ll > lo) & (ll < hi), "50 %"), ("Húmedo", ll >= hi, "25 %")]
        sc = [(n, round(6 * float(np.mean(u[s])), 1) if s.any() else float("nan"), pr) for n, s, pr in grupos]
        peor = min(range(3), key=lambda i: sc[i][1])
        notas = {"Seco": "Más riego", "Normal": "Más probable", "Húmedo": "Más enfermedades"}
        return [{"name": n, "prob": pr, "score": s, "note": "Peor caso" if i == peor else notas[n], "worst": i == peor}
                for i, (n, s, pr) in enumerate(sc)]

    def textos_ejes(self, m: dict[str, float], base: dict[str, float] | None, v: list[float], ids: list[str]) -> list[dict]:
        w = self.ent.pesos
        s = sum(w.values())
        pct = lambda k: f"{round(100 * w.get(k, 0) / s)} %"
        hoy = (lambda k, f: f" (hoy {f(base[k])})") if base else (lambda k, f: "")
        cob = [self.C[c].nombre.split(" (")[0].lower() for c in ids if self.C[c].solo_descanso or self.C[c].fija_n]
        txt_suelo = (f"IBS medido en las ventanas de suelo desnudo: {m['ibs']:.0f}{hoy('ibs', lambda x: f'{x:.0f}')}. "
                     + (f"La cobertura de {', '.join(cob)} protege el suelo entre cosecha y siembra." if cob
                        else "Sin cobertura entre campañas el suelo queda labrado y desnudo."))
        txt_agua = (f"Riego para no tener estrés: {m['riego']:.0f} mm/año{hoy('riego', lambda x: f'{x:.0f}')}; "
                    f"en un año seco, {m['riego_p90']:.0f} mm.")
        txt_san = (f"Pérdida esperada por plagas y enfermedades: {100 * m['perdida']:.0f} %"
                   f"{hoy('perdida', lambda x: f'{100 * x:.0f} %')}. Cambiar de familia corta el ciclo de plagas.")
        txt_ero = f"Erosión estimada: {m['erosion']:.1f} t/ha/año{hoy('erosion', lambda x: f'{x:.1f}')}."
        txt_din = ("Ingreso bruto estimado: " + (f"{m['ingreso'] / 1e6:.1f} millones por hectárea al año" if np.isfinite(m["ingreso"])
                                                 else "sin dato (falta el precio de algún cultivo)") + ". No descuenta costos.")
        coma = lambda t: t.replace(".", ",").replace(", ", "; ", 0)
        txt_ero, txt_din = coma(txt_ero[:-1]) + ".", txt_din.replace(f"{m['ingreso'] / 1e6:.1f}", f"{m['ingreso'] / 1e6:.1f}".replace(".", ",")) if np.isfinite(m["ingreso"]) else txt_din
        return [{"name": "Salud del suelo", "weight": pct("suelo"), "v": v[0], "why": txt_suelo},
                {"name": "Agua", "weight": pct("agua"), "v": v[1], "why": txt_agua},
                {"name": "Riesgo sanitario", "weight": "", "v": v[2], "why": txt_san},
                {"name": "Erosión", "weight": pct("erosion"), "v": v[3], "why": txt_ero},
                {"name": "Economía", "weight": pct("ingreso"), "v": v[4], "why": txt_din}]

    def dinero(self, camps: list[dict], area_ha: float) -> dict:
        items, mx = [], max([c["total"] for c in camps] + [0.1])
        for j, c in enumerate(camps):
            sin = any(s["gain"] is None for s in c["split"])
            solo_abono = all(s["abono_verde"] for s in c["split"] if s["present"])
            items.append({"crop": " + ".join(c["crops"]), "camp": f"Campaña {j + 1}", "solo_abono": solo_abono,
                          "lote": None if sin and c["total"] == 0 else c["total"],
                          "ha": None if sin and c["total"] == 0 else round(c["total"] / max(area_ha, 1e-6), 1),
                          "bar": round(100 * c["total"] / mx),
                          "split": " · ".join(f"{s['name']} ≈ " + ("sin precio" if s["gain"] is None else
                                              ("abono verde, no se vende" if s["abono_verde"] else f"{s['gain']:.1f} millones".replace(".", ",")))
                                              for s in c["split"] if s["present"])})
        return {"total": round(sum(c["total"] for c in camps), 1), "items": items}

    def calendario(self, camps: list[dict], g: NDArray[np.int32]) -> list[dict]:
        out = []
        for j, c in enumerate(camps):
            tareas = []
            ks = [k for k in np.unique(g) if j < len(self.d.evals[k].siembras)]
            tras_cob = any(self.d.evals[k].siembras[j].tras_cobertura for k in ks)
            sin_arar = any(self.d.planes[k].descanso is not None for k in ks)   # sistema con cobertura
            soca = all(self.d.planes[k].actual and self.d.planes[k].produccion == "cana_azucar" for k in ks)
            if soca:
                antes = ("Dejar la hojarasca y esperar el rebrote", "La soca rebrota sola; no queme la hojarasca.")
            elif tras_cob:
                antes = ("Sembrar directo sobre la cobertura", "Corte el abono verde en floración, déjelo sobre el suelo y siembre sin arar.")
            elif sin_arar and j > 0:
                antes = ("Sembrar directo sobre el rastrojo", f"Deje {self.ent.dias_barbecho} días de barbecho con el rastrojo encima y siembre sin arar.")
            else:
                antes = ("Preparar el suelo", f"Deje {self.ent.dias_barbecho} días entre la cosecha anterior y la siembra.")
            tareas.append({"when": "Antes de sembrar", "title": antes[0], "note": antes[1]})
            tareas.append({"when": f"Siembra · desde el {c['siembra']}", "title": "Sembrar " + " y ".join(x.lower() for x in c["crops"])
                           + (" en sus zonas" if len(c["crops"]) > 1 else ""),
                           "note": "Fechas simuladas con el clima de 2001–2023; confírmelas con el calendario del ICA."})
            riego = []
            for s in c["split"]:
                vals = []
                for k in np.unique(g):
                    se = self.d.evals[k].siembras
                    if j < len(se) and se[j].cultivo == s["id"]:
                        vals.append(_med_d(se[j].riego))
                if vals:
                    riego.append(f"{s['name']} ≈ {float(np.nanmean(np.concatenate(vals))):.0f} mm")
            tareas.append({"when": "Mientras crece", "title": "Planear el riego de apoyo",
                           "note": "Agua que evitaría el estrés: " + ", ".join(riego) + "."})
            av = [s["name"] for s in c["split"] if s["abono_verde"]]
            tareas.append({"when": f"Hasta el {c['cosecha']}",
                           "title": "Cortar el abono verde y dejarlo como cobertura" if av and len(av) == len(c["split"]) else "Cosechar y dejar el rastrojo",
                           "note": "No queme ni retire el residuo: es el alimento de la vida del suelo."})
            out.append({"crop": " + ".join(c["crops"]), "when": f"Campaña {j + 1}", "now": j == 0, "tasks": tareas})
        return out

    # ------------------------------------------------------------ suelo
    def suelo(self, g_actual: NDArray[np.int32] | None) -> dict:
        N = len(self.m.modelada)
        ibs = self.m.ibs_obs
        hay = np.isfinite(ibs)
        pct = lambda m: round(100 * float(m.sum()) / max(int(hay.sum()), 1))
        m_ibs = float(np.nanmean(ibs)) if hay.any() else float("nan")
        mo = float(np.nanmean(self.f.soc) * 1.724)
        clases = [str(x) for x in self.f.clase]
        clase = max(set(clases), key=clases.count) if clases else "—"
        salud = {
            "valores": [None if not np.isfinite(v) else round(float(v), 1) for v in ibs],
            "t": [None if not np.isfinite(v) else round(float(np.clip((v - 20) / 70, 0, 1)), 3) for v in ibs],
            "mapTag": (f"Salud: {m_ibs:.0f} de 100 · " + ("Alta" if m_ibs >= 70 else "Media" if m_ibs >= 50 else "Baja"))
            if np.isfinite(m_ibs) else "Salud: sin dato",
            "distTitle": "Salud del suelo en el lote (IBS 2020–2025)",
            "dist": [{"name": "Alta", "pct": pct(ibs >= 70), "tono": 0.9}, {"name": "Media", "pct": pct((ibs >= 50) & (ibs < 70)), "tono": 0.5},
                     {"name": "Baja", "pct": pct(ibs < 50), "tono": 0.1}],
            "tiles": [{"label": "Materia orgánica", "value": f"{mo:.1f} %".replace(".", ",")},
                      {"label": "Arcilla", "value": f"{np.nanmean(self.f.arcilla):.0f} %"},
                      {"label": "Clase IGAC", "value": clase}],
            "tips": [t for t in [
                "En las zonas claras (salud baja) siembre primero la leguminosa de cobertura: aporta nitrógeno y protege el suelo."
                if pct(ibs < 50) >= 20 else None,
                "Deje el residuo de la cosecha sobre el suelo: no lo queme ni lo retire.",
                "Evite dejar el suelo desnudo entre campañas: es cuando más se pierde la vida del suelo.",
                f"Sin dato en el {100 - round(100 * hay.mean())} % del lote (nubes o no es cultivo)." if hay.mean() < 0.95 else None] if t],
            "sin_dato": round(100 * (1 - float(hay.mean()))),
        }
        g = g_actual if g_actual is not None else self.d.asignacion.astype(np.int32)
        ero_pix = _med_d(self._sel(g, "erosion"))
        cob = float(np.nanmean(_med_d(self._sel(g, "cobertura")))) if self.d.evals[g[0]].cobertura is not None else float("nan")
        ero = np.full(N, np.nan)
        for k, ip in enumerate(self.m.idx_finca):
            if ip >= 0:
                ero[k] = ero_pix[ip]
        hayE = np.isfinite(ero)
        pe = lambda m: round(100 * float(m.sum()) / max(int(hayE.sum()), 1))
        m_e = float(np.nanmean(ero))
        erosion = {
            "valores": [None if not np.isfinite(v) else round(float(v), 2) for v in ero],
            "t": [None if not np.isfinite(v) else round(float(np.clip(v / 5, 0, 1)), 3) for v in ero],
            "mapTag": f"Erosión: {m_e:.1f} t/ha/año".replace(".", ","),
            "distTitle": "Erosión con el manejo de hoy" if g_actual is not None else "Erosión con el plan recomendado",
            "dist": [{"name": "Baja", "pct": pe(ero < 1), "tono": 0.15}, {"name": "Media", "pct": pe((ero >= 1) & (ero < 2.5)), "tono": 0.45},
                     {"name": "Alta", "pct": pe((ero >= 2.5) & (ero < 5)), "tono": 0.75}, {"name": "Muy alta", "pct": pe(ero >= 5), "tono": 1.0}],
            "tiles": [{"label": "Pendiente media", "value": f"{np.nanmean(self.f.pendiente):.1f} %".replace(".", ",")},
                      {"label": "Lluvia al año", "value": f"{self.lluvia:,.0f} mm".replace(",", ".")},
                      {"label": "Tiempo con cobertura", "value": f"{100 * cob:.0f} %" if np.isfinite(cob) else "—"}],
            "tips": ["Mantenga cobertura viva o residuo sobre el suelo antes de las lluvias de abril–mayo y octubre–noviembre.",
                     "En las zonas oscuras, siembre en contorno y deje franjas de pasto o barreras vivas."
                     + (" Su lote tiene clase IGAC 4: estas prácticas son obligatorias para cultivos transitorios." if "4" in clases else "")],
        }
        return {"salud": salud, "erosion": erosion}

    # ------------------------------------------------------------ todo
    def construir(self, area_ha: float) -> dict:
        asigs = self.asignaciones()
        base = self._metricas(np.full(self.P, self.A, np.int32)) if self.A is not None else None
        ing_max = max([self._metricas(g)["ingreso"] for _, g, _ in asigs if np.isfinite(self._metricas(g)["ingreso"])] + [1.0])
        expl = explicar(self.d, self.C, self.pl, self.ent, self.calib, self.f)
        planes = []
        for tag, g, rec in asigs:
            m = self._metricas(g)
            u = self._util(g)
            camps = self.campanas(g)
            ids = sorted({sp["id"] for c in camps for sp in c["split"]})
            ax = self.ejes(m, base, ing_max) if not (base and tag.startswith("3 · Lo")) else [0.0] * 5
            pm = float(np.mean([self.d.p_mejor[g[p], p] for p in range(self.P)]))
            es_actual = all(self.d.planes[k].actual for k in np.unique(g))
            conf = "alta" if es_actual or pm >= 0.6 else "media" if pm >= 0.3 else "baja"
            alertas = []
            ant = self.ent.cultivo_anterior
            if ant and any(sp["id"] == ant for sp in camps[0]["split"]):
                alertas.append(f"Alerta en sanidad: repite {self.C[ant].nombre.lower()}, su cultivo anterior; sus plagas siguen en el lote.")
            if m["riego_p90"] > 500:
                alertas.append(f"Alerta de agua: en un año seco necesitaría unos {m['riego_p90']:.0f} mm de riego.")
            clave = "C" if rec else ("A" if es_actual else "B")
            planes.append({
                "tag": tag, "recommended": rec, "conf": conf, "p_mejor": round(pm, 2),
                "score": round(6 * float(np.mean(u)), 1), "axes": ax,
                "alert": " ".join(alertas) or None,
                "camps": [c["crops"] for c in camps], "campanas": camps,
                "ejes": self.textos_ejes(m, None if es_actual else base, ax, ids),
                "escenarios": self.escenarios(u), "dinero": self.dinero(camps, area_ha),
                "calendario": self.calendario(camps, g),
                "explicacion": expl.get(clave) or expl.get("B"),
                "metricas": {k: (None if not np.isfinite(v) else round(v, 3)) for k, v in m.items()},
            })
        return {"planes": planes, "suelo": self.suelo(np.full(self.P, self.A, np.int32) if self.A is not None else None)}
