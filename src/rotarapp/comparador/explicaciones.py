"""Las 5 explicaciones que exige el reto para cada estrategia: por qué, evidencia, beneficios,
vacíos de información y riesgos para revisión local. Se generan a partir de los resultados."""

from __future__ import annotations

import numpy as np

from rotarapp.comparador.decision import Decision, Entrada, Finca, Plan
from rotarapp.datos.cultivos import Cultivo, Plagas

NOMBRE_COMP = {"ingreso": "ingreso", "agua": "ahorro de agua", "suelo": "salud del suelo", "erosion": "control de erosión"}


def nombre_plan(p: Plan, c: dict[str, Cultivo]) -> str:
    if p.actual:
        return f"Seguir con {c[p.produccion].nombre.lower()}"
    if p.descanso is None:
        return f"{c[p.produccion].nombre} sin descanso"
    return f"{c[p.descanso].nombre} (descanso) → {c[p.produccion].nombre.lower()}"


def _plan(d: Decision, pid: str) -> int:
    return next(i for i, p in enumerate(d.planes) if p.id == pid)


def _fmt(x: float, unidad: str, dec: int = 0) -> str:
    return f"{x:,.{dec}f} {unidad}".replace(",", " ")


def explicar(d: Decision, cultivos: dict[str, Cultivo], plagas: Plagas, entrada: Entrada,
             calib: dict, finca: Finca) -> dict[str, dict[str, list[str]]]:
    val = calib["metricas"]["val"]["simulador_bayesiano"]
    out: dict[str, dict[str, list[str]]] = {}
    A = d.estrategias.get("A")
    for clave, e in d.estrategias.items():
        idx = [_plan(d, pid) for pid in e["planes"]]
        planes = [d.planes[i] for i in idx]
        ids = sorted({c for p in planes for c in (p.descanso, p.produccion) if c})
        m = e["metricas"]
        por_que, evid, benef, vacios, riesgos = [], [], [], [], []

        if clave == "A":
            por_que.append(f"Es lo que hay hoy: {cultivos[planes[0].produccion].nombre.lower()} sin cambios. Es la línea base contra la que se comparan las otras.")
        else:
            pesos = ", ".join(f"{NOMBRE_COMP[k]} {v:.0%}" for k, v in entrada.pesos.items())
            if clave == "B":
                por_que.append(f"Es el plan único con mayor utilidad esperada en toda la finca según tus prioridades ({pesos}).")
            else:
                zonas = "; ".join(f"{nombre_plan(d.planes[_plan(d, z['plan'])], cultivos)}: {z['area_ha']} ha" for z in d.zonas)
                por_que.append(f"Cada zona recibe el plan que más probablemente es el mejor en esos píxeles ({zonas}).")
                por_que.append(f"Probabilidad de que la rotación por zonas supere al mejor plan único: {e.get('p_mejor_que_B', 0):.0%}.")

        evid.append(f"Simulador por píxel calibrado con {calib['n_train']} curvas de rebrote de caña observadas por NASA HLS (2020–2023). "
                    f"En validación (2024): error RMSE de NDVI {val['rmse']:.3f}, el intervalo del 80 % contuvo el {val['cobertura_80']:.0%} de lo observado.")
        evid.append("Clima: NASA POWER 2001–2023 corregido por altitud (23 escenarios de 2 años). Suelo: IGAC-CVC 1:25.000. Erosión: RUSLE con pendiente NASADEM. Salud del suelo: IBS (validado en Iowa).")
        no_cal = [cultivos[c].nombre for c in ids if c != "cana_azucar"]
        if no_cal:
            evid.append(f"{', '.join(no_cal)}: no aparecen en las imágenes de la zona; se simulan con los rangos agronómicos de la tabla (prior) y los parámetros de sensor calibrados en caña.")

        if A is not None and clave != "A":
            dr = m["riego"]["p50"] - A["metricas"]["riego"]["p50"]
            di = m["ibs"]["p50"] - A["metricas"]["ibs"]["p50"]
            de = m["erosion"]["p50"] - A["metricas"]["erosion"]["p50"]
            benef.append(f"Agua: necesitaría {_fmt(m['riego']['p50'], 'mm/año')} de riego para no tener estrés ({'+' if dr >= 0 else '−'}{_fmt(abs(dr), 'mm')} frente a hoy).")
            ds = m["salud"]["p50"] - A["metricas"]["salud"]["p50"]
            benef.append(f"Salud del suelo: índice comparativo {m['salud']['p50']:.0f}/100 ({ds:+.0f} frente a hoy); "
                         f"IBS {m['ibs']['p50']:.0f} ({di:+.0f}; 50 = lo típico de ese suelo, que aquí es caña).")
            benef.append(f"Erosión: {m['erosion']['p50']:.2f} t/ha/año ({de:+.2f} frente a hoy).")
        elif clave == "A":
            benef.append("No exige cambiar manejo, maquinaria ni mercado.")
            benef.append(f"Riego para no tener estrés: {_fmt(m['riego']['p50'], 'mm/año')}; IBS {m['ibs']['p50']:.0f}; erosión {m['erosion']['p50']:.2f} t/ha/año.")
        if np.isfinite(m["ingreso"]["p50"]):
            benef.append(f"Ingreso bruto: {_fmt(m['ingreso']['p50'] / 1e6, 'millones COP/ha/año', 1)} (año malo {m['ingreso']['p10'] / 1e6:.1f}, año bueno {m['ingreso']['p90'] / 1e6:.1f}).")

        vacios.append("No hay costos de producción: el ingreso es bruto, no ganancia.")
        sin_precio = [cultivos[c].nombre for c in ids if cultivos[c].precio_cop_kg is None and cultivos[c].rend_pot_kg_ha > 0]
        if sin_precio:
            vacios.append(f"Sin precio SIPSA: {', '.join(sin_precio)} (su ingreso no se cuenta).")
        proxys = [f"{cultivos[c].nombre}: {cultivos[c].producto_precio}" for c in ids if "proxy" in cultivos[c].producto_precio]
        if proxys:
            vacios.append("Precios aproximados (mayorista, no en finca): " + "; ".join(proxys) + ".")
        sin_pl = sorted({c for i in idx for c in d.evals[i].sin_datos_plagas})
        if sin_pl:
            vacios.append(f"Sin datos de plagas en el JSON: {', '.join(cultivos[c].nombre for c in sin_pl)} (se asume presión baja).")
        vacios.append("El IBS no está validado en el Valle (no hay datos de manejo); el aporte de N de las leguminosas y la mineralización son supuestos.")

        ant = entrada.cultivo_anterior
        for c in ids:
            if ant and c != ant and plagas.presion(ant, c):
                riesgos.append(f"{cultivos[c].nombre} comparte plagas con {cultivos[ant].nombre.lower()} (cultivo anterior): revisar con el técnico.")
        if clave == "A" and ant:
            riesgos.append(f"Repetir {cultivos[ant].nombre.lower()} mantiene todo el ciclo de sus plagas en el lote.")
        for pl in entrada.plagas_anteriores:
            hosp = plagas.hospedantes_por_plaga.get(pl.lower(), set())
            si = [cultivos[c].nombre.lower() for c in ids if c in hosp]
            if si:
                riesgos.append(f"{pl} (reportada) también ataca {', '.join(si)}: la rotación no corta su ciclo.")
            elif hosp:
                benef.append(f"{pl} (reportada) no ataca a ningún cultivo de este plan según el JSON de plagas: la rotación corta su ciclo.")
        if m["riego"]["p90"] > 1.3 * max(m["riego"]["p50"], 1):
            riesgos.append(f"En un año seco el riego necesario sube a {_fmt(m['riego']['p90'], 'mm/año')}.")
        c4 = float(np.mean([str(k) == "4" for k in finca.clase]))
        if clave != "A" and c4 > 0:
            riesgos.append(f"El {c4:.0%} de la finca es clase IGAC 4: transitorios solo con prácticas de conservación (siembra en contorno, cobertura permanente).")
        riesgos.append("Validar fechas de siembra, variedades y mercado local con un agrónomo antes de decidir.")
        out[clave] = {"por_que": por_que, "evidencia": evid, "beneficios": benef, "vacios": vacios, "riesgos": riesgos}
    return out
