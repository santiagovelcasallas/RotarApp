"""Decisión bayesiana por píxel: probabilidad de que cada plan de rotación sea el mejor.

Para cada plan (leguminosa de descanso → cultivo de producción, repetido en un horizonte de
2 años) se simula cada píxel de la finca en D muestras = escenarios climáticos (años 2001-2023
de NASA POWER) × muestras del posterior de parámetros. Con las prioridades del agricultor se
calcula una utilidad por muestra y el plan ganador; la frecuencia con que gana cada plan es
P(plan es el mejor | datos), el "mapa de calor".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import NDArray

from rotarapp.datos.cultivos import Cultivo, Plagas
from rotarapp.datos.power import ClimaDecadal, fecha_decada
from rotarapp.economia import rendimiento as ec
from rotarapp.erosion.rusle import erosion
from rotarapp.filtro.aptitud import aptitud, f_ph, f_textura, rotacion_permitida
from rotarapp.microbiota import NDTI_TIPICO, Componentes, Referencia, componentes_desde_serie, puntuar
from rotarapp.simulador.dinamica import V_COSECHA, Parametros, Siembra, simular

F = NDArray[np.float32]
HORIZONTE = 72                 # décadas (2 años): el IBS necesita al menos 2 años para C3
HA_POR_PIXEL = 0.09            # 30 m × 30 m
REF_RIEGO_MM = 800.0
REF_EROSION_T = 10.0
APTITUD_MIN = 0.25
NDVI_SUELO_LABRADO = 0.18      # supuesto: el NDVI de suelo calibrado (~0,3) es lote de caña con hojarasca
# Abono verde manejado como cobertura: se corta y se deja en superficie y el cultivo siguiente se
# siembra directo, sin labranza. NDTI supuesto = hojarasca de caña observada (0,14); calibrar en campo.
NDTI_ABONO_COBERTURA = 0.14
PENALIZA_CLASE4 = 0.30         # clase IGAC 4: utilidad × (1 − 0,3 × fracción del tiempo en cultivo de venta)


@dataclass(frozen=True)
class Finca:
    fila: NDArray[np.int32]
    col: NDArray[np.int32]
    clase: NDArray[np.object_]
    ph: F
    alt: F
    arcilla: F
    soc: F
    pendiente: F
    awc: F
    grupo_suelo: NDArray[np.int32]
    ibs_obs: F

    @property
    def n(self) -> int:
        return int(self.fila.size)


@dataclass(frozen=True)
class Entrada:
    cultivo_anterior: str | None = None
    plagas_anteriores: list[str] = field(default_factory=list)
    pesos: dict[str, float] = field(default_factory=lambda: {"ingreso": 0.25, "agua": 0.25,
                                                              "suelo": 0.25, "erosion": 0.25})
    decada_siembra: int = 32   # 21 de noviembre
    dias_barbecho: int = 40    # suelo sin cultivo entre cosecha y siembra (supuesto; en caña se observan ~50 días de baja cobertura)
    precios: dict[str, float] = field(default_factory=dict)   # COP/kg que da el agricultor (p. ej. caña)

    def precio(self, c: Cultivo) -> float | None:
        return self.precios.get(c.id, c.precio_cop_kg)


@dataclass(frozen=True)
class Plan:
    descanso: str | None
    produccion: str
    actual: bool = False

    @property
    def id(self) -> str:
        if self.actual:
            return f"actual:{self.produccion}"
        return f"{self.descanso or 'sin_descanso'}>{self.produccion}"


@dataclass
class Evaluacion:
    plan: Plan
    ingreso: F         # (D, P) COP/ha/año (NaN si un cultivo de venta no tiene precio)
    riego: F           # (D, P) mm/año que evitarían el estrés (simulación en secano)
    ibs: F             # (D, P) 0..100, IBS medido solo en las ventanas de suelo desnudo (C2, C3)
    erosion: F         # (D, P) t/ha/año
    rendimientos: dict[str, F]   # cultivo → (D, P) kg/ha por ciclo (media de ciclos)
    factible: NDArray[np.bool_]  # (P,)
    factor_clase: F              # (P,)
    sin_datos_plagas: list[str]


# ---------------------------------------------------------------- escenarios y parámetros
@dataclass(frozen=True)
class Escenarios:
    lluvia: F       # (T, D, 1)
    et0: F
    temp: F
    hr_media_ciclo: F   # (T, D) para promediar por ciclo
    mojado: F           # (T, D) fracción de días con lluvia ≥ 1 mm
    calor: dict[int, F]  # umbral °C → (T, D, 1) fracción de días con Tmax > umbral
    lluvia_anual: F     # (D, 1)
    th: Parametros      # muestras para cultivos sin calibrar (prior + parámetros de sensor)
    th_cana: Parametros  # posterior completo de caña
    fechas: list[date]
    hr_media: float
    mojado_medio: float


def escenarios(clima: ClimaDecadal, post_muestras: dict[str, F], post_pesos: F,
               s0: int, k_post: int = 4, semilla: int = 0) -> Escenarios:
    rng = np.random.default_rng(semilla)
    serie = {k: getattr(clima, k).reshape(-1) for k in ("lluvia", "et0", "t_media", "hr", "frac_lluvia")}
    inicios = [a * 36 + s0 for a in range(len(clima.anios)) if a * 36 + s0 + HORIZONTE <= serie["lluvia"].size]
    Y, D = len(inicios), len(inicios) * k_post
    idx = np.array(inicios)[None, :] + np.arange(HORIZONTE)[:, None]            # (T, Y)
    rep = lambda a: np.repeat(a[idx], k_post, axis=1).astype(np.float32)        # (T, D)
    ll, et, tt, hr = rep(serie["lluvia"]), rep(serie["et0"]), rep(serie["t_media"]), rep(serie["hr"])
    moj = rep(serie["frac_lluvia"])
    calor = {u: rep(v.reshape(-1))[:, :, None] for u, v in clima.frac_calor.items()}

    j = rng.choice(post_pesos.size, size=D, p=post_pesos / post_pesos.sum())
    col = lambda v: np.asarray(v, np.float32).reshape(-1, 1)
    sensor = {k: col(post_muestras[k][j]) for k in ("ndvi_suelo", "ndvi_max")}
    th = Parametros(k_mult=col(rng.uniform(0.7, 1.3, D)), p_agot=col(rng.uniform(0.4, 0.6, D)),
                    v0=col(rng.uniform(0.03, 0.10, D)), riego=np.zeros((D, 1), np.float32),
                    latencia=col(rng.uniform(0.5, 1.5, D)), **sensor)            # emergencia tras la siembra
    th_cana = Parametros(k_mult=col(post_muestras["k_mult"][j]), p_agot=col(post_muestras["p_agot"][j]),
                         v0=col(post_muestras["v0"][j]), riego=np.zeros((D, 1), np.float32),
                         latencia=col(post_muestras["latencia"][j]), **sensor)   # rebrote de soca
    anio0 = clima.anios[0]
    fechas = [fecha_decada(anio0 + (s0 + t) // 36, (s0 + t) % 36) for t in range(HORIZONTE)]
    return Escenarios(ll[:, :, None], et[:, :, None], tt[:, :, None], hr, moj, calor,
                      (ll.sum(axis=0) / (HORIZONTE / 36)).reshape(-1, 1), th, th_cana, fechas,
                      float(np.nanmean(serie["hr"])), float(np.nanmean(serie["frac_lluvia"])))


# ---------------------------------------------------------------- secuencias
def secuencia(plan: Plan, cultivos: dict[str, Cultivo], T: int = HORIZONTE, barbecho: int = 4) -> list[Siembra]:
    """Siembras del plan en el horizonte, con `barbecho` décadas sin cultivo entre cosecha y
    siembra. La soca de caña rebrota sin resiembra (su ventana desnuda es la latencia)."""
    dec = lambda c: max(int(round(c.ciclo_dias / 10)), 1)
    out: list[Siembra] = []
    t = 0
    if plan.produccion == "cana_azucar":
        # actual: socas sin interrupción; renovación: leguminosa y luego caña nueva con sus socas
        if plan.descanso:
            d = cultivos[plan.descanso]
            out.append(Siembra(d, 0, min(dec(d), T))); t = dec(d) + barbecho
        c = cultivos["cana_azucar"]
        while t < T:
            out.append(Siembra(c, t, min(t + dec(c), T))); t += dec(c)
        return out
    orden = ([plan.descanso] if plan.descanso else []) + [plan.produccion]
    while t < T:
        for cid in orden:
            c = cultivos[cid]
            if t >= T:
                break
            out.append(Siembra(c, t, min(t + dec(c), T)))
            t += dec(c) + barbecho
    return out


def ndti_programa(sec: list[Siembra], anterior: Cultivo | None, T: int = HORIZONTE,
                  sin_labranza: bool = False) -> F:
    """NDTI por década según el estado del lote (solo cuenta donde NDVI < 0,3).

    sin_labranza: sistema con leguminosa de cobertura → siembra directa en todas las siembras
    (sin herbicidas, la cobertura es lo que controla las arvenses; sin ella hay que labrar)."""
    x = np.full(T, NDTI_TIPICO["suelo_labrado"], np.float32)
    def res(c: Cultivo) -> float:
        return NDTI_ABONO_COBERTURA if c.solo_descanso else NDTI_TIPICO.get(c.residuo, NDTI_TIPICO["suelo_labrado"])
    residuo = res(anterior) if anterior else float(x[0])
    previo = anterior
    t_sig = {s.inicio for s in sec}
    for t in range(T):
        activo = next((s for s in sec if s.inicio <= t < s.fin), None)
        soca = activo is not None and activo.cultivo.id == "cana_azucar" and (
            t > 0 or (anterior is not None and anterior.id == "cana_azucar"))
        directa = sin_labranza or (previo is not None and previo.solo_descanso)   # siembra directa
        if t in t_sig and directa:
            x[t] = residuo
        elif t in t_sig and not soca:
            x[t] = NDTI_TIPICO["suelo_labrado"]                    # preparación / siembra
        elif activo is None or t == activo.fin - 1:
            x[t] = residuo
        else:
            x[t] = 0.20                                            # dosel verde: no se usa
        if activo and t == activo.fin - 1:
            residuo, previo = res(activo.cultivo), activo.cultivo
    return x


# ---------------------------------------------------------------- evaluación
def evaluar(plan: Plan, finca: Finca, esc: Escenarios, cultivos: dict[str, Cultivo], plagas: Plagas,
            ref: Referencia, entrada: Entrada, t_media: float) -> Evaluacion:
    sec = secuencia(plan, cultivos, barbecho=max(int(round(entrada.dias_barbecho / 10)), 0))
    anterior0 = cultivos.get(entrada.cultivo_anterior or "")
    th = esc.th_cana if plan.actual and plan.produccion == "cana_azucar" else esc.th
    from rotarapp.datos.prototipo import drenaje_mm_decada
    r = simular(sec, esc.lluvia, esc.et0, esc.temp, finca.awc[None, :], th, calor=esc.calor,
                drenaje_mm=drenaje_mm_decada(finca.arcilla)[None, :])
    D, P = r.ndvi.shape[1:]
    anios = HORIZONTE / 36

    hosp_rep = set().union(*[plagas.hospedantes_por_plaga.get(p.lower(), set())
                             for p in entrada.plagas_anteriores]) if entrada.plagas_anteriores else set()

    ingreso = np.zeros((D, P), np.float32)
    rend: dict[str, list[F]] = {}
    sin_datos: set[str] = set()
    sin_precio = False
    anterior, bio_ant = anterior0, np.float32(1.0)
    for i, s in enumerate(sec):
        c = s.cultivo
        ftex = f_textura(finca.arcilla, c)[None, :]
        n_disp = ec.n_disponible(c, finca.ibs_obs[None, :], anterior, bio_ant)
        pres, con_datos = ec.presion_plagas(c, anterior, plagas, (c.id in hosp_rep) and i <= 1)
        if not con_datos:   # registra el cultivo al que le faltan datos, no al que sí los tiene
            sin_datos.update(x.id for x in (c, anterior) if x is not None and plagas.n_por_cultivo.get(x.id, 0) == 0)
        hr_ciclo = esc.hr_media_ciclo[s.inicio:s.fin].mean(axis=0)[:, None]          # (D, 1)
        moj_ciclo = esc.mojado[s.inicio:s.fin].mean(axis=0)[:, None]
        perdida = ec.perdida_plagas(pres, plagas.frac_humeda.get(c.id), moj_ciclo, esc.mojado_medio,
                                    hr_ciclo, esc.hr_media)
        bio = np.nan_to_num(r.rend_relativo[i]) * ftex * f_ph(finca.ph, c)[None, :]
        y = (c.rend_pot_kg_ha * bio * ec.f_nitrogeno(c, n_disp) * (1 - perdida)).astype(np.float32)
        # ciclos truncados por el horizonte cuentan en proporción
        frac = (s.fin - s.inicio) / max(int(round(c.ciclo_dias / 10)), 1)
        if c.rend_pot_kg_ha > 0:
            rend.setdefault(c.id, []).append(y)
            precio = entrada.precio(c)
            if precio is not None:
                ingreso += y * precio * min(frac, 1.0)
            elif c.rol != "Ambos" or c.id == plan.produccion:
                sin_precio = True       # cultivo de venta sin precio: ingreso desconocido
        anterior, bio_ant = c, bio

    # Suelo y erosión: en caña el lote cosechado queda con hojarasca (NDVI de suelo calibrado);
    # en una rotación el lote se prepara y queda desnudo (NDVI_SUELO_LABRADO)
    if plan.actual and plan.produccion == "cana_azucar":
        ndvi_s = r.ndvi
    else:
        ndvi_s = (NDVI_SUELO_LABRADO + (th.ndvi_max - NDVI_SUELO_LABRADO) * r.V).astype(np.float32)
    ndti = ndti_estado(ndti_programa(sec, anterior0, sin_labranza=plan.descanso is not None), r.V)
    ibs = ibs_suelo_desnudo(ndvi_s, ndti, r.V, esc.fechas, finca.grupo_suelo, ref)
    # NDTI 0,20 marca dosel verde (no residuo): para el subfactor de cobertura solo cuenta sin dosel
    ndti_suelo = np.where(r.V <= V_COSECHA + 1e-6, ndti, NDTI_TIPICO["suelo_labrado"])
    ero = erosion(esc.lluvia_anual, finca.arcilla[None, :], finca.soc[None, :], finca.pendiente[None, :], ndvi_s,
                  ndti_serie=ndti_suelo)

    ids = {s.cultivo.id for s in sec}
    apt = np.stack([aptitud(cultivos[c], t_media, finca.ph, finca.alt, finca.clase) for c in ids])
    if plan.actual:
        factible = np.ones(P, bool)
        fclase = np.ones(P, np.float32)
    else:
        factible = (apt >= APTITUD_MIN).all(axis=0)
        rot = rotacion_permitida(finca.clase)
        t_venta = sum(min(x.fin, HORIZONTE) - x.inicio for x in sec if not x.cultivo.solo_descanso)
        fclase = np.where(rot == "restringida", 1 - PENALIZA_CLASE4 * t_venta / HORIZONTE, 1.0).astype(np.float32)
    return Evaluacion(plan, np.where(sin_precio, np.nan, ingreso / anios).astype(np.float32),
                      (r.deficit_mm.sum(axis=0) / anios).astype(np.float32), ibs.astype(np.float32),
                      ero, {k: np.mean(v, axis=0) for k, v in rend.items()}, factible, fclase,
                      sorted(sin_datos))


def ndti_estado(programa: F, V: F) -> F:
    """NDTI por década y muestra: donde no hay dosel (ventana desnuda: barbecho, latencia tras la
    siembra o la cosecha) vale el estado del suelo vigente (labrado, rastrojo, cobertura); con dosel
    no se usa (0,20). El estado se arrastra desde la última siembra o cosecha."""
    estado = programa.copy()
    for t in range(1, estado.size):
        if abs(estado[t] - 0.20) < 1e-6:
            estado[t] = estado[t - 1]
    desnudo = V <= V_COSECHA + 1e-6
    return np.where(desnudo, estado[:, None, None], 0.20).astype(np.float32)


def ibs_suelo_desnudo(ndvi: F, ndti: F, V: F, fechas: list[date], grupo: NDArray[np.integer],
                      ref: Referencia) -> F:
    """IBS con C2 (residuo) y C3 (perturbación) medidos solo en las ventanas sin dosel, frente a lo
    normal del mismo suelo. Sin C4: el índice no se mide con plantas puestas. Si el NDVI de la
    ventana no baja de 0,3 (hojarasca de caña), se usa el NDTI de esa misma ventana."""
    comp = componentes_desde_serie(ndvi, ndti, None, fechas, np.zeros(len(fechas)))
    desnudo = V <= V_COSECHA + 1e-6
    anios = np.array([d.year for d in fechas])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(desnudo, ndti, np.nan), axis=0)
        minimo = np.nanmean(np.stack([np.nanmin(np.where(desnudo[anios == a], ndti[anios == a], np.nan), axis=0)
                                      for a in np.unique(anios)]), axis=0)
    v = dict(comp.valores)
    v["C2"] = np.where(np.isfinite(v["C2"]), v["C2"], med).astype(np.float32)
    v["C3"] = np.where(np.isfinite(v["C3"]), v["C3"], minimo).astype(np.float32)
    return puntuar(Componentes(v, comp.n_obs), grupo, ref).ibs_sin_c4.astype(np.float32)


def planes_candidatos(cultivos: dict[str, Cultivo], entrada: Entrada) -> list[Plan]:
    descansos = [c.id for c in cultivos.values() if c.fija_n and c.rol in ("Descanso", "Ambos")]
    prod = [c.id for c in cultivos.values() if c.rol in ("Producción", "Ambos")
            and c.tipo == "Transitorio" and c.precio_cop_kg is not None]
    planes = [Plan(None, p) for p in prod] + [Plan(d, p) for p in prod for d in descansos if d != p]
    if entrada.cultivo_anterior in cultivos:
        planes.insert(0, Plan(None, entrada.cultivo_anterior, actual=True))
    if entrada.cultivo_anterior == "cana_azucar":
        # renovación de la caña con leguminosa (la oportunidad de rotación del Valle)
        planes += [Plan(d, "cana_azucar") for d in descansos]
    return planes


# ---------------------------------------------------------------- utilidad y decisión
@dataclass
class Decision:
    planes: list[Plan]
    evals: list[Evaluacion]
    utilidad: F          # (K, D, P), −inf si no es elegible
    p_mejor: F           # (K, P)
    asignacion: NDArray[np.int32]   # (P,) índice de plan por píxel (zonificado)
    zonas: list[dict]
    estrategias: dict[str, dict]


def componentes(e: Evaluacion, ref_ingreso: float) -> dict[str, F]:
    return {"ingreso": np.clip(e.ingreso / max(ref_ingreso, 1.0), 0, 1),
            "agua": 1 - np.clip(e.riego / REF_RIEGO_MM, 0, 1),
            "suelo": np.nan_to_num(e.ibs, nan=50.0) / 100.0,
            "erosion": 1 - np.clip(e.erosion / REF_EROSION_T, 0, 1)}


def utilidad(e: Evaluacion, pesos: dict[str, float], ref_ingreso: float) -> F:
    comp = componentes(e, ref_ingreso)
    s = sum(pesos.values())
    u = sum((pesos[k] / s) * np.nan_to_num(comp[k], nan=0.0) for k in pesos)
    return (u * e.factor_clase[None, :]).astype(np.float32)


def resumen(x: F) -> dict[str, float]:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(x, [10, 50, 90])
    return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def decidir(planes: list[Plan], evals: list[Evaluacion], entrada: Entrada,
            min_px_zona: int = 6, max_zonas: int = 3) -> Decision:
    candidatos = [i for i, p in enumerate(planes) if not p.actual]
    ref_ing = max((float(np.nanmedian(evals[i].ingreso)) for i in candidatos
                   if np.isfinite(evals[i].ingreso).any()), default=1.0)
    K, (D, P) = len(planes), evals[0].ibs.shape
    U = np.full((K, D, P), -np.inf, np.float32)
    for i in range(K):
        e = evals[i]
        u = utilidad(e, entrada.pesos, ref_ing)
        elegible = e.factible[None, :] & ~planes[i].actual
        if entrada.pesos.get("ingreso", 0) > 0:
            elegible = elegible & np.isfinite(e.ingreso)
        U[i] = np.where(elegible, u, -np.inf)
    gana = np.argmax(U, axis=0)                                              # (D, P)
    ok = np.isfinite(U.max(axis=0))
    p_mejor = np.stack([((gana == k) & ok).mean(axis=0) for k in range(K)]).astype(np.float32)
    eu = np.where(np.isfinite(U), U, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eu_m = np.nanmean(eu, axis=1)                                        # (K, P)

    # zonificación: plan de mayor probabilidad por píxel, máx. 3 zonas de área mínima
    pix = np.argmax(p_mejor + 1e-6 * np.nan_to_num(eu_m, nan=-1), axis=0)
    cuenta = np.bincount(pix, minlength=K)
    keep = [k for k in np.argsort(-cuenta) if cuenta[k] >= min_px_zona][:max_zonas] or [int(np.argmax(cuenta))]
    sub = np.nan_to_num(eu_m[keep], nan=-np.inf)
    asign = np.array(keep)[np.argmax(sub, axis=0)].astype(np.int32)

    # estrategia B: mejor plan uniforme (máxima utilidad esperada promedio en la finca)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prom = np.nanmean(np.where(np.isfinite(eu_m), eu_m, np.nan), axis=1)
    prom = np.where(np.array([p.actual for p in planes]), -np.inf, np.nan_to_num(prom, nan=-np.inf))
    b = int(np.argmax(prom))
    a = next((i for i, p in enumerate(planes) if p.actual), None)

    def metricas_estrategia(g: NDArray[np.int32]) -> dict[str, dict[str, float]]:
        sel = lambda campo: np.stack([getattr(evals[g[p]], campo)[:, p] for p in range(P)], axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = {k: resumen(np.nanmean(sel(k), axis=1)) for k in ("ingreso", "riego", "ibs", "erosion")}
        if not all(planes[k].actual for k in set(g.tolist())):
            u = np.stack([U[g[p], :, p] for p in range(P)], axis=1)
            out["utilidad"] = resumen(np.where(np.isfinite(u), u, np.nan).mean(axis=1))
        return out

    estr: dict[str, dict] = {}
    if a is not None:
        estr["A"] = {"nombre": "Actual", "planes": [planes[a].id], "metricas": metricas_estrategia(np.full(P, a, np.int32))}
    estr["B"] = {"nombre": "Mejor rotación para toda la finca", "planes": [planes[b].id],
                 "metricas": metricas_estrategia(np.full(P, b, np.int32))}
    estr["C"] = {"nombre": "Rotación por zonas", "planes": [planes[k].id for k in keep],
                 "metricas": metricas_estrategia(asign)}
    uB = np.where(np.isfinite(U[b]), U[b], np.nan).mean(axis=1)
    uC = np.stack([U[asign[p], :, p] for p in range(P)], axis=1)
    uC = np.where(np.isfinite(uC), uC, np.nan).mean(axis=1)
    estr["C"]["p_mejor_que_B"] = float(np.mean(uC > uB + 1e-9))

    zonas = []
    for k in keep:
        m = asign == k
        zonas.append({"plan": planes[k].id, "pixeles": int(m.sum()), "area_ha": round(float(m.sum()) * HA_POR_PIXEL, 2),
                      "p_mejor_media": float(p_mejor[k, m].mean()) if m.any() else 0.0})
    return Decision(planes, evals, U, p_mejor, asign, zonas, estr)
