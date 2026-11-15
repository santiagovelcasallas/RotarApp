"""Amplía Rangos_cultivos_Valle_del_Cauca.xlsx: leguminosas de descanso, columnas para el
modelo, precios SIPSA 2026 (data.csv) y plagas compartidas (plagas_valle_del_cauca.json).
Uso: python -I ampliar_cultivos.py <xlsx_original> <data.csv> <plagas.json> <xlsx_salida>"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

src_xlsx, src_csv, src_json, out_xlsx = sys.argv[1:5]

# ---------- 1. Filas originales (texto) ----------
wb0 = openpyxl.load_workbook(src_xlsx, data_only=True)
ws0 = wb0["Rangos"]
rows0 = [r for r in ws0.iter_rows(values_only=True) if any(c is not None for c in r)]
head_txt, orig = list(rows0[0]), [list(r) for r in rows0[1:]]
notas0 = [r[0] for r in wb0["Notas"].iter_rows(values_only=True) if r[0]][1:]

ID_ORIG = ["cana_azucar", "maiz", "arroz", "sorgo", "algodon", "hortalizas", "aji", "berenjena",
           "platano", "cacao", "frijol", "cafe", "cana_panelera", "guayaba", "chontaduro",
           "borojo", "aguacate"]
assert len(orig) == len(ID_ORIG), len(orig)

# Filas nuevas en el mismo formato de texto que el original (conocimiento general, no verificado)
NUEVAS_TXT = {
    "soya": ["Soya", "20–32 (24–28)", "500–1.000", "450–700 mm por ciclo; sensible al déficit en llenado de grano", "60–80", "16–22 MJ/m²/día", "5,5–7,5 (6,0–7,0)", "Franca a franco-arcillosa, bien drenada", "0–1.200 (óptimo 0–1.000)"],
    "mani": ["Maní", "20–32 (25–30)", "500–1.000", "450–700 mm por ciclo; requiere suelo seco en cosecha", "50–80", "16–22 MJ/m²/día", "5,8–7,0 (6,0–6,5)", "Franco-arenosa, suelta (en arcillas pesadas la cosecha se dificulta)", "0–1.200"],
    "caupi": ["Fríjol caupí", "20–35 (25–30)", "400–900", "300–500 mm por ciclo; tolera sequía", "50–80", "16–22 MJ/m²/día", "5,5–7,5 (6,0–7,0)", "Amplia, bien drenada", "0–1.500"],
    "canavalia": ["Canavalia (abono verde)", "20–32 (24–30)", "700–2.000", "400–800 mm por ciclo; tolera sequía", "50–85", "14–22 MJ/m²/día", "4,5–8,0 (5,5–7,0)", "Amplia, incluso arcillosa", "0–1.800"],
    "crotalaria": ["Crotalaria juncea (abono verde)", "20–35 (25–30)", "500–1.500", "300–600 mm por ciclo", "50–80", "16–22 MJ/m²/día", "5,0–7,5 (6,0–7,0)", "Bien drenada; no tolera encharcamiento", "0–1.500"],
    "mucuna": ["Mucuna (abono verde)", "20–30 (24–28)", "1.000–2.500", "500–900 mm por ciclo", "60–85", "14–20 MJ/m²/día", "5,0–7,5 (5,5–7,0)", "Amplia, incluso arcillosa", "0–1.500"],
    "guandul": ["Guandul", "18–35 (24–30)", "600–1.500", "400–700 mm por ciclo; muy tolerante a sequía", "50–80", "16–22 MJ/m²/día", "5,0–7,5 (5,5–7,0)", "Bien drenada; no tolera encharcamiento", "0–2.000"],
}

# ---------- 2. Atributos de rotación ----------
# tipo, rol, familia, fija_N, ciclo (días; texto), Kc medio (FAO-56 o supuesto), rendimiento típico, aporte/nota
A = {
 "cana_azucar":  ("Semiperenne", "Producción", "Poaceae", "No", "~365 por corte; renovación cada 5–6 cortes", 1.25, "100–130 t caña/ha por corte", "La ventana de renovación es la oportunidad de rotación"),
 "maiz":         ("Transitorio", "Producción", "Poaceae", "No", "110–130", 1.20, "4–7 t grano/ha", "Alta demanda de N: conviene después de una leguminosa"),
 "arroz":        ("Transitorio", "Producción", "Poaceae", "No", "120–140", 1.20, "5–7 t paddy/ha", "Inundado: alto consumo de agua, choca con el objetivo de mínimo riego"),
 "sorgo":        ("Transitorio", "Producción", "Poaceae", "No", "100–120", 1.05, "3–5 t grano/ha", "Tolera sequía; comparte barrenador y pulgón con caña y maíz"),
 "algodon":      ("Transitorio", "Producción", "Malvaceae", "No", "150–180", 1.18, "2–3 t algodón semilla/ha", "Rompe el ciclo de gramíneas; plagas propias (picudo)"),
 "hortalizas":   ("Transitorio", "Producción", "Varias", "No", "60–120", 1.05, "Varía por especie", "Afinar por especie; familia depende de la especie"),
 "aji":          ("Transitorio", "Producción", "Solanaceae", "No", "150–210", 1.05, "10–20 t fruto/ha", "No seguir con otra solanácea (berenjena, tomate)"),
 "berenjena":    ("Transitorio", "Producción", "Solanaceae", "No", "150–240", 1.05, "20–30 t fruto/ha", "No seguir con otra solanácea (ají, tomate)"),
 "platano":      ("Permanente", "Producción", "Musaceae", "No", "Perenne (primer racimo 12–14 meses)", 1.10, "15–25 t/ha/año", "Perenne: cambio de uso, no rotación"),
 "cacao":        ("Permanente", "Producción", "Malvaceae", "No", "Perenne", 1.05, "0,5–0,8 t grano seco/ha/año", "Perenne: cambio de uso, no rotación"),
 "frijol":       ("Transitorio", "Ambos", "Fabaceae", "Sí", "75–90", 1.15, "1,0–1,5 t grano/ha", "Única leguminosa del Excel original; marginal en la zona plana por temperatura"),
 "cafe":         ("Permanente", "Producción", "Rubiaceae", "No", "Perenne", 1.00, "1,0–1,5 t café pergamino/ha/año", "Perenne de ladera: fuera de la zona plana"),
 "cana_panelera":("Semiperenne", "Producción", "Poaceae", "No", "14–18 meses por corte", 1.25, "60–100 t caña/ha por corte", "Mismas plagas que la caña de azúcar"),
 "guayaba":      ("Permanente", "Producción", "Myrtaceae", "No", "Perenne", 0.90, "15–25 t/ha/año", "Perenne: cambio de uso, no rotación"),
 "chontaduro":   ("Permanente", "Producción", "Arecaceae", "No", "Perenne", 1.00, "6–10 t/ha/año", "Perenne del Pacífico: fuera de la zona plana"),
 "borojo":       ("Permanente", "Producción", "Rubiaceae", "No", "Perenne", 1.00, "Sin dato", "Perenne del Pacífico: fuera de la zona plana"),
 "aguacate":     ("Permanente", "Producción", "Lauraceae", "No", "Perenne", 0.85, "8–15 t/ha/año", "Rangos del Excel son de Hass; en zona plana aplicaría el antillano"),
 "soya":         ("Transitorio", "Ambos", "Fabaceae", "Sí", "100–120", 1.15, "2,0–2,5 t grano/ha", "Leguminosa de grano con mercado; históricamente rotada con caña y maíz en el Valle"),
 "mani":         ("Transitorio", "Ambos", "Fabaceae", "Sí", "110–130", 1.15, "1,5–2,5 t vaina/ha", "Limitado por suelos arcillosos del valle"),
 "caupi":        ("Transitorio", "Ambos", "Fabaceae", "Sí", "65–90", 1.05, "0,8–1,5 t grano/ha", "Doble propósito: grano o abono verde; ciclo corto"),
 "canavalia":    ("Descanso", "Descanso", "Fabaceae", "Sí", "90–120 (corte en floración)", 1.00, "4–8 t materia seca/ha; aporta 100–200 kg N/ha", "Grano tóxico: solo abono verde. Tolera arcilla y sequía"),
 "crotalaria":   ("Descanso", "Descanso", "Fabaceae", "Sí", "60–90 (corte en floración)", 1.00, "4–8 t materia seca/ha; aporta 80–150 kg N/ha", "Supresora de nematodos agalladores (útil antes de plátano u hortalizas)"),
 "mucuna":       ("Descanso", "Descanso", "Fabaceae", "Sí", "120–150", 1.00, "6–10 t materia seca/ha; aporta 150–250 kg N/ha", "Controla malezas sin herbicida; trepadora, no asociar sin manejo"),
 "guandul":      ("Semiperenne", "Ambos", "Fabaceae", "Sí", "150–270", 1.00, "0,8–1,5 t grano/ha; aporta 40–100 kg N/ha", "Raíz pivotante profunda: ayuda contra la compactación"),
}

# ---------- 3. Parámetros numéricos (para el modelo) ----------
# T_min, T_opt_min, T_opt_max, T_max, P_anual_min, P_anual_max, agua_min, agua_max, agua_unidad,
# HR_min, HR_max, pH_min, pH_opt_min, pH_opt_max, pH_max, alt_min, alt_opt_max, alt_max
N = {
 "cana_azucar":  (22, 24, 28, 30, 1200, 1800, 1400, 1800, "año", 60, 80, 5.5, 6.0, 7.5, 8.0, 0, 1100, 1600),
 "maiz":         (20, 24, 28, 30, 600, 1000, 500, 800, "ciclo", 50, 80, 5.8, 5.8, 7.0, 7.0, 0, 1000, 1800),
 "arroz":        (24, 25, 30, 32, 1000, 2000, 800, 1200, "ciclo", 70, 85, 5.5, 6.0, 6.5, 7.0, 0, 1000, 1000),
 "sorgo":        (20, 25, 32, 38, 400, 800, 450, 650, "ciclo", 40, 70, 5.5, 6.0, 7.5, 8.0, 0, 1500, 1500),
 "algodon":      (22, 25, 30, 32, 600, 1000, 700, 1000, "ciclo", 50, 70, 6.0, 6.0, 7.5, 7.5, 0, 1000, 1000),
 "hortalizas":   (12, 15, 28, 30, 600, 1500, 400, 700, "ciclo", 60, 80, 6.0, 6.0, 7.0, 7.0, 0, 2800, 2800),
 "aji":          (20, 24, 28, 30, 600, 1250, 500, 700, "ciclo", 60, 80, 5.8, 5.8, 7.0, 7.0, 0, 1500, 1500),
 "berenjena":    (22, 24, 28, 30, 600, 1200, 500, 800, "ciclo", 60, 75, 5.5, 6.0, 6.5, 6.8, 0, 1500, 1500),
 "platano":      (22, 26, 28, 30, 1800, 2500, 1200, 1440, "año", 70, 85, 5.5, 6.0, 6.5, 7.0, 0, 1000, 1500),
 "cacao":        (22, 24, 26, 28, 1500, 2500, None, None, "año", 70, 85, 5.5, 6.0, 6.5, 7.0, 0, 800, 1000),
 "frijol":       (16, 18, 24, 24, 500, 1200, 300, 500, "ciclo", 60, 75, 6.0, 6.0, 7.0, 7.0, 0, 1600, 1600),
 "cafe":         (18, 19, 22, 24, 1800, 2800, 1200, None, "año", 70, 85, 5.0, 5.0, 6.0, 6.0, 1000, 1800, 2000),
 "cana_panelera":(18, 20, 26, 28, 1200, 2000, None, None, "año", 70, 85, 5.5, 6.0, 7.0, 7.5, 800, 1600, 1800),
 "guayaba":      (18, 23, 28, 30, 1000, 2000, None, None, "año", 60, 80, 5.5, 6.0, 7.0, 7.5, 0, 1800, 1800),
 "chontaduro":   (22, 24, 28, 28, 2000, 5000, None, None, "año", 80, 90, 4.5, 4.5, 6.0, 6.0, 0, 800, 1200),
 "borojo":       (24, 24, 28, 28, 3000, 7000, None, None, "año", 85, 90, 4.5, 4.5, 6.0, 6.0, 0, 500, 800),
 "aguacate":     (16, 18, 22, 24, 1200, 2000, None, None, "año", 60, 80, 5.5, 6.0, 6.0, 6.5, 1800, 2600, 2600),
 "soya":         (20, 24, 28, 32, 500, 1000, 450, 700, "ciclo", 60, 80, 5.5, 6.0, 7.0, 7.5, 0, 1000, 1200),
 "mani":         (20, 25, 30, 32, 500, 1000, 450, 700, "ciclo", 50, 80, 5.8, 6.0, 6.5, 7.0, 0, 1200, 1200),
 "caupi":        (20, 25, 30, 35, 400, 900, 300, 500, "ciclo", 50, 80, 5.5, 6.0, 7.0, 7.5, 0, 1500, 1500),
 "canavalia":    (20, 24, 30, 32, 700, 2000, 400, 800, "ciclo", 50, 85, 4.5, 5.5, 7.0, 8.0, 0, 1800, 1800),
 "crotalaria":   (20, 25, 30, 35, 500, 1500, 300, 600, "ciclo", 50, 80, 5.0, 6.0, 7.0, 7.5, 0, 1500, 1500),
 "mucuna":       (20, 24, 28, 30, 1000, 2500, 500, 900, "ciclo", 60, 85, 5.0, 5.5, 7.0, 7.5, 0, 1500, 1500),
 "guandul":      (18, 24, 30, 35, 600, 1500, 400, 700, "ciclo", 50, 80, 5.0, 5.5, 7.0, 7.5, 0, 2000, 2000),
}
NCOLS = ["T_min", "T_opt_min", "T_opt_max", "T_max", "P_anual_min_mm", "P_anual_max_mm",
         "agua_min_mm", "agua_max_mm", "agua_unidad", "HR_min", "HR_max", "pH_min", "pH_opt_min",
         "pH_opt_max", "pH_max", "alt_min_m", "alt_opt_max_m", "alt_max_m"]

# ---------- 4. Precios SIPSA 2026 (data.csv) ----------
SIPSA = {
 "maiz": "Maíz amarillo trillado", "arroz": "Arroz de primera", "platano": "Plátano hartón verde",
 "frijol": "Fríjol calima", "guayaba": "Guayaba pera", "aguacate": "Aguacate común",
 "hortalizas": "Tomate chonto regional", "aji": "Pimentón",
}
precios = defaultdict(list)
mercados = defaultdict(set)
fechas = set()
with open(src_csv, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        art = r["ARTÍCULO"]
        precios[art].append(float(r["PROMEDIO"].replace(".", "").replace(",", ".")))
        mercados[art].add(r["FUENTE"])
        fechas.add(r["FECHA"][:10])
f0, f1 = min(fechas), max(fechas)

def q(v, p):
    v = sorted(v); i = (len(v) - 1) * p; lo = int(i)
    return v[lo] if lo + 1 >= len(v) else v[lo] + (v[lo + 1] - v[lo]) * (i - lo)

precio_row = {}
for cid, art in SIPSA.items():
    v = precios[art]
    assert v, art
    precio_row[cid] = (art, round(st.median(v)), round(q(v, .1)), round(q(v, .9)), len(v), len(mercados[art]))

# ---------- 5. Plagas compartidas (plagas_valle_del_cauca.json) ----------
pj = json.load(open(src_json, encoding="utf-8"))
import re
def genero(agente):
    # Agrupa por género del agente para no contar dos veces la misma plaga con nombres distintos
    t = re.split(r"[\s,(/]+", agente.strip())[0].lower()
    return agente.strip().lower() if t in ("por", "virus", "género", "sugarcane", "pokkah") else t

compart = defaultdict(lambda: defaultdict(set))  # a -> b -> {género}
n_plagas = defaultdict(int)
for p in pj["plagas"]:
    a = p["cultivo_id"]; n_plagas[a] += 1
    g = genero(p["agente"])
    for b in p.get("otros_hospedantes") or []:
        if b != a:
            compart[a][b].add(g); compart[b][a].add(g)
for m in pj["matriz_hospedantes"]:
    g = genero(m["plaga"])
    cs = list(m["cultivos"])
    for a in cs:
        for b in cs:
            if a != b:
                compart[a][b].add(g)
GEN = compart

# ---------- 6. Chequeo preliminar zona plana ----------
REF = {"T": 24.0, "P": 1050, "alt": 1000, "HR": 72}  # Palmira aprox.; supuesto a reemplazar por NASA POWER
def zona_plana(cid):
    t0, to0, to1, t1, p0 = N[cid][0:5]
    hr0, hr1 = N[cid][9:11]
    a0, aopt, a1 = N[cid][15:18]
    T, alt = REF["T"], REF["alt"]
    no, marg = [], []
    if not (a0 <= alt <= a1): no.append("altitud")
    if not (t0 < T < t1): (no if not (t0 <= T <= t1) else marg).append("temperatura en el límite" if t0 <= T <= t1 else "temperatura")
    elif not (to0 <= T <= to1): marg.append("temperatura fuera del óptimo")
    if alt > aopt and not no: marg.append("altitud fuera del óptimo")
    if not (hr0 <= REF["HR"] <= hr1): marg.append("humedad relativa")
    riego = A[cid][0] in ("Permanente", "Semiperenne") and REF["P"] < p0
    if no: base = "No apto (" + ", ".join(no + marg) + ")"
    elif marg: base = "Marginal (" + ", ".join(marg) + ")"
    else: base = "Apto"
    return base + ("; necesita riego" if riego and not no else "")

# ---------- 7. Escribir Excel ----------
wb = openpyxl.Workbook()
H = Font(bold=True, color="FFFFFF"); HF = PatternFill("solid", fgColor="2F7D4F")
NEW = PatternFill("solid", fgColor="EAF4EC")
def header(ws, cols):
    ws.append(cols)
    for c in ws[1]:
        c.font = H; c.fill = HF; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
def widths(ws, w=22):
    for i in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")

ids = ID_ORIG + list(NUEVAS_TXT)
txt = {cid: orig[i] for i, cid in enumerate(ID_ORIG)} | NUEVAS_TXT

ws = wb.active; ws.title = "Rangos"
header(ws, ["id"] + head_txt + ["Origen", "Tipo", "Rol en rotación", "Familia", "Fija N",
            "Ciclo (días)", "Kc medio", "Rendimiento típico", "Nota de rotación",
            "N.º plagas en JSON", "Comparte plagas con", "Zona plana (preliminar)"])
for cid in ids:
    tipo, rol, fam, fn, ciclo, kc, rend, nota = A[cid]
    comp = ", ".join(f"{b} ({len(s)})" for b, s in sorted(compart[cid].items(), key=lambda x: -len(x[1])))
    ws.append([cid] + list(txt[cid]) + ["Agregado" if cid in NUEVAS_TXT else "Original", tipo, rol, fam, fn,
              ciclo, kc, rend, nota, n_plagas.get(cid, 0) or "Sin datos en el JSON", comp or "—", zona_plana(cid)])
    if cid in NUEVAS_TXT:
        for c in ws[ws.max_row]: c.fill = NEW
widths(ws)

ws = wb.create_sheet("Parametros_modelo")
header(ws, ["id", "Cultivo"] + NCOLS + ["Kc_medio", "fija_N", "rol"])
for cid in ids:
    ws.append([cid, txt[cid][0]] + list(N[cid]) + [A[cid][5], A[cid][3] == "Sí", A[cid][1]])
widths(ws, 14)

ws = wb.create_sheet("Precios_SIPSA_2026")
header(ws, ["id", "Artículo SIPSA usado", "Mediana COP/kg", "P10 COP/kg", "P90 COP/kg", "Registros", "Mercados"])
for cid in ids:
    if cid in precio_row: ws.append([cid, *precio_row[cid]])
    else: ws.append([cid, "Sin artículo en data.csv", None, None, None, None, None])
widths(ws, 18)

ws = wb.create_sheet("Plagas_compartidas")
cids_p = [c for c in ids if c in compart or c in n_plagas]
header(ws, ["Cultivo anterior \\ siguiente"] + cids_p)
for a in cids_p:
    ws.append([a] + [len(compart[a][b]) if b != a else "—" for b in cids_p])
widths(ws, 12)

ws = wb.create_sheet("Notas")
ws.append(["Notas"]); ws["A1"].font = Font(bold=True)
for n in notas0: ws.append([n])
for n in [
    f"v2 (ampliación): se agregan 7 leguminosas (filas en verde). Sus rangos, ciclos, rendimientos y aportes de N son conocimiento general, no verificados: validar con Agrosavia.",
    "Kc medio: FAO-56 para cultivos con valor tabulado; guayaba, chontaduro, borojó y abonos verdes son supuestos.",
    "Rendimientos típicos: orden de magnitud para manejo convencional; el manejo agroecológico puede rendir distinto. Calibrar con EVA (UPRA) por municipio.",
    f"Precios: SIPSA (DANE), mayoristas del Valle, {f0} a {f1}. Son precios de mercado mayorista, NO costos de producción ni precio en finca. Maíz y arroz son productos procesados (trillado, de primera); hortalizas usa tomate chonto como referencia y ají usa pimentón como aproximación.",
    "Faltan: costos de producción por cultivo (no están en data.csv) y precios de caña, sorgo, algodón, cacao, café, panela, soya, maní, caupí y guandul.",
    "Plagas compartidas: número de agentes (por género) comunes entre dos cultivos según 'otros_hospedantes' y 'matriz_hospedantes' del JSON. Varias relaciones son inferencia (ver 'nota' en el JSON). Las leguminosas nuevas no están en el JSON.",
    f"Zona plana (preliminar): comparación con un clima de referencia supuesto (T media {REF['T']} °C, {REF['P']} mm/año, {REF['alt']} m). Se reemplaza por NASA POWER por píxel en el modelo.",
]:
    ws.append([n])
ws.column_dimensions["A"].width = 140
for row in ws.iter_rows(min_row=2):
    row[0].alignment = Alignment(wrap_text=True)

wb.save(out_xlsx)
print("OK", out_xlsx, len(ids), "cultivos", f0, f1)
for cid in ids: print(f"{cid:14s} {zona_plana(cid)}")
print({k: v[1:4] for k, v in precio_row.items()})
