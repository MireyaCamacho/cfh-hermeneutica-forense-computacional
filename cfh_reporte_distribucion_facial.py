# -*- coding: utf-8 -*-
"""
cfh_reporte_distribucion_facial.py — SOLO LECTURA
Genera la TABLA DE REFERENCIA de intensidad de Action Units de distress
(AU1, AU4, AU15, AU17) via MediaPipe FaceLandmarker en el corpus del Macrocaso 003.

Produce caracterizacion descriptiva (no documentada antes para habla judicial en
espanol) para citar en Cap.5 y eventual paper:
  · distribucion global del distress (mu, sigma, percentiles, rango)
  · por AU individual (cada una de las 4 de distress + AU6, AU12 de referencia)
  · por subcaso
  · por compareciente (los 47)
Guarda un CSV con la tabla por compareciente y un TXT con el resumen citable.

Uso: python cfh_reporte_distribucion_facial.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
CAPA3 = BASE / "outputs" / "capa3"
MIN_VENTANAS = 25
PREF = {"Catatumbo":"catatumbo","Dabeiba":"dabeiba","Casanare":"casanare",
        "Huila":"huila","CostaCaribe":"costa_caribe"}
AUS_DISTRESS = ["AU1","AU4","AU15","AU17"]
AUS_TODAS = ["AU1","AU4","AU6","AU12","AU15","AU17"]

def t_a_seg(s):
    s=str(s).strip()
    if ":" not in s: return None
    p=[int(x) for x in s.split(":")]
    return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]

# ── recolectar por compareciente: media de cada AU + distress agregado ──
filas = []
for sub, pref in PREF.items():
    aus_csv = CAPA3 / f"aus_{pref}_v3.csv"
    inv_csv = BASE / "data" / "marcacion" / f"inventario_{sub}.csv"
    if not aus_csv.exists() or not inv_csv.exists():
        print(f"[{sub}] sin datos -- saltado"); continue
    aus = pd.read_csv(aus_csv)
    tiene_ident = "identidad" in aus.columns
    inv = pd.read_csv(inv_csv)
    inv = inv[inv["uso"].astype(str).str.upper()=="ANALISIS"].copy()
    inv["ini_s"]=inv["inicio"].apply(t_a_seg); inv["fin_s"]=inv["fin"].apply(t_a_seg)
    inv = inv[inv["ini_s"].notna() & inv["fin_s"].notna()]
    acc = {}
    for _, r in inv.iterrows():
        a0,a1,ident = r["ini_s"], r["fin_s"], r["identidad"]
        if tiene_ident:
            subf = aus[(aus["identidad"]==ident) & (aus["end"]>a0) & (aus["start"]<a1)]
        else:
            subf = aus[(aus["end"]>a0) & (aus["start"]<a1)]
        if len(subf) < MIN_VENTANAS: continue
        prev = acc.get(ident)
        medias = {au: float(subf[au].fillna(0).mean()) for au in AUS_TODAS if au in subf.columns}
        n = len(subf)
        if prev is None:
            acc[ident] = {**medias, "_n": n}
        else:
            for au in medias:
                acc[ident][au] = (acc[ident][au]*acc[ident]["_n"] + medias[au]*n)/(acc[ident]["_n"]+n)
            acc[ident]["_n"] += n
    for ident, m in acc.items():
        d = np.mean([m[au] for au in AUS_DISTRESS if au in m])
        fila = {"subcaso": sub, "compareciente": ident, "n_frames": m["_n"],
                "distress": d}
        for au in AUS_TODAS:
            fila[au] = m.get(au, np.nan)
        filas.append(fila)

df = pd.DataFrame(filas)
if df.empty:
    print("Sin datos."); raise SystemExit

# ── guardar tabla por compareciente ──
out_csv = CAPA3 / "referencia_AUs_distress_corpus.csv"
df.sort_values("distress").to_csv(out_csv, index=False, encoding="utf-8-sig")

# ── resumen citable ──
lineas = []
def P(s=""): lineas.append(s); print(s)

P("="*78)
P("TABLA DE REFERENCIA — Intensidad de Action Units de distress")
P("MediaPipe FaceLandmarker | Corpus Macrocaso 003 JEP | habla judicial espanol")
P("="*78)
P(f"\nN = {len(df)} comparecientes | {int(df['n_frames'].sum())} frames | 5 subcasos")
P(f"AUs de distress: AU1 (inner brow raiser), AU4 (brow lowerer),")
P(f"                 AU15 (lip corner depressor), AU17 (chin raiser)")
P(f"  (Ekman et al. 2002: expresion prototipica de tristeza/distress)")

d = df["distress"]
P(f"\n-- DISTRESS AGREGADO (media de las 4 AUs) --")
P(f"  media (mu)      = {d.mean():.4f}")
P(f"  desv std (sigma)= {d.std():.4f}")
P(f"  minimo          = {d.min():.4f}")
P(f"  percentil 25    = {d.quantile(.25):.4f}")
P(f"  mediana         = {d.median():.4f}")
P(f"  percentil 75    = {d.quantile(.75):.4f}")
P(f"  percentil 90    = {d.quantile(.90):.4f}")
P(f"  maximo          = {d.max():.4f}")

P(f"\n-- POR ACTION UNIT (media +/- std sobre {len(df)} comparecientes) --")
P(f"  {'AU':6s} {'media':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
for au in AUS_TODAS:
    if au in df.columns:
        c = df[au]
        P(f"  {au:6s} {c.mean():8.4f} {c.std():8.4f} {c.min():8.4f} {c.max():8.4f}")

P(f"\n-- POR SUBCASO (distress medio) --")
P(f"  {'subcaso':14s} {'n':>4s} {'mu_distress':>12s} {'std':>8s}")
for sub, g in df.groupby("subcaso"):
    P(f"  {sub:14s} {len(g):4d} {g['distress'].mean():12.4f} {g['distress'].std():8.4f}")

P(f"\n-- VALORES PARA NORMALIZACION ICM (z-score congelado) --")
P(f"  FACIAL_MU    = {d.mean():.4f}")
P(f"  FACIAL_SIGMA = {d.std():.4f}")
P(f"  (estos son los valores que el ICM usa para z = (distress-mu)/sigma)")

P(f"\n[GUARDADO] tabla por compareciente: {out_csv.name}")
P("\nNOTA PARA TESIS/PAPER: valores especificos de este corpus, configuracion de")
P("MediaPipe y poblacion (mestiza/afrocolombiana). NO es umbral universal; sirve")
P("como referencia descriptiva para contextos similares (justicia transicional,")
P("espanol). Transferibilidad requiere validacion (Sen et al. 2024; Buolamwini &")
P("Gebru 2018 — auditoria intersectional pendiente).")

out_txt = CAPA3 / "referencia_AUs_distress_resumen.txt"
out_txt.write_text("\n".join(lineas), encoding="utf-8")
print(f"\n[GUARDADO] resumen citable: {out_txt}")
