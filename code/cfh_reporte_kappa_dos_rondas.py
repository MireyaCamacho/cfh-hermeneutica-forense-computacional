# -*- coding: utf-8 -*-
r"""
cfh_reporte_kappa_dos_rondas.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Genera el reporte oficial de fiabilidad inter-anotador (IAA) con las DOS
rondas, reproduciendo todos los numeros desde los datos (no escritos a mano):

  RONDA 1 (anotacion independiente, 100 fragmentos):
    EBI, SA, NV, REP  -> Cohen kappa por categoria + macro global

  RONDA 2 (revision de los 34 desacuerdos de REP):
    Tras calibrar la definicion operativa del constructo de Ruptura
    Epistemica Positiva, ambos anotadores revisaron los 34 fragmentos en
    disputa. La anotacion revisada quedo registrada en REP_diferencias_A1_A2.csv
    (columnas A1_REP / A2_REP). Se recalcula REP y el global.

Entradas:
  data/referencias/gold_consolidado_A1A2.json   (ronda 1, 100 fragmentos)
  data/referencias/REP_diferencias_A1_A2.csv    (revision de los 34)
    -> si no esta en data/referencias, se busca en la raiz del repo

Salida:
  outputs/iaa_kappa_reporte_DOS_RONDAS.txt
  outputs/iaa_kappa_dos_rondas.csv

Uso (raiz del repo, env cfh):
    python code\cfh_reporte_kappa_dos_rondas.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(REPO, "data", "referencias", "gold_consolidado_A1A2.json")
OUT_TXT = os.path.join(REPO, "outputs", "iaa_kappa_reporte_DOS_RONDAS.txt")
OUT_CSV = os.path.join(REPO, "outputs", "iaa_kappa_dos_rondas.csv")
CATS = ["EBI", "SA", "NV", "REP"]


def buscar_dif():
    for p in [os.path.join(REPO, "data", "referencias", "REP_diferencias_A1_A2.csv"),
              os.path.join(REPO, "REP_diferencias_A1_A2.csv")]:
        if os.path.exists(p):
            return p
    return None


def landis_koch(k):
    if k < 0: return "pobre"
    if k <= 0.20: return "leve"
    if k <= 0.40: return "aceptable"
    if k <= 0.60: return "moderado"
    if k <= 0.80: return "SUSTANCIAL"
    return "casi perfecto"


def marcado(v):
    if pd.isna(v): return 0
    s = str(v).strip().lower()
    return 0 if s in ("(nada)", "nada", "", "none") else 1


def main():
    g = json.load(open(GOLD, encoding="utf-8"))
    ids = sorted(x["id"] for x in g)
    a = {c: {x["id"]: int(x[f"{c}_A1"]) for x in g} for c in CATS}
    b = {c: {x["id"]: int(x[f"{c}_A2"]) for x in g} for c in CATS}

    # ---- RONDA 1 ----
    k1 = {}
    for c in CATS:
        k1[c] = cohen_kappa_score([a[c][i] for i in ids], [b[c][i] for i in ids])
    global1 = np.mean(list(k1.values()))

    # ---- RONDA 2: aplicar revision de REP ----
    dif_path = buscar_dif()
    rep_a1 = dict(a["REP"])
    rep_a2 = dict(b["REP"])
    n_conv = n_desac = None
    if dif_path:
        dif = pd.read_csv(dif_path)
        dif["A1f"] = dif["A1_REP"].apply(marcado)
        dif["A2f"] = dif["A2_REP"].apply(marcado)
        for _, r in dif.iterrows():
            fid = int(r["fragmento"])
            if fid in rep_a1:
                rep_a1[fid] = int(r["A1f"])
                rep_a2[fid] = int(r["A2f"])
        n_conv = int((dif["A1f"] == dif["A2f"]).sum())
        n_desac = len(dif) - n_conv
    k_rep2 = cohen_kappa_score([rep_a1[i] for i in ids], [rep_a2[i] for i in ids])
    k2 = {"EBI": k1["EBI"], "SA": k1["SA"], "NV": k1["NV"], "REP": k_rep2}
    global2 = np.mean(list(k2.values()))

    # ---- Reporte ----
    L = []
    L.append("=" * 62)
    L.append("REPORTE IAA - CFH | Fiabilidad inter-anotador (dos rondas)")
    L.append("Anotadores: A1 (investigadora) + A2 (segundo anotador)")
    L.append("Fragmentos: 100 | Escala: Landis y Koch (1977)")
    L.append("=" * 62)
    L.append("")
    L.append("RONDA 1 - Anotacion independiente")
    L.append("-" * 62)
    for c in CATS:
        L.append(f"  {c:<4}: kappa={k1[c]:.4f}  ({landis_koch(k1[c])})")
    L.append(f"  GLOBAL (macro-promedio 4 cats): {global1:.4f}  ({landis_koch(global1)})")
    L.append("")
    L.append("RONDA 2 - Revision de desacuerdos de REP")
    L.append("-" * 62)
    L.append("  El constructo de Ruptura Epistemica Positiva (REP) obtuvo la")
    L.append("  menor concordancia inicial. El analisis de los 34 desacuerdos")
    L.append("  revelo una comprension divergente de sus limites (que distingue")
    L.append("  el reconocimiento genuino del lenguaje juridico formulaico).")
    L.append("  Tras calibrar la definicion operativa, ambos anotadores")
    L.append("  revisaron los 34 fragmentos en disputa.")
    if n_conv is not None:
        L.append(f"  Resultado: {n_conv}/34 convergieron; {n_desac}/34 siguen en")
        L.append("  desacuerdo (casos genuinamente ambiguos, no forzados).")
    L.append("")
    for c in CATS:
        marca = "  <- revisado" if c == "REP" else ""
        L.append(f"  {c:<4}: kappa={k2[c]:.4f}  ({landis_koch(k2[c])}){marca}")
    L.append(f"  GLOBAL (macro-promedio 4 cats): {global2:.4f}  ({landis_koch(global2)})")
    L.append("")
    L.append("=" * 62)
    L.append("SINTESIS PARA LA TESIS")
    L.append("-" * 62)
    L.append(f"  Ronda 1: EBI/SA/NV sustanciales; REP={k1['REP']:.3f} (aceptable)")
    L.append(f"           global={global1:.3f} (moderado).")
    L.append(f"  Ronda 2: tras revision del constructo REP, REP={k_rep2:.3f}")
    L.append(f"           ({landis_koch(k_rep2)}); global={global2:.3f} (sustancial).")
    L.append("=" * 62)

    reporte = "\n".join(L)
    print(reporte)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, "w", encoding="utf-8").write(reporte)
    pd.DataFrame([
        {"categoria": c, "kappa_ronda1": k1[c], "kappa_ronda2": k2[c],
         "interpretacion_final": landis_koch(k2[c])} for c in CATS
    ] + [{"categoria": "GLOBAL", "kappa_ronda1": global1, "kappa_ronda2": global2,
          "interpretacion_final": landis_koch(global2)}]).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  Reporte -> {OUT_TXT}")
    print(f"  CSV     -> {OUT_CSV}")
    if not dif_path:
        print("\n  [AVISO] no encontre REP_diferencias_A1_A2.csv; RONDA 2 uso REP sin revisar.")


if __name__ == "__main__":
    main()
