# -*- coding: utf-8 -*-
"""
cfh_analizar_desacuerdos_rep.py
================================
Analiza los desacuerdos de la categoria REP entre los dos anotadores (A1
investigador, A2 segundo anotador) del archivo de refinamiento, para entender
POR QUE el kappa de REP es bajo (0.31).

NOTA: ambos anotadores ajustaban el puntaje con nombres propios (de victimas y
comparecientes), asi que el desacuerdo NO viene de un criterio de nombres
distinto. Este script muestra los desacuerdos REALES (que marco cada uno en
los fragmentos donde difieren) para diagnosticar la causa sin asumir hipotesis.

Para cada fragmento donde A1 y A2 difieren en REP, muestra:
  - quien marco REP (solo A1 o solo A2)
  - el span que cada uno marco
  - un extracto del texto del fragmento

Salida:
  outputs/desacuerdos_rep.csv
  consola: resumen + casos para inspeccion manual.

Uso:
    python cfh_analizar_desacuerdos_rep.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
XLSX = BASE / "Refinamiento_CFH_IAA_Validacion_A2.xlsx"

# nombre propio heuristico: 2+ palabras con inicial mayuscula seguidas,
# o palabra toda en mayusculas de >=3 letras (nombres en autos van asi)
NOMBRE_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"
                       r"|[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\b")


def fid_valido(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def presencia(cell):
    return pd.notna(cell) and str(cell).strip() != ""


def contiene_nombre(txt):
    if not txt or pd.isna(txt):
        return False
    return bool(NOMBRE_RE.search(str(txt)))


def main():
    a1 = pd.read_excel(XLSX, sheet_name="A1_INVESTIGADOR")
    a2 = pd.read_excel(XLSX, sheet_name="ANOTACION")

    # indexar por #
    a1d, a2d = {}, {}
    for _, r in a1.iterrows():
        fid = fid_valido(r["#"])
        if fid is not None:
            a1d[fid] = r
    for _, r in a2.iterrows():
        fid = fid_valido(r["#"])
        if fid is not None:
            a2d[fid] = r

    comunes = sorted(set(a1d) & set(a2d))

    filas = []
    for i in comunes:
        r1, r2 = a1d[i], a2d[i]
        p1 = presencia(r1.get("REP_A1"))
        p2 = presencia(r2.get("REP"))
        if p1 != p2:  # discordancia en REP
            span1 = str(r1.get("REP_A1")) if p1 else ""
            span2 = str(r2.get("REP")) if p2 else ""
            texto = str(r2.get("FRAGMENTO (texto original)", ""))[:200]
            quien = "solo_A1" if p1 else "solo_A2"
            span_marcado = span1 if p1 else span2
            filas.append({
                "#": i, "quien_marco_REP": quien,
                "span_A1": span1, "span_A2": span2,
                "texto_frag": texto,
            })

    df = pd.DataFrame(filas)
    df.to_csv(BASE / "outputs" / "desacuerdos_rep.csv",
              index=False, encoding="utf-8-sig")

    print("=" * 64)
    print(f"DESACUERDOS EN REP: {len(df)} fragmentos")
    print("=" * 64)
    n_a1 = (df["quien_marco_REP"] == "solo_A1").sum()
    n_a2 = (df["quien_marco_REP"] == "solo_A2").sum()
    print(f"  solo A1 (Mireya) marcó REP: {n_a1}")
    print(f"  solo A2 marcó REP:          {n_a2}")

    print("\n  CASOS 'solo A1' (Mireya marcó REP, A2 no):")
    for _, r in df[df["quien_marco_REP"] == "solo_A1"].iterrows():
        print(f"    #{r['#']:3d}  REP_A1='{r['span_A1'][:60]}'")
        print(f"          texto: {r['texto_frag'][:110]}")

    print("\n  CASOS 'solo A2' (A2 marcó REP, Mireya no):")
    for _, r in df[df["quien_marco_REP"] == "solo_A2"].iterrows():
        print(f"    #{r['#']:3d}  REP_A2='{r['span_A2'][:60]}'")
        print(f"          texto: {r['texto_frag'][:110]}")

    print(f"\n  Guardado: outputs/desacuerdos_rep.csv (con texto completo)")
    print("\n  LECTURA: revisa los spans para ver si el desacuerdo es de criterio")
    print("  (que cuenta como reparacion) o de deteccion (uno lo vio, otro no).")


if __name__ == "__main__":
    main()
