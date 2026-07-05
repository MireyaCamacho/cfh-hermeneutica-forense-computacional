# -*- coding: utf-8 -*-
"""
cfh_recalcular_kappa_iaa.py
============================
Recalcula el Cohen kappa del IAA desde el archivo REAL de refinamiento
(Refinamiento_CFH_IAA_Validacion_A2.xlsx), que contiene las anotaciones de
los DOS anotadores en dos hojas:
  - A1_INVESTIGADOR: anotaciones de Mireya (cols EBI_A1, SA_A1, NV_A1, REP_A1)
  - ANOTACION:       anotaciones del segundo anotador (cols EBI, SA, NV, REP)

Cada celda contiene el/los span(s) marcados (texto) o NaN si no se marcó esa
categoría en ese fragmento. Se convierte a presencia/ausencia binaria por
fragmento y categoría, se alinean los fragmentos comunes por '#', y se calcula:
  - Cohen kappa por categoría (EBI, SA, NV, REP)
  - kappa global = macro-promedio de las 4
  - acuerdo observado por categoría
  - conteos de concordancia/discordancia

Interpretación por Landis y Koch (1977):
  <0.00 pobre | 0.00-0.20 leve | 0.21-0.40 aceptable | 0.41-0.60 moderado
  0.61-0.80 SUSTANCIAL | 0.81-1.00 casi perfecto

Salida:
  outputs/iaa_kappa_reporte.txt
  outputs/iaa_kappa_resultados.csv

Uso:
    python cfh_recalcular_kappa_iaa.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path(".")
XLSX = BASE / "Refinamiento_CFH_IAA_Validacion_A2.xlsx"
OUT_TXT = BASE / "outputs" / "iaa_kappa_reporte.txt"
OUT_CSV = BASE / "outputs" / "iaa_kappa_resultados.csv"

CATEGORIAS = ["EBI", "SA", "NV", "REP"]


def landis_koch(k):
    if np.isnan(k):
        return "N/A"
    if k < 0.00:
        return "pobre"
    if k <= 0.20:
        return "leve"
    if k <= 0.40:
        return "aceptable"
    if k <= 0.60:
        return "moderado"
    if k <= 0.80:
        return "SUSTANCIAL"
    return "casi perfecto"


def presencia(cell):
    """1 si la celda tiene texto (marcó la categoría), 0 si NaN/vacía."""
    return int(pd.notna(cell) and str(cell).strip() != "")


def main():
    a1 = pd.read_excel(XLSX, sheet_name="A1_INVESTIGADOR")
    a2 = pd.read_excel(XLSX, sheet_name="ANOTACION")

    # matrices binarias por fragmento (#) y categoria
    # (se saltan filas cuyo # no sea numerico: A1 trae filas de estadisticas al final)
    def fid_valido(v):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    a1_bin, a2_bin = {}, {}
    for _, r in a1.iterrows():
        fid = fid_valido(r["#"])
        if fid is None:
            continue
        a1_bin[fid] = {c: presencia(r.get(c + "_A1")) for c in CATEGORIAS}
    for _, r in a2.iterrows():
        fid = fid_valido(r["#"])
        if fid is None:
            continue
        a2_bin[fid] = {c: presencia(r.get(c)) for c in CATEGORIAS}

    comunes = sorted(set(a1_bin) & set(a2_bin))

    rep = []
    rep.append("=" * 62)
    rep.append("REPORTE IAA — Hermenéutica Forense Computacional (CFH)")
    rep.append("Fuente: Refinamiento_CFH_IAA_Validacion_A2.xlsx")
    rep.append(f"Anotadores: 2 (A1 investigador + A2 segundo anotador)")
    rep.append(f"Fragmentos comunes evaluados: {len(comunes)}")
    rep.append("Escala: Landis y Koch (1977)")
    rep.append("=" * 62)

    filas, kappas = [], []
    for c in CATEGORIAS:
        y1 = [a1_bin[i][c] for i in comunes]
        y2 = [a2_bin[i][c] for i in comunes]
        try:
            k = cohen_kappa_score(y1, y2)
        except Exception:
            k = float("nan")
        po = sum(a == b for a, b in zip(y1, y2)) / len(comunes)
        n1, n2 = sum(y1), sum(y2)
        ambos = sum(a == 1 and b == 1 for a, b in zip(y1, y2))
        ninguno = sum(a == 0 and b == 0 for a, b in zip(y1, y2))
        disc = len(comunes) - ambos - ninguno
        rep.append(f"\n--- {c} ---")
        rep.append(f"  kappa Cohen:     {k:.4f}  -> {landis_koch(k)}")
        rep.append(f"  acuerdo obs.:    {po:.4f} ({po*100:.1f}%)")
        rep.append(f"  A1 marcó:        {n1} fragmentos")
        rep.append(f"  A2 marcó:        {n2} fragmentos")
        rep.append(f"  ambos presente:  {ambos}   ambos ausente: {ninguno}   discord.: {disc}")
        filas.append({"categoria": c, "kappa": round(k, 4) if not np.isnan(k) else None,
                      "acuerdo_obs": round(po, 4), "n_A1": n1, "n_A2": n2,
                      "ambos_presentes": ambos, "discordancias": disc,
                      "interpretacion": landis_koch(k)})
        kappas.append(k if not np.isnan(k) else 0)

    k_global = float(np.mean(kappas))
    rep.append("\n" + "=" * 62)
    rep.append(f"kappa GLOBAL (macro-promedio 4 categorías): {k_global:.4f}")
    rep.append(f"  -> {landis_koch(k_global)} (Landis y Koch 1977)")
    rep.append("=" * 62)

    # tambien un kappa "colapsado" (cualquier categoria marcada vs no) como referencia
    yy1 = [int(any(a1_bin[i][c] for c in CATEGORIAS)) for i in comunes]
    yy2 = [int(any(a2_bin[i][c] for c in CATEGORIAS)) for i in comunes]
    try:
        k_any = cohen_kappa_score(yy1, yy2)
        rep.append(f"\n(Referencia) kappa presencia-de-cualquier-marca: {k_any:.4f} "
                   f"-> {landis_koch(k_any)}")
    except Exception:
        pass

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(rep), encoding="utf-8")
    pd.DataFrame(filas).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n".join(rep))
    print(f"\nGuardado: {OUT_TXT}")
    print(f"          {OUT_CSV}")


if __name__ == "__main__":
    main()
