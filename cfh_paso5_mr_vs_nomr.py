# -*- coding: utf-8 -*-
"""
cfh_paso5_mr_vs_nomr.py
========================
PASO 5 (guia de cierre) — la conclusion de la tesis.

Compara los indices por compareciente entre MAXIMOS RESPONSABLES (MR) y
NO MAXIMOS RESPONSABLES (no-MR), usando Mann-Whitney U (no parametrico,
apropiado para N pequeno y sin supuesto de normalidad).

VERIFICADO:
  - Etiquetas: data/mr_asignacion_final.csv  (col 'compareciente', 'etiqueta_MR')
  - Indices por persona: outputs/capa3/icm_tricanal_final.csv
      (col 'identidad'; metricas: icm_tricanal, y10_rep, icm_facial, icm_vocal)
  - DIS/IEI NO estan por compareciente (viven por bloque en otro CSV), por eso
    aqui se analizan los indices que SI existen a nivel persona.

Cruce por nombre normalizado (tildes/espacios), con respaldo por apellidos.

Salidas:
  data/paso5_mr_resultados.csv     (test por indice)
  data/paso5_mr_por_persona.csv    (tabla con etiqueta + indices, base del test)
  consola: descriptivos, test, tamano de efecto, tabla por subcaso.

Uso:
    python cfh_paso5_mr_vs_nomr.py
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = Path(".")
ETIQ = BASE / "data" / "mr_asignacion_final.csv"
ICM = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"

INDICES = [
    ("icm_tricanal", "ICM tri-canal"),
    ("y10_rep",      "REP (verbal)"),
    ("icm_facial",   "ICM facial"),
    ("icm_vocal",    "ICM vocal"),
]


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def clave_apellidos(nombre):
    p = norm(nombre).split()
    return " ".join(p[-2:]) if len(p) >= 2 else norm(nombre)


def main():
    et = pd.read_csv(ETIQ)
    icm = pd.read_csv(ICM)

    et["_key"] = et["compareciente"].apply(norm)
    icm["_key"] = icm["identidad"].apply(norm)

    # merge exacto por nombre normalizado
    df = icm.merge(et[["_key", "etiqueta_MR", "subcaso"]], on="_key",
                   how="left", suffixes=("", "_et"))

    # respaldo por apellidos para los que no matchearon exacto
    faltan = df["etiqueta_MR"].isna()
    if faltan.any():
        et["_ap"] = et["compareciente"].apply(clave_apellidos)
        mapa_ap = dict(zip(et["_ap"], et["etiqueta_MR"]))
        df.loc[faltan, "etiqueta_MR"] = df.loc[faltan, "identidad"].apply(
            lambda x: mapa_ap.get(clave_apellidos(x), np.nan))

    sin = df["etiqueta_MR"].isna().sum()
    print("=" * 68)
    print("PASO 5 — MR vs no-MR  (Mann-Whitney U)")
    print("=" * 68)
    print(f"  comparecientes en ICM: {len(df)}")
    print(f"  etiquetados MR/no-MR:  {(~df['etiqueta_MR'].isna()).sum()}")
    if sin:
        print(f"  [AVISO] sin etiqueta ({sin}):")
        for x in df[df["etiqueta_MR"].isna()]["identidad"]:
            print(f"      {x}")

    df = df[df["etiqueta_MR"].isin(["MR", "NO_MR"])].copy()
    n_mr = (df["etiqueta_MR"] == "MR").sum()
    n_no = (df["etiqueta_MR"] == "NO_MR").sum()
    print(f"\n  N: MR={n_mr}  no-MR={n_no}")

    # guardar base por persona
    cols_guardar = ["subcaso", "identidad", "etiqueta_MR"] + \
                   [c for c, _ in INDICES] + ["n_tokens", "robustez"]
    cols_guardar = [c for c in cols_guardar if c in df.columns]
    df[cols_guardar].to_csv(BASE / "data" / "paso5_mr_por_persona.csv",
                            index=False, encoding="utf-8-sig")

    # test por indice
    print("\n" + "=" * 68)
    print(f"  {'Indice':16s} {'MR med':>8s} {'noMR med':>9s} {'U':>8s} "
          f"{'p':>8s} {'r':>7s}  sig")
    print("-" * 68)
    resultados = []
    for col, label in INDICES:
        a = df[df["etiqueta_MR"] == "MR"][col].dropna()
        b = df[df["etiqueta_MR"] == "NO_MR"][col].dropna()
        if len(a) < 3 or len(b) < 3:
            print(f"  {label:16s}  (n insuficiente)")
            continue
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        # tamano de efecto r = Z / sqrt(N)
        n = len(a) + len(b)
        mu = len(a) * len(b) / 2
        sigma = np.sqrt(len(a) * len(b) * (n + 1) / 12)
        z = (U - mu) / sigma
        r = abs(z) / np.sqrt(n)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:16s} {a.median():>8.3f} {b.median():>9.3f} "
              f"{U:>8.1f} {p:>8.4f} {r:>7.3f}  {sig}")
        resultados.append({
            "indice": label, "col": col,
            "mr_median": round(a.median(), 4), "mr_mean": round(a.mean(), 4),
            "nomr_median": round(b.median(), 4), "nomr_mean": round(b.mean(), 4),
            "n_mr": len(a), "n_nomr": len(b),
            "U": round(U, 1), "p_value": round(p, 5),
            "r_efecto": round(r, 3), "sig": sig,
        })

    pd.DataFrame(resultados).to_csv(BASE / "data" / "paso5_mr_resultados.csv",
                                    index=False, encoding="utf-8-sig")

    # descriptivos por grupo
    print("\n" + "=" * 68)
    print("  DESCRIPTIVOS POR GRUPO (media +- sd)")
    print("=" * 68)
    for col, label in INDICES:
        a = df[df["etiqueta_MR"] == "MR"][col].dropna()
        b = df[df["etiqueta_MR"] == "NO_MR"][col].dropna()
        print(f"  {label:16s}  MR: {a.mean():.3f}+-{a.std():.3f}   "
              f"no-MR: {b.mean():.3f}+-{b.std():.3f}")

    # tabla por subcaso
    print("\n" + "=" * 68)
    print("  DISTRIBUCION MR/no-MR POR SUBCASO")
    print("=" * 68)
    tab = df.groupby(["subcaso", "etiqueta_MR"]).size().unstack(fill_value=0)
    print(tab.to_string())

    print("\n  Guardado: data/paso5_mr_resultados.csv, data/paso5_mr_por_persona.csv")
    print("\n  LECTURA: los indices con p<0.05 y r>=0.3 son diferencias sustantivas")
    print("  entre MR y no-MR. El REP (verbal) y el ICM son los centrales para la")
    print("  conclusion sobre reconocimiento por calidad de responsabilidad.")


if __name__ == "__main__":
    main()
