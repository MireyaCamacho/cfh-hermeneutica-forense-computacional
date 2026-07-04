# -*- coding: utf-8 -*-
"""
cfh_disociacion_sin_version_libre.py
=====================================
Rehace el analisis de disociacion multimodal EXCLUYENDO a Henry Torres
Escalante (Casanare), que corresponde a una VERSION VOLUNTARIA (2020-02-06),
NO a una audiencia publica de reconocimiento como los otros 4 subcasos.

Justificacion (verificada):
  - SUBCASO_META en cfh_unificar_corpus_c.py marca casanare_torres como
    tipo="version_voluntaria", fecha 2020-02-06.
  - La fecha (2020) es anterior a todas las audiencias de reconocimiento
    (Catatumbo 2022, Costa Caribe 2022, Dabeiba 2023, Huila 2024).
  - En version voluntaria el compareciente aporta su relato/descargo, no el
    reconocimiento publico ante victimas; su REP verbal bajo (0.06) refleja el
    GENERO DISCURSIVO, no un patron de (in)congruencia comparable.

Por eso el analisis de disociacion se hace SOLO sobre audiencias de
reconocimiento (Catatumbo, Dabeiba, Huila, Costa Caribe), y Torres se reporta
aparte como caso de tipo de diligencia distinto.

Salida:
  data/disociacion_sin_torres.csv
  consola: tests sin Torres + ficha de Torres aparte.

Uso:
    python cfh_disociacion_sin_version_libre.py
"""

import itertools
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal

BASE = Path(".")
ICM = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
ETIQ = BASE / "data" / "mr_asignacion_final.csv"

CANALES = ["icm_facial", "icm_vocal", "y10_rep"]
# comparecientes en version voluntaria (no audiencia de reconocimiento)
VERSION_LIBRE = {"henry torres escalante"}


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def sig_str(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def r_ef(a, b, U):
    n = len(a) + len(b)
    mu = len(a)*len(b)/2
    sg = np.sqrt(len(a)*len(b)*(n+1)/12)
    return abs((U-mu)/sg)/np.sqrt(n)


def main():
    df = pd.read_csv(ICM)
    et = pd.read_csv(ETIQ)
    df["_k"] = df["identidad"].apply(norm)
    et["_k"] = et["compareciente"].apply(norm)
    df = df.merge(et[["_k", "etiqueta_MR"]], on="_k", how="left")

    # separar version libre
    es_libre = df["_k"].isin(VERSION_LIBRE)
    torres = df[es_libre]
    d_all = df[~es_libre].dropna(subset=CANALES).copy()

    print("=" * 66)
    print("DISOCIACION — SOLO AUDIENCIAS DE RECONOCIMIENTO (sin version libre)")
    print("=" * 66)
    print(f"  excluido (version voluntaria): {torres['identidad'].tolist()}")
    print(f"  n audiencias de reconocimiento con 3 canales: {len(d_all)}")

    # estandarizar SOBRE la submuestra homogenea
    for c in CANALES:
        mu, sd = d_all[c].mean(), d_all[c].std()
        d_all[c + "_z"] = (d_all[c] - mu) / (sd if sd > 1e-9 else 1)
    zc = [c + "_z" for c in CANALES]
    d_all["disociacion"] = d_all[zc].std(axis=1)

    d_all.sort_values("disociacion", ascending=False)[
        ["subcaso", "identidad", "etiqueta_MR"] + CANALES + ["disociacion"]
    ].to_csv(BASE / "data" / "disociacion_sin_torres.csv",
             index=False, encoding="utf-8-sig")

    # por subcaso
    print("\n  DISOCIACION POR SUBCASO")
    print("  " + "-" * 58)
    grupos, nombres = [], []
    for sub, g in d_all.groupby("subcaso"):
        x = g["disociacion"].dropna()
        print(f"    {str(sub):14s} n={len(x):2d}  media={x.mean():.3f}  mediana={x.median():.3f}")
        if len(x) >= 2:
            grupos.append(x.values); nombres.append(sub)
    if len(grupos) >= 3:
        H, p = kruskal(*grupos)
        print(f"\n    Kruskal-Wallis: H={H:.3f}  p={p:.4f}  {sig_str(p)}")

    # pareadas
    print("\n  COMPARACIONES PAREADAS")
    print("  " + "-" * 58)
    porsub = {s: g["disociacion"].dropna().values
              for s, g in d_all.groupby("subcaso") if len(g) >= 3}
    for s1, s2 in itertools.combinations(porsub.keys(), 2):
        a, b = porsub[s1], porsub[s2]
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        marca = "  <--" if p < 0.05 else ""
        print(f"    {s1:12s}({np.median(a):.2f}) vs {s2:12s}({np.median(b):.2f}): "
              f"p={p:.4f} r={r_ef(a,b,U):.3f} {sig_str(p)}{marca}")

    # MR vs no-MR
    print("\n  MR vs no-MR (sin version libre)")
    print("  " + "-" * 58)
    a = d_all[d_all["etiqueta_MR"] == "MR"]["disociacion"].dropna()
    b = d_all[d_all["etiqueta_MR"] == "NO_MR"]["disociacion"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"    MR: {a.median():.3f} (n={len(a)})  no-MR: {b.median():.3f} (n={len(b)})"
              f"  p={p:.4f} r={r_ef(a,b,U):.3f} {sig_str(p)}")

    # ranking extremos
    print("\n  RANKING (extremos, sin version libre)")
    print("  " + "-" * 58)
    top = d_all.sort_values("disociacion", ascending=False)
    for _, r in top.head(5).iterrows():
        print(f"    + {str(r['identidad'])[:30]:30s} {str(r['subcaso'])[:10]:10s} "
              f"disoc={r['disociacion']:.3f} (f={r['icm_facial']:.2f} "
              f"v={r['icm_vocal']:.2f} verb={r['y10_rep']:.2f})")
    for _, r in top.tail(3).iterrows():
        print(f"    - {str(r['identidad'])[:30]:30s} {str(r['subcaso'])[:10]:10s} "
              f"disoc={r['disociacion']:.3f} (f={r['icm_facial']:.2f} "
              f"v={r['icm_vocal']:.2f} verb={r['y10_rep']:.2f})")

    # ficha de Torres aparte
    print("\n" + "=" * 66)
    print("  CASO APARTE — VERSION VOLUNTARIA (no comparable)")
    print("=" * 66)
    for _, r in torres.iterrows():
        print(f"    {r['identidad']} [{r['subcaso']}]")
        print(f"      facial={r['icm_facial']:.2f} vocal={r['icm_vocal']:.2f} "
              f"verbal={r['y10_rep']:.2f}")
        print(f"      -> version voluntaria 2020: el REP verbal bajo refleja el")
        print(f"         genero (relato/descargo), no un patron de reconocimiento.")

    print("\n  Guardado: data/disociacion_sin_torres.csv")


if __name__ == "__main__":
    main()
