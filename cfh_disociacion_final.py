# -*- coding: utf-8 -*-
"""
cfh_disociacion_final.py
=========================
Analisis FINAL de la disociacion multimodal, sobre la muestra completa de
comparecientes con los 3 canales (n=44). El control de artefacto previo mostro
que la disociacion NO correlaciona con la cobertura (rho~0, p>0.19 en tokens,
facial y vocal), por lo que NO es artefacto de captura y se usa la muestra
completa.

Disociacion = sd de los 3 canales estandarizados (facial_z, vocal_z, verbal_z)
por compareciente. Alta = canales dispares (incongruencia entre lo dicho y lo
expresado); baja = canales alineados (congruencia).

Analisis:
  1. Kruskal-Wallis de disociacion entre SUBCASOS (a plena potencia, n=44).
  2. Comparaciones pareadas entre subcasos (Mann-Whitney) para localizar el
     contraste (p.ej. Huila vs Costa Caribe).
  3. MR vs no-MR (para confirmar que la calidad juridica no explica).
  4. Ranking completo.

Salida:
  data/disociacion_final.csv
  consola: tests + tabla.

Uso:
    python cfh_disociacion_final.py
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


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def sig_str(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def r_efecto(a, b, U):
    n = len(a) + len(b)
    mu = len(a) * len(b) / 2
    sigma = np.sqrt(len(a) * len(b) * (n + 1) / 12)
    return abs((U - mu) / sigma) / np.sqrt(n)


def main():
    df = pd.read_csv(ICM)
    et = pd.read_csv(ETIQ)
    df["_key"] = df["identidad"].apply(norm)
    et["_key"] = et["compareciente"].apply(norm)
    df = df.merge(et[["_key", "etiqueta_MR"]], on="_key", how="left")

    d = df.dropna(subset=CANALES).copy()
    for c in CANALES:
        mu, sd = d[c].mean(), d[c].std()
        d[c + "_z"] = (d[c] - mu) / (sd if sd > 1e-9 else 1)
    zcols = [c + "_z" for c in CANALES]
    d["disociacion"] = d[zcols].std(axis=1)

    d.sort_values("disociacion", ascending=False)[
        ["subcaso", "identidad", "etiqueta_MR"] + CANALES + ["disociacion"]
    ].to_csv(BASE / "data" / "disociacion_final.csv", index=False, encoding="utf-8-sig")

    print("=" * 68)
    print(f"DISOCIACION MULTIMODAL — MUESTRA COMPLETA (n={len(d)})")
    print("=" * 68)

    # 1. Kruskal-Wallis por subcaso
    print("\n  1. DISOCIACION POR SUBCASO")
    print("  " + "-" * 60)
    grupos, nombres = [], []
    for sub, g in d.groupby("subcaso"):
        x = g["disociacion"].dropna()
        print(f"    {str(sub):14s} n={len(x):2d}  media={x.mean():.3f}  "
              f"mediana={x.median():.3f}  sd={x.std():.3f}")
        if len(x) >= 2:
            grupos.append(x.values)
            nombres.append(sub)
    if len(grupos) >= 3:
        H, p = kruskal(*grupos)
        print(f"\n    Kruskal-Wallis: H={H:.3f}  p={p:.4f}  {sig_str(p)}")

    # 2. comparaciones pareadas
    print("\n  2. COMPARACIONES PAREADAS ENTRE SUBCASOS (Mann-Whitney)")
    print("  " + "-" * 60)
    porsub = {s: g["disociacion"].dropna().values
              for s, g in d.groupby("subcaso") if len(g) >= 3}
    for s1, s2 in itertools.combinations(porsub.keys(), 2):
        a, b = porsub[s1], porsub[s2]
        if len(a) >= 3 and len(b) >= 3:
            U, p = mannwhitneyu(a, b, alternative="two-sided")
            r = r_efecto(a, b, U)
            marca = "  <--" if p < 0.05 else ""
            print(f"    {s1:12s} ({np.median(a):.2f}) vs {s2:12s} "
                  f"({np.median(b):.2f}):  p={p:.4f}  r={r:.3f}  {sig_str(p)}{marca}")

    # 3. MR vs no-MR
    print("\n  3. DISOCIACION: MR vs no-MR")
    print("  " + "-" * 60)
    a = d[d["etiqueta_MR"] == "MR"]["disociacion"].dropna()
    b = d[d["etiqueta_MR"] == "NO_MR"]["disociacion"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        r = r_efecto(a, b, U)
        print(f"    MR:    media {a.mean():.3f}  mediana {a.median():.3f}  n={len(a)}")
        print(f"    no-MR: media {b.mean():.3f}  mediana {b.median():.3f}  n={len(b)}")
        print(f"    U={U:.1f}  p={p:.4f}  r={r:.3f}  {sig_str(p)}")

    # 4. ranking (top y bottom)
    print("\n  4. RANKING (extremos)")
    print("  " + "-" * 60)
    top = d.sort_values("disociacion", ascending=False)
    print("    MAS DISOCIADOS (canales dispares):")
    for _, r in top.head(6).iterrows():
        print(f"      {str(r['identidad'])[:30]:30s} {str(r['subcaso'])[:11]:11s} "
              f"{str(r.get('etiqueta_MR',''))[:5]:5s} disoc={r['disociacion']:.3f} "
              f"(fac={r['icm_facial']:.2f} voc={r['icm_vocal']:.2f} verb={r['y10_rep']:.2f})")
    print("    MAS CONGRUENTES (canales alineados):")
    for _, r in top.tail(4).iterrows():
        print(f"      {str(r['identidad'])[:30]:30s} {str(r['subcaso'])[:11]:11s} "
              f"{str(r.get('etiqueta_MR',''))[:5]:5s} disoc={r['disociacion']:.3f} "
              f"(fac={r['icm_facial']:.2f} voc={r['icm_vocal']:.2f} verb={r['y10_rep']:.2f})")

    print("\n  Guardado: data/disociacion_final.csv")
    print("\n  LECTURA: si Kruskal-Wallis por subcaso es significativo y MR/no-MR no,")
    print("  la conclusion es que la (in)congruencia multimodal del reconocimiento")
    print("  varia por CONTEXTO TERRITORIAL (subcaso), no por calidad juridica.")


if __name__ == "__main__":
    main()
