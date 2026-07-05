# -*- coding: utf-8 -*-
"""
cfh_disociacion_multimodal_v2.py
=================================
Analiza la DISOCIACION MULTIMODAL por compareciente CONTROLANDO el artefacto
de captura: una intervencion corta (pocos tokens, pocas ventanas facial/vocal)
puede producir canales dispares por FALTA DE DATOS, no por disociacion real.

Este script:
  1. Calcula la disociacion (sd de los 3 canales estandarizados por persona).
  2. La cruza con la robustez de captura (n_facial, n_vocal, n_tokens, robustez).
  3. Verifica si la disociacion CORRELACIONA con la cobertura (si los mas
     disociados son los de menos datos -> artefacto).
  4. Recalcula el analisis SOLO sobre los comparecientes "solido" (buena
     cobertura), para separar disociacion real de artefacto.

Salida:
  data/disociacion_multimodal_v2.csv  (con robustez y flag_artefacto)
  consola: correlacion disociacion~cobertura, ranking anotado, analisis
           filtrado (solo solido) MR vs no-MR y por subcaso.

Uso:
    python cfh_disociacion_multimodal_v2.py
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr

BASE = Path(".")
ICM = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
ETIQ = BASE / "data" / "mr_asignacion_final.csv"

CANALES = ["icm_facial", "icm_vocal", "y10_rep"]
NOMBRES = {"icm_facial": "facial", "icm_vocal": "vocal", "y10_rep": "verbal"}

# umbrales de cobertura minima (bajo estos, la captura es escasa)
MIN_TOKENS = 100
MIN_FACIAL = 100    # ventanas faciales
MIN_VOCAL = 100     # ~100 s de audio


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def test_mw(a, b):
    if len(a) < 3 or len(b) < 3:
        return None
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    n = len(a) + len(b)
    mu = len(a) * len(b) / 2
    sigma = np.sqrt(len(a) * len(b) * (n + 1) / 12)
    r = abs((U - mu) / sigma) / np.sqrt(n)
    return U, p, r


def sig_str(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    df = pd.read_csv(ICM)
    et = pd.read_csv(ETIQ)
    df["_key"] = df["identidad"].apply(norm)
    et["_key"] = et["compareciente"].apply(norm)
    df = df.merge(et[["_key", "etiqueta_MR"]], on="_key", how="left")

    completos = df.dropna(subset=CANALES).copy()
    print("=" * 70)
    print("DISOCIACION MULTIMODAL — CON CONTROL DE ARTEFACTO DE CAPTURA")
    print("=" * 70)
    print(f"  con los 3 canales: {len(completos)} / {len(df)}")

    # estandarizar canales
    for c in CANALES:
        mu, sd = completos[c].mean(), completos[c].std()
        completos[c + "_z"] = (completos[c] - mu) / (sd if sd > 1e-9 else 1)
    zcols = [c + "_z" for c in CANALES]
    completos["disociacion"] = completos[zcols].std(axis=1)

    # cobertura y flag de artefacto
    for col in ["n_facial", "n_vocal", "n_tokens"]:
        if col not in completos.columns:
            completos[col] = np.nan
    completos["captura_escasa"] = (
        (completos["n_tokens"].fillna(0) < MIN_TOKENS) |
        (completos["n_facial"].fillna(0) < MIN_FACIAL) |
        (completos["n_vocal"].fillna(0) < MIN_VOCAL))
    if "robustez" in completos.columns:
        completos["flag_artefacto"] = (
            completos["captura_escasa"] |
            completos["robustez"].astype(str).str.contains("minimo", case=False, na=False))
    else:
        completos["flag_artefacto"] = completos["captura_escasa"]

    # guardar
    out_cols = ["subcaso", "identidad", "etiqueta_MR"] + CANALES + \
               ["disociacion", "n_facial", "n_vocal", "n_tokens",
                "robustez", "flag_artefacto"]
    out_cols = [c for c in out_cols if c in completos.columns]
    completos[out_cols].sort_values("disociacion", ascending=False).to_csv(
        BASE / "data" / "disociacion_multimodal_v2.csv", index=False, encoding="utf-8-sig")

    # 1. CORRELACION disociacion ~ cobertura (la prueba del artefacto)
    print("\n" + "=" * 70)
    print("  ¿LA DISOCIACION ES ARTEFACTO DE CAPTURA?")
    print("=" * 70)
    for cov in ["n_tokens", "n_facial", "n_vocal"]:
        sub = completos.dropna(subset=[cov, "disociacion"])
        if len(sub) >= 5:
            rho, p = spearmanr(sub[cov], sub["disociacion"])
            flag = "  <- correlacion NEGATIVA: menos datos, mas disociacion (ARTEFACTO)" \
                   if (rho < -0.3 and p < 0.05) else ""
            print(f"  disociacion ~ {cov:10s}: rho={rho:+.3f}  p={p:.4f}{flag}")
    print(f"\n  comparecientes flag_artefacto (captura escasa o robustez minima): "
          f"{int(completos['flag_artefacto'].sum())} / {len(completos)}")

    # 2. RANKING anotado con flag
    print("\n" + "=" * 70)
    print("  RANKING POR DISOCIACION (con marca de captura)")
    print("=" * 70)
    print(f"  {'compareciente':32s} {'sub':10s} {'MR':6s} "
          f"{'disoc':>6s} {'tok':>5s} {'nf':>4s} {'nv':>4s} {'flag':>5s}")
    top = completos.sort_values("disociacion", ascending=False)
    for _, r in top.iterrows():
        fl = "ART" if r["flag_artefacto"] else ""
        print(f"  {str(r['identidad'])[:32]:32s} {str(r['subcaso'])[:10]:10s} "
              f"{str(r.get('etiqueta_MR',''))[:6]:6s} {r['disociacion']:>6.3f} "
              f"{int(r['n_tokens']) if pd.notna(r['n_tokens']) else 0:>5d} "
              f"{int(r['n_facial']) if pd.notna(r['n_facial']) else 0:>4d} "
              f"{int(r['n_vocal']) if pd.notna(r['n_vocal']) else 0:>4d} {fl:>5s}")

    # 3. ANALISIS SOLO SOBRE SOLIDOS
    solido = completos[~completos["flag_artefacto"]].copy()
    print("\n" + "=" * 70)
    print(f"  ANALISIS SOLO SOBRE CAPTURA SOLIDA (n={len(solido)})")
    print("=" * 70)

    a = solido[solido["etiqueta_MR"] == "MR"]["disociacion"].dropna()
    b = solido[solido["etiqueta_MR"] == "NO_MR"]["disociacion"].dropna()
    print(f"\n  MR vs no-MR (solo solido):")
    print(f"    MR:    media {a.mean():.3f}  mediana {a.median():.3f}  n={len(a)}")
    print(f"    no-MR: media {b.mean():.3f}  mediana {b.median():.3f}  n={len(b)}")
    res = test_mw(a, b)
    if res:
        U, p, r = res
        print(f"    U={U:.1f}  p={p:.4f}  r={r:.3f}  {sig_str(p)}")

    print(f"\n  Por subcaso (solo solido):")
    grupos = []
    for sub, g in solido.groupby("subcaso"):
        d = g["disociacion"].dropna()
        print(f"    {str(sub):14s} n={len(d):2d}  media={d.mean():.3f}  mediana={d.median():.3f}")
        if len(d) >= 2:
            grupos.append(d.values)
    if len(grupos) >= 3:
        H, p = kruskal(*grupos)
        print(f"    Kruskal-Wallis: H={H:.3f}  p={p:.4f}  {sig_str(p)}")

    print("\n  Guardado: data/disociacion_multimodal_v2.csv")
    print("\n  LECTURA:")
    print("  - Si disociacion correlaciona NEGATIVO con cobertura -> parte de la")
    print("    disociacion es artefacto; el analisis sobre 'solido' es el valido.")
    print("  - Si NO correlaciona -> la disociacion es real y el ranking completo vale.")


if __name__ == "__main__":
    main()
