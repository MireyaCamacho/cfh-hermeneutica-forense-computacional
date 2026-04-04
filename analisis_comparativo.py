"""
CFH · Análisis comparativo — Corpus A vs Corpus B
Ejecutar desde la raíz: python analisis_comparativo.py
Requiere: data/features/indicators_corpus_a.csv
          data/features/indicators_corpus_b.csv
"""
import sys
sys.path.insert(0, "code/src")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_a = pd.read_csv("data/features/indicators_corpus_a.csv")
df_b = pd.read_csv("data/features/indicators_corpus_b.csv")
df_all = pd.concat([df_a, df_b], ignore_index=True)

print("=" * 65)
print("ANÁLISIS COMPARATIVO CFH — CORPUS A (Justicia Ordinaria) vs B (JEP)")
print("=" * 65)

# ── 1. Resumen general ────────────────────────────────────────────────────────
print("\n── 1. Resumen general ──")
print(f"{'':30s} {'Corpus A':>10s} {'Corpus B':>10s}")
print(f"{'Documentos':30s} {df_a['doc_id'].nunique():>10d} {df_b['doc_id'].nunique():>10d}")
print(f"{'Secciones target':30s} {len(df_a):>10d} {len(df_b):>10d}")

# ── 2. Estadísticas por indicador ────────────────────────────────────────────
print("\n── 2. Medias por indicador (Corpus A vs B) ──")
indicadores = ["y4_nv", "y10_rep", "y2_sa", "y3_civil"]
labels = {"y4_nv": "NV Score (y₄)", "y10_rep": "REP Score (y₁₀)",
          "y2_sa": "SA Score (y₂)", "y3_civil": "Dist. Civil (y₃)"}

print(f"\n{'Indicador':25s} {'Media A':>8s} {'Std A':>8s} {'Media B':>8s} {'Std B':>8s} {'p-valor':>10s} {'Sig':>5s}")
print("-" * 75)
for col in indicadores:
    m_a, s_a = df_a[col].mean(), df_a[col].std()
    m_b, s_b = df_b[col].mean(), df_b[col].std()
    t_stat, p_val = stats.mannwhitneyu(df_a[col], df_b[col], alternative='two-sided')
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"{labels[col]:25s} {m_a:>8.3f} {s_a:>8.3f} {m_b:>8.3f} {s_b:>8.3f} {p_val:>10.4f} {sig:>5s}")

# ── 3. Distribución por subsistema ───────────────────────────────────────────
print("\n── 3. Medias por subsistema (corpus_type) ──")
subsistemas = df_all.groupby("corpus_type")[["y4_nv", "y10_rep", "y2_sa"]].mean().round(3)
print(subsistemas.to_string())

# ── 4. Análisis por sección ──────────────────────────────────────────────────
print("\n── 4. NV y REP por sección — Corpus A ──")
secs_a = df_a.groupby("section_id")[["y4_nv", "y10_rep"]].agg(["mean", "count"]).round(3)
print(secs_a.to_string())

print("\n── 5. NV y REP por sección — Corpus B ──")
secs_b = df_b.groupby("section_id")[["y4_nv", "y10_rep"]].agg(["mean", "count"]).round(3)
print(secs_b.to_string())

# ── 6. Brecha discursiva ──────────────────────────────────────────────────────
print("\n── 6. Brecha discursiva (hipótesis H₃) ──")
rep_a = df_a["y10_rep"].mean()
rep_b = df_b["y10_rep"].mean()
nv_a  = df_a["y4_nv"].mean()
nv_b  = df_b["y4_nv"].mean()
brecha_rep = rep_b - rep_a
brecha_nv  = nv_a - nv_b

print(f"REP medio Corpus A: {rep_a:.3f}")
print(f"REP medio Corpus B: {rep_b:.3f}")
print(f"Brecha REP (B-A):   {brecha_rep:+.3f}  {'✓ B > A (esperado)' if brecha_rep > 0 else '✗ A > B (inesperado)'}")
print()
print(f"NV medio Corpus A:  {nv_a:.3f}")
print(f"NV medio Corpus B:  {nv_b:.3f}")
print(f"Brecha NV (A-B):    {brecha_nv:+.3f}  {'✓ A > B (esperado)' if brecha_nv > 0 else '✗ B > A (inesperado)'}")

# ── 7. Top secciones REP puro ────────────────────────────────────────────────
print("\n── 7. Secciones REP puro (REP>0.3, NV<0.2) por corpus ──")
for label, df in [("Corpus A", df_a), ("Corpus B", df_b)]:
    rep_puro = df[(df["y10_rep"] > 0.3) & (df["y4_nv"] < 0.2)]
    print(f"\n{label}: N={len(rep_puro)}")
    if len(rep_puro) > 0:
        print(rep_puro[["section_id", "y4_nv", "y10_rep"]].head(5).to_string(index=False))

print("\n" + "=" * 65)
print("Nota: Mann-Whitney U (no paramétrico, sin supuesto de normalidad)")
print("* p<0.05  ** p<0.01  *** p<0.001  n.s. no significativo")
print("=" * 65)
