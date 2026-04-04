"""
CFH · Análisis de secciones REP puro en Corpus A
¿Son las sentencias más recientes post-2016?
Ejecutar: python analisis_rep_puro_a.py
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/features/indicators_corpus_a.csv")

# Secciones REP puro
rep_puro = df[(df["y10_rep"] > 0.3) & (df["y4_nv"] < 0.2)].copy()

print("=" * 65)
print("ANÁLISIS — 36 SECCIONES REP PURO EN CORPUS A")
print("REP > 0.3 y NV < 0.2")
print("=" * 65)

print(f"\nTotal secciones REP puro: {len(rep_puro)}")
print(f"Total secciones corpus A: {len(df)}")
print(f"Proporción: {len(rep_puro)/len(df):.1%}")

# Por tipo de sección
print("\n── Por sección ──")
print(rep_puro["section_id"].value_counts().to_string())

# Por subsistema
print("\n── Por subsistema ──")
print(rep_puro["corpus_type"].value_counts().to_string())

# Por año — ¿son más recientes?
print("\n── Distribución temporal (año) ──")
print(rep_puro["year"].value_counts().sort_index().to_string())

# Comparar año promedio REP puro vs resto
año_rep_puro = rep_puro["year"].mean()
año_resto = df[~df.index.isin(rep_puro.index)]["year"].mean()
print(f"\nAño promedio REP puro: {año_rep_puro:.1f}")
print(f"Año promedio resto corpus A: {año_resto:.1f}")
print(f"Diferencia: {año_rep_puro - año_resto:+.1f} años")

# ¿Post-2016?
post_2016 = rep_puro[rep_puro["year"] >= 2016]
pre_2016 = rep_puro[rep_puro["year"] < 2016]
print(f"\nPost-2016: {len(post_2016)} ({len(post_2016)/len(rep_puro):.0%})")
print(f"Pre-2016:  {len(pre_2016)} ({len(pre_2016)/len(rep_puro):.0%})")

# REP promedio por período en corpus A completo
print("\n── REP promedio por período — Corpus A completo ──")
df["periodo"] = pd.cut(df["year"],
    bins=[1993, 2008, 2016, 2023],
    labels=["pre-escándalo (≤2008)", "post-escándalo (2009-2016)", "post-acuerdo (2017+)"])
periodo_stats = df.groupby("periodo", observed=True)["y10_rep"].agg(["mean", "count"]).round(3)
print(periodo_stats.to_string())

# Top 10 secciones REP puro con mayor REP
print("\n── Top 10 secciones REP puro (mayor y₁₀) ──")
top10 = rep_puro.nlargest(10, "y10_rep")[["section_id", "corpus_type", "year", "y4_nv", "y10_rep"]]
print(top10.to_string(index=False))

# Correlación entre año y REP en corpus A
from scipy import stats
df_con_año = df.dropna(subset=["year"])
corr, pval = stats.spearmanr(df_con_año["year"], df_con_año["y10_rep"])
print(f"\n── Correlación Spearman año vs REP (Corpus A) ──")
print(f"ρ = {corr:.3f}, p = {pval:.4f}")
print(f"{'Correlación positiva significativa ✓' if corr > 0 and pval < 0.05 else 'Sin correlación temporal significativa'}")
