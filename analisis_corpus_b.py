"""
CFH · Análisis exploratorio — Corpus B (JEP)
Ejecutar desde la raíz: python analisis_corpus_b.py
"""
import sys
sys.path.insert(0, "code/src")

import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("data/features/indicators_corpus_b.csv")

print("=" * 60)
print("ANÁLISIS EXPLORATORIO — CORPUS B (JEP)")
print("=" * 60)
print(f"\nTotal secciones: {len(df)}")
print(f"Total documentos: {df['doc_id'].nunique()}")

print("\n── Distribución por sección target ──")
print(df["section_id"].value_counts().to_string())

print("\n── Estadísticas por indicador ──")
indicadores = ["y2_sa", "y3_civil", "y4_nv", "y10_rep", "y6_period"]
stats = df[indicadores].describe().round(3)
print(stats.to_string())

print("\n── Medias por sección target ──")
medias = df.groupby("section_id")[["y4_nv", "y10_rep"]].mean().round(3)
medias = medias.sort_values("y10_rep", ascending=False)
print(medias.to_string())

print("\n── Top 5 secciones con mayor REP ──")
top_rep = df.nlargest(5, "y10_rep")[["doc_id", "section_id", "y4_nv", "y10_rep"]]
print(top_rep.to_string(index=False))

print("\n── Top 5 secciones con mayor NV ──")
top_nv = df.nlargest(5, "y4_nv")[["doc_id", "section_id", "y4_nv", "y10_rep"]]
print(top_nv.to_string(index=False))

print("\n── Correlación NV vs REP ──")
corr = df[["y4_nv", "y10_rep"]].corr().round(3)
print(corr.to_string())

print("\n── Secciones con REP alto (>0.3) y NV bajo (<0.2) ──")
rep_puro = df[(df["y10_rep"] > 0.3) & (df["y4_nv"] < 0.2)]
print(f"N = {len(rep_puro)}")
print(rep_puro[["section_id", "y4_nv", "y10_rep"]].to_string(index=False))
