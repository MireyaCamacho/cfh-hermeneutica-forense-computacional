"""
CFH — Cohen's d con Corpus B ampliado (145 secciones)
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\outputs")

a = pd.read_csv(BASE / "nivel1_dis_iei_AB.csv")
b = pd.read_csv(BASE / "indicators_corpus_b_v2.csv")

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return abs(x.mean() - y.mean()) / pooled

a2 = a[a['corpus_type'].isin(['A-CE', 'A-CSJ'])]

print(f"N_A={len(a2)}, N_B={len(b)}")
print()

for ind in ['y2_sa', 'y4_nv', 'y10_rep']:
    xa = a2[ind].dropna()
    xb = b[ind].dropna()
    d = cohen_d(xa, xb)
    u, p = mannwhitneyu(xa, xb)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"{ind}: A={xa.mean():.3f}, B={xb.mean():.3f}, d={d:.3f}, p={p:.4f} {sig}")

# DIS calculado con los mismos pesos
print()
print("DIS sintético (0.35×SA + 0.35×NV + 0.30×(1-REP)):")

def calc_dis(df):
    sa  = df['y2_sa'].fillna(df['y2_sa'].mean())
    nv  = df['y4_nv'].fillna(df['y4_nv'].mean())
    rep = df['y10_rep'].fillna(df['y10_rep'].mean())
    # Normalizar por percentil
    sa_n  = (sa  - sa.min())  / (sa.max()  - sa.min()  + 1e-9)
    nv_n  = (nv  - nv.min())  / (nv.max()  - nv.min()  + 1e-9)
    rep_n = (rep - rep.min()) / (rep.max() - rep.min() + 1e-9)
    return 0.35*sa_n + 0.35*nv_n + 0.30*(1 - rep_n)

dis_a = calc_dis(a2)
dis_b = calc_dis(b)

d = cohen_d(dis_a, dis_b)
u, p = mannwhitneyu(dis_a, dis_b)
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
print(f"DIS: A={dis_a.mean():.3f}, B={dis_b.mean():.3f}, d={d:.3f}, p={p:.4f} {sig}")

print()
print("[CFH] Completado.")
