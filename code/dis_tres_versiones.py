"""
Pesos DIS_v3 — proporcionales a variación ENTRE subcasos del Corpus C
(no dentro del corpus completo, que produce 0.333/0.333/0.333)
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')

# Medias por subcaso
medias = df.groupby('audio')[['y2_sa','y4_nv','y10_rep']].mean()
print("Medias por subcaso:")
print(medias.round(3))

# Rango entre subcasos (max_subcaso - min_subcaso)
print("\nVariación ENTRE subcasos:")
for col in ['y2_sa','y4_nv','y10_rep']:
    rango = medias[col].max() - medias[col].min()
    cv    = medias[col].std() / medias[col].mean()
    print(f"  {col}: rango={rango:.3f}  CV={cv:.3f}")

rango_sa  = medias['y2_sa'].max()  - medias['y2_sa'].min()
rango_nv  = medias['y4_nv'].max()  - medias['y4_nv'].min()
rango_rep = medias['y10_rep'].max() - medias['y10_rep'].min()
total = rango_sa + rango_nv + rango_rep

w_sa_v3  = rango_sa  / total
w_nv_v3  = rango_nv  / total
w_rep_v3 = rango_rep / total

print(f"\nPesos DIS_v3 (proporcionales al rango entre subcasos):")
print(f"  w_SA  = {w_sa_v3:.3f}  (teórico v1: 0.350)")
print(f"  w_NV  = {w_nv_v3:.3f}  (teórico v1: 0.350)")
print(f"  w_REP = {w_rep_v3:.3f}  (teórico v1: 0.300)")

# Función normalización min-max sobre medias de subcasos
def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

medias_n = medias.copy()
for col in ['y2_sa','y4_nv','y10_rep']:
    medias_n[col+'_n'] = norm(medias[col])

# DIS_v1 teórico
medias_n['DIS_v1'] = (0.35*medias_n['y2_sa_n'] +
                      0.35*medias_n['y4_nv_n'] +
                      0.30*(1-medias_n['y10_rep_n']))

# DIS_v3 empírico Corpus C
medias_n['DIS_v3'] = (w_sa_v3*medias_n['y2_sa_n'] +
                      w_nv_v3*medias_n['y4_nv_n'] +
                      w_rep_v3*(1-medias_n['y10_rep_n']))

print(f"\nComparación DIS_v1 vs DIS_v3 por subcaso:")
print(f"{'Subcaso':15} {'DIS_v1':10} {'DIS_v3':10}")
for idx, row in medias_n.iterrows():
    print(f"  {idx:13} {row['DIS_v1']:10.3f} {row['DIS_v3']:10.3f}")

rho, _ = spearmanr(medias_n['DIS_v1'], medias_n['DIS_v3'])
print(f"\nCorrelación Spearman v1 vs v3: rho={rho:.3f}")

# Parsimonia DIS_v3 sobre Corpus C
print(f"\n=== PARSIMONIA DIS_v3 (sobre medias de subcasos) ===")
n_est=0; n_tot=0
base = medias_n['DIS_v3'].values
for w2 in np.arange(0.10, 0.71, 0.05):
    for w4 in np.arange(0.10, 0.71, 0.05):
        w10 = round(1-w2-w4, 3)
        if 0.10 <= w10 <= 0.60:
            alt = w2*medias_n['y2_sa_n']+w4*medias_n['y4_nv_n']+w10*(1-medias_n['y10_rep_n'])
            rho_alt,_ = spearmanr(base, alt)
            n_tot += 1
            if rho_alt >= 0.90: n_est += 1
print(f"  {n_est}/{n_tot} combinaciones con rho>=0.90 ({100*n_est/n_tot:.0f}%)")

# Resumen tres versiones
print(f"\n=== RESUMEN TRES VERSIONES DIS ===")
print(f"{'Versión':12} {'w_SA':8} {'w_NV':8} {'w_REP':8} {'Propósito':35} {'Parsimonia'}")
print("-"*85)
print(f"  DIS_v1      0.350    0.350    0.300    {'Corpus C — pesos teóricos (Habermas)':35} 89% rho>=0.90")
print(f"  DIS_v2      0.250    0.045    0.705    {'A vs B — pesos empíricos d Cohen':35} d=0.481 p=0.0003")
print(f"  DIS_v3     {w_sa_v3:.3f}   {w_nv_v3:.3f}   {w_rep_v3:.3f}    {'Corpus C — pesos empíricos rango':35} ρ={rho:.3f} vs v1")
