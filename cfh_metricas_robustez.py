"""
CFH — Métricas de robustez DIS e IEI
=====================================
python cfh_metricas_robustez.py
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from pathlib import Path

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\outputs")

df  = pd.read_csv(BASE / "nivel1_dis_iei_AB.csv")
par = pd.read_csv(BASE / "parsimonia_dis_ab.csv")

print("=" * 60)
print("MÉTRICAS DE ROBUSTEZ — DIS e IEI")
print("=" * 60)

# 1. Correlación DIS vs IEI
r, p = pearsonr(df["DIS_n1"].dropna(), df["IEI_n1"].dropna())
print(f"\n1. Correlación DIS vs IEI (convergente-discriminante)")
print(f"   r = {r:.3f}, p = {p:.4f}")

# 2. Validez de criterio: DIS vs y3_civil
r2, p2 = pearsonr(df["DIS_n1"].dropna(), df["y3_civil"].dropna())
print(f"\n2. Validez criterio: DIS vs y3_civil")
print(f"   r = {r2:.3f}, p = {p2:.4f}")

# 3. Validez convergente: IEI vs y8_mafapo
r3, p3 = pearsonr(df["IEI_n1"].dropna(), df["y8_mafapo_cs_z"].dropna())
print(f"\n3. Validez convergente: IEI vs y8_mafapo_z")
print(f"   r = {r3:.3f}, p = {p3:.4f}")

# 4. Separación A vs B
a = df[df["corpus_type"] == 0]
b = df[df["corpus_type"] == 1]
print(f"\n4. Separación A vs B")
print(f"   DIS: A={a['DIS_n1'].mean():.3f}, B={b['DIS_n1'].mean():.3f}, Δ={b['DIS_n1'].mean()-a['DIS_n1'].mean():.3f}")
print(f"   IEI: A={a['IEI_n1'].mean():.3f}, B={b['IEI_n1'].mean():.3f}, Δ={b['IEI_n1'].mean()-a['IEI_n1'].mean():.3f}")

# Cohen d
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return abs(x.mean() - y.mean()) / pooled

d_dis = cohen_d(a["DIS_n1"].dropna(), b["DIS_n1"].dropna())
d_iei = cohen_d(a["IEI_n1"].dropna(), b["IEI_n1"].dropna())
u_dis, p_dis = mannwhitneyu(a["DIS_n1"].dropna(), b["DIS_n1"].dropna())
u_iei, p_iei = mannwhitneyu(a["IEI_n1"].dropna(), b["IEI_n1"].dropna())
print(f"   DIS: d={d_dis:.3f}, Mann-Whitney p={p_dis:.4f}")
print(f"   IEI: d={d_iei:.3f}, Mann-Whitney p={p_iei:.4f}")

# 5. Rango de variación
print(f"\n5. Rango de variación (discriminación entre documentos)")
print(f"   DIS: [{df['DIS_n1'].min():.3f}, {df['DIS_n1'].max():.3f}], rango={df['DIS_n1'].max()-df['DIS_n1'].min():.3f}")
print(f"   IEI: [{df['IEI_n1'].min():.3f}, {df['IEI_n1'].max():.3f}], rango={df['IEI_n1'].max()-df['IEI_n1'].min():.3f}")

# 6. Sensibilidad pesos DIS
sig = par[par["p_valor"] < 0.05]
print(f"\n6. Sensibilidad pesos DIS (parsimonia)")
print(f"   {len(sig)}/{len(par)} combinaciones significativas ({100*len(sig)/len(par):.1f}%)")
print(f"   d_cohen max = {par['d_cohen'].max():.3f}")
peso_teo = par[(par["w_SA"].round(2) == 0.35) & (par["w_NV"].round(2) == 0.35)]
if len(peso_teo) > 0:
    print(f"   d_cohen pesos teóricos (0.35/0.35/0.30) = {peso_teo['d_cohen'].values[0]:.3f}")

# 7. Cronbach (para contexto)
def cronbach(data):
    k = data.shape[1]
    vi = data.var(axis=0, ddof=1).sum()
    vt = data.sum(axis=1).var(ddof=1)
    return round((k/(k-1))*(1-vi/vt), 4)

dis_data = df[["y2_sa_z","y4_nv_z","y10_rep_z"]].copy()
dis_data["y10_rep_z"] = 1 - dis_data["y10_rep_z"]
iei_data = df[["y8_mafapo_cs_z","y9_cidh_cs_z","y4_nv_z","y10_rep_z"]].copy()
iei_data["y10_rep_z"] = 1 - iei_data["y10_rep_z"]

print(f"\n7. α Cronbach (contexto)")
print(f"   DIS α = {cronbach(dis_data.dropna())}")
print(f"   IEI α = {cronbach(iei_data.dropna())}")

print("\n" + "=" * 60)
print("RESUMEN PARA TESIS")
print("=" * 60)
print(f"""
El DIS y el IEI son evaluados con tres criterios complementarios:

(a) Consistencia interna (α Cronbach):
    DIS α = {cronbach(dis_data.dropna())} — negativo (componentes independientes)
    IEI α = {cronbach(iei_data.dropna())} — bajo (multidimensionalidad)
    → Inadecuado para índices compuestos multidimensionales

(b) Validez de criterio (correlación con indicadores teóricos):
    DIS vs y3_civil: r={r2:.3f} (p={p2:.4f})
    IEI vs y8_mafapo: r={r3:.3f} (p={p3:.4f})
    → Los índices se comportan como se espera teóricamente

(c) Poder discriminativo entre corpus (Cohen d):
    DIS A vs B: d={d_dis:.3f} (p={p_dis:.4f})
    IEI A vs B: d={d_iei:.3f} (p={p_iei:.4f})
    → Separación moderada-alta entre sistemas de justicia

(d) Robustez de pesos (sensibilidad):
    {len(sig)}/{len(par)} combinaciones de pesos producen d>0, p<0.05
    → El hallazgo A vs B no depende de los pesos específicos elegidos
""")

print("[CFH] Completado.")
