"""
Diagnóstico: qué discrimina A vs B individualmente
vs qué discrimina el índice compuesto DIS
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv('data/features/indicators_completo_conflibert.csv')
df['corpus'] = df['corpus_type'].apply(lambda x: 'A' if x.startswith('A') else 'B')

def sigmoid(x): return 1/(1+np.exp(-x))

print("=== DISCRIMINACIÓN POR INDICADOR INDIVIDUAL (A vs B) ===")
print(f"N_A={len(df[df.corpus=='A'])}  N_B={len(df[df.corpus=='B'])}")
print()

cols = ['y2_sa','y4_nv','y10_rep','y8_mafapo_cs','y9_cidh_cs']
nombres = ['SA','NV','REP','MAFAPO','CIDH']
efecto = {}

for col, nom in zip(cols, nombres):
    a = df[df.corpus=='A'][col].dropna()
    b = df[df.corpus=='B'][col].dropna()
    _,p = mannwhitneyu(a, b, alternative='two-sided')
    d = abs(a.mean()-b.mean())/np.sqrt((a.std()**2+b.std()**2)/2+1e-9)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
    efecto[col] = d
    print(f"  {nom:10} A={a.mean():.3f}  B={b.mean():.3f}  "
          f"Δ={b.mean()-a.mean():+.3f}  d={d:.3f}  p={p:.4f}  {sig}")

print()
print("=== PROBLEMA DEL DIS COMPUESTO ===")
print(f"  SA   d={efecto['y2_sa']:.3f}  peso=0.35 → contribución={0.35*efecto['y2_sa']:.3f}")
print(f"  NV   d={efecto['y4_nv']:.3f}  peso=0.35 → contribución={0.35*efecto['y4_nv']:.3f}")
print(f"  REP  d={efecto['y10_rep']:.3f}  peso=0.30 → contribución={0.30*efecto['y10_rep']:.3f}")
total = 0.35*efecto['y2_sa']+0.35*efecto['y4_nv']+0.30*efecto['y10_rep']
print(f"  DIS compuesto d efectivo ≈ {total:.3f}")
print(f"  → SA y NV diluyen el efecto de REP en el índice")

print()
print("=== OPCIONES ===")
print("  Opción 1: DIS con solo REP (d=0.372) → discrimina A vs B")
print("  Opción 2: DIS con pesos empíricos A vs B (SA=0.25,NV=0.04,REP=0.71)")
print("  Opción 3: Reportar indicadores individualmente, no DIS como índice")
print("  Opción 4: DIS para corpus C (donde NV y SA SÍ discriminan entre subcasos)")

# Verificar con pesos empíricos A vs B
mu = {}; std = {}
for col in cols:
    mu[col]  = df[col].mean()
    std[col] = df[col].std()+1e-9
    df[col+'_z'] = sigmoid((df[col]-mu[col])/std[col])

# Pesos empíricos proporcionales a d de Cohen A vs B
d_sa=efecto['y2_sa']; d_nv=efecto['y4_nv']; d_rep=efecto['y10_rep']
tot=d_sa+d_nv+d_rep
w_sa=d_sa/tot; w_nv=d_nv/tot; w_rep=d_rep/tot
print(f"\n  Pesos empíricos A vs B: SA={w_sa:.3f} NV={w_nv:.3f} REP={w_rep:.3f}")

df['DIS_emp'] = w_sa*df['y2_sa_z'] + w_nv*df['y4_nv_z'] + w_rep*(1-df['y10_rep_z'])
a_e = df[df.corpus=='A']['DIS_emp']
b_e = df[df.corpus=='B']['DIS_emp']
_,p_e = mannwhitneyu(a_e, b_e, alternative='two-sided')
d_e = abs(a_e.mean()-b_e.mean())/np.sqrt((a_e.std()**2+b_e.std()**2)/2+1e-9)
print(f"  DIS empírico: A={a_e.mean():.3f} B={b_e.mean():.3f} d={d_e:.3f} p={p_e:.4f}")

# Solo REP
df['DIS_rep'] = 1-df['y10_rep_z']
a_r = df[df.corpus=='A']['DIS_rep']
b_r = df[df.corpus=='B']['DIS_rep']
_,p_r = mannwhitneyu(a_r, b_r, alternative='two-sided')
d_r = abs(a_r.mean()-b_r.mean())/np.sqrt((a_r.std()**2+b_r.std()**2)/2+1e-9)
print(f"  Solo 1-REP:   A={a_r.mean():.3f} B={b_r.mean():.3f} d={d_r:.3f} p={p_r:.4f}")

# IEI para comparar
df['IEI'] = (0.35*df['y8_mafapo_cs_z'] + 0.20*df['y9_cidh_cs_z'] +
             0.25*df['y4_nv_z'] + 0.20*(1-df['y10_rep_z']))
a_i = df[df.corpus=='A']['IEI']
b_i = df[df.corpus=='B']['IEI']
_,p_i = mannwhitneyu(a_i, b_i, alternative='two-sided')
d_i = abs(a_i.mean()-b_i.mean())/np.sqrt((a_i.std()**2+b_i.std()**2)/2+1e-9)
print(f"  IEI (ref):    A={a_i.mean():.3f} B={b_i.mean():.3f} d={d_i:.3f} p={p_i:.4f}")
