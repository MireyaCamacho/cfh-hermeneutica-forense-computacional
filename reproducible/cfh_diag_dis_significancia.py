# -*- coding: utf-8 -*-
"""
CFH — Diagnóstico transparente de significancia del DIS (A vs B)
================================================================
NO busca "hacer significativo" el DIS. Calcula TODAS las variantes de
normalización razonables y muestra el p-valor de cada una, para decidir
con criterio metodológico (no por conveniencia) cuál es la correcta.

Variantes:
  V1. z-score+sigmoid conjunto A+B+C        (actual)
  V2. z-score+sigmoid solo A+B              (sin Corpus C en la distribución)
  V3. DIS sobre indicadores CRUDOS          (Mann-Whitney no necesita normalizar)
  V4. componentes individuales y2/y4/y10    (¿el compuesto diluye?)

Base: indicators_completo_conflibert.csv (A+B) + indicators_corpus_c_unificado.csv (C)
"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

def sigmoid(x): return 1/(1+np.exp(-x))
def stars(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

# ── Cargar A+B ──
df_ab = pd.read_csv('data/features/indicators_completo_conflibert.csv')
df_ab['corpus'] = df_ab['corpus_type'].apply(lambda x: 'A' if str(x).startswith('A') else 'B')
COLS = ['y2_sa','y4_nv','y10_rep']

a = df_ab[df_ab.corpus=='A']
b = df_ab[df_ab.corpus=='B']
print(f"Corpus A: {len(a)}  Corpus B: {len(b)}\n")

# ════════════════════════════════════════════════════════════════════
print("="*64)
print("V1. z-score+sigmoid CONJUNTO A+B (la comparación es A vs B)")
print("="*64)
# Nota: para A vs B, lo correcto es normalizar sobre A+B (no meter C que no se compara)
dfab = df_ab.copy()
for col in COLS:
    mu,sd = dfab[col].mean(), dfab[col].std()+1e-9
    dfab[col+'_z'] = sigmoid((dfab[col]-mu)/sd)
dfab['DIS'] = 0.35*dfab['y2_sa_z']+0.35*dfab['y4_nv_z']+0.30*(1-dfab['y10_rep_z'])
_,p = mannwhitneyu(dfab[dfab.corpus=='A']['DIS'], dfab[dfab.corpus=='B']['DIS'], alternative='two-sided')
print(f"  DIS A={dfab[dfab.corpus=='A']['DIS'].mean():.3f} vs B={dfab[dfab.corpus=='B']['DIS'].mean():.3f}  p={p:.4f} {stars(p)}")

# ════════════════════════════════════════════════════════════════════
print("\n"+"="*64)
print("V2. z-score+sigmoid CONJUNTO A+B+C (lo actual de la tesis)")
print("="*64)
df_c = pd.read_csv('data/indicators_corpus_c_unificado.csv')
df_c['corpus']='C'
allc = pd.concat([df_ab[['corpus']+COLS], df_c[['corpus']+COLS]], ignore_index=True)
for col in COLS:
    mu,sd = allc[col].mean(), allc[col].std()+1e-9
    allc[col+'_z'] = sigmoid((allc[col]-mu)/sd)
allc['DIS'] = 0.35*allc['y2_sa_z']+0.35*allc['y4_nv_z']+0.30*(1-allc['y10_rep_z'])
_,p = mannwhitneyu(allc[allc.corpus=='A']['DIS'], allc[allc.corpus=='B']['DIS'], alternative='two-sided')
print(f"  DIS A={allc[allc.corpus=='A']['DIS'].mean():.3f} vs B={allc[allc.corpus=='B']['DIS'].mean():.3f}  p={p:.4f} {stars(p)}")

# ════════════════════════════════════════════════════════════════════
print("\n"+"="*64)
print("V3. DIS sobre indicadores CRUDOS (Mann-Whitney NO necesita normalizar)")
print("="*64)
# DIS crudo: combinación directa de los indicadores sin normalizar
for col in COLS:
    df_ab[col+'_raw'] = df_ab[col]
df_ab['DIS_raw'] = 0.35*df_ab['y2_sa']+0.35*df_ab['y4_nv']+0.30*(1-df_ab['y10_rep'])
_,p = mannwhitneyu(df_ab[df_ab.corpus=='A']['DIS_raw'], df_ab[df_ab.corpus=='B']['DIS_raw'], alternative='two-sided')
print(f"  DIS_crudo A={df_ab[df_ab.corpus=='A']['DIS_raw'].mean():.3f} vs B={df_ab[df_ab.corpus=='B']['DIS_raw'].mean():.3f}  p={p:.4f} {stars(p)}")
print("  (Este es el resultado 'verdadero' — Mann-Whitney compara rangos, inmune a la normalización monótona)")

# ════════════════════════════════════════════════════════════════════
print("\n"+"="*64)
print("V4. COMPONENTES INDIVIDUALES (¿el DIS compuesto diluye la señal?)")
print("="*64)
print(f"  {'Indicador':12} {'A':>8} {'B':>8} {'p':>10} {'sig':>6}")
for col,nom in [('y2_sa','y2 SA'),('y4_nv','y4 NV'),('y10_rep','y10 REP')]:
    _,p = mannwhitneyu(a[col], b[col], alternative='two-sided')
    print(f"  {nom:12} {a[col].mean():8.3f} {b[col].mean():8.3f} {p:10.4f} {stars(p):>6}")

# ════════════════════════════════════════════════════════════════════
print("\n"+"="*64)
print("INTERPRETACIÓN")
print("="*64)
print("""  - Si V3 (crudo) da n.s.: la diferencia NO existe en los datos; ninguna
    normalización legítima la 'creará'. El DIS compuesto realmente no separa A de B.
  - Si V3 da significativo pero V2 (A+B+C) no: la normalización conjunta con C
    está aplastando la señal → V1/V2 sobre A+B es más correcto para comparar A vs B.
  - Si V4 muestra que y2 o y4 separan pero el DIS no: el índice compuesto diluye;
    el hallazgo está a nivel de indicador, no de índice agregado.
  Decisión: usar la variante teóricamente correcta, NO la que dé menor p.""")
