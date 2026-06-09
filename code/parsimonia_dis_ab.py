"""
Parsimonia DIS_AB — búsqueda exhaustiva de pesos
=================================================
Para cada combinación de pesos (w_SA, w_NV, w_REP) con suma=1:
  - Calcula DIS_AB sobre corpus A+B
  - Mide d de Cohen A vs B
  - Mide p-valor Mann-Whitney
  - Reporta % combinaciones con d>=0.30 (umbral "efecto medio")
  - Muestra si los pesos teóricos son cercanos al óptimo empírico
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv('data/features/indicators_completo_conflibert.csv')
df['corpus'] = df['corpus_type'].apply(lambda x: 'A' if x.startswith('A') else 'B')

def sigmoid(x): return 1/(1+np.exp(-x))

# Normalizar con z-score sobre A+B
for col in ['y2_sa','y4_nv','y10_rep']:
    mu=df[col].mean(); std=df[col].std()+1e-9
    df[col+'_z'] = sigmoid((df[col]-mu)/std)

a_mask = df.corpus=='A'
b_mask = df.corpus=='B'

# ── Grid search de pesos ──────────────────────────────────────────────────────
results = []
for w_sa in np.arange(0.05, 0.91, 0.05):
    for w_nv in np.arange(0.05, 0.91, 0.05):
        w_rep = round(1 - w_sa - w_nv, 3)
        if not (0.05 <= w_rep <= 0.90):
            continue

        dis = w_sa*df['y2_sa_z'] + w_nv*df['y4_nv_z'] + w_rep*(1-df['y10_rep_z'])
        a_dis = dis[a_mask].values
        b_dis = dis[b_mask].values

        d = abs(a_dis.mean()-b_dis.mean()) / np.sqrt(
            (a_dis.std()**2 + b_dis.std()**2)/2 + 1e-9)
        _,p = mannwhitneyu(a_dis, b_dis, alternative='two-sided')

        results.append({
            'w_SA':  round(w_sa, 2),
            'w_NV':  round(w_nv, 2),
            'w_REP': round(w_rep, 2),
            'd_cohen': round(d, 4),
            'p_valor': round(p, 4),
            'sig': '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.',
        })

df_res = pd.DataFrame(results)
n_total = len(df_res)

print(f"=== PARSIMONIA DIS_AB — {n_total} combinaciones ===\n")

# Umbrales de efecto
for umbral, label in [(0.20,'pequeño'), (0.30,'medio'), (0.40,'medio-alto'), (0.50,'grande')]:
    n = (df_res['d_cohen'] >= umbral).sum()
    print(f"  d >= {umbral} ({label:10}): {n}/{n_total} ({100*n/n_total:.0f}%)")

n_sig = (df_res['p_valor'] < 0.05).sum()
print(f"  p < 0.05:                {n_sig}/{n_total} ({100*n_sig/n_total:.0f}%)")
print(f"  p < 0.001:               {(df_res.p_valor<0.001).sum()}/{n_total} ({100*(df_res.p_valor<0.001).mean():.0f}%)")

# Pesos óptimos
best = df_res.nlargest(5, 'd_cohen')
print(f"\nTop 5 combinaciones (mayor d Cohen):")
print(best[['w_SA','w_NV','w_REP','d_cohen','p_valor','sig']].to_string(index=False))

# Pesos teóricos actuales
teo = df_res[(df_res.w_SA==0.35)&(df_res.w_NV==0.35)&(df_res.w_REP==0.30)]
if len(teo)>0:
    print(f"\nPesos teóricos (0.35/0.35/0.30):")
    print(teo[['w_SA','w_NV','w_REP','d_cohen','p_valor','sig']].to_string(index=False))
else:
    # buscar más cercano
    df_res['dist_teo'] = ((df_res.w_SA-0.35)**2 +
                          (df_res.w_NV-0.35)**2 +
                          (df_res.w_REP-0.30)**2)**0.5
    cercano = df_res.nsmallest(1,'dist_teo')
    print(f"\nCombinación más cercana a teórico (0.35/0.35/0.30):")
    print(cercano[['w_SA','w_NV','w_REP','d_cohen','p_valor','sig']].to_string(index=False))

# Pesos empíricos (proporcionales a d Cohen individual)
# SA=0.250, NV=0.045, REP=0.705
emp = df_res[(df_res.w_SA==0.25)&(df_res.w_NV==0.05)&(df_res.w_REP==0.70)]
if len(emp)>0:
    print(f"\nPesos empíricos A vs B (0.25/0.05/0.70):")
    print(emp[['w_SA','w_NV','w_REP','d_cohen','p_valor','sig']].to_string(index=False))

# Distribución del d Cohen
print(f"\nDistribución d Cohen:")
print(f"  media={df_res.d_cohen.mean():.3f}  "
      f"std={df_res.d_cohen.std():.3f}  "
      f"min={df_res.d_cohen.min():.3f}  "
      f"max={df_res.d_cohen.max():.3f}")

# Qué peso importa más
print(f"\nCorrelación peso → d Cohen:")
for col in ['w_SA','w_NV','w_REP']:
    corr = df_res[col].corr(df_res['d_cohen'])
    print(f"  {col}: r={corr:+.3f}")

# Guardar
df_res.to_csv('outputs/parsimonia_dis_ab.csv',index=False,encoding='utf-8-sig')
print(f"\n✓ outputs/parsimonia_dis_ab.csv ({n_total} combinaciones)")
