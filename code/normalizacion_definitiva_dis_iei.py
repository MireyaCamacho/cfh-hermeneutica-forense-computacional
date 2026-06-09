"""
CFH — Normalización definitiva DIS e IEI
=========================================
Corpus A+B: indicators_completo_conflibert.csv (n=873)
Corpus C:   data/features/indicators_corpus_c.csv (n=588)
            + data/indicators_corpus_c_capa1_v2.csv (y2,y4,y10)

Normalización: z-score+sigmoid sobre distribución conjunta A+B+C (n=1461)
Sin valores 0.000 ni 1.000 absolutos.

DIS = 0.35×SA_z + 0.35×NV_z + 0.30×(1-REP_z)
IEI = 0.35×MAFAPO_z + 0.20×CIDH_z + 0.25×NV_z + 0.20×(1-REP_z)
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

def sigmoid(x): return 1/(1+np.exp(-x))

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_ab  = pd.read_csv('data/features/indicators_completo_conflibert.csv')
df_c1  = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')   # y2,y4,y10
df_c2  = pd.read_csv('data/features/indicators_corpus_c.csv')   # y8,y9

df_ab['corpus'] = df_ab['corpus_type'].apply(lambda x: 'A' if x.startswith('A') else 'B')
df_c1['corpus'] = 'C'
df_c2['corpus'] = 'C'

print(f"A+B: {len(df_ab)}  C_capa1: {len(df_c1)}  C_capa2: {len(df_c2)}")

# Merge Corpus C — unir por posición ya que los bloques corresponden
# Normalizar nombres de audio
df_c2['audio_norm'] = df_c2['audio'].str.replace('_antioquia','').str.replace('_torres','')
df_c1['audio_norm'] = df_c1['audio']

# Columnas A+B para unir
cols_ab = ['corpus','corpus_type','y2_sa','y4_nv','y10_rep',
           'y8_mafapo_cs','y9_cidh_cs']

# Columnas C
df_c1_sub = df_c1[['audio','audio_norm','corpus','y2_sa','y4_nv','y10_rep']].copy()
df_c2_sub = df_c2[['audio','audio_norm','corpus','y8_mafapo_cs','y9_cidh_cs']].copy()

# Merge C por audio_norm
df_c = pd.merge(df_c1_sub, df_c2_sub[['audio_norm','y8_mafapo_cs','y9_cidh_cs']],
                on='audio_norm', how='left')
df_c['corpus_type'] = 'C'

print(f"C merged: {len(df_c)}  y8 disponible: {df_c['y8_mafapo_cs'].notna().sum()}")

# Unir A+B+C
df_all = pd.concat([
    df_ab[cols_ab],
    df_c[cols_ab],
], ignore_index=True)

print(f"Total A+B+C: {len(df_all)}")
print(f"  A: {(df_all.corpus=='A').sum()}  B: {(df_all.corpus=='B').sum()}  "
      f"C: {(df_all.corpus=='C').sum()}")

# ── Z-score+sigmoid sobre distribución conjunta A+B+C ────────────────────────
print(f"\n=== PARÁMETROS NORMALIZACIÓN (A+B+C conjunta) ===")
COLS = ['y2_sa','y4_nv','y10_rep','y8_mafapo_cs','y9_cidh_cs']
params = {}
for col in COLS:
    v    = df_all[col].dropna()
    mu   = v.mean()
    std  = v.std() + 1e-9
    params[col] = (mu, std)
    df_all[col+'_z'] = sigmoid((df_all[col]-mu)/std)
    print(f"  {col:20} mu={mu:.3f}  std={std:.3f}  "
          f"→ range_z=[{df_all[col+'_z'].min():.3f},{df_all[col+'_z'].max():.3f}]")

# ── DIS e IEI ─────────────────────────────────────────────────────────────────
df_all['DIS'] = (0.35*df_all['y2_sa_z'] +
                 0.35*df_all['y4_nv_z'] +
                 0.30*(1-df_all['y10_rep_z']))

df_all['IEI'] = (0.35*df_all['y8_mafapo_cs_z'] +
                 0.20*df_all['y9_cidh_cs_z'] +
                 0.25*df_all['y4_nv_z'] +
                 0.20*(1-df_all['y10_rep_z']))

# ── Resultados por corpus ─────────────────────────────────────────────────────
print(f"\n=== DIS e IEI POR CORPUS ===")
print(f"{'Corpus':8} {'N':6} {'DIS_mean':10} {'DIS_std':10} {'IEI_mean':10} {'IEI_std':10}")
for corp in ['A','B','C']:
    s = df_all[df_all.corpus==corp]
    print(f"  {corp:6} {len(s):6} {s['DIS'].mean():10.3f} {s['DIS'].std():10.3f} "
          f"{s['IEI'].dropna().mean():10.3f} {s['IEI'].dropna().std():10.3f}")

# ── Tests A vs B ──────────────────────────────────────────────────────────────
print(f"\n=== TESTS A vs B ===")
a_dis = df_all[df_all.corpus=='A']['DIS']
b_dis = df_all[df_all.corpus=='B']['DIS']
a_iei = df_all[df_all.corpus=='A']['IEI'].dropna()
b_iei = df_all[df_all.corpus=='B']['IEI'].dropna()

_,p_dis = mannwhitneyu(a_dis, b_dis, alternative='two-sided')
_,p_iei = mannwhitneyu(a_iei, b_iei, alternative='two-sided')
d_dis = abs(a_dis.mean()-b_dis.mean())/np.sqrt((a_dis.std()**2+b_dis.std()**2)/2+1e-9)
d_iei = abs(a_iei.mean()-b_iei.mean())/np.sqrt((a_iei.std()**2+b_iei.std()**2)/2+1e-9)

print(f"  DIS A={a_dis.mean():.3f} vs B={b_dis.mean():.3f}  d={d_dis:.3f}  p={p_dis:.4f}  {'***' if p_dis<0.001 else '**' if p_dis<0.01 else '*' if p_dis<0.05 else 'n.s.'}")
print(f"  IEI A={a_iei.mean():.3f} vs B={b_iei.mean():.3f}  d={d_iei:.3f}  p={p_iei:.4f}  {'***' if p_iei<0.001 else '**' if p_iei<0.01 else '*' if p_iei<0.05 else 'n.s.'}")

# ── DIS e IEI por subcaso Corpus C ────────────────────────────────────────────
print(f"\n=== DIS e IEI POR SUBCASO CORPUS C ===")
df_c_res = df_all[df_all.corpus=='C'].copy()

# Normalizar nombres
audio_map = {
    'casanare':'casanare','catatumbo':'catatumbo',
    'dabeiba':'dabeiba','huila':'huila','costa_caribe':'costa_caribe'
}

if 'audio_norm' in df_c.columns:
    df_c_res = df_c_res.copy()
    df_c_res['audio_norm'] = df_c['audio_norm'].values[:len(df_c_res)]

subs = ['casanare','catatumbo','dabeiba','huila','costa_caribe']
resultados = []
for sub in subs:
    # buscar por audio_norm
    mask = df_c_res.get('audio_norm', pd.Series(dtype=str)) == sub
    if mask.sum() == 0:
        # intentar con audio original
        raw = df_c1[df_c1['audio']==sub]
        if len(raw)==0: continue
        mu_sa,std_sa   = params['y2_sa']
        mu_nv,std_nv   = params['y4_nv']
        mu_rep,std_rep = params['y10_rep']
        sa_z  = sigmoid((raw['y2_sa']  -mu_sa) /std_sa).mean()
        nv_z  = sigmoid((raw['y4_nv']  -mu_nv) /std_nv).mean()
        rep_z = sigmoid((raw['y10_rep']-mu_rep)/std_rep).mean()
        # y8,y9 del corpus_c.csv
        c2sub = df_c2[df_c2['audio_norm']==sub]
        if len(c2sub)>0:
            mu_y8,std_y8 = params['y8_mafapo_cs']
            mu_y9,std_y9 = params['y9_cidh_cs']
            y8_z = sigmoid((c2sub['y8_mafapo_cs']-mu_y8)/std_y8).mean()
            y9_z = sigmoid((c2sub['y9_cidh_cs']  -mu_y9)/std_y9).mean()
        else:
            y8_z = y9_z = np.nan
    else:
        s    = df_c_res[mask]
        sa_z  = s['y2_sa_z'].mean()
        nv_z  = s['y4_nv_z'].mean()
        rep_z = s['y10_rep_z'].mean()
        y8_z  = s['y8_mafapo_cs_z'].mean() if 'y8_mafapo_cs_z' in s else np.nan
        y9_z  = s['y9_cidh_cs_z'].mean()   if 'y9_cidh_cs_z' in s else np.nan

    dis = 0.35*sa_z + 0.35*nv_z + 0.30*(1-rep_z)
    iei = (0.35*y8_z + 0.20*y9_z + 0.25*nv_z + 0.20*(1-rep_z)
           if not np.isnan(y8_z) else np.nan)

    resultados.append({
        'subcaso': sub,
        'SA_z':    round(sa_z,3),
        'NV_z':    round(nv_z,3),
        'REP_z':   round(rep_z,3),
        'y8_z':    round(y8_z,3) if not np.isnan(y8_z) else None,
        'y9_z':    round(y9_z,3) if not np.isnan(y9_z) else None,
        'DIS':     round(dis,3),
        'IEI':     round(iei,3) if not np.isnan(iei) else None,
    })

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

# Correlación con versión anterior
dis_ant = {'casanare':0.808,'catatumbo':0.110,'dabeiba':0.490,
           'huila':0.228,'costa_caribe':0.464}
iei_ant = {'casanare':0.517,'catatumbo':0.624,'dabeiba':0.299,
           'huila':0.081,'costa_caribe':0.231}

rho_dis,_ = spearmanr([dis_ant[s] for s in subs], df_res['DIS'].values)
rho_iei,_ = spearmanr([iei_ant[s] for s in subs],
                       [df_res[df_res.subcaso==s]['IEI'].values[0] for s in subs])

print(f"\nCorrelación Spearman rankings anterior vs nuevo:")
print(f"  DIS: rho={rho_dis:.3f}")
print(f"  IEI: rho={rho_iei:.3f}")

# Verificar zeros/ones
print(f"\nVerificación valores extremos (A+B+C):")
for col in ['DIS','IEI']:
    v = df_all[col].dropna().values
    print(f"  {col}: zeros={100*(v<0.01).mean():.1f}%  ones={100*(v>0.99).mean():.1f}%  "
          f"range=[{v.min():.3f},{v.max():.3f}]")

# Guardar
df_all.to_csv('data/dis_iei_corpus_abc_definitivo.csv',index=False,encoding='utf-8-sig')
df_res.to_csv('data/dis_iei_corpus_c_v3.csv',index=False,encoding='utf-8-sig')
print(f"\n✓ data/dis_iei_corpus_abc_definitivo.csv")
print(f"✓ data/dis_iei_corpus_c_v3.csv")
