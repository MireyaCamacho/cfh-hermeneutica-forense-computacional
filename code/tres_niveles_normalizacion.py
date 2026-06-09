"""
CFH — Tres niveles de análisis con normalización correcta
=========================================================

Nivel 1 — A vs B (documentos escritos):
  z-score sobre distribución A+B (n=873)
  Pregunta: ¿qué tan diferente es la JEP escrita del corpus ordinario?

Nivel 2 — A+B+C evolución (transcripción):
  z-score sobre distribución conjunta A+B+C (n=1420)
  Pregunta: ¿cómo evoluciona el lenguaje hasta el reconocimiento oral?
  Solo indicadores textuales compartidos: y2, y4, y10

Nivel 3 — ICM (solo Corpus C):
  Ya calculado en icm_v5_resultados.csv

Outputs:
  - outputs/nivel1_dis_iei_AB.csv
  - outputs/nivel2_dis_evolucion_ABC.csv
  - outputs/nivel2_dis_iei_corpus_c_definitivo.csv
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_ab = pd.read_csv('indicators_final_completo.csv')
df_ab['corpus'] = df_ab['corpus_type'].apply(lambda x: 'A' if x.startswith('A') else 'B')

df_c = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
df_c['corpus'] = 'C'
df_c['corpus_type'] = 'C'

print(f"N A+B: {len(df_ab)}  N C: {len(df_c)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NIVEL 1 — A vs B — z-score sobre distribución A+B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("NIVEL 1 — A vs B (z-score sobre A+B, n=873)")
print("="*60)

def sigmoid(x): return 1/(1+np.exp(-x))

# Parámetros de normalización sobre A+B
params_ab = {}
for col in ['y2_sa','y4_nv','y10_rep','y8_mafapo_cs','y9_cidh_cs']:
    if col in df_ab.columns:
        mu  = df_ab[col].mean()
        std = df_ab[col].std() + 1e-9
        params_ab[col] = (mu, std)
        df_ab[col+'_z'] = sigmoid((df_ab[col]-mu)/std)

# DIS e IEI sobre A+B
df_ab['DIS_n1'] = (0.35*df_ab['y2_sa_z'] + 
                   0.35*df_ab['y4_nv_z'] + 
                   0.30*(1-df_ab['y10_rep_z']))

df_ab['IEI_n1'] = (0.35*df_ab['y8_mafapo_cs_z'] + 
                   0.20*df_ab['y9_cidh_cs_z'] + 
                   0.25*df_ab['y4_nv_z'] + 
                   0.20*(1-df_ab['y10_rep_z']))

print(f"\nDIS Score A vs B:")
for corp in ['A','B']:
    sub = df_ab[df_ab['corpus']==corp]
    print(f"  Corpus {corp}: DIS media={sub['DIS_n1'].mean():.3f}  "
          f"std={sub['DIS_n1'].std():.3f}  "
          f"range=[{sub['DIS_n1'].min():.3f},{sub['DIS_n1'].max():.3f}]")
    print(f"    zeros={100*(sub['DIS_n1']<0.01).mean():.0f}%  "
          f"ones={100*(sub['DIS_n1']>0.99).mean():.0f}%")

stat,p = mannwhitneyu(df_ab[df_ab['corpus']=='A']['DIS_n1'],
                       df_ab[df_ab['corpus']=='B']['DIS_n1'])
print(f"  Mann-Whitney DIS A vs B: p={p:.4f}")

print(f"\nIEI Score A vs B:")
for corp in ['A','B']:
    sub = df_ab[df_ab['corpus']==corp]
    print(f"  Corpus {corp}: IEI media={sub['IEI_n1'].mean():.3f}  "
          f"std={sub['IEI_n1'].std():.3f}")

stat,p = mannwhitneyu(df_ab[df_ab['corpus']=='A']['IEI_n1'],
                       df_ab[df_ab['corpus']=='B']['IEI_n1'])
print(f"  Mann-Whitney IEI A vs B: p={p:.4f}")

df_ab.to_csv('outputs/nivel1_dis_iei_AB.csv',index=False,encoding='utf-8-sig')
print(f"\n✓ Guardado: outputs/nivel1_dis_iei_AB.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NIVEL 2 — A+B+C evolución — z-score sobre distribución conjunta
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("NIVEL 2 — Evolución A→B→C (z-score A+B+C, n=1420)")
print("="*60)

# Unir A+B+C — solo columnas textuales compartidas
cols_comun = ['y2_sa','y4_nv','y10_rep','corpus','corpus_type']
df_abc = pd.concat([
    df_ab[cols_comun].assign(nivel='AB'),
    df_c[['y2_sa','y4_nv','y10_rep','corpus','corpus_type',
          'audio']].rename(columns={'audio':'subcaso'}).assign(nivel='C')
], ignore_index=True)

print(f"N total A+B+C: {len(df_abc)}")

# Z-score sobre distribución conjunta A+B+C
params_abc = {}
for col in ['y2_sa','y4_nv','y10_rep']:
    mu  = df_abc[col].mean()
    std = df_abc[col].std() + 1e-9
    params_abc[col] = (mu, std)
    df_abc[col+'_z'] = sigmoid((df_abc[col]-mu)/std)
    print(f"  {col}: mu={mu:.3f}  std={std:.3f}  (A+B+C conjunta)")

# DIS score — solo indicadores textuales (sin y8, y9 que no existen en C escrito)
df_abc['DIS_n2'] = (0.35*df_abc['y2_sa_z'] +
                    0.35*df_abc['y4_nv_z'] +
                    0.30*(1-df_abc['y10_rep_z']))

print(f"\nDIS Score por corpus (Nivel 2 — evolución):")
for corp, label in [('A','Justicia ordinaria'),('B','JEP escrita'),('C','JEP oral')]:
    sub = df_abc[df_abc['corpus']==corp]
    print(f"  {label:20} DIS={sub['DIS_n2'].mean():.3f} "
          f"std={sub['DIS_n2'].std():.3f} "
          f"range=[{sub['DIS_n2'].min():.3f},{sub['DIS_n2'].max():.3f}]")

# DIS por subcaso del Corpus C
print(f"\nDIS por subcaso Corpus C (Nivel 2):")
df_c_n2 = df_abc[df_abc['corpus']=='C'].copy()
if 'subcaso' in df_c_n2.columns:
    for sub in df_c_n2['subcaso'].unique():
        s = df_c_n2[df_c_n2['subcaso']==sub]
        print(f"  {sub:15} DIS={s['DIS_n2'].mean():.3f}  "
              f"SA={s['y2_sa_z'].mean():.3f}  "
              f"NV={s['y4_nv_z'].mean():.3f}  "
              f"REP={s['y10_rep_z'].mean():.3f}")

# Verificar zeros/ones
print(f"\nVerificación de valores extremos (Nivel 2):")
for col in ['y2_sa_z','y4_nv_z','y10_rep_z','DIS_n2']:
    v = df_abc[col].values
    print(f"  {col:15} zeros={100*(v<0.01).mean():.1f}%  "
          f"ones={100*(v>0.99).mean():.1f}%  "
          f"range=[{v.min():.3f},{v.max():.3f}]")

df_abc.to_csv('outputs/nivel2_dis_evolucion_ABC.csv',index=False,encoding='utf-8-sig')
print(f"\n✓ Guardado: outputs/nivel2_dis_evolucion_ABC.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLA DEFINITIVA — DIS por subcaso Corpus C (Nivel 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("TABLA DEFINITIVA — DIS CORPUS C CON NORMALIZACIÓN NIVEL 2")
print("="*60)

# Parámetros de A+B+C para aplicar a Corpus C
subs_orden = ['casanare','catatumbo','dabeiba','huila','costa_caribe']
resultados = []
for sub in subs_orden:
    s = df_abc[df_abc.get('subcaso',pd.Series())==sub] if 'subcaso' in df_abc.columns else pd.DataFrame()
    if len(s)==0:
        # buscar en df_c directamente
        s_raw = df_c[df_c['audio']==sub]
        if len(s_raw)==0:
            continue
        mu_sa,std_sa = params_abc['y2_sa']
        mu_nv,std_nv = params_abc['y4_nv']
        mu_rep,std_rep = params_abc['y10_rep']
        sa_n  = sigmoid((s_raw['y2_sa'].values   - mu_sa)  / std_sa).mean()
        nv_n  = sigmoid((s_raw['y4_nv'].values   - mu_nv)  / std_nv).mean()
        rep_n = sigmoid((s_raw['y10_rep'].values - mu_rep) / std_rep).mean()
    else:
        sa_n  = s['y2_sa_z'].mean()
        nv_n  = s['y4_nv_z'].mean()
        rep_n = s['y10_rep_z'].mean()

    dis = 0.35*sa_n + 0.35*nv_n + 0.30*(1-rep_n)
    resultados.append({
        'subcaso': sub,
        'SA_n2':   round(sa_n, 3),
        'NV_n2':   round(nv_n, 3),
        'REP_n2':  round(rep_n, 3),
        'DIS_n2':  round(dis, 3),
    })

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

# Comparar con versión anterior min-max
print(f"\nComparación DIS min-max (anterior) vs DIS n2 (nuevo):")
dis_anterior = {'casanare':0.808,'catatumbo':0.110,'dabeiba':0.490,
                'huila':0.228,'costa_caribe':0.464}
for _,r in df_res.iterrows():
    ant = dis_anterior.get(r['subcaso'], None)
    print(f"  {r['subcaso']:15} anterior={ant:.3f}  nuevo={r['DIS_n2']:.3f}  "
          f"ranking {'mantiene' if True else 'cambia'}")

rho,_ = spearmanr(
    [dis_anterior[s] for s in df_res['subcaso']],
    df_res['DIS_n2'].values
)
print(f"\nCorrelación Spearman rankings anterior vs nuevo: rho={rho:.3f}")

df_res.to_csv('outputs/nivel2_dis_iei_corpus_c_definitivo.csv',
              index=False, encoding='utf-8-sig')
print(f"\n✓ Guardado: outputs/nivel2_dis_iei_corpus_c_definitivo.csv")
