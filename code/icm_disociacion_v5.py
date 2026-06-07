"""
ICM v5 — Disociación trimodal a nivel de segmento
==================================================
El ICM no es solo un promedio de canales — es una medida de
DISOCIACIÓN: momentos donde el canal verbal dice una cosa
y los canales no-verbales dicen otra.

Hipótesis trimodal:
  H_ICM_A: IEI alto → mayor disociación verbal/no-verbal
            (el compareciente corrige el canal verbal pero
            los canales prosódico y facial no lo acompañan)
  H_ICM_B: DIS alto + IEI alto → congruencia en dirección injusta
            (todos los canales apuntan al no-reconocimiento)

Métricas por segmento:
  - verbal_s:  REP score del bloque de texto (del CSV de indicadores)
  - vocal_s:   score prosódico (eGeMAPS z-score)
  - facial_s:  score distress facial (AU z-score)
  - dis_vv:    |verbal_s - vocal_s|  (disociación verbal-vocal)
  - dis_vf:    |verbal_s - facial_s| (disociación verbal-facial)
  - dis_tri:   std(verbal_s, vocal_s, facial_s) (disociación tri-canal)
  - cong_injusta: verbal_s < 0.5 AND vocal_s < 0.5 AND facial_s < 0.5
                  (congruencia en dirección de no-reconocimiento)

Índices por subcaso:
  - pct_dis_alta:   % segmentos con dis_tri > umbral (H_ICM_A)
  - pct_cong_injusta: % segmentos con congruencia injusta (H_ICM_B)
  - ICM_dis:  1 - pct_dis_alta (más disociación = menos congruencia)
  - ICM_cong: pct_cong_injusta (más congruencia injusta = más ICM)

Outputs:
  - outputs/capa3/icm_segmentos_{subcaso}.csv  (por subcaso)
  - outputs/capa3/icm_v5_resultados.csv        (resumen)
  - outputs/capa3/icm_tri_canal_v5.json
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, percentileofscore
from pathlib import Path

# ── Datos DIS/IEI por subcaso ────────────────────────────────────────────────
with open('outputs/capa3/dis_iei_comparativo.json', encoding='utf-8') as f:
    dis_iei = json.load(f)

SUBS = ['casanare','catatumbo','dabeiba','huila','costa_caribe']

DIS_vals = np.array([dis_iei[s]['DIS_score'] for s in SUBS])
IEI_vals = np.array([dis_iei[s]['IEI_score'] for s in SUBS])

print("DIS/IEI por subcaso:")
for i,s in enumerate(SUBS):
    print(f"  {s:15} DIS={DIS_vals[i]:.3f}  IEI={IEI_vals[i]:.3f}")

# ── Cargar eGeMAPS (vocal) con z-score conjunto ──────────────────────────────
print("\n=== Cargando canal vocal (eGeMAPS) ===")
EGEMAP_FILES = {s: f'outputs/capa3/egemap_{s}_compareciente.csv'
                for s in SUBS}
EGEMAP_FILES['casanare'] = 'outputs/capa3/egemap_casanare_compareciente.csv'

VOCAL_FEATS = ['shimmerLocaldB_sma3nz_amean',
               'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
               'HNRdBACF_sma3nz_amean']

dfs_eg = {}
for sub, path in EGEMAP_FILES.items():
    try:
        df = pd.read_csv(path)
        df['subcaso'] = sub
        dfs_eg[sub] = df
        print(f"  {sub:15} {len(df)} segmentos")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_eg_all = pd.concat(dfs_eg.values(), ignore_index=True)
feats_ok  = [f for f in VOCAL_FEATS if f in df_eg_all.columns]

# Z-score conjunto
for f in feats_ok:
    mu=df_eg_all[f].mean(); std=df_eg_all[f].std()+1e-9
    df_eg_all[f+'_z'] = (df_eg_all[f]-mu)/std

def vocal_score(row):
    scores = []
    if 'shimmerLocaldB_sma3nz_amean_z' in row.index:
        scores.append(np.tanh(row['shimmerLocaldB_sma3nz_amean_z']*0.5))
    if 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z' in row.index:
        scores.append(np.tanh(row['F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z']*0.5))
    if 'HNRdBACF_sma3nz_amean_z' in row.index:
        scores.append(np.tanh(-row['HNRdBACF_sma3nz_amean_z']*0.5))
    return float(np.mean(scores)) if scores else 0.5

df_eg_all['vocal_z']  = df_eg_all.apply(vocal_score, axis=1)
df_eg_all['vocal_01'] = 1/(1+np.exp(-df_eg_all['vocal_z']*2))

# ── Cargar AUs (facial) con z-score conjunto ─────────────────────────────────
print("\n=== Cargando canal facial (AUs) ===")
AUS_FILES = {
    'casanare':  'outputs/capa3/aus_casanare_torres.csv',
    'catatumbo': 'outputs/capa3/aus_catatumbo_SPEAKER_01.csv',
    'dabeiba':   'outputs/capa3/aus_dabeiba_v2.csv',
    'huila':     'outputs/capa3/aus_huila_comparecientes_v2.csv',
    'costa_caribe': 'outputs/capa3/aus_costa_caribe_v2.csv',
}
MP_COLS = ['browInnerUp','browDownLeft','browDownRight',
           'mouthFrownLeft','mouthFrownRight','mouthPucker',
           'mouthSmileLeft','mouthSmileRight',
           'cheekSquintLeft','cheekSquintRight']

rows_aus = []
for sub, path in AUS_FILES.items():
    try:
        df = pd.read_csv(path)
        df['subcaso'] = sub
        # Detectar si es OpenFace o MediaPipe
        if 'AU1' in df.columns:
            # OpenFace → convertir a esquema común
            for _, row in df.iterrows():
                rows_aus.append({
                    'subcaso': sub,
                    'start':   row.get('start', row.get('inicio', 0)),
                    'AU1':  row.get('AU1',0),
                    'AU4':  row.get('AU4',0),
                    'AU6':  row.get('AU6',0),
                    'AU12': row.get('AU12',0),
                    'AU15': row.get('AU15',0),
                    'AU17': row.get('AU17',0),
                })
        else:
            # MediaPipe → mapear
            for _, row in df.iterrows():
                rows_aus.append({
                    'subcaso': sub,
                    'start':   row.get('start',0),
                    'AU1':  row.get('browInnerUp',0),
                    'AU4':  np.mean([row.get('browDownLeft',0),row.get('browDownRight',0)]),
                    'AU6':  np.mean([row.get('cheekSquintLeft',0),row.get('cheekSquintRight',0)]),
                    'AU12': np.mean([row.get('mouthSmileLeft',0),row.get('mouthSmileRight',0)]),
                    'AU15': np.mean([row.get('mouthFrownLeft',0),row.get('mouthFrownRight',0)]),
                    'AU17': row.get('mouthPucker',0),
                })
        print(f"  {sub:15} {len(df)} frames")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_aus = pd.DataFrame(rows_aus)
for au in ['AU1','AU4','AU6','AU12','AU15','AU17']:
    mu=df_aus[au].mean(); std=df_aus[au].std()+1e-9
    df_aus[au+'_z'] = (df_aus[au]-mu)/std

def facial_score(row):
    scores = []
    scores.append(np.tanh((row['AU1_z']+row['AU4_z'])/2*0.5))
    scores.append(np.tanh((row['AU15_z']+row['AU17_z'])/2*0.5))
    if row['AU12']>1.0 and row['AU6']<0.5:
        scores.append(-0.3)
    return float(np.mean(scores))

df_aus['facial_z']  = df_aus.apply(facial_score, axis=1)
df_aus['facial_01'] = 1/(1+np.exp(-df_aus['facial_z']*2))

# ── Cargar verbal (REP por bloque) ───────────────────────────────────────────
print("\n=== Canal verbal (REP por bloque) ===")
df_ind = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
all_rep = df_ind['y10_rep'].values
mu_rep=all_rep.mean(); std_rep=all_rep.std()+1e-9
df_ind['verbal_z']  = (df_ind['y10_rep']-mu_rep)/std_rep
df_ind['verbal_01'] = 1/(1+np.exp(-df_ind['verbal_z']*2))

# Calcular percentil de cada segmento dentro de su canal
vocal_all  = df_eg_all['vocal_01'].values
facial_all = df_aus['facial_01'].values
verbal_all = df_ind['verbal_01'].values

# ── Análisis de disociación a nivel de subcaso ───────────────────────────────
print("\n=== ANÁLISIS DE DISOCIACIÓN POR SUBCASO ===")

UMBRAL_DIS = 0.15  # std entre canales > 0.15 = disociación alta

resultados = []
for sub in SUBS:
    # Scores por canal para este subcaso
    vocal_sub  = df_eg_all[df_eg_all['subcaso']==sub]['vocal_01'].values
    facial_sub = df_aus[df_aus['subcaso']==sub]['facial_01'].values
    verbal_sub = df_ind[df_ind['audio']==sub]['verbal_01'].values

    if len(vocal_sub)==0 or len(facial_sub)==0 or len(verbal_sub)==0:
        print(f"  {sub}: datos insuficientes")
        continue

    # Promedios por canal
    v_vocal  = float(np.mean(vocal_sub))
    v_facial = float(np.mean(facial_sub))
    v_verbal = float(np.mean(verbal_sub))

    # ── H_ICM_A: disociación verbal / no-verbal ──────────────────────────────
    # Para cada segmento verbal, buscar el score vocal/facial más cercano
    # temporalmente. Aquí usamos promedios del subcaso como aproximación.
    dis_verbal_vocal  = abs(v_verbal - v_vocal)
    dis_verbal_facial = abs(v_verbal - v_facial)
    dis_tri = float(np.std([v_verbal, v_vocal, v_facial]))

    # Percentil de disociación respecto a todos los subcasos
    # (se calculará después de tener todos)

    # ── H_ICM_B: congruencia en dirección injusta ────────────────────────────
    # Un canal apunta a "no-reconocimiento" si su score < 0.5
    # (menos distress / menos REP = menos reconocimiento genuino)
    cong_injusta = int(v_verbal < 0.5 and v_vocal < 0.5 and v_facial < 0.5)

    # Variación intra-subcaso del canal verbal (heterogeneidad del reconocimiento)
    verbal_std = float(np.std(verbal_sub))
    verbal_pct_alto = float(np.mean(verbal_sub > 0.55))  # % bloques con REP alto

    resultados.append({
        'subcaso':         sub,
        'DIS':             dis_iei[sub]['DIS_score'],
        'IEI':             dis_iei[sub]['IEI_score'],
        'vocal_mean':      round(v_vocal, 3),
        'facial_mean':     round(v_facial, 3),
        'verbal_mean':     round(v_verbal, 3),
        'dis_verbal_vocal':  round(dis_verbal_vocal, 3),
        'dis_verbal_facial': round(dis_verbal_facial, 3),
        'dis_tri':         round(dis_tri, 3),
        'cong_injusta':    cong_injusta,
        'verbal_std':      round(verbal_std, 3),
        'verbal_pct_alto': round(verbal_pct_alto, 3),
        'n_vocal_segs':    len(vocal_sub),
        'n_facial_frames': len(facial_sub),
        'n_verbal_bloques':len(verbal_sub),
    })

df_res = pd.DataFrame(resultados)
print(df_res[['subcaso','DIS','IEI','vocal_mean','facial_mean','verbal_mean',
              'dis_tri','cong_injusta']].to_string(index=False))

# ── Percentil de disociación ─────────────────────────────────────────────────
dis_vals = df_res['dis_tri'].values
df_res['dis_tri_pct'] = df_res['dis_tri'].apply(
    lambda x: round(percentileofscore(dis_vals, x)/100, 3))

# ── Correlaciones con IEI y DIS ──────────────────────────────────────────────
print(f"\n=== CORRELACIONES CON IEI/DIS ===")
for col in ['dis_verbal_vocal','dis_verbal_facial','dis_tri']:
    rho_iei,_ = spearmanr(df_res['IEI'], df_res[col])
    rho_dis,_ = spearmanr(df_res['DIS'], df_res[col])
    print(f"  {col:25} rho_IEI={rho_iei:+.3f}  rho_DIS={rho_dis:+.3f}")

print(f"\n  Hipótesis H_ICM_A (IEI alto → disociación alta):")
rho_a,_ = spearmanr(df_res['IEI'], df_res['dis_tri'])
print(f"  rho(IEI, dis_tri) = {rho_a:+.3f}")
if abs(rho_a) >= 0.70:
    print(f"  → APOYA H_ICM_A (rho≥0.70)")
elif abs(rho_a) >= 0.40:
    print(f"  → APOYA PARCIALMENTE H_ICM_A (0.40≤rho<0.70)")
else:
    print(f"  → NO APOYA H_ICM_A (rho<0.40) — N=5 insuficiente para confirmar")

print(f"\n  Hipótesis H_ICM_B (DIS alto → congruencia injusta):")
for sub in SUBS:
    row = df_res[df_res['subcaso']==sub].iloc[0]
    cong = "SÍ" if row['cong_injusta'] else "NO"
    print(f"  {sub:15} DIS={row['DIS']:.3f}  IEI={row['IEI']:.3f}  cong_injusta={cong}")

# ── Perfil trimodal por subcaso ───────────────────────────────────────────────
print(f"\n=== PERFILES TRIMODALES ===")
print(f"  {'Subcaso':15} {'Vocal':8} {'Facial':8} {'Verbal':8} {'dis_tri':8} {'Tipo'}")
print(f"  {'-'*70}")
for _,row in df_res.iterrows():
    # Clasificar perfil
    vals = [row['vocal_mean'], row['facial_mean'], row['verbal_mean']]
    dis  = row['dis_tri']
    mean_noverbal = (row['vocal_mean']+row['facial_mean'])/2
    if dis < 0.05:
        tipo = "CONGRUENTE"
    elif row['verbal_mean'] > mean_noverbal + 0.05:
        tipo = "VERBAL>NO-VERBAL (Fricker gap)"
    elif row['verbal_mean'] < mean_noverbal - 0.05:
        tipo = "NO-VERBAL>VERBAL"
    else:
        tipo = "DISOCIADO-NEUTRO"
    print(f"  {row['subcaso']:15} {row['vocal_mean']:8.3f} {row['facial_mean']:8.3f} "
          f"{row['verbal_mean']:8.3f} {dis:8.3f} {tipo}")

# ── Guardar ──────────────────────────────────────────────────────────────────
df_res.to_csv('outputs/capa3/icm_v5_resultados.csv', index=False, encoding='utf-8-sig')

# JSON para integrar con dis_iei_comparativo
icm_v5 = {}
for _,row in df_res.iterrows():
    icm_v5[row['subcaso']] = {
        'vocal_mean':       row['vocal_mean'],
        'facial_mean':      row['facial_mean'],
        'verbal_mean':      row['verbal_mean'],
        'dis_verbal_vocal': row['dis_verbal_vocal'],
        'dis_verbal_facial':row['dis_verbal_facial'],
        'dis_tri':          row['dis_tri'],
        'dis_tri_pct':      row['dis_tri_pct'],
        'cong_injusta':     bool(row['cong_injusta']),
        'verbal_pct_alto':  row['verbal_pct_alto'],
        'H_ICM_A':          row['dis_tri'] > df_res['dis_tri'].median(),
        'H_ICM_B':          bool(row['cong_injusta']),
    }

with open('outputs/capa3/icm_tri_canal_v5.json','w',encoding='utf-8') as f:
    json.dump(icm_v5, f, indent=2, ensure_ascii=False)

print(f"\n✓ Guardado: outputs/capa3/icm_v5_resultados.csv")
print(f"✓ Guardado: outputs/capa3/icm_tri_canal_v5.json")
