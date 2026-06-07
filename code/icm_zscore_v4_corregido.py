"""
ICM v4 corregido — unifica OpenFace (AU1/AU4...) y MediaPipe (browInnerUp...)
en un esquema común antes del z-score conjunto.

Mapeo:
  AU1  ↔ browInnerUp
  AU4  ↔ browDownLeft + browDownRight
  AU6  ↔ cheekSquintLeft + cheekSquintRight
  AU12 ↔ mouthSmileLeft + mouthSmileRight
  AU15 ↔ mouthFrownLeft + mouthFrownRight
  AU17 ↔ mouthPucker
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

# ── Cargar y unificar AUs ─────────────────────────────────────────────────────
OPENFACE_FILES = {
    'casanare':  'outputs/capa3/aus_casanare_torres.csv',
    'catatumbo': 'outputs/capa3/aus_catatumbo_SPEAKER_01.csv',
    'dabeiba':   'outputs/capa3/aus_dabeiba_v2.csv',
    'huila':     'outputs/capa3/aus_huila_comparecientes_v2.csv',
}
MEDIAPIPE_FILE = {'costa_caribe': 'outputs/capa3/aus_costa_caribe_v2.csv'}

# Mapeo MediaPipe → esquema común
MP_MAP = {
    'AU1':  ['browInnerUp'],
    'AU4':  ['browDownLeft','browDownRight'],
    'AU6':  ['cheekSquintLeft','cheekSquintRight'],
    'AU12': ['mouthSmileLeft','mouthSmileRight'],
    'AU15': ['mouthFrownLeft','mouthFrownRight'],
    'AU17': ['mouthPucker'],
}

rows_facial = []

# OpenFace — ya tiene AU1, AU4, etc.
for sub, path in OPENFACE_FILES.items():
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rows_facial.append({
                'subcaso': sub,
                'AU1':  row.get('AU1', 0),
                'AU4':  row.get('AU4', 0),
                'AU6':  row.get('AU6', 0),
                'AU12': row.get('AU12', 0),
                'AU15': row.get('AU15', 0),
                'AU17': row.get('AU17', 0),
                'start': row.get('start', 0),
            })
        print(f"  {sub:15} {len(df)} frames (OpenFace)")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

# MediaPipe — convertir blendshapes a AU equivalentes
for sub, path in MEDIAPIPE_FILE.items():
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rows_facial.append({
                'subcaso': sub,
                'AU1':  row.get('browInnerUp', 0),
                'AU4':  np.mean([row.get('browDownLeft',0), row.get('browDownRight',0)]),
                'AU6':  np.mean([row.get('cheekSquintLeft',0), row.get('cheekSquintRight',0)]),
                'AU12': np.mean([row.get('mouthSmileLeft',0), row.get('mouthSmileRight',0)]),
                'AU15': np.mean([row.get('mouthFrownLeft',0), row.get('mouthFrownRight',0)]),
                'AU17': row.get('mouthPucker', 0),
                'start': row.get('start', 0),
            })
        print(f"  {sub:15} {len(df)} frames (MediaPipe→AU)")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_facial = pd.DataFrame(rows_facial)
print(f"  Total frames: {len(df_facial)}")

# Z-score sobre corpus completo
for au in ['AU1','AU4','AU6','AU12','AU15','AU17']:
    mu  = df_facial[au].mean()
    std = df_facial[au].std() + 1e-9
    df_facial[au+'_z'] = (df_facial[au] - mu) / std

# Score facial por frame
def facial_score(row):
    scores = []
    # Distress superior (AU1 inner brow + AU4 brow lowerer)
    scores.append(np.tanh((row['AU1_z'] + row['AU4_z']) / 2 * 0.5))
    # Tristeza (AU15 lip corner + AU17 chin)
    scores.append(np.tanh((row['AU15_z'] + row['AU17_z']) / 2 * 0.5))
    # Penalizar sonrisa social (AU12 sin AU6)
    if row['AU12'] > 1.0 and row['AU6'] < 0.5:
        scores.append(-0.3)
    return float(np.mean(scores))

df_facial['facial_z'] = df_facial.apply(facial_score, axis=1)
df_facial['facial_01'] = 1 / (1 + np.exp(-df_facial['facial_z'] * 2))

facial_por_subcaso = df_facial.groupby('subcaso')['facial_01'].mean()
print(f"\nScore facial unificado por subcaso:")
for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
    print(f"  {s:15} {facial_por_subcaso.get(s, 0):.3f}")

# ── Canal vocal — igual que antes ────────────────────────────────────────────
print(f"\n=== CANAL VOCAL ===")
EGEMAP_FILES = {
    'casanare':     'outputs/capa3/egemap_casanare_compareciente.csv',
    'catatumbo':    'outputs/capa3/egemap_catatumbo_compareciente.csv',
    'dabeiba':      'outputs/capa3/egemap_dabeiba_compareciente.csv',
    'huila':        'outputs/capa3/egemap_huila_compareciente.csv',
    'costa_caribe': 'outputs/capa3/egemap_costa_caribe_compareciente.csv',
}
VOCAL_FEATS = [
    'shimmerLocaldB_sma3nz_amean',
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    'HNRdBACF_sma3nz_amean',
    'loudness_sma3_amean',
]
dfs_eg = {}
for sub, path in EGEMAP_FILES.items():
    try:
        df = pd.read_csv(path); df['subcaso'] = sub
        dfs_eg[sub] = df
    except: pass

df_eg_all = pd.concat(dfs_eg.values(), ignore_index=True)
feats_ok  = [f for f in VOCAL_FEATS if f in df_eg_all.columns]
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
vocal_por_subcaso = df_eg_all.groupby('subcaso')['vocal_01'].mean()

print(f"Score vocal z-score por subcaso:")
for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
    print(f"  {s:15} {vocal_por_subcaso.get(s,0):.3f}")

# ── Canal verbal ──────────────────────────────────────────────────────────────
print(f"\n=== CANAL VERBAL ===")
try:
    df_ind = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
    all_rep = df_ind['y10_rep'].values
    mu_rep=all_rep.mean(); std_rep=all_rep.std()+1e-9
    verbal_por_subcaso = {}
    for sub, grp in df_ind.groupby('audio'):
        zs = (grp['y10_rep'].values - mu_rep)/std_rep
        verbal_por_subcaso[sub] = float((1/(1+np.exp(-zs*2))).mean())
    print(f"Score verbal z-score por subcaso:")
    for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
        print(f"  {s:15} {verbal_por_subcaso.get(s,0):.3f}")
except Exception as e:
    print(f"ERROR: {e}")
    verbal_por_subcaso = {
        'casanare':0.368,'catatumbo':0.452,'dabeiba':0.388,
        'huila':0.464,'costa_caribe':0.454
    }

# ── ICM v4 ────────────────────────────────────────────────────────────────────
print(f"\n=== ICM v4 FINAL ===")
subs = ['casanare','catatumbo','huila','dabeiba','costa_caribe']
facial_v = np.array([facial_por_subcaso.get(s, 0.5) for s in subs])
vocal_v  = np.array([vocal_por_subcaso.get(s, 0.5)  for s in subs])
verbal_v = np.array([verbal_por_subcaso.get(s, 0.5) for s in subs])

icm_v4 = 0.40*facial_v + 0.40*vocal_v + 0.20*verbal_v

print(f"  {'Subcaso':15} {'Facial':8} {'Vocal':8} {'Verbal':8} {'ICM v4':8}")
for i,s in enumerate(subs):
    print(f"  {s:15} {facial_v[i]:.3f}    {vocal_v[i]:.3f}    {verbal_v[i]:.3f}    {icm_v4[i]:.3f}")

# Rangos
print(f"\nRangos por canal (mayor = más discriminante):")
for nombre, v in [('Facial',facial_v),('Vocal',vocal_v),('Verbal',verbal_v)]:
    print(f"  {nombre:8} rango={v.max()-v.min():.3f}  CV={v.std()/v.mean():.3f}")

# Parsimonia
n_est=0; n_tot=0
for wf in np.arange(0.10,0.71,0.05):
    for wv in np.arange(0.10,0.71,0.05):
        wb=round(1-wf-wv,3)
        if 0.05<=wb<=0.60:
            alt=wf*facial_v+wv*vocal_v+wb*verbal_v
            rho,_=spearmanr(icm_v4,alt)
            n_tot+=1
            if rho>=0.90: n_est+=1
print(f"\nSensibilidad: {n_est}/{n_tot} combinaciones con rho>=0.90 ({100*n_est/n_tot:.0f}%)")

# Leave-one-out
print(f"\nLeave-one-out:")
for nombre,(wf,wv,wb) in {
    'Base (F+V+Vb)': (0.40,0.40,0.20),
    'Sin facial':    (0.00,0.60,0.40),
    'Sin vocal':     (0.60,0.00,0.40),
    'Sin verbal':    (0.50,0.50,0.00),
}.items():
    alt=wf*facial_v+wv*vocal_v+wb*verbal_v
    rho,_=spearmanr(icm_v4,alt)
    print(f"  {nombre:15} rho={rho:.3f}")

# Pesos empíricos
std_f=facial_v.std(); std_v=vocal_v.std(); std_b=verbal_v.std()
total=std_f+std_v+std_b
opt=std_f/total*facial_v+std_v/total*vocal_v+std_b/total*verbal_v
rho_opt,_=spearmanr(icm_v4,opt)
print(f"\nPesos empíricos (por std):")
print(f"  Facial={std_f/total:.3f} Vocal={std_v/total:.3f} Verbal={std_b/total:.3f}")
print(f"  rho vs teóricos: {rho_opt:.3f}")

# Guardar
icm_out={}
for i,s in enumerate(subs):
    icm_out[s]={
        'icm_facial_v4': round(float(facial_v[i]),3),
        'icm_vocal_v4':  round(float(vocal_v[i]),3),
        'icm_verbal_v4': round(float(verbal_v[i]),3),
        'icm_v4':        round(float(icm_v4[i]),3),
    }
with open('outputs/capa3/icm_tri_canal_v4.json','w',encoding='utf-8') as f:
    json.dump(icm_out,f,indent=2,ensure_ascii=False)
print(f"\n✓ Guardado: outputs/capa3/icm_tri_canal_v4.json")
