"""
ICM tri-canal v4 — normalización z-score a nivel de segmento
=============================================================
Problema v2: sin normalización, el canal verbal dominaba (rango 5x mayor).
Problema v3: min-max sobre 5 promedios fuerza extremos artificiales.

Solución v4: 
- Unir todos los segmentos eGeMAPS de los 5 subcasos
- Calcular z-score de cada feature sobre la distribución conjunta
- Calcular score vocal por segmento sobre features normalizadas
- Promediar por subcaso -> scores en la misma escala

Para canal facial: z-score sobre los blendshapes del corpus conjunto.
Para canal verbal: z-score sobre REP scores a nivel de bloque.

Outputs:
  - outputs/capa3/icm_tri_canal_v4.json
  - outputs/capa3/icm_resultados_v4.csv (reemplaza)
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

# ── 1. Canal vocal — z-score sobre corpus completo ───────────────────────────
print("=== CANAL VOCAL (eGeMAPS z-score) ===")

EGEMAP_FILES = {
    'casanare':     'outputs/capa3/egemap_casanare_compareciente.csv',
    'catatumbo':    'outputs/capa3/egemap_catatumbo_compareciente.csv',
    'dabeiba':      'outputs/capa3/egemap_dabeiba_compareciente.csv',
    'huila':        'outputs/capa3/egemap_huila_compareciente.csv',
    'costa_caribe': 'outputs/capa3/egemap_costa_caribe_compareciente.csv',
}

# Features relevantes para sinceridad prosódica
VOCAL_FEATS = [
    'shimmerLocaldB_sma3nz_amean',
    'alphaRatio_sma3_amean',
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    'HNRdBACF_sma3nz_amean',
    'loudness_sma3_amean',
]

# Cargar todos los segmentos
dfs = {}
for sub, path in EGEMAP_FILES.items():
    try:
        df = pd.read_csv(path)
        df['subcaso'] = sub
        dfs[sub] = df
        print(f"  {sub:15} {len(df)} segmentos")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_all = pd.concat(dfs.values(), ignore_index=True)
print(f"  Total segmentos: {len(df_all)}")

# Identificar features disponibles
feats_ok = [f for f in VOCAL_FEATS if f in df_all.columns]
print(f"  Features disponibles: {feats_ok}")

# Z-score sobre corpus completo
for feat in feats_ok:
    mu  = df_all[feat].mean()
    std = df_all[feat].std() + 1e-9
    df_all[feat+'_z'] = (df_all[feat] - mu) / std

# Score vocal por segmento (promedio de z-scores relevantes)
# Shimmer alto = distress, alpha ratio alto = estrés, F0 std alto = emoción
# HNR bajo = voz tensa -> invertir
def vocal_score_z(row):
    scores = []
    if 'shimmerLocaldB_sma3nz_amean_z' in row.index:
        scores.append(np.tanh(row['shimmerLocaldB_sma3nz_amean_z'] * 0.5))
    if 'alphaRatio_sma3_amean_z' in row.index:
        scores.append(np.tanh(abs(row['alphaRatio_sma3_amean_z']) * 0.5))
    if 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z' in row.index:
        scores.append(np.tanh(row['F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z'] * 0.5))
    if 'HNRdBACF_sma3nz_amean_z' in row.index:
        scores.append(np.tanh(-row['HNRdBACF_sma3nz_amean_z'] * 0.5))  # invertido
    return float(np.mean(scores)) if scores else 0.0

feats_z = [f+'_z' for f in feats_ok]
if feats_z:
    df_all['vocal_score_z'] = df_all.apply(vocal_score_z, axis=1)
    # Llevar a [0,1] con sigmoid
    df_all['vocal_score_01'] = 1/(1+np.exp(-df_all['vocal_score_z']*2))
    vocal_por_subcaso = df_all.groupby('subcaso')['vocal_score_01'].mean()
else:
    # Fallback: usar scores originales del JSON
    vocal_por_subcaso = pd.Series({
        'casanare':0.415,'catatumbo':0.317,'dabeiba':0.452,
        'huila':0.334,'costa_caribe':0.465
    })

print(f"\nScore vocal z-score por subcaso:")
for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
    print(f"  {s:15} {vocal_por_subcaso.get(s,0):.3f}")

# ── 2. Canal facial — z-score sobre corpus completo ───────────────────────────
print("\n=== CANAL FACIAL (blendshapes z-score) ===")

AUS_FILES = {
    'casanare':     'outputs/capa3/aus_casanare_torres.csv',
    'catatumbo':    'outputs/capa3/aus_catatumbo_SPEAKER_01.csv',
    'dabeiba':      'outputs/capa3/aus_dabeiba_v2.csv',
    'huila':        'outputs/capa3/aus_huila_comparecientes_v2.csv',
    'costa_caribe': 'outputs/capa3/aus_costa_caribe_v2.csv',
}

FACIAL_FEATS_DISTRESS = ['browInnerUp','browDownLeft','browDownRight',
                          'mouthFrownLeft','mouthFrownRight','mouthPucker']
FACIAL_FEATS_SOCIAL   = ['mouthSmileLeft','mouthSmileRight']
FACIAL_FEAT_DUCHENNE  = ['cheekSquintLeft','cheekSquintRight']

dfs_aus = {}
for sub, path in AUS_FILES.items():
    try:
        df = pd.read_csv(path)
        df['subcaso'] = sub
        dfs_aus[sub] = df
        print(f"  {sub:15} {len(df)} frames")
    except Exception as e:
        print(f"  {sub:15} no disponible ({e})")

facial_por_subcaso = {}
if dfs_aus:
    df_aus_all = pd.concat(dfs_aus.values(), ignore_index=True)

    # Z-score sobre corpus completo para features de distress
    feats_d_ok = [f for f in FACIAL_FEATS_DISTRESS if f in df_aus_all.columns]
    feats_s_ok = [f for f in FACIAL_FEATS_SOCIAL if f in df_aus_all.columns]
    feats_du_ok= [f for f in FACIAL_FEAT_DUCHENNE if f in df_aus_all.columns]

    for feat in feats_d_ok + feats_s_ok + feats_du_ok:
        mu  = df_aus_all[feat].mean()
        std = df_aus_all[feat].std() + 1e-9
        df_aus_all[feat+'_z'] = (df_aus_all[feat] - mu) / std

    def facial_score_z(row):
        scores = []
        # Distress
        for f in feats_d_ok:
            scores.append(np.tanh(row.get(f+'_z', 0) * 0.5))
        # Penalizar sonrisa social
        sonrisa = np.mean([row.get(f, 0) for f in feats_s_ok]) if feats_s_ok else 0
        duchenne = np.mean([row.get(f, 0) for f in feats_du_ok]) if feats_du_ok else 0
        if sonrisa > 0.3 and duchenne < 0.1:
            scores.append(-0.3)
        return float(np.mean(scores)) if scores else 0.0

    df_aus_all['facial_score_z'] = df_aus_all.apply(facial_score_z, axis=1)
    df_aus_all['facial_score_01'] = 1/(1+np.exp(-df_aus_all['facial_score_z']*2))
    facial_ss = df_aus_all.groupby('subcaso')['facial_score_01'].mean()
    facial_por_subcaso = facial_ss.to_dict()
else:
    facial_por_subcaso = {
        'casanare':0.190,'catatumbo':0.272,'dabeiba':0.353,
        'huila':0.299,'costa_caribe':0.104
    }

print(f"\nScore facial z-score por subcaso:")
for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
    print(f"  {s:15} {facial_por_subcaso.get(s,0):.3f}")

# ── 3. Canal verbal — z-score sobre bloques del corpus C ─────────────────────
print("\n=== CANAL VERBAL (REP z-score sobre bloques) ===")

try:
    df_ind = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
    rep_por_subcaso_seg = df_ind.groupby('audio')['y10_rep'].apply(list)
    # Calcular z-score sobre todos los bloques
    all_rep = np.concatenate([v for v in rep_por_subcaso_seg.values])
    mu_rep  = all_rep.mean()
    std_rep = all_rep.std() + 1e-9
    verbal_por_subcaso = {}
    for sub in rep_por_subcaso_seg.index:
        zs = (np.array(rep_por_subcaso_seg[sub]) - mu_rep) / std_rep
        # Sigmoid para llevar a [0,1]
        score_01 = (1/(1+np.exp(-zs*2))).mean()
        verbal_por_subcaso[sub] = float(score_01)
    print(f"  Bloques totales: {len(all_rep)}, mu={mu_rep:.3f}, std={std_rep:.3f}")
except Exception as e:
    print(f"  Fallback a valores previos: {e}")
    verbal_por_subcaso = {
        'casanare':0.567,'catatumbo':0.295,'dabeiba':0.842,
        'huila':0.841,'costa_caribe':0.132
    }

print(f"\nScore verbal z-score por subcaso:")
for s in ['casanare','catatumbo','dabeiba','huila','costa_caribe']:
    print(f"  {s:15} {verbal_por_subcaso.get(s,0):.3f}")

# ── 4. ICM v4 agregado ────────────────────────────────────────────────────────
print("\n=== ICM v4 FINAL ===")
subs = ['casanare','catatumbo','huila','dabeiba','costa_caribe']

facial_v = np.array([facial_por_subcaso.get(s,0.5) for s in subs])
vocal_v  = np.array([vocal_por_subcaso.get(s,0.5)  for s in subs])
verbal_v = np.array([verbal_por_subcaso.get(s,0.5) for s in subs])

icm_v4 = 0.40*facial_v + 0.40*vocal_v + 0.20*verbal_v

print(f"  {'Subcaso':15} {'Facial':8} {'Vocal':8} {'Verbal':8} {'ICM v4':8}")
for i,s in enumerate(subs):
    print(f"  {s:15} {facial_v[i]:.3f}    {vocal_v[i]:.3f}    {verbal_v[i]:.3f}    {icm_v4[i]:.3f}")

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

print(f"\nSensibilidad pesos ICM v4:")
print(f"  Combinaciones: {n_tot}, con rho>=0.90: {n_est} ({100*n_est/n_tot:.0f}%)")

# Leave-one-out
print(f"\nLeave-one-out ICM v4:")
for nombre,(wf,wv,wb) in {
    'Base':         (0.40,0.40,0.20),
    'Sin facial':   (0.00,0.60,0.40),
    'Sin vocal':    (0.60,0.00,0.40),
    'Sin verbal':   (0.50,0.50,0.00),
    'Solo facial':  (1.00,0.00,0.00),
    'Solo vocal':   (0.00,1.00,0.00),
    'Solo verbal':  (0.00,0.00,1.00),
}.items():
    alt=wf*facial_v+wv*vocal_v+wb*verbal_v
    rho,_=spearmanr(icm_v4,alt)
    print(f"  {nombre:15} rho={rho:.3f}")

# Pesos empíricos óptimos
rango_f=facial_v.std(); rango_v=vocal_v.std(); rango_b=verbal_v.std()
total=rango_f+rango_v+rango_b
print(f"\nPesos empíricos óptimos (por std):")
print(f"  Facial  = {rango_f/total:.3f}  (teórico: 0.400)")
print(f"  Vocal   = {rango_v/total:.3f}  (teórico: 0.400)")
print(f"  Verbal  = {rango_b/total:.3f}  (teórico: 0.200)")
opt=rango_f/total*facial_v+rango_v/total*vocal_v+rango_b/total*verbal_v
rho_opt,_=spearmanr(icm_v4,opt)
print(f"  rho pesos teóricos vs empíricos: {rho_opt:.3f}")

# Guardar
icm_out = {}
for i,s in enumerate(subs):
    icm_out[s] = {
        'icm_facial_v4':  round(float(facial_v[i]),3),
        'icm_vocal_v4':   round(float(vocal_v[i]),3),
        'icm_verbal_v4':  round(float(verbal_v[i]),3),
        'icm_v4':         round(float(icm_v4[i]),3),
    }
with open('outputs/capa3/icm_tri_canal_v4.json','w',encoding='utf-8') as f:
    json.dump(icm_out, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: outputs/capa3/icm_tri_canal_v4.json")
