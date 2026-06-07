"""
ICM v5 corregido — mapeo correcto de nombres y columnas
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, percentileofscore
from pathlib import Path

# ── DIS/IEI ──────────────────────────────────────────────────────────────────
with open('outputs/capa3/dis_iei_comparativo.json', encoding='utf-8') as f:
    dis_iei = json.load(f)

SUBS = ['casanare','catatumbo','dabeiba','huila','costa_caribe']

# ── Canal facial — cargar con mapeo de nombres ────────────────────────────────
print("=== CANAL FACIAL ===")
AUS_FILES = {
    'casanare':     ('outputs/capa3/aus_casanare_torres.csv',    'casanare_torres', 'openface'),
    'catatumbo':    ('outputs/capa3/aus_catatumbo_SPEAKER_01.csv','catatumbo',       'openface'),
    'dabeiba':      ('outputs/capa3/aus_dabeiba_v2.csv',         'dabeiba',         'openface'),
    'huila':        ('outputs/capa3/aus_huila_comparecientes_v2.csv','huila',        'openface'),
    'costa_caribe': ('outputs/capa3/aus_costa_caribe_v2.csv',    'costa_caribe',    'mediapipe'),
}

rows_aus = []
for sub, (path, audio_id, tipo) in AUS_FILES.items():
    try:
        df = pd.read_csv(path)
        print(f"  {sub:15} {len(df)} frames ({tipo}) cols={df.columns.tolist()[:5]}")
        for _, row in df.iterrows():
            if tipo == 'openface':
                rows_aus.append({
                    'subcaso': sub,
                    'start':   row.get('start', row.get('inicio', 0)),
                    'AU1':  float(row.get('AU1', 0)),
                    'AU4':  float(row.get('AU4', 0)),
                    'AU6':  float(row.get('AU6', 0)),
                    'AU12': float(row.get('AU12', 0)),
                    'AU15': float(row.get('AU15', 0)),
                    'AU17': float(row.get('AU17', 0)),
                })
            else:  # mediapipe
                rows_aus.append({
                    'subcaso': sub,
                    'start':   row.get('start', 0),
                    'AU1':  float(row.get('browInnerUp', 0)),
                    'AU4':  float(np.mean([row.get('browDownLeft',0), row.get('browDownRight',0)])),
                    'AU6':  float(np.mean([row.get('cheekSquintLeft',0), row.get('cheekSquintRight',0)])),
                    'AU12': float(np.mean([row.get('mouthSmileLeft',0), row.get('mouthSmileRight',0)])),
                    'AU15': float(np.mean([row.get('mouthFrownLeft',0), row.get('mouthFrownRight',0)])),
                    'AU17': float(row.get('mouthPucker', 0)),
                })
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_aus = pd.DataFrame(rows_aus)
print(f"  Total: {len(df_aus)} frames")
print(f"  Subcasos en AUs: {df_aus['subcaso'].unique()}")

# Z-score conjunto sobre AUs
for au in ['AU1','AU4','AU6','AU12','AU15','AU17']:
    mu=df_aus[au].mean(); std=df_aus[au].std()+1e-9
    df_aus[au+'_z'] = (df_aus[au]-mu)/std

def facial_score(row):
    scores = []
    scores.append(float(np.tanh((row['AU1_z']+row['AU4_z'])/2*0.5)))
    scores.append(float(np.tanh((row['AU15_z']+row['AU17_z'])/2*0.5)))
    if row['AU12'] > 1.0 and row['AU6'] < 0.5:
        scores.append(-0.3)
    return float(np.mean(scores))

df_aus['facial_z']  = df_aus.apply(facial_score, axis=1)
df_aus['facial_01'] = 1/(1+np.exp(-df_aus['facial_z']*2))

facial_por_sub = df_aus.groupby('subcaso')['facial_01'].mean()
print(f"\nScore facial por subcaso:")
for s in SUBS:
    print(f"  {s:15} {facial_por_sub.get(s, float('nan')):.3f}")

# ── Canal vocal ───────────────────────────────────────────────────────────────
print("\n=== CANAL VOCAL ===")
EGEMAP_FILES = {
    'casanare':     'outputs/capa3/egemap_casanare_compareciente.csv',
    'catatumbo':    'outputs/capa3/egemap_catatumbo_compareciente.csv',
    'dabeiba':      'outputs/capa3/egemap_dabeiba_compareciente.csv',
    'huila':        'outputs/capa3/egemap_huila_compareciente.csv',
    'costa_caribe': 'outputs/capa3/egemap_costa_caribe_compareciente.csv',
}
VOCAL_FEATS = ['shimmerLocaldB_sma3nz_amean',
               'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
               'HNRdBACF_sma3nz_amean']

dfs_eg = []
for sub, path in EGEMAP_FILES.items():
    try:
        df = pd.read_csv(path)
        df['subcaso'] = sub
        dfs_eg.append(df)
        print(f"  {sub:15} {len(df)} segmentos")
    except Exception as e:
        print(f"  {sub:15} ERROR: {e}")

df_eg = pd.concat(dfs_eg, ignore_index=True)
feats_ok = [f for f in VOCAL_FEATS if f in df_eg.columns]
for f in feats_ok:
    mu=df_eg[f].mean(); std=df_eg[f].std()+1e-9
    df_eg[f+'_z'] = (df_eg[f]-mu)/std

def vocal_score(row):
    scores = []
    if 'shimmerLocaldB_sma3nz_amean_z' in row.index:
        scores.append(float(np.tanh(row['shimmerLocaldB_sma3nz_amean_z']*0.5)))
    if 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z' in row.index:
        scores.append(float(np.tanh(row['F0semitoneFrom27.5Hz_sma3nz_stddevNorm_z']*0.5)))
    if 'HNRdBACF_sma3nz_amean_z' in row.index:
        scores.append(float(np.tanh(-row['HNRdBACF_sma3nz_amean_z']*0.5)))
    return float(np.mean(scores)) if scores else 0.5

df_eg['vocal_z']  = df_eg.apply(vocal_score, axis=1)
df_eg['vocal_01'] = 1/(1+np.exp(-df_eg['vocal_z']*2))
vocal_por_sub = df_eg.groupby('subcaso')['vocal_01'].mean()

print(f"\nScore vocal por subcaso:")
for s in SUBS:
    print(f"  {s:15} {vocal_por_sub.get(s, float('nan')):.3f}")

# ── Canal verbal ──────────────────────────────────────────────────────────────
print("\n=== CANAL VERBAL ===")
df_ind = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
all_rep = df_ind['y10_rep'].values
mu_rep=all_rep.mean(); std_rep=all_rep.std()+1e-9
df_ind['verbal_z']  = (df_ind['y10_rep']-mu_rep)/std_rep
df_ind['verbal_01'] = 1/(1+np.exp(-df_ind['verbal_z']*2))
verbal_por_sub = df_ind.groupby('audio')['verbal_01'].mean()

print(f"Score verbal por subcaso:")
for s in SUBS:
    print(f"  {s:15} {verbal_por_sub.get(s, float('nan')):.3f}")

# ── Análisis de disociación ───────────────────────────────────────────────────
print("\n=== ANÁLISIS DE DISOCIACIÓN ===")
resultados = []
for sub in SUBS:
    v_vocal  = float(vocal_por_sub.get(sub,  np.nan))
    v_facial = float(facial_por_sub.get(sub, np.nan))
    v_verbal = float(verbal_por_sub.get(sub, np.nan))

    if any(np.isnan([v_vocal, v_facial, v_verbal])):
        print(f"  {sub}: facial={v_facial:.3f} vocal={v_vocal:.3f} verbal={v_verbal:.3f}")
        # Si falta facial, calcular dis solo verbal-vocal
        if np.isnan(v_facial):
            dis_vv  = abs(v_verbal - v_vocal)
            dis_tri = dis_vv  # aproximación con 2 canales
            cong    = int(v_verbal < 0.5 and v_vocal < 0.5)
            v_facial_rep = np.nan
        else:
            dis_vv  = abs(v_verbal - v_vocal)
            dis_tri = float(np.std([v for v in [v_verbal,v_vocal,v_facial] if not np.isnan(v)]))
            cong    = int(v_verbal < 0.5 and v_vocal < 0.5)
            v_facial_rep = v_facial
    else:
        dis_vv  = abs(v_verbal - v_vocal)
        dis_tri = float(np.std([v_verbal, v_vocal, v_facial]))
        cong    = int(v_verbal < 0.5 and v_vocal < 0.5 and v_facial < 0.5)
        v_facial_rep = v_facial

    # Variación interna del canal verbal
    verbal_segs = df_ind[df_ind['audio']==sub]['verbal_01'].values
    verbal_std  = float(np.std(verbal_segs)) if len(verbal_segs) > 0 else np.nan
    verbal_pct_alto = float(np.mean(verbal_segs > 0.55)) if len(verbal_segs) > 0 else np.nan

    # Perfil
    if np.isnan(v_facial_rep):
        mean_noverbal = v_vocal
    else:
        mean_noverbal = (v_vocal + v_facial_rep) / 2

    if dis_tri < 0.05:
        tipo = "CONGRUENTE"
    elif v_verbal > mean_noverbal + 0.05:
        tipo = "VERBAL>NO-VERBAL (Fricker gap)"
    elif v_verbal < mean_noverbal - 0.05:
        tipo = "NO-VERBAL>VERBAL"
    else:
        tipo = "DISOCIADO-NEUTRO"

    resultados.append({
        'subcaso':       sub,
        'DIS':           dis_iei[sub]['DIS_score'],
        'IEI':           dis_iei[sub]['IEI_score'],
        'vocal':         round(v_vocal, 3),
        'facial':        round(v_facial_rep, 3) if not np.isnan(v_facial_rep) else None,
        'verbal':        round(v_verbal, 3),
        'dis_verbal_vocal':  round(dis_vv, 3),
        'dis_tri':       round(dis_tri, 3),
        'cong_injusta':  bool(cong),
        'verbal_std':    round(verbal_std, 3) if not np.isnan(verbal_std) else None,
        'verbal_pct_alto': round(verbal_pct_alto, 3) if not np.isnan(verbal_pct_alto) else None,
        'tipo_perfil':   tipo,
        'canales_disponibles': 2 if np.isnan(v_facial_rep) else 3,
    })

df_res = pd.DataFrame(resultados)

print(f"\n{'Subcaso':15} {'DIS':6} {'IEI':6} {'Vocal':7} {'Facial':7} {'Verbal':7} {'dis_tri':8} {'Perfil'}")
print("-"*80)
for _,r in df_res.iterrows():
    fac = f"{r['facial']:.3f}" if r['facial'] is not None else " NaN "
    print(f"{r['subcaso']:15} {r['DIS']:6.3f} {r['IEI']:6.3f} "
          f"{r['vocal']:7.3f} {fac:7} {r['verbal']:7.3f} "
          f"{r['dis_tri']:8.3f} {r['tipo_perfil']}")

# ── Correlaciones (solo subcasos con 3 canales) ───────────────────────────────
df_3c = df_res[df_res['canales_disponibles']==3]
df_2c = df_res[df_res['canales_disponibles']==2]

print(f"\nSubcasos con 3 canales: {df_3c['subcaso'].tolist()}")
print(f"Subcasos con 2 canales: {df_2c['subcaso'].tolist()}")

if len(df_3c) >= 3:
    rho_a,_ = spearmanr(df_3c['IEI'], df_3c['dis_tri'])
    rho_b,_ = spearmanr(df_3c['DIS'], df_3c['dis_tri'])
    print(f"\nCorrelaciones (N={len(df_3c)}, solo subcasos con 3 canales):")
    print(f"  rho(IEI, dis_tri) = {rho_a:+.3f}  [H_ICM_A]")
    print(f"  rho(DIS, dis_tri) = {rho_b:+.3f}")
else:
    print(f"\nInsuficientes subcasos con 3 canales para correlación (N={len(df_3c)})")

# Correlación sobre dis_verbal_vocal (disponible para todos)
rho_vv_iei,_ = spearmanr(df_res['IEI'], df_res['dis_verbal_vocal'])
rho_vv_dis,_ = spearmanr(df_res['DIS'], df_res['dis_verbal_vocal'])
print(f"\nCorrelaciones dis_verbal_vocal (N=5, todos los subcasos):")
print(f"  rho(IEI, dis_verbal_vocal) = {rho_vv_iei:+.3f}  [H_ICM_A parcial]")
print(f"  rho(DIS, dis_verbal_vocal) = {rho_vv_dis:+.3f}")

print(f"\nH_ICM_B (congruencia injusta):")
for _,r in df_res.iterrows():
    print(f"  {r['subcaso']:15} DIS={r['DIS']:.3f} IEI={r['IEI']:.3f} cong_injusta={r['cong_injusta']}")

# ── Guardar ───────────────────────────────────────────────────────────────────
df_res.to_csv('outputs/capa3/icm_v5_resultados.csv', index=False, encoding='utf-8-sig')

icm_v5 = {}
for _,r in df_res.iterrows():
    icm_v5[r['subcaso']] = {
        'vocal':             r['vocal'],
        'facial':            r['facial'],
        'verbal':            r['verbal'],
        'dis_verbal_vocal':  r['dis_verbal_vocal'],
        'dis_tri':           r['dis_tri'],
        'cong_injusta':      r['cong_injusta'],
        'tipo_perfil':       r['tipo_perfil'],
        'canales':           int(r['canales_disponibles']),
        'H_ICM_A':           bool(r['dis_tri'] > float(df_res['dis_tri'].median())),
        'H_ICM_B':           r['cong_injusta'],
    }

with open('outputs/capa3/icm_tri_canal_v5.json','w',encoding='utf-8') as f:
    json.dump(icm_v5, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✓ outputs/capa3/icm_v5_resultados.csv")
print(f"✓ outputs/capa3/icm_tri_canal_v5.json")
