"""
ICM tri-canal v3 — normalización min-max por canal
===================================================
Problema detectado en v2: el canal verbal (rango=0.710) dominaba
el índice porque los tres canales tenían rangos muy distintos.
Solución: normalizar cada canal min-max dentro del Corpus C
antes de agregar, igual que DIS e IEI.

Outputs:
  - outputs/capa3/icm_tri_canal_v3.json
  - outputs/capa3/icm_resultados_v4.csv
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

d = json.load(open('outputs/capa3/icm_tri_canal_v2.json'))
subs   = ['casanare','catatumbo','huila','dabeiba','costa_caribe']
facial = np.array([d[s]['icm_facial']    for s in subs])
vocal  = np.array([d[s]['icm_vocal']     for s in subs])
verbal = np.array([d[s]['icm_verbal_v2'] for s in subs])

# Normalización min-max por canal dentro del Corpus C
def norm(v): return (v - v.min()) / (v.max() - v.min() + 1e-9)

facial_n = norm(facial)
vocal_n  = norm(vocal)
verbal_n = norm(verbal)

print("Valores normalizados por canal:")
print(f"  {'Subcaso':15} {'Facial_n':10} {'Vocal_n':10} {'Verbal_n':10}")
for i,s in enumerate(subs):
    print(f"  {s:15} {facial_n[i]:.3f}      {vocal_n[i]:.3f}      {verbal_n[i]:.3f}")

# ICM v3 con pesos teóricos sobre valores normalizados
icm_v3 = 0.40*facial_n + 0.40*vocal_n + 0.20*verbal_n

print(f"\nICM v3 (normalizado, pesos 0.40/0.40/0.20):")
for i,s in enumerate(subs):
    print(f"  {s:15} {icm_v3[i]:.3f}")

# Análisis de parsimonia sobre valores normalizados
n_est=0; n_tot=0
for wf in np.arange(0.10,0.71,0.05):
    for wv in np.arange(0.10,0.71,0.05):
        wb=round(1-wf-wv,3)
        if 0.05<=wb<=0.60:
            alt=wf*facial_n+wv*vocal_n+wb*verbal_n
            rho,_=spearmanr(icm_v3,alt)
            n_tot+=1
            if rho>=0.90: n_est+=1

print(f"\nSensibilidad de pesos ICM v3 (normalizado):")
print(f"  Combinaciones: {n_tot}")
print(f"  Con rho>=0.90: {n_est} ({100*n_est/n_tot:.0f}%)")

# Leave-one-out sobre normalizados
print(f"\nLeave-one-out (normalizado):")
configs = {
    'Base (F+V+Vb)':  (0.40, 0.40, 0.20),
    'Sin facial':     (0.00, 0.60, 0.40),
    'Sin vocal':      (0.60, 0.00, 0.40),
    'Sin verbal':     (0.50, 0.50, 0.00),
    'Solo facial':    (1.00, 0.00, 0.00),
    'Solo vocal':     (0.00, 1.00, 0.00),
    'Solo verbal':    (0.00, 0.00, 1.00),
}
for nombre,(wf,wv,wb) in configs.items():
    alt=wf*facial_n+wv*vocal_n+wb*verbal_n
    rho,_=spearmanr(icm_v3,alt)
    print(f"  {nombre:20} rho={rho:.3f}")

# Pesos empíricos óptimos sobre normalizados
rango_f = facial_n.max()-facial_n.min()
rango_v = vocal_n.max()-vocal_n.min()
rango_b = verbal_n.max()-verbal_n.min()
total   = rango_f+rango_v+rango_b
wf_opt  = rango_f/total
wv_opt  = rango_v/total
wb_opt  = rango_b/total
icm_opt = wf_opt*facial_n+wv_opt*vocal_n+wb_opt*verbal_n
rho_opt,_ = spearmanr(icm_v3, icm_opt)

print(f"\nPesos empíricos óptimos (normalizados):")
print(f"  Facial  = {wf_opt:.3f}  (teórico: 0.400)")
print(f"  Vocal   = {wv_opt:.3f}  (teórico: 0.400)")
print(f"  Verbal  = {wb_opt:.3f}  (teórico: 0.200)")
print(f"  rho pesos teóricos vs empíricos: {rho_opt:.3f}")

# Comparación rankings v2 vs v3
icm_v2 = 0.40*facial + 0.40*vocal + 0.20*verbal
def rank(v): return np.argsort(np.argsort(v))+1
rk_v2 = rank(icm_v2)
rk_v3 = rank(icm_v3)
print(f"\nCambios de ranking v2 (sin norm) vs v3 (normalizado):")
print(f"  {'Subcaso':15} {'ICM v2':8} {'Rk v2':8} {'ICM v3':8} {'Rk v3':8} {'Cambio':8}")
for i,s in enumerate(subs):
    cambio = "SI" if rk_v2[i]!=rk_v3[i] else "no"
    print(f"  {s:15} {icm_v2[i]:.3f}    {rk_v2[i]:5}    {icm_v3[i]:.3f}    {rk_v3[i]:5}    {cambio}")

# Guardar JSON v3
icm_data_v3 = {}
for i,s in enumerate(subs):
    icm_data_v3[s] = {
        'icm_facial':     round(float(facial[i]), 3),
        'icm_vocal':      round(float(vocal[i]), 3),
        'icm_verbal':     round(float(verbal[i]), 3),
        'icm_facial_norm':round(float(facial_n[i]), 3),
        'icm_vocal_norm': round(float(vocal_n[i]), 3),
        'icm_verbal_norm':round(float(verbal_n[i]), 3),
        'icm_v3':         round(float(icm_v3[i]), 3),
    }
with open('outputs/capa3/icm_tri_canal_v3.json','w',encoding='utf-8') as f:
    json.dump(icm_data_v3, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: outputs/capa3/icm_tri_canal_v3.json")

# Guardar CSV v4
df = pd.read_csv('outputs/capa3/icm_resultados_v3.csv')
df_v4 = df.copy()
orden = {'casanare_torres':0,'catatumbo':1,'huila':2,'dabeiba':3,'costa_caribe':4}
for i,s in enumerate(subs):
    audio_id = 'casanare_torres' if s=='casanare' else s
    df_v4.loc[df_v4['audio']==audio_id, 'icm_score'] = round(float(icm_v3[i]),3)
df_v4.to_csv('outputs/capa3/icm_resultados_v4.csv', index=False, encoding='utf-8-sig')
print(f"✓ Guardado: outputs/capa3/icm_resultados_v4.csv")

print(f"\n=== CORPUS C ICM v3 FINAL ===")
print(df_v4.to_string(index=False))
