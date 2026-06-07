"""
Análisis de parsimonia ICM — 5 subcasos Corpus C
"""
import json, numpy as np
from scipy.stats import spearmanr

d = json.load(open('outputs/capa3/icm_tri_canal_v2.json'))
subs   = ['casanare','catatumbo','huila','dabeiba','costa_caribe']
facial = np.array([d[s]['icm_facial']    for s in subs])
vocal  = np.array([d[s]['icm_vocal']     for s in subs])
verbal = np.array([d[s]['icm_verbal_v2'] for s in subs])

print('Valores por canal:')
for i,s in enumerate(subs):
    print(f'  {s:15} F={facial[i]:.3f} V={vocal[i]:.3f} Vb={verbal[i]:.3f}')

base = 0.40*facial + 0.40*vocal + 0.20*verbal
print(f'\nICM con pesos teóricos (0.40/0.40/0.20):')
for i,s in enumerate(subs):
    print(f'  {s:15} {base[i]:.3f}')

n_est=0; n_tot=0
for wf in np.arange(0.10,0.71,0.05):
    for wv in np.arange(0.10,0.71,0.05):
        wb = round(1-wf-wv,3)
        if 0.05<=wb<=0.60:
            alt  = wf*facial + wv*vocal + wb*verbal
            rho,_ = spearmanr(base, alt)
            n_tot += 1
            if rho >= 0.90: n_est += 1

print(f'\nSensibilidad de pesos ICM:')
print(f'  Combinaciones evaluadas: {n_tot}')
print(f'  Con rho>=0.90: {n_est} ({100*n_est/n_tot:.0f}%)')

# Leave-one-out
print(f'\nLeave-one-out:')
configs = {
    'Base (F+V+Vb)':   (0.40,0.40,0.20),
    'Sin facial':       (0.00,0.60,0.40),
    'Sin vocal':        (0.60,0.00,0.40),
    'Sin verbal':       (0.50,0.50,0.00),
    'Solo facial':      (1.00,0.00,0.00),
    'Solo vocal':       (0.00,1.00,0.00),
    'Solo verbal':      (0.00,0.00,1.00),
}
for nombre,(wf,wv,wb) in configs.items():
    alt  = wf*facial + wv*vocal + wb*verbal
    rho,_ = spearmanr(base, alt)
    print(f'  {nombre:20} rho={rho:.3f}')
