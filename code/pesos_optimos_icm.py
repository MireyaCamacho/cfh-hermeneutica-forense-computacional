"""
Pesos óptimos ICM — análisis de variación entre subcasos
"""
import json, numpy as np
from scipy.stats import spearmanr

d = json.load(open('outputs/capa3/icm_tri_canal_v2.json'))
subs   = ['casanare','catatumbo','huila','dabeiba','costa_caribe']
facial = np.array([d[s]['icm_facial']    for s in subs])
vocal  = np.array([d[s]['icm_vocal']     for s in subs])
verbal = np.array([d[s]['icm_verbal_v2'] for s in subs])

# Rango y CV por canal (poder discriminante entre subcasos)
print("Poder discriminante por canal:")
for nombre, vals in [('Facial',facial),('Vocal',vocal),('Verbal',verbal)]:
    rango = vals.max()-vals.min()
    cv    = vals.std()/vals.mean()
    print(f"  {nombre:8} rango={rango:.3f}  CV={cv:.3f}  min={vals.min():.3f}  max={vals.max():.3f}")

# Pesos empíricos óptimos proporcionales al rango
rango_f = facial.max()-facial.min()
rango_v = vocal.max()-vocal.min()
rango_b = verbal.max()-verbal.min()
total   = rango_f+rango_v+rango_b
wf_opt  = rango_f/total
wv_opt  = rango_v/total
wb_opt  = rango_b/total

print(f"\nPesos empíricos óptimos (proporcionales al rango):")
print(f"  Facial  = {wf_opt:.3f}  (teórico: 0.400)")
print(f"  Vocal   = {wv_opt:.3f}  (teórico: 0.400)")
print(f"  Verbal  = {wb_opt:.3f}  (teórico: 0.200)")

base     = 0.40*facial + 0.40*vocal + 0.20*verbal
opt      = wf_opt*facial + wv_opt*vocal + wb_opt*verbal
rho_opt,_ = spearmanr(base, opt)
print(f"\nCorrelación pesos teóricos vs empíricos: rho={rho_opt:.3f}")

# Tres versiones como en DIS
print(f"\nComparación tres versiones de pesos:")
v1 = 0.40*facial + 0.40*vocal + 0.20*verbal  # teórico
v2 = wf_opt*facial + wv_opt*vocal + wb_opt*verbal  # empírico
v3 = 0.35*facial + 0.35*vocal + 0.30*verbal  # compromiso
print(f"{'Subcaso':15} {'v1 teórico':12} {'v2 empírico':12} {'v3 comproms':12}")
for i,s in enumerate(subs):
    print(f"  {s:13} {v1[i]:.3f}        {v2[i]:.3f}        {v3[i]:.3f}")

r12,_ = spearmanr(v1,v2)
r13,_ = spearmanr(v1,v3)
print(f"\nrho v1-v2: {r12:.3f}")
print(f"rho v1-v3: {r13:.3f}")

# Ranking
def rank(v): return np.argsort(np.argsort(v))+1
r1=rank(v1); r2=rank(v2); r3=rank(v3)
print(f"\nRankings (1=menor ICM):")
print(f"{'Subcaso':15} {'Rk v1':8} {'Rk v2':8} {'Rk v3':8} {'Cambia?':8}")
for i,s in enumerate(subs):
    cambio = "SI" if r1[i]!=r2[i] else "no"
    print(f"  {s:13} {r1[i]:8} {r2[i]:8} {r3[i]:8} {cambio:8}")
