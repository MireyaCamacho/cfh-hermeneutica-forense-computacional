"""
Diagnóstico dominancia canal verbal en ICM
"""
import json, numpy as np
from scipy.stats import spearmanr

d = json.load(open('outputs/capa3/icm_tri_canal_v2.json'))
subs   = ['casanare','catatumbo','huila','dabeiba','costa_caribe']
facial = np.array([d[s]['icm_facial']    for s in subs])
vocal  = np.array([d[s]['icm_vocal']     for s in subs])
verbal = np.array([d[s]['icm_verbal_v2'] for s in subs])

def rank(v): return np.argsort(np.argsort(v))+1

print("Rankings por canal individual:")
print(f"  {'Subcaso':15} {'Facial':8} {'Vocal':8} {'Verbal':8} {'ICM base':8}")
rb = rank(0.40*facial+0.40*vocal+0.20*verbal)
rf = rank(facial); rv = rank(vocal); rvb = rank(verbal)
for i,s in enumerate(subs):
    print(f"  {s:15} {rf[i]:8} {rv[i]:8} {rvb[i]:8} {rb[i]:8}")

print(f"\nCorrelacion canal individual vs ICM base:")
for nombre, vals in [('Facial',facial),('Vocal',vocal),('Verbal',verbal)]:
    rho,_ = spearmanr(vals, 0.40*facial+0.40*vocal+0.20*verbal)
    print(f"  {nombre:8} rho={rho:.3f}")

print(f"\nDiagnostico:")
print(f"  Si verbal rho con base ~ 1.0 -> verbal domina el indice")
print(f"  El ρ=1.000 entre versiones de pesos NO es robustez real")
print(f"  Es artefacto de que un canal tiene rango 5x mayor que los otros")

print(f"\nRangos normalizados:")
for nombre, vals in [('Facial',facial),('Vocal',vocal),('Verbal',verbal)]:
    vn = (vals-vals.min())/(vals.max()-vals.min()+1e-9)
    print(f"  {nombre:8} norm: {[round(x,3) for x in vn]}")
