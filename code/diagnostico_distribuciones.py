"""
Diagnóstico distribuciones y2_sa, y4_nv, y10_rep en corpus A+B
"""
import pandas as pd, numpy as np

df = pd.read_csv('indicators_final_completo.csv')
ab = df[df.corpus_type.isin(['A-CE','A-CSJ','B'])]

print(f"N corpus A+B: {len(ab)}")
print()

for col in ['y2_sa','y4_nv','y10_rep']:
    v = ab[col].values
    pcts = [1,5,10,25,50,75,90,95,99]
    print(f"=== {col} ===")
    print(f"  zeros:  {100*(v==0).mean():.1f}%")
    print(f"  ones:   {100*(v==1).mean():.1f}%")
    print(f"  mean:   {v.mean():.3f}  std: {v.std():.3f}")
    for p in pcts:
        print(f"  p{p:02d}:   {np.percentile(v,p):.3f}")
    print()

# Comparar estrategias de normalización sobre Corpus C
print("=== ESTRATEGIAS DE NORMALIZACIÓN SOBRE CORPUS C ===")
print("(usando mu/std de A+B como referencia)")
print()

cc = pd.read_csv('data/indicators_corpus_c_capa1_v2.csv')
print(f"N Corpus C bloques: {len(cc)}")

for col, cc_col in [('y2_sa','y2_sa'),('y4_nv','y4_nv'),('y10_rep','y10_rep')]:
    mu  = ab[col].mean()
    std = ab[col].std() + 1e-9
    p5  = np.percentile(ab[col], 5)
    p95 = np.percentile(ab[col], 95)

    if cc_col not in cc.columns:
        print(f"  {col}: columna no encontrada en corpus C")
        continue

    v_cc = cc[cc_col].values

    # Z-score con mu/std de A+B -> sigmoid
    z    = (v_cc - mu) / std
    sig  = 1/(1+np.exp(-z))

    # Winsorized min-max (p5-p95 de A+B)
    v_w  = np.clip(v_cc, p5, p95)
    wmm  = (v_w - p5) / (p95 - p5 + 1e-9)

    # Min-max original dentro de C
    mm   = (v_cc - v_cc.min()) / (v_cc.max() - v_cc.min() + 1e-9)

    print(f"\n{col}:")
    print(f"  Ref A+B:  mu={mu:.3f}  std={std:.3f}  p5={p5:.3f}  p95={p95:.3f}")
    print(f"  Corpus C: mu={v_cc.mean():.3f}  std={v_cc.std():.3f}")
    print(f"  --- z-score+sigmoid (A+B ref) ---")
    print(f"    range [{sig.min():.3f}, {sig.max():.3f}]  zeros={100*(sig<0.01).mean():.0f}%  ones={100*(sig>0.99).mean():.0f}%")
    print(f"  --- winsorized min-max p5-p95 (A+B ref) ---")
    print(f"    range [{wmm.min():.3f}, {wmm.max():.3f}]  zeros={100*(wmm<0.01).mean():.0f}%  ones={100*(wmm>0.99).mean():.0f}%")
    print(f"  --- min-max dentro de C (actual) ---")
    print(f"    range [{mm.min():.3f}, {mm.max():.3f}]  zeros={100*(mm<0.01).mean():.0f}%  ones={100*(mm>0.99).mean():.0f}%")

# Por subcaso con z-score+sigmoid
print("\n=== DIS SCORE POR SUBCASO — Z-SCORE+SIGMOID (A+B ref) ===")
mu_sa  = ab['y2_sa'].mean();   std_sa  = ab['y2_sa'].std()+1e-9
mu_nv  = ab['y4_nv'].mean();   std_nv  = ab['y4_nv'].std()+1e-9
mu_rep = ab['y10_rep'].mean(); std_rep = ab['y10_rep'].std()+1e-9

def sig(x): return 1/(1+np.exp(-x))

for sub in cc['audio'].unique():
    s = cc[cc['audio']==sub]
    sa_n  = sig((s['y2_sa'].values   - mu_sa)  / std_sa).mean()
    nv_n  = sig((s['y4_nv'].values   - mu_nv)  / std_nv).mean()
    rep_n = sig((s['y10_rep'].values - mu_rep) / std_rep).mean()
    dis   = 0.35*sa_n + 0.35*nv_n + 0.30*(1-rep_n)
    print(f"  {sub:15} SA={sa_n:.3f} NV={nv_n:.3f} REP={rep_n:.3f} → DIS={dis:.3f}")
