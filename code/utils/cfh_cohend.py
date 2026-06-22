import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv(r'C:/PROYECTOS 2026/TESIS 2026/CFH_Hermeneutica_Forense_Computacional/outputs/nivel1_dis_iei_AB.csv')

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2)/(nx+ny-2))
    return abs(x.mean()-y.mean())/pooled

a = df[df['corpus_type'].isin(['A-CE','A-CSJ'])]
b = df[df['corpus_type']=='B']

print(f'N_A={len(a)}, N_B={len(b)}')
d_dis = cohen_d(a['DIS_n1'].dropna(), b['DIS_n1'].dropna())
d_iei = cohen_d(a['IEI_n1'].dropna(), b['IEI_n1'].dropna())
print(f'DIS: A={a["DIS_n1"].mean():.3f}, B={b["DIS_n1"].mean():.3f}, d={d_dis:.3f}')
print(f'IEI: A={a["IEI_n1"].mean():.3f}, B={b["IEI_n1"].mean():.3f}, d={d_iei:.3f}')
u,p = mannwhitneyu(a['DIS_n1'].dropna(), b['DIS_n1'].dropna())
print(f'DIS Mann-Whitney p={p:.4f}')
u,p = mannwhitneyu(a['IEI_n1'].dropna(), b['IEI_n1'].dropna())
print(f'IEI Mann-Whitney p={p:.4f}')
