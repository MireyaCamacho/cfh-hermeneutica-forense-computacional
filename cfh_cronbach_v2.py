"""
CFH — Cálculo α de Cronbach desde CSVs existentes
===================================================
Ejecutar:
  python cfh_cronbach_v2.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\outputs")

def cronbach_alpha(df):
    df = df.dropna()
    k = df.shape[1]
    if k < 2 or len(df) < 3:
        return None, len(df)
    var_items = df.var(axis=0, ddof=1).sum()
    var_total = df.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return None, len(df)
    alpha = (k / (k - 1)) * (1 - var_items / var_total)
    return round(float(alpha), 4), len(df)

def mostrar_cols(df, nombre):
    print(f"\n  Columnas en {nombre}:")
    for c in df.columns:
        print(f"    {c}: min={df[c].min():.3f} max={df[c].max():.3f} mean={df[c].mean():.3f}")

# ── Cargar archivos ───────────────────────────────────────────────────────
archivos = {
    "nivel1_dis_iei_AB":          BASE / "nivel1_dis_iei_AB.csv",
    "nivel2_dis_iei_corpus_c":    BASE / "nivel2_dis_iei_corpus_c_definitivo.csv",
    "parsimonia_dis_ab":          BASE / "parsimonia_dis_ab.csv",
}

dfs = {}
for nombre, ruta in archivos.items():
    try:
        df = pd.read_csv(ruta)
        dfs[nombre] = df
        print(f"✓ {nombre}: {df.shape[0]} filas × {df.shape[1]} cols")
        print(f"  Columnas: {list(df.columns)}")
    except Exception as e:
        print(f"✗ {nombre}: {e}")

print("\n" + "="*60)
print("CÁLCULO α DE CRONBACH")
print("="*60)

# Componentes teóricos
# DIS = 0.35×SA_norm + 0.35×NV_norm + 0.30×(1−REP_norm)
# IEI = 0.35×MAFAPO_norm + 0.20×CIDH_norm + 0.25×NV_norm + 0.20×(1−REP_norm)

ALIAS_DIS = ['sa', 'sa_norm', 'sa_score', 'y2', 'y2_norm',
             'nv', 'nv_norm', 'nv_score', 'y4', 'y4_norm',
             'rep', 'rep_norm', 'rep_score', 'y10', 'y10_norm']

ALIAS_IEI = ['mafapo', 'dist_mafapo', 'y8', 'y8_norm',
             'cidh', 'dist_cidh', 'y9', 'y9_norm',
             'nv', 'nv_norm', 'nv_score', 'y4', 'y4_norm',
             'rep', 'rep_norm', 'rep_score', 'y10', 'y10_norm']

for nombre, df in dfs.items():
    cols_lower = {c.lower(): c for c in df.columns}
    numeric_df = df.select_dtypes(include=[np.number])

    # Buscar componentes DIS
    dis_found = [cols_lower[a] for a in ALIAS_DIS if a in cols_lower]
    # Buscar componentes IEI
    iei_found = [cols_lower[a] for a in ALIAS_IEI if a in cols_lower]

    # Eliminar duplicados manteniendo orden
    dis_found = list(dict.fromkeys(dis_found))
    iei_found = list(dict.fromkeys(iei_found))

    print(f"\n[{nombre}]")

    if len(dis_found) >= 2:
        dis_data = numeric_df[dis_found].copy()
        # Invertir REP (polo opuesto)
        for c in dis_found:
            if any(x in c.lower() for x in ['rep', 'y10']):
                dis_data[c] = 1 - dis_data[c]
        alpha, n = cronbach_alpha(dis_data)
        print(f"  DIS α Cronbach = {alpha}  (n={n}, k={len(dis_found)} ítems: {dis_found})")
    else:
        print(f"  DIS — columnas insuficientes: {dis_found}")
        # Mostrar todas las columnas numéricas para diagnóstico
        mostrar_cols(numeric_df, nombre)

    if len(iei_found) >= 2:
        iei_data = numeric_df[iei_found].copy()
        # Invertir REP
        for c in iei_found:
            if any(x in c.lower() for x in ['rep', 'y10']):
                iei_data[c] = 1 - iei_data[c]
        alpha, n = cronbach_alpha(iei_data)
        print(f"  IEI α Cronbach = {alpha}  (n={n}, k={len(iei_found)} ítems: {iei_found})")
    else:
        print(f"  IEI — columnas insuficientes: {iei_found}")

print("\n[CFH] Cálculo completado.")
