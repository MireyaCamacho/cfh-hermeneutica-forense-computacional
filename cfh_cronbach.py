"""
CFH — Cálculo α de Cronbach para DIS e IEI
============================================
Ejecutar:
  python cfh_cronbach.py
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\data\cfh.db")

def cronbach_alpha(df):
    """Calcula α de Cronbach para un DataFrame de ítems (columnas = ítems)."""
    df = df.dropna()
    k = df.shape[1]
    if k < 2:
        return None, 0
    var_items = df.var(axis=0, ddof=1).sum()
    var_total = df.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return None, len(df)
    alpha = (k / (k - 1)) * (1 - var_items / var_total)
    return round(float(alpha), 4), len(df)

conn = sqlite3.connect(DB)

# Ver tablas disponibles
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tablas = [r[0] for r in cur.fetchall()]
print(f"Tablas en cfh.db: {tablas}\n")

# Buscar columnas relevantes
for tabla in tablas:
    cur.execute(f"PRAGMA table_info({tabla})")
    cols = [r[1] for r in cur.fetchall()]
    relevantes = [c for c in cols if any(x in c.lower() for x in
                  ['sa', 'nv', 'rep', 'ebi', 'dis', 'iei', 'y2', 'y4', 'y8', 'y9', 'y10',
                   'mafapo', 'cidh', 'score', 'dist'])]
    if relevantes:
        print(f"Tabla '{tabla}' — columnas relevantes: {relevantes}")

print()

# Intentar calcular con las columnas que existan
for tabla in tablas:
    try:
        df = pd.read_sql(f"SELECT * FROM {tabla} LIMIT 5000", conn)
        cols = df.columns.tolist()

        # Componentes DIS: SA(y2), NV(y4), REP(y10) invertido
        dis_cols = [c for c in cols if any(x in c.lower() for x in
                    ['sa_score','sa_norm','y2','nv_score','nv_norm','y4','rep_score','rep_norm','y10'])]

        # Componentes IEI: MAFAPO(y8), CIDH(y9), NV(y4), REP(y10) invertido
        iei_cols = [c for c in cols if any(x in c.lower() for x in
                    ['mafapo','y8','cidh','y9','nv_score','nv_norm','y4','rep_score','rep_norm','y10'])]

        if len(dis_cols) >= 2:
            print(f"\n[{tabla}] DIS — columnas encontradas: {dis_cols}")
            dis_data = df[dis_cols].apply(pd.to_numeric, errors='coerce')
            alpha_dis, n = cronbach_alpha(dis_data)
            print(f"  α Cronbach DIS = {alpha_dis}  (n={n})")

        if len(iei_cols) >= 2:
            print(f"\n[{tabla}] IEI — columnas encontradas: {iei_cols}")
            iei_data = df[iei_cols].apply(pd.to_numeric, errors='coerce')
            alpha_iei, n = cronbach_alpha(iei_data)
            print(f"  α Cronbach IEI = {alpha_iei}  (n={n})")

    except Exception as e:
        print(f"  ✗ {tabla}: {e}")

conn.close()
print("\n[CFH] Cálculo completado.")
