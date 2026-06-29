"""
DIAGNÓSTICO DE IMPACTO: centroide v5 en DIS/IEI
================================================
Toma el recálculo de y8/y9 con el centroide v5 (generado por
cfh_verificar_y8y9_centroide_v5.py) y calcula qué DIS/IEI producirían,
comparándolos con los valores ACTUALES de la tesis.

NO modifica nada. Solo compara y reporta el impacto. Permite decidir
con datos si vale la pena migrar al centroide v5.

Requiere haber corrido antes:
    python cfh_verificar_y8y9_centroide_v5.py
    python cfh_unificar_corpus_c.py   (para tener el unificado actual)

Uso:
    conda activate cfh
    python cfh_diag_impacto_centroide_v5.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Rutas ──
RECALC_V5   = "data/indicators_corpus_c_y8y9_v5_VERIFICACION.csv"   # y8_v5, y9_v5 por bloque
UNIF_ACTUAL = "data/indicators_corpus_c_unificado.csv"              # base unificada actual (y2,y4,y10,y8,y9)
AB_ACTUAL   = "data/features/indicators_completo_conflibert.csv"    # Corpus A+B

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def zsig(serie, mu, sd):
    return sigmoid((serie - mu) / sd)

# ── PASO 1: Cargar datos ──
print("== DIAGNÓSTICO DE IMPACTO: centroide v5 en DIS/IEI ==\n")
for f in [RECALC_V5, UNIF_ACTUAL]:
    assert os.path.exists(f), f"Falta {f}. Corre primero los scripts previos."

recalc = pd.read_csv(RECALC_V5)       # tiene bloque_id, y8_v5, y9_v5
unif   = pd.read_csv(UNIF_ACTUAL)     # tiene bloque_id, y2, y4, y10, y8, y9 (actuales)

# Normalizar nombres de columnas y8/y9 actuales
col_y8_act = next((c for c in unif.columns if c.lower() in ("y8","y8_mafapo_cs","y8_mafapo")), None)
col_y9_act = next((c for c in unif.columns if c.lower() in ("y9","y9_cidh_cs","y9_cidh")), None)
col_y2 = next((c for c in unif.columns if c.lower().startswith("y2")), None)
col_y4 = next((c for c in unif.columns if c.lower().startswith("y4")), None)
col_y10= next((c for c in unif.columns if c.lower().startswith("y10")), None)
print(f"Columnas detectadas en unificado: y2={col_y2}, y4={col_y4}, y10={col_y10}, y8={col_y8_act}, y9={col_y9_act}")

# Unir el recálculo v5 con el unificado actual por bloque_id
m = unif.merge(recalc[["bloque_id","y8_v5","y9_v5"]], on="bloque_id", how="inner")
print(f"Bloques comparables: {len(m)}\n")

# ── PASO 2: Cargar A+B para construir la distribución conjunta (normalización) ──
ab = pd.read_csv(AB_ACTUAL)
# Detectar columnas equivalentes en A+B
def find_col(df, prefijos):
    for c in df.columns:
        for p in prefijos:
            if c.lower().startswith(p) or c.lower()==p:
                return c
    return None
ab_y2  = find_col(ab, ["y2","sa"])
ab_y4  = find_col(ab, ["y4","nv"])
ab_y10 = find_col(ab, ["y10","rep"])
ab_y8  = find_col(ab, ["y8","y8_mafapo","mafapo"])
ab_y9  = find_col(ab, ["y9","y9_cidh","cidh"])

def construir_dis_iei(y8_col_C, etiqueta):
    """Calcula DIS/IEI por subcaso usando y8_col_C como fuente de y8 en el Corpus C."""
    # Distribución conjunta A+B+C para cada indicador (z-score+sigmoid)
    # Para y2,y4,y10,y9: usar los actuales (no cambian). Para y8: usar la columna indicada.
    res = {}
    for ind, colC, colAB in [
        ("y2", col_y2, ab_y2), ("y4", col_y4, ab_y4),
        ("y10", col_y10, ab_y10), ("y9", col_y9_act, ab_y9),
    ]:
        vals = pd.concat([ab[colAB], m[colC]]) if colAB else m[colC]
        mu, sd = vals.mean(), vals.std()
        res[ind] = zsig(m[colC], mu, sd)
    # y8 con la fuente indicada
    vals8 = pd.concat([ab[ab_y8], m[y8_col_C]]) if ab_y8 else m[y8_col_C]
    mu8, sd8 = vals8.mean(), vals8.std()
    res["y8"] = zsig(m[y8_col_C], mu8, sd8)

    dis = 0.35*res["y2"] + 0.35*res["y4"] + 0.30*(1-res["y10"])
    iei = 0.35*res["y8"] + 0.20*res["y9"] + 0.25*res["y4"] + 0.20*(1-res["y10"])
    out = m[["audio"]].copy()
    out["DIS"], out["IEI"] = dis, iei
    return out.groupby("audio")[["DIS","IEI"]].mean().round(3)

# ── PASO 3: Comparar ACTUAL vs v5 ──
print("== DIS/IEI por subcaso ==\n")
actual = construir_dis_iei(col_y8_act, "actual")
nuevo  = construir_dis_iei("y8_v5", "v5")

comp = actual.join(nuevo, lsuffix="_ACTUAL", rsuffix="_v5")
comp["Δ_DIS"] = (comp["DIS_v5"] - comp["DIS_ACTUAL"]).round(3)
comp["Δ_IEI"] = (comp["IEI_v5"] - comp["IEI_ACTUAL"]).round(3)
comp["patron_ACT"] = np.where(comp["IEI_ACTUAL"]>comp["DIS_ACTUAL"], "IEI>DIS", "DIS>IEI")
comp["patron_v5"]  = np.where(comp["IEI_v5"]>comp["DIS_v5"], "IEI>DIS", "DIS>IEI")
print(comp.to_string())

# ── PASO 4: Veredicto sobre el hallazgo clave ──
print("\n== IMPACTO EN EL HALLAZGO CLAVE (Catatumbo IEI>DIS) ==")
for caso in ["catatumbo"]:
    if caso in comp.index:
        r = comp.loc[caso]
        print(f"  {caso}:")
        print(f"    ACTUAL: DIS={r['DIS_ACTUAL']} IEI={r['IEI_ACTUAL']} → {r['patron_ACT']}")
        print(f"    v5:     DIS={r['DIS_v5']} IEI={r['IEI_v5']} → {r['patron_v5']}")
        if r['patron_ACT'] == r['patron_v5']:
            print(f"    ✓ El patrón SE MANTIENE con v5")
        else:
            print(f"    ⚠ El patrón CAMBIA con v5 — afectaría el hallazgo central")

# ── PASO 5: ¿Cambian los rankings? ──
print("\n== RANKINGS ==")
print("  DIS actual:", list(actual.sort_values("DIS", ascending=False).index))
print("  DIS v5:    ", list(nuevo.sort_values("DIS", ascending=False).index))
print("  IEI actual:", list(actual.sort_values("IEI", ascending=False).index))
print("  IEI v5:    ", list(nuevo.sort_values("IEI", ascending=False).index))

print("\n== RESUMEN ==")
print(f"  Δ IEI promedio: {comp['Δ_IEI'].abs().mean():.4f}")
print(f"  Δ DIS promedio: {comp['Δ_DIS'].abs().mean():.4f}  (debería ser ~0: y8 no entra en DIS)")
print("\n  Decisión sugerida: si los patrones y rankings se mantienen y Δ es pequeño,")
print("  migrar a v5 es un refinamiento de bajo riesgo. Si algún patrón cambia,")
print("  hay que discutir la narrativa antes de migrar.")
