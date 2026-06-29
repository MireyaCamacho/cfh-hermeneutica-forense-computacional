"""
MIGRACIÓN A CENTROIDE v5: regenera indicators_corpus_c.csv
============================================================
Reemplaza el y8 (dist. MAFAPO) del CSV canónico por el calculado con el
centroide MAFAPO v5 (293 textos). El y9 (CIDH) se mantiene: el centroide
CIDH v3 no cambió (diferencia verificada ~0.007, despreciable).

Hace backup del CSV original antes de sobrescribir. Trazable y reversible.

Requiere haber corrido antes:
    python cfh_verificar_y8y9_centroide_v5.py
    (que generó data/indicators_corpus_c_y8y9_v5_VERIFICACION.csv)

Uso:
    conda activate cfh
    python cfh_migrar_centroide_v5.py
"""

import os
import shutil
import pandas as pd
from datetime import datetime

CANONICO   = "data/features/indicators_corpus_c.csv"
RECALC_V5  = "data/indicators_corpus_c_y8y9_v5_VERIFICACION.csv"

print("== MIGRACIÓN A CENTROIDE v5: indicators_corpus_c.csv ==\n")

assert os.path.exists(CANONICO), f"Falta {CANONICO}"
assert os.path.exists(RECALC_V5), f"Falta {RECALC_V5}. Corre primero cfh_verificar_y8y9_centroide_v5.py"

# ── PASO 1: Backup del original ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = CANONICO.replace(".csv", f"_BACKUP_pre_v5_{ts}.csv")
shutil.copy2(CANONICO, backup)
print(f"✓ Backup creado: {backup}\n")

# ── PASO 2: Cargar ambos ──
canon = pd.read_csv(CANONICO)
recalc = pd.read_csv(RECALC_V5)
print(f"Canónico: {len(canon)} filas")
print(f"Recálculo v5: {len(recalc)} filas")

# Detectar columna y8 en el canónico
col_y8 = next((c for c in canon.columns if c.lower() in ("y8_mafapo_cs","y8","y8_mafapo")), None)
assert col_y8, "No se encontró columna y8 en el canónico"
print(f"Columna y8 a reemplazar: {col_y8}\n")

# ── PASO 3: Reemplazar y8 por el v5, casando por bloque_id ──
m = canon.merge(recalc[["bloque_id","y8_v5"]], on="bloque_id", how="left", validate="one_to_one")
n_sin_match = m["y8_v5"].isna().sum()
if n_sin_match > 0:
    print(f"⚠ {n_sin_match} bloques sin match en el recálculo v5 — conservan su y8 anterior")

# y8 antiguo para el reporte
y8_old_mean = m[col_y8].mean()
# Sustituir donde haya v5
m[col_y8] = m["y8_v5"].fillna(m[col_y8])
y8_new_mean = m[col_y8].mean()
m = m.drop(columns=["y8_v5"])

print(f"y8 media: {y8_old_mean:.4f} (anterior) → {y8_new_mean:.4f} (v5)")
print(f"  Δ = {y8_new_mean - y8_old_mean:+.4f}\n")

# ── PASO 4: Guardar (sobrescribe el canónico, con backup ya hecho) ──
m.to_csv(CANONICO, index=False, encoding="utf-8-sig")
print(f"✓ {CANONICO} regenerado con y8 del centroide v5")
print(f"  (backup del original en {backup})")

# ── PASO 5: Verificación por subcaso ──
print("\n== y8 por subcaso (v5) ==")
if "audio" in m.columns:
    print(m.groupby("audio")[col_y8].mean().round(4).to_string())
print("\nSiguiente paso: correr cfh_unificar_corpus_c.py para recalcular DIS/IEI con el y8 nuevo.")
