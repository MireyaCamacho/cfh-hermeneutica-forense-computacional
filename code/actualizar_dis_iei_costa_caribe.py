# -*- coding: utf-8 -*-
"""
CFH — actualizar_dis_iei_costa_caribe.py
Reemplaza las 120 filas de Costa Caribe por las 129 del TXT completo
y recalcula DIS/IEI con normalizacion dentro del Corpus C.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(".")

CAPA1_PATH  = BASE / "data/indicators_corpus_c_capa1.csv"
CC_V2_PATH  = BASE / "data/indicators_costa_caribe_v2.csv"
Y89_PATH    = BASE / "indicators_corpus_c.csv"
OUT_CAPA1   = BASE / "data/indicators_corpus_c_capa1_v2.csv"
OUT_DIS_IEI = BASE / "data/dis_iei_corpus_c_v2.csv"

print("=" * 60)
print("CFH - actualizar_dis_iei_costa_caribe.py")
print("=" * 60)

df_capa1 = pd.read_csv(CAPA1_PATH, encoding="utf-8")
df_cc_v2 = pd.read_csv(CC_V2_PATH, encoding="utf-8")

print(f"Capa1 original: {len(df_capa1)} filas")
print(f"  Costa Caribe original: {(df_capa1['audio']=='costa_caribe').sum()} bloques")
print(f"Costa Caribe v2: {len(df_cc_v2)} filas")
print(f"  Columnas CC v2: {df_cc_v2.columns.tolist()}")

# Construir filas CC v2 compatibles con capa1
cols_capa1 = df_capa1.columns.tolist()
df_cc_compat = pd.DataFrame(np.nan, index=range(len(df_cc_v2)), columns=cols_capa1)
df_cc_compat["audio"] = "costa_caribe"

# Mapear columnas disponibles
mapeo = {
    "bloque_id": "bloque_id",
    "y2_sa":     "y2_sa",
    "y3_civil":  "y3_civil",
    "y4_nv":     "y4_nv",
    "y10_rep":   "y10_rep",
    "y8_mafapo_cs": "y8_mafapo_cs",
    "y9_cidh_cs":   "y9_cidh_cs",
}
for col_v2, col_c1 in mapeo.items():
    if col_v2 in df_cc_v2.columns and col_c1 in cols_capa1:
        df_cc_compat[col_c1] = df_cc_v2[col_v2].values

# Reemplazar en capa1
df_sin_cc   = df_capa1[df_capa1["audio"] != "costa_caribe"].copy()
df_capa1_v2 = pd.concat([df_sin_cc, df_cc_compat], ignore_index=True)
df_capa1_v2.to_csv(OUT_CAPA1, index=False, encoding="utf-8")

print(f"\nCapa1 actualizado: {len(df_capa1_v2)} filas")
print(f"  Distribucion: {df_capa1_v2['audio'].value_counts().to_dict()}")
print(f"Guardado: {OUT_CAPA1}")

# Estadisticas por subcaso
c1 = df_capa1_v2.groupby("audio").agg(
    y2_sa   = ("y2_sa",  "mean"),
    y4_nv   = ("y4_nv",  "mean"),
    y10_rep = ("y10_rep","mean"),
    n       = ("bloque_id","count"),
).reset_index()

# y8/y9: usar columnas del capa1_v2 si estan disponibles
y8_col = "y8_mafapo_cs" if "y8_mafapo_cs" in df_capa1_v2.columns else None
y9_col = "y9_cidh_cs"   if "y9_cidh_cs"   in df_capa1_v2.columns else None

if y8_col and y9_col:
    y89 = df_capa1_v2.groupby("audio").agg(
        y8_mafapo=(y8_col,"mean"),
        y9_cidh  =(y9_col,"mean"),
    ).reset_index()
elif Y89_PATH.exists():
    df_y89 = pd.read_csv(Y89_PATH, encoding="utf-8")
    NM = {"casanare_torres":"casanare","dabeiba_antioquia":"dabeiba",
          "costa_caribe":"costa_caribe","huila":"huila","catatumbo":"catatumbo"}
    df_y89["audio_norm"] = df_y89["audio"].map(NM)
    y89 = df_y89.groupby("audio_norm").agg(
        y8_mafapo=("y8_mafapo_cs","mean"),
        y9_cidh  =("y9_cidh_cs","mean"),
    ).reset_index().rename(columns={"audio_norm":"audio"})
else:
    df_orig = pd.read_csv(BASE/"data/dis_iei_corpus_c.csv")
    y89 = df_orig[["audio","y8_mafapo","y9_cidh"]].copy()

df = c1.merge(y89, on="audio", how="left")

# Normalizacion dentro del Corpus C (igual que calcular_iei.py)
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

df["y2_norm"]  = norm(df["y2_sa"])
df["y4_norm"]  = norm(df["y4_nv"])
df["y10_norm"] = norm(df["y10_rep"])
df["y8_norm"]  = norm(df["y8_mafapo"])
df["y9_norm"]  = norm(df["y9_cidh"])

df["DIS_score"] = 0.35*df["y2_norm"] + 0.35*df["y4_norm"] + 0.30*(1-df["y10_norm"])
df["IEI_score"] = 0.35*df["y8_norm"] + 0.20*df["y9_norm"] + 0.25*df["y4_norm"] + 0.20*(1-df["y10_norm"])
df["disociacion"] = (df["DIS_score"] - df["IEI_score"]).abs()
df["tipo_disociacion"] = df.apply(lambda r: (
    "DIS>IEI" if r["DIS_score"] > r["IEI_score"] + 0.08
    else "IEI>DIS" if r["IEI_score"] > r["DIS_score"] + 0.08
    else "Coherente"
), axis=1)

ORDER = ["casanare","catatumbo","dabeiba","huila","costa_caribe"]
df = df.set_index("audio").reindex(ORDER).reset_index()

# Comparar v1 vs v2
df_orig = pd.read_csv(BASE/"data/dis_iei_corpus_c.csv")
df_orig = df_orig.set_index("audio").reindex(ORDER).reset_index()

print(f"\n{'='*70}")
print("COMPARACION v1 (120 bloques) vs v2 (129 bloques):")
print(f"{'Subcaso':15} {'DIS_v1':8} {'DIS_v2':8} {'dDIS':7} {'IEI_v1':8} {'IEI_v2':8} {'dIEI':7}")
print("-"*70)
for sc in ORDER:
    r2 = df[df["audio"]==sc].iloc[0]
    r1 = df_orig[df_orig["audio"]==sc].iloc[0]
    dd = r2["DIS_score"] - r1["DIS_score"]
    di = r2["IEI_score"] - r1["IEI_score"]
    marca = " <-- NUEVO" if sc == "costa_caribe" else ""
    print(f"{sc:15} {r1['DIS_score']:8.3f} {r2['DIS_score']:8.3f} {dd:+7.3f} "
          f"{r1['IEI_score']:8.3f} {r2['IEI_score']:8.3f} {di:+7.3f}{marca}")

# Guardar
df_out = df[["audio","n","y2_sa","y4_nv","y10_rep","y8_mafapo","y9_cidh",
             "DIS_score","IEI_score","disociacion","tipo_disociacion"]]
df_out.to_csv(OUT_DIS_IEI, index=False, encoding="utf-8")

print(f"\nTabla 5.14 v2:")
print(f"{'Subcaso':15} {'DIS':7} {'IEI':7} {'Delta':7} {'Dir.'}")
print("-"*50)
for _, r in df.iterrows():
    print(f"{r['audio']:15} {r['DIS_score']:7.3f} {r['IEI_score']:7.3f} "
          f"{r['disociacion']:7.3f}  {r['tipo_disociacion']}")

print(f"\nGuardado: {OUT_DIS_IEI}")
