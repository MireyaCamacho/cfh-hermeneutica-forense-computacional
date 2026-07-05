# -*- coding: utf-8 -*-
r"""
cfh_dis_iei_paso2.py  (v2 - merge por posicion)
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

PASO 2 - DIS/IEI tri-corpus con indicadores ajustados y normalizacion conjunta.

CORRECCION: el merge de A por doc_id+section_id explotaba a 95k filas porque
esa clave NO es unica (819 filas -> 608 claves). Los tres archivos (base,
y1_recalc, y10_recalc) tienen 873 filas en EL MISMO ORDEN (verificado), asi
que se alinea y1/y10 v5 POR POSICION, sin merge. A queda en 819 exactas.

Ensamblaje:
  A (819): base[y2,y4,y8_cs,y9_cs] + y1/y10_v5 por posicion
  B ( 80): corpus_b_indicadores_COMPLETO.csv
  C ( 47): CSV del SEM (y8_mafapo/y9_cidh -> _cs)

Normalizacion: z-score + sigmoide sobre A+B+C conjunta.

FORMULAS:
  DIS   = 0.40*EBI_z + 0.30*SA_z + 0.30*(1-REP_z)
  IEI_A = 0.40*MAF_z + 0.30*CIDH_z + 0.30*NV_z              (sin REP)
  IEI_B = 0.30*MAF_z + 0.25*CIDH_z + 0.25*NV_z + 0.20*(1-REP_z)  (con REP)

Uso:
    python code\cfh_dis_iei_paso2.py
"""

import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_BASE = os.path.join(REPO, "data", "features", "indicators_completo_conflibert.csv")
F_Y1 = os.path.join(REPO, "outputs", "y1_ebi_AB_recalculado.csv")
F_Y10 = os.path.join(REPO, "outputs", "y10_rep_v5_AB_recalculado.csv")
F_B = os.path.join(REPO, "outputs", "corpus_b_indicadores_COMPLETO.csv")
F_C = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")

OUT_UNID = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
OUT_RES = os.path.join(REPO, "outputs", "dis_iei_resumen_por_corpus.csv")
OUT_COR = os.path.join(REPO, "outputs", "dis_iei_correlaciones.csv")

IND = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    print("=" * 66)
    print("PASO 2 (v2) - DIS/IEI tri-corpus | merge A por posicion")
    print("=" * 66)

    base = pd.read_csv(F_BASE).reset_index(drop=True)
    y1 = pd.read_csv(F_Y1).reset_index(drop=True)
    y10 = pd.read_csv(F_Y10).reset_index(drop=True)

    assert (base["doc_id"].values == y1["doc_id"].values).all(), "y1 desalineado"
    assert (base["doc_id"].values == y10["doc_id"].values).all(), "y10 desalineado"

    base["y1_ebi"] = y1["y1_ebi"].values
    base["y10_rep_v5"] = y10["y10_rep_v5"].values

    a = base[base["corpus_type"].str.startswith("A")].copy()
    a_out = pd.DataFrame({
        "unidad": a["doc_id"].str[:12] + "_" + a["section_id"].astype(str)
                  + "_" + a.groupby(["doc_id", "section_id"]).cumcount().astype(str),
        "corpus": "A",
        "y1_ebi": a["y1_ebi"], "y2_sa": a["y2_sa"], "y4_nv": a["y4_nv"],
        "y8_mafapo_cs": a["y8_mafapo_cs"], "y9_cidh_cs": a["y9_cidh_cs"],
        "y10_rep": a["y10_rep_v5"],
    })
    print(f"\nA: {len(a_out)} secciones  (esperado 819)")

    b = pd.read_csv(F_B)
    b_out = pd.DataFrame({
        "unidad": b["doc"].astype(str) + "_" + b["seccion"].astype(str)
                  + "_" + b.groupby(["doc", "seccion"]).cumcount().astype(str),
        "corpus": "B",
        "y1_ebi": b["y1_ebi"], "y2_sa": b["y2_sa"], "y4_nv": b["y4_nv"],
        "y8_mafapo_cs": b["y8_mafapo_cs"], "y9_cidh_cs": b["y9_cidh_cs"],
        "y10_rep": b["y10_rep_v5"],
    })
    print(f"B: {len(b_out)} secciones  (esperado 80)")

    c = pd.read_csv(F_C)
    c_out = pd.DataFrame({
        "unidad": c["identidad"].astype(str),
        "corpus": "C",
        "y1_ebi": c["y1_ebi"], "y2_sa": c["y2_sa"], "y4_nv": c["y4_nv"],
        "y8_mafapo_cs": c["y8_mafapo"], "y9_cidh_cs": c["y9_cidh"],
        "y10_rep": c["y10_rep"],
    })
    c_icm = c[["identidad", "icm_tricanal"]].copy()
    print(f"C: {len(c_out)} comparecientes  (esperado 47)")

    df = pd.concat([a_out, b_out, c_out], ignore_index=True)
    print(f"TOTAL A+B+C: {len(df)}  ({df['corpus'].value_counts().to_dict()})")

    print("\nNormalizando (z-score+sigmoide sobre A+B+C conjunto)...")
    for col in IND:
        v = df[col].astype(float)
        mu, sd = v.mean(), v.std() + 1e-9
        df[col + "_z"] = sigmoid((v - mu) / sd)

    df["DIS"] = 0.40 * df["y1_ebi_z"] + 0.30 * df["y2_sa_z"] + 0.30 * (1 - df["y10_rep_z"])
    df["IEI_A"] = 0.40 * df["y8_mafapo_cs_z"] + 0.30 * df["y9_cidh_cs_z"] + 0.30 * df["y4_nv_z"]
    df["IEI_B"] = (0.30 * df["y8_mafapo_cs_z"] + 0.25 * df["y9_cidh_cs_z"]
                   + 0.25 * df["y4_nv_z"] + 0.20 * (1 - df["y10_rep_z"]))

    print("\n" + "=" * 66)
    print("DIS / IEI por corpus  (mayor = mas injusticia)")
    print("=" * 66)
    res = df.groupby("corpus")[["DIS", "IEI_A", "IEI_B"]].agg(["mean", "std"]).round(4)
    print(res.to_string())
    print("\n  Orden por indice:")
    for idx in ["DIS", "IEI_A", "IEI_B"]:
        m = df.groupby("corpus")[idx].mean()
        print(f"    {idx:<6}: " + " < ".join(f"{k}={v:.3f}" for k, v in m.sort_values().items()))

    print("\n" + "=" * 66)
    print("CORRELACIONES entre indices (Spearman)")
    print("=" * 66)
    cor = df[["DIS", "IEI_A", "IEI_B"]].corr(method="spearman").round(3)
    print("  DIS vs IEI (A+B+C):")
    print(cor.to_string())

    dfc = df[df["corpus"] == "C"].merge(c_icm, left_on="unidad",
                                        right_on="identidad", how="left")
    cor_icm = dfc[["DIS", "IEI_A", "IEI_B", "icm_tricanal"]].corr(method="spearman").round(3)
    print("\n  con ICM (solo C, n=47):")
    print(cor_icm.to_string())
    print("\n  >> correlaciones bajas = dimensiones distintas (objetivo del framework)")

    df.to_csv(OUT_UNID, index=False, encoding="utf-8")
    res.to_csv(OUT_RES, encoding="utf-8-sig")
    cor.to_csv(OUT_COR, encoding="utf-8-sig")
    print(f"\n  Por unidad    -> {OUT_UNID}")
    print(f"  Resumen       -> {OUT_RES}")
    print(f"  Correlaciones -> {OUT_COR}")


if __name__ == "__main__":
    main()
