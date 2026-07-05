# -*- coding: utf-8 -*-
r"""
cfh_heterogeneidad_H3.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Explora si la relacion estructural H3 (injusticia -> transicion epistemica)
es HETEROGENEA por rol (maximo responsable) y por subcaso, antes de aceptar
el resultado nulo agregado (los 47).

CONTEXTO: el SEM sobre los 47 dio beta_23=-0.327 pero NO robusto (path
observado = -0.055). Hipotesis: la relacion existe pero SEGMENTADA; el
agregado da ~0 porque relaciones opuestas se cancelan.

DOS AJUSTES sobre el analisis anterior:
  1. INVERSION de y8/y9 -> injusticia = 1 - cercania. Asi eta1 mide
     INJUSTICIA EPISTEMICA (distancia al lenguaje de victimas), coherente
     con la etiqueta del Cap. 6. Signos esperados: H1 (violencia->injusticia)
     POSITIVO; H3 (injusticia->transicion) NEGATIVO.
  2. PARTICION por etiqueta_MR (27 MR vs 20 NO_MR) ademas de por subcaso.

Indices observados (sin latentes, que n=47 no soporta):
  viol      = media z (y1_ebi, y2_sa, y4_nv)              [violencia discursiva]
  injust    = media z (1-y8_mafapo, 1-y9_cidh)            [injusticia epistemica]
  transic   = media z (y10_rep, y11_conv_rest)            [transicion epistemica]

Uso (raiz del repo, env cfh):
    python code\cfh_heterogeneidad_H3.py

Entradas:
  data\referencias\indicadores_sem_compareciente.csv
  data\mr_asignacion_final.csv   (columnas: compareciente/identidad, etiqueta_MR, subcaso)

Salida: consola + cfh_heterogeneidad_H3_resultados.txt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_SEM = REPO / "data" / "referencias" / "indicadores_sem_compareciente.csv"
CSV_MR = REPO / "data" / "mr_asignacion_final.csv"
OUT = REPO / "cfh_heterogeneidad_H3_resultados.txt"

N_MIN = 10  # minimo para correlaciones interpretables


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def z(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 1e-9 else 1.0)


def corr_seg(sub, a, b):
    """Correlacion re-estandarizando dentro del subgrupo."""
    if len(sub) < 3:
        return np.nan
    return z(sub[a]).corr(z(sub[b]))


def bloque(nombre, sub, n_min=N_MIN):
    print("\n" + "-" * 66)
    print(f"{nombre}  (n={len(sub)})")
    print("-" * 66)
    # medias descriptivas siempre
    print("  Medias:  viol={:.3f}  injust={:.3f}  transic={:.3f}".format(
        sub["viol"].mean(), sub["injust"].mean(), sub["transic"].mean()))
    print("           y1_ebi={:.3f}  y10_rep={:.3f}  y11={:.3f}".format(
        sub["y1_ebi"].mean(), sub["y10_rep"].mean(), sub["y11_conv_rest"].mean()))
    if len(sub) < n_min:
        print(f"  [n<{n_min}] correlaciones NO reportadas (ruido).")
        return None
    h1 = corr_seg(sub, "viol", "injust")
    h3 = corr_seg(sub, "injust", "transic")
    vt = corr_seg(sub, "viol", "transic")
    ebirep = corr_seg(sub, "y1_ebi", "y10_rep")
    print(f"  H1  viol ~ injust   : {h1:+.3f}   (esperado + )")
    print(f"  H3  injust ~ transic: {h3:+.3f}   (esperado - )  <- CLAVE")
    print(f"      viol ~ transic  : {vt:+.3f}")
    print(f"      EBI ~ REP       : {ebirep:+.3f}")
    return h3


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    df = pd.read_csv(CSV_SEM)
    mr = pd.read_csv(CSV_MR)

    # Detectar columna de nombre en mr
    col_nom = None
    for c in ["compareciente", "identidad", "nombre"]:
        if c in mr.columns:
            col_nom = c
            break
    if col_nom is None:
        print("  [ERROR] no encuentro columna de nombre en mr_asignacion_final.csv")
        print(f"          columnas: {list(mr.columns)}")
        return

    # Merge por nombre
    mr_slim = mr[[col_nom, "etiqueta_MR"]].rename(columns={col_nom: "identidad"})
    df = df.merge(mr_slim, on="identidad", how="left")
    sin = df["etiqueta_MR"].isna().sum()

    print("=" * 66)
    print("CFH - Heterogeneidad de H3 (rol MR y subcaso) | y8/y9 INVERTIDOS")
    print("=" * 66)
    print(f"Comparecientes: {len(df)}  |  sin etiqueta_MR tras merge: {sin}")
    print("Distribucion etiqueta_MR:")
    print(df["etiqueta_MR"].value_counts(dropna=False).to_string())

    if sin > 0:
        print("\n  [aviso] algunos no emparejaron por nombre. Revisar grafias.")
        print("  No emparejados:", df[df["etiqueta_MR"].isna()]["identidad"].tolist())

    # --- Indices observados (INVIRTIENDO y8/y9) ---
    df["viol"] = (z(df["y1_ebi"]) + z(df["y2_sa"]) + z(df["y4_nv"])) / 3
    # injusticia = 1 - cercania (invertir antes de estandarizar)
    inj8 = 1.0 - df["y8_mafapo"]
    inj9 = 1.0 - df["y9_cidh"]
    df["injust"] = (z(inj8) + z(inj9)) / 2
    df["transic"] = (z(df["y10_rep"]) + z(df["y11_conv_rest"])) / 2

    # ===== AGREGADO =====
    print("\n" + "=" * 66)
    print("AGREGADO (los 47) - con y8/y9 invertidos")
    print("=" * 66)
    h3_all = corr_seg(df, "injust", "transic")
    print(f"  H1  viol ~ injust   : {corr_seg(df,'viol','injust'):+.3f}   (esperado +)")
    print(f"  H3  injust ~ transic: {h3_all:+.3f}   (esperado -)")

    # ===== POR ROL MR =====
    print("\n" + "=" * 66)
    print("PARTICION 1: POR ROL (Maximo Responsable)")
    print("=" * 66)
    h3_mr = bloque("MAXIMO RESPONSABLE (MR)", df[df["etiqueta_MR"] == "MR"])
    h3_nomr = bloque("NO MAXIMO RESPONSABLE (NO_MR)", df[df["etiqueta_MR"] == "NO_MR"])

    # ===== POR SUBCASO =====
    print("\n" + "=" * 66)
    print("PARTICION 2: POR SUBCASO")
    print("=" * 66)
    h3_sub = {}
    for sc in df["subcaso"].value_counts().index:
        h = bloque(f"SUBCASO {sc}", df[df["subcaso"] == sc])
        if h is not None:
            h3_sub[sc] = h

    # ===== SINTESIS =====
    print("\n" + "=" * 66)
    print("SINTESIS - H3 (injust -> transic) por grupo")
    print("=" * 66)
    print(f"  AGREGADO (47)     : {h3_all:+.3f}")
    if h3_mr is not None:
        print(f"  MR (n=27)         : {h3_mr:+.3f}")
    if h3_nomr is not None:
        print(f"  NO_MR (n=20)      : {h3_nomr:+.3f}")
    for sc, h in h3_sub.items():
        print(f"  {sc:<18}: {h:+.3f}")
    print("\n  LECTURA:")
    print("  - Si H3 tiene SIGNOS OPUESTOS entre MR y NO_MR (o entre subcasos),")
    print("    el agregado ~0 se explica por cancelacion -> HALLAZGO de")
    print("    heterogeneidad: la estructura del reconocimiento depende del rol.")
    print("  - Si H3 es ~0 en todos los grupos -> el nulo es genuino; el peso")
    print("    de la evidencia se traslada al contraste entre corpus A/B/C.")

    print(f"\n  Log -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
