# -*- coding: utf-8 -*-
"""
CFH — Ensamblar dataset SEM por compareciente (n=47)
====================================================
Consolida todos los indicadores calculados por compareciente en un unico CSV
maestro para el SEM multimodal de Corpus C.

Indicadores incluidos:
  xi1 (Violencia Discursiva):   y2_sa, y4_nv   (y3 excluido: saturado/no correla)
  eta1 (Injusticia Epistemica): y8_mafapo, y9_cidh
  eta2 (Transicion Epistemica): y10_rep, y11_conv_rest, y12_acustico
  (y1_ebi y y7_surprisal: no disponibles - requieren CFH-BERT/BETO, placeholders)

NOTA: dataset PROVISIONAL. Los extractores lexicos deben tunearse contra la
doble anotacion (IAA) antes de fijar el modelo definitivo.

Uso:
    python code/cfh_ensamblar_dataset_sem.py
"""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "data" / "referencias"
CAPA3 = REPO / "outputs" / "capa3"
OUT = REF / "indicadores_sem_compareciente.csv"

KEYS = ["subcaso", "identidad"]


def main():
    print("=" * 64)
    print("CFH — Ensamblar dataset SEM por compareciente (n=47)")
    print("=" * 64)

    # Base: identificadores + y10 + icm
    base = pd.read_csv(CAPA3 / "icm_tricanal_final.csv")
    base = base[KEYS + ["y10_rep", "icm_tricanal", "icm_facial", "icm_vocal",
                        "n_tokens", "robustez"]].copy()

    lex = pd.read_csv(REF / "indicadores_lexicos_compareciente.csv")[
        KEYS + ["y2_sa", "y3_civil", "y4_nv"]]
    y8y9 = pd.read_csv(REF / "y8_y9_compareciente.csv")[
        KEYS + ["y8_mafapo", "y9_cidh"]]
    y11 = pd.read_csv(REF / "y11_restaurativo.csv")[
        KEYS + ["y11_conv_rest"]]
    y12 = pd.read_csv(REF / "y12_acustico_compareciente.csv")[
        KEYS + ["y12_acustico"]]

    # Merge secuencial
    df = base.merge(lex, on=KEYS, how="left")
    df = df.merge(y8y9, on=KEYS, how="left")
    df = df.merge(y11, on=KEYS, how="left")
    df = df.merge(y12, on=KEYS, how="left")

    print(f"\n  Comparecientes: {len(df)}")

    # Verificar completitud de indicadores del SEM
    indicadores_sem = ["y2_sa", "y4_nv", "y8_mafapo", "y9_cidh",
                       "y10_rep", "y11_conv_rest", "y12_acustico"]
    print("\n  Completitud de indicadores del SEM:")
    for col in indicadores_sem:
        n_ok = df[col].notna().sum()
        marca = "OK" if n_ok == len(df) else f"** {len(df)-n_ok} NaN **"
        print(f"    {col:16s}: {n_ok}/{len(df)}  {marca}")

    # y3 se conserva pero marcado (excluido del modelo)
    print(f"\n  y3_civil (excluido de xi1): {df['y3_civil'].notna().sum()}/{len(df)} "
          f"[conservado para referencia, no entra al SEM]")

    # Orden de columnas
    cols = KEYS + indicadores_sem + ["y3_civil", "icm_tricanal",
                                     "icm_facial", "icm_vocal", "n_tokens", "robustez"]
    df = df[[c for c in cols if c in df.columns]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n  Guardado: {OUT}")

    print("\n  Estadisticas de indicadores SEM:")
    print(df[indicadores_sem].describe().round(4).to_string())

    print("\n  Matriz de correlaciones (indicadores SEM):")
    print(df[indicadores_sem].corr().round(2).to_string())

    print("=" * 64)


if __name__ == "__main__":
    main()
