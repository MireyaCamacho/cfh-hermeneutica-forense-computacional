# -*- coding: utf-8 -*-
"""
cfh_sem_completo.py
===================
Estima el modelo SEM CFH LATENTE COMPLETO, resolviendo los dos problemas
que hacian que el modelo diera CFI negativo y beta absurdo:

  PROBLEMA 1: y8/y9 estaban vacios en indicators_corpus_a/b.csv.
              -> Se traen desde indicators_completo_conflibert_v3b.csv
                 por merge (doc_id, section_id). Cobertura verificada 100%.

  PROBLEMA 2: los indicadores estaban en escalas incompatibles
              (distancias ~0.2, anios ~2013, surprisal ~-5) -> covarianzas
              mal condicionadas -> CFI<0, beta explotado.
              -> Se ESTANDARIZAN (z-score) todos los indicadores antes de estimar.

MODELO (latente completo, el que usa y7):
    xi1  =~ y2 + y3 + y4        (Violencia Discursiva)
    xi2  =~ y5 + y6             (Contexto Institucional)
    eta1 =~ y7 + y8 + y9        (DIS Score)   <- y7 recien calculado + y8/y9 traidos
    eta2 =~ y10                 (Transicion Epistemica; y11 vacio, se omite)
    eta1 ~ xi1 + xi2
    eta2 ~ eta1                 (H3: beta_23)
    xi1 ~~ xi2

NOTA: y1 (EBI) se omite (placeholder 0.0, no operativo segun guia).
      y11 (conv_rest) esta vacio -> eta2 se identifica con y10.
      y8/y9 son v3b (centroide anterior). Para ESTE test de convergencia sirve;
      si converge, se re-corre con v5 cuando y8/y9 v5 esten en A/B.

Uso:
    python cfh_sem_completo.py
    python cfh_sem_completo.py --no-standardize   # para comparar (dara mal)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEM_A = "data/features/indicators_corpus_a.csv"
SEM_B = "data/features/indicators_corpus_b.csv"
FUENTE_Y8Y9 = "data/features/canonico/indicators_completo_conflibert_v3b.csv"

# Mapeo columnas CSV -> nombres del modelo
COLMAP = {
    "y2_sa": "y2",
    "y3_civil": "y3",
    "y4_nv": "y4",
    "y5_corpus_type": "y5",
    "y6_period": "y6",
    "y7_surprisal": "y7",
    "y8_mafapo": "y8",
    "y9_cidh": "y9",
    "y10_rep": "y10",
}

MODELO_SPEC = """
    xi1 =~ y2 + y3 + y4
    xi2 =~ y5 + y6
    eta1 =~ y7 + y8 + y9
    eta2 =~ y10
    eta1 ~ xi1 + xi2
    eta2 ~ eta1
    xi1 ~~ xi2
"""

INDICADORES = ["y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]


def cargar_y_unir():
    """Carga A+B del SEM y trae y8/y9 desde v3b por merge."""
    print("=" * 70)
    print("1. CARGA Y MERGE DE y8/y9")
    print("=" * 70)

    dfs = []
    for nombre, path in [("A", SEM_A), ("B", SEM_B)]:
        d = pd.read_csv(path)
        d["_corpus"] = nombre
        dfs.append(d)
        print(f"  {path}: {len(d)} filas")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total: {len(df)} filas")

    # traer y8/y9 desde v3b
    src = pd.read_csv(FUENTE_Y8Y9)[["doc_id", "section_id", "y8_mafapo", "y9_cidh"]]
    src = src.drop_duplicates(subset=["doc_id", "section_id"])
    print(f"\n  fuente y8/y9 (v3b): {len(src)} llaves unicas")

    # merge: sobreescribe las columnas vacias y8/y9 del SEM
    df = df.drop(columns=[c for c in ["y8_mafapo", "y9_cidh"] if c in df.columns])
    df = df.merge(src, on=["doc_id", "section_id"], how="left")

    print(f"  tras merge -> y8_mafapo no-NaN: {df['y8_mafapo'].notna().sum()} / {len(df)}")
    print(f"               y9_cidh   no-NaN: {df['y9_cidh'].notna().sum()} / {len(df)}")
    return df


def preparar(df, standardize=True):
    """Renombra, selecciona indicadores, elimina NaN y estandariza."""
    print("\n" + "=" * 70)
    print("2. PREPARACION" + ("  (con z-score)" if standardize else "  (SIN estandarizar)"))
    print("=" * 70)

    d = df.rename(columns=COLMAP)
    faltan = [c for c in INDICADORES if c not in d.columns]
    if faltan:
        print(f"  [AVISO] columnas ausentes, se crean NaN: {faltan}")
        for c in faltan:
            d[c] = np.nan

    d = d[INDICADORES].copy()

    # eliminar filas con NaN en cualquier indicador del modelo
    antes = len(d)
    d = d.dropna(subset=INDICADORES)
    print(f"  filas: {antes} -> {len(d)} (eliminadas {antes-len(d)} por NaN)")

    # reporte de escalas ANTES de estandarizar
    print("\n  escalas originales (min | media | max):")
    for c in INDICADORES:
        print(f"    {c:5s}: {d[c].min():8.3f} | {d[c].mean():8.3f} | {d[c].max():8.3f}")

    if standardize:
        # z-score: (x - mean) / std, columna por columna
        for c in INDICADORES:
            sd = d[c].std()
            if sd > 1e-9:
                d[c] = (d[c] - d[c].mean()) / sd
            else:
                print(f"    [AVISO] {c} tiene varianza ~0 (constante), se deja igual")
        print("\n  -> indicadores estandarizados (media~0, sd~1)")

    return d


def estimar(d):
    """Estima el SEM con semopy y reporta ajuste + H3."""
    print("\n" + "=" * 70)
    print("3. ESTIMACION DEL SEM LATENTE COMPLETO")
    print("=" * 70)

    import semopy
    from semopy import Model

    model = Model(MODELO_SPEC)
    model.fit(d, obj="MLW")

    # indices de ajuste
    stats = semopy.calc_stats(model)

    def _get(name):
        try:
            v = stats[name]
            return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
        except Exception:
            return float("nan")

    cfi = _get("CFI")
    rmsea = _get("RMSEA")
    tli = _get("TLI")
    chi2 = _get("chi2")

    print(f"\n  N = {len(d)}")
    print(f"  CFI:   {cfi:7.3f}   {'OK (>=0.90)' if cfi>=0.90 else 'bajo'}")
    print(f"  TLI:   {tli:7.3f}")
    print(f"  RMSEA: {rmsea:7.3f}   {'OK (<=0.08)' if rmsea<=0.08 else 'alto'}")
    print(f"  chi2:  {chi2:7.3f}")

    # parametros
    ins = model.inspect(std_est=True)
    # H3: eta2 ~ eta1
    print("\n" + "-" * 70)
    print("  HIPOTESIS H3:  eta2 ~ eta1  (beta_23)")
    print("-" * 70)
    fila = ins[(ins["lval"] == "eta2") & (ins["op"] == "~") & (ins["rval"] == "eta1")]
    if len(fila):
        est = float(fila["Estimate"].iloc[0])
        est_std = float(fila["Est. Std"].iloc[0]) if "Est. Std" in fila.columns else float("nan")
        se = float(fila["Std. Err"].iloc[0]) if "Std. Err" in fila.columns else float("nan")
        pval = float(fila["p-value"].iloc[0]) if "p-value" in fila.columns else float("nan")
        print(f"  beta_23 (crudo):        {est:8.3f}")
        print(f"  beta_23 (estandarizado):{est_std:8.3f}")
        print(f"  SE = {se:.3f}   p = {pval:.4f}")
        soportada = (est < 0) and (pval < 0.01)
        print(f"  H3 (beta_23 < 0, p<0.01): {'APOYADA' if soportada else 'NO apoyada'}")
    else:
        print("  [no se encontro el path eta2~eta1 en la salida]")

    # cargas factoriales
    print("\n" + "-" * 70)
    print("  CARGAS FACTORIALES (std):")
    print("-" * 70)
    cargas = ins[ins["op"] == "=~"]
    for _, r in cargas.iterrows():
        est_std = r.get("Est. Std", r.get("Estimate", np.nan))
        try:
            est_std = float(est_std)
        except Exception:
            est_std = float("nan")
        flag = "" if abs(est_std) >= 0.40 else "  <- debil (<0.40)"
        print(f"    {r['lval']:5s} =~ {r['rval']:5s}: {est_std:7.3f}{flag}")

    print("\n" + "=" * 70)
    return cfi, rmsea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-standardize", action="store_true",
                    help="NO estandarizar (para comparar; dara mal ajuste)")
    args = ap.parse_args()

    df = cargar_y_unir()
    d = preparar(df, standardize=not args.no_standardize)

    try:
        cfi, rmsea = estimar(d)
    except Exception as e:
        print(f"\n[ERROR en estimacion] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nLECTURA:")
    if cfi >= 0.90 and rmsea <= 0.08:
        print("  El modelo latente CONVERGE con buen ajuste. SEM recuperado.")
    elif cfi >= 0 and cfi < 0.90:
        print("  Converge pero ajuste mediocre. El SEM corre y y7 ya esta dentro,")
        print("  pero no alcanza umbrales de publicacion. Reportable junto al path analysis.")
    else:
        print("  Ajuste aun problematico. Revisar especificacion (quiza eta2 con un solo")
        print("  indicador y10 no se identifica bien, o falta y11).")
    print("=" * 70)


if __name__ == "__main__":
    main()
