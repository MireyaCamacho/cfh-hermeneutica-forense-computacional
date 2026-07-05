# -*- coding: utf-8 -*-
"""
cfh_sem_eta2_completo.py
=========================
Re-estima el SEM CFH con eta2 (Transicion Epistemica) COMPLETO, usando los
indicadores y11/y12/y13 que ya estan calculados en data/indicators_y11_y12_y13.csv.

Resuelve la sub-identificacion de eta2: antes solo tenia y10 (un indicador).
Ahora tiene y10 + y11 + y12 + y13 -> identificable.

PROBLEMA DE LLAVES (verificado):
  - y11/y12/y13 estan por bloque (doc_id_corto + _bNNN, bloques de 2000 chars)
  - el SEM esta por doc_id_largo (64 chars) + section_id semantico
  - el doc_id corto (16 chars) es PREFIJO del largo
  SOLUCION: se agregan y11/y12/y13 promediando los bloques por doc_id_corto,
  y se unen al SEM mapeando doc_id_largo -> sus primeros 16 chars.

INDICADORES eta2 (del CSV de bloques):
  y10 = y10_rep (ya en el SEM)
  y11 = y11_quotes        (convergencia restaurativa / citas de voz de victima)
  y12 = y12_judgment      (juicio)
  y13 = y13_evidential    (evidencialidad)

MODELO:
    xi1  =~ y2 + y3 + y4
    xi2  =~ y5 + y6
    eta1 =~ y7 + y8 + y9
    eta2 =~ y10 + y11 + y12 + y13     <- ahora identificable
    eta1 ~ xi1 + xi2
    eta2 ~ eta1                        (H3: beta_23)
    xi1 ~~ xi2

Todos los indicadores se estandarizan (z-score) antes de estimar.

Uso:
    python cfh_sem_eta2_completo.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEM_A = "data/features/indicators_corpus_a.csv"
SEM_B = "data/features/indicators_corpus_b.csv"
Y8Y9 = "data/features/canonico/indicators_completo_conflibert_v3b.csv"
Y11 = "data/indicators_y11_y12_y13.csv"

COLMAP = {
    "y2_sa": "y2", "y3_civil": "y3", "y4_nv": "y4",
    "y5_corpus_type": "y5", "y6_period": "y6",
    "y7_surprisal": "y7", "y8_mafapo": "y8", "y9_cidh": "y9",
    "y10_rep": "y10",
}

MODELO_SPEC = """
    xi1 =~ y2 + y3 + y4
    xi2 =~ y5 + y6
    eta1 =~ y7 + y8 + y9
    eta2 =~ y10 + y11 + y12 + y13
    eta1 ~ xi1 + xi2
    eta2 ~ eta1
    xi1 ~~ xi2
"""

INDICADORES = ["y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9",
               "y10", "y11", "y12", "y13"]


def cargar_sem():
    print("=" * 70)
    print("1. CARGA SEM A+B + merge y8/y9")
    print("=" * 70)
    dfs = []
    for nombre, path in [("A", SEM_A), ("B", SEM_B)]:
        d = pd.read_csv(path)
        d["_corpus"] = nombre
        dfs.append(d)
        print(f"  {path}: {len(d)} filas")
    df = pd.concat(dfs, ignore_index=True)

    # traer y8/y9 (v3b) por doc_id+section_id
    src = pd.read_csv(Y8Y9)[["doc_id", "section_id", "y8_mafapo", "y9_cidh"]]
    src = src.drop_duplicates(subset=["doc_id", "section_id"])
    df = df.drop(columns=[c for c in ["y8_mafapo", "y9_cidh"] if c in df.columns])
    df = df.merge(src, on=["doc_id", "section_id"], how="left")
    print(f"  y8 no-NaN: {df['y8_mafapo'].notna().sum()}  "
          f"y9 no-NaN: {df['y9_cidh'].notna().sum()}")
    return df


def cargar_y11_agregado():
    """Agrega y11/y12/y13 por doc_id_corto promediando bloques."""
    print("\n" + "=" * 70)
    print("2. CARGA y11/y12/y13 (agregado por documento)")
    print("=" * 70)
    y = pd.read_csv(Y11)
    print(f"  {Y11}: {len(y)} bloques  ({y['corpus_type'].value_counts().to_dict()})")

    # columnas a usar como y11/y12/y13
    cols_y = {"y11_quotes": "y11", "y12_judgment": "y12", "y13_evidential": "y13"}
    agg = (y.groupby("doc_id")[list(cols_y.keys())]
           .mean()
           .rename(columns=cols_y)
           .reset_index())
    agg = agg.rename(columns={"doc_id": "doc_id_corto"})
    print(f"  documentos unicos con y11/y12/y13: {len(agg)}")
    print(f"  y11 rango: {agg['y11'].min():.3f}..{agg['y11'].max():.3f}")
    print(f"  y12 rango: {agg['y12'].min():.3f}..{agg['y12'].max():.3f}")
    print(f"  y13 rango: {agg['y13'].min():.3f}..{agg['y13'].max():.3f}")
    return agg


def unir(df, agg):
    print("\n" + "=" * 70)
    print("3. UNION por doc_id (largo -> corto 16 chars)")
    print("=" * 70)
    df["doc_id_corto"] = df["doc_id"].astype(str).str[:16]
    df = df.drop(columns=[c for c in ["y11", "y12", "y13"] if c in df.columns])
    df = df.merge(agg, on="doc_id_corto", how="left")
    for c in ["y11", "y12", "y13"]:
        print(f"  {c} no-NaN tras merge: {df[c].notna().sum()} / {len(df)}")
    return df


def preparar(df):
    print("\n" + "=" * 70)
    print("4. PREPARACION (rename + dropna + z-score)")
    print("=" * 70)
    d = df.rename(columns=COLMAP)
    faltan = [c for c in INDICADORES if c not in d.columns]
    if faltan:
        print(f"  [AVISO] faltan columnas: {faltan}")
        for c in faltan:
            d[c] = np.nan
    d = d[INDICADORES].copy()
    antes = len(d)
    d = d.dropna(subset=INDICADORES)
    print(f"  filas: {antes} -> {len(d)} (eliminadas {antes-len(d)} por NaN)")

    print("\n  escalas originales (min | media | max):")
    for c in INDICADORES:
        print(f"    {c:4s}: {d[c].min():8.3f} | {d[c].mean():8.3f} | {d[c].max():8.3f}")

    for c in INDICADORES:
        sd = d[c].std()
        if sd > 1e-9:
            d[c] = (d[c] - d[c].mean()) / sd
        else:
            print(f"    [AVISO] {c} varianza ~0 (constante)")
    print("  -> estandarizado (z-score)")
    return d


def estimar(d):
    print("\n" + "=" * 70)
    print("5. ESTIMACION SEM (eta2 completo: y10+y11+y12+y13)")
    print("=" * 70)
    import semopy
    from semopy import Model

    model = Model(MODELO_SPEC)
    model.fit(d, obj="MLW")
    stats = semopy.calc_stats(model)

    def g(name):
        try:
            v = stats[name]
            return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
        except Exception:
            return float("nan")

    cfi, tli, rmsea, chi2 = g("CFI"), g("TLI"), g("RMSEA"), g("chi2")
    print(f"\n  N = {len(d)}")
    print(f"  CFI:   {cfi:7.3f}   {'OK (>=0.90)' if cfi>=0.90 else 'bajo'}")
    print(f"  TLI:   {tli:7.3f}")
    print(f"  RMSEA: {rmsea:7.3f}   {'OK (<=0.08)' if rmsea<=0.08 else 'alto'}")
    print(f"  chi2:  {chi2:7.3f}")

    ins = model.inspect(std_est=True)
    print("\n  " + "-" * 60)
    print("  H3:  eta2 ~ eta1  (beta_23 < 0 esperado)")
    print("  " + "-" * 60)
    fila = ins[(ins["lval"] == "eta2") & (ins["op"] == "~") & (ins["rval"] == "eta1")]
    if len(fila):
        est = float(fila["Estimate"].iloc[0])
        est_std = float(fila["Est. Std"].iloc[0]) if "Est. Std" in fila.columns else float("nan")
        pval = float(fila["p-value"].iloc[0]) if "p-value" in fila.columns else float("nan")
        print(f"  beta_23 crudo:         {est:8.3f}")
        print(f"  beta_23 estandarizado: {est_std:8.3f}")
        print(f"  p = {pval:.4f}")
        soportada = (est < 0) and (pval < 0.05)
        print(f"  H3: {'APOYADA' if soportada else 'NO apoyada'}")

    print("\n  " + "-" * 60)
    print("  CARGAS FACTORIALES (std):")
    print("  " + "-" * 60)
    for _, r in ins[ins["op"] == "=~"].iterrows():
        try:
            es = float(r.get("Est. Std", r.get("Estimate", np.nan)))
        except Exception:
            es = float("nan")
        flag = "" if abs(es) >= 0.40 else "  <- debil"
        print(f"    {r['lval']:5s} =~ {r['rval']:5s}: {es:7.3f}{flag}")
    return cfi, rmsea


def main():
    df = cargar_sem()
    agg = cargar_y11_agregado()
    df = unir(df, agg)
    d = preparar(df)
    try:
        cfi, rmsea = estimar(d)
    except Exception as e:
        print(f"\n[ERROR estimacion] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("LECTURA:")
    if cfi >= 0.90 and rmsea <= 0.08:
        print("  El SEM CONVERGE con buen ajuste. eta2 completo lo resolvio.")
    elif cfi >= 0.60:
        print("  Mejoro respecto al modelo con eta2=y10 solo (CFI 0.42 antes).")
        print("  Revisar cargas: si y11/y12/y13 cargan debil, eta2 sigue floja.")
    else:
        print("  Ajuste aun bajo. Revisar si y11/y12/y13 covarian con y10.")
    print("=" * 70)


if __name__ == "__main__":
    main()
