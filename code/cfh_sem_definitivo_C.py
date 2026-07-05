# -*- coding: utf-8 -*-
r"""
cfh_sem_definitivo_C.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

SEM DEFINITIVO - Corpus C por compareciente (n=47), indicadores TUNEADOS.

Ya NO es provisional: los indicadores que estaban excluidos o colineales en
el modelo anterior fueron rescatados/tuneados en esta cadena de trabajo:
  - y1_ebi  : rescatado por gazetteer (antes 0.0 en todo).
  - y10_rep : extractor v5 (Opcion B) + normalizado z-score/sigmoide sobre C.
  - y11_conv_rest : redisenado como densidad dialogica/testimonial.
    La colinealidad critica y8/y9/y11 (r>0.9) que amenazaba con colapsar
    eta1 y eta2 quedo resuelta: y11 vs y8 = 0.13, y11 vs y9 = 0.33.

ESPECIFICACION (dos versiones comparadas):
  xi1  =~ y1_ebi + y2_sa + y4_nv        (Violencia Discursiva)
  eta1 =~ y8_mafapo + y9_cidh           (Injusticia Epistemica)
  eta2 =~ y10_rep + y11_conv_rest [+ y12_acustico]   (Transicion Epistemica)
  eta1 ~ xi1                            (H1)
  eta2 ~ eta1                           (H3: beta_23 < 0  <- CENTRAL)

  V_dentro : eta2 =~ y10_rep + y11_conv_rest + y12_acustico
  V_fuera  : eta2 =~ y10_rep + y11_conv_rest

ADVERTENCIA HONESTA (n=47):
  Con 47 casos y un modelo de 3 latentes, se esta en el limite inferior de
  estimacion estable de un SEM (la regla usual pide ~10-20 casos por
  parametro). Por eso el Cap. 6 reencuadra el SEM como EXPLORATORIO /
  analisis de senderos, no confirmatorio. Los indices de ajuste se reportan
  con esa cautela; no deben sobre-interpretarse.

Uso (raiz del repo, env cfh):
    python code\cfh_sem_definitivo_C.py

Salidas:
  - reporte por consola (correlaciones, ambas versiones, H3, indices)
  - cfh_sem_definitivo_resultados.txt  (log completo)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "data" / "referencias" / "indicadores_sem_compareciente.csv"
OUT_TXT = REPO / "cfh_sem_definitivo_resultados.txt"

IND_BASE = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo", "y9_cidh",
            "y10_rep", "y11_conv_rest", "y12_acustico"]

SPEC_DENTRO = """
# Medicion
xi1  =~ y1_ebi + y2_sa + y4_nv
eta1 =~ y8_mafapo + y9_cidh
eta2 =~ y10_rep + y11_conv_rest + y12_acustico
# Estructural
eta1 ~ xi1
eta2 ~ eta1
"""

SPEC_FUERA = """
# Medicion
xi1  =~ y1_ebi + y2_sa + y4_nv
eta1 =~ y8_mafapo + y9_cidh
eta2 =~ y10_rep + y11_conv_rest
# Estructural
eta1 ~ xi1
eta2 ~ eta1
"""


class Tee:
    """Escribe a consola y a archivo a la vez."""
    def __init__(self, fh):
        self.fh = fh
    def write(self, s):
        sys.__stdout__.write(s)
        self.fh.write(s)
    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()


def estandarizar(df, cols):
    dfz = df.copy()
    for c in cols:
        s = df[c].std()
        dfz[c] = (df[c] - df[c].mean()) / (s if s > 1e-9 else 1.0)
    return dfz


def correr_version(nombre, spec, dfz):
    import semopy
    from semopy import Model

    print("\n" + "=" * 66)
    print(f"VERSION: {nombre}")
    print("=" * 66)
    print(spec)

    model = Model(spec)
    obj_usado = "MLW"
    try:
        model.fit(dfz, obj="MLW")
    except Exception as e:
        print(f"  [aviso] MLW fallo ({e}); reintento con ULS...")
        model = Model(spec)
        model.fit(dfz, obj="ULS")
        obj_usado = "ULS"

    ins = model.inspect(std_est=True)
    print(f"\n  Estimador: {obj_usado}")
    print("\n  PARAMETROS (std):")
    print(ins.to_string())

    # H3: eta2 ~ eta1
    b = ins[(ins["lval"] == "eta2") & (ins["rval"] == "eta1") & (ins["op"] == "~")]
    beta23 = None
    if len(b):
        beta23 = float(b.iloc[0]["Estimate"])
        pv = b.iloc[0].get("p-value", np.nan)
        print("\n  " + "-" * 60)
        print(f"  H3  beta_23 (eta1 -> eta2) = {beta23:+.4f}   p={pv}")
        signo = "CONSISTENTE con H3 (<0)" if beta23 < 0 else "OPUESTO a H3 (>0)"
        print(f"      {signo}")
        print("  " + "-" * 60)

    # H1: eta1 ~ xi1
    g = ins[(ins["lval"] == "eta1") & (ins["rval"] == "xi1") & (ins["op"] == "~")]
    if len(g):
        gamma = float(g.iloc[0]["Estimate"])
        pv = g.iloc[0].get("p-value", np.nan)
        print(f"  H1  gamma_11 (xi1 -> eta1) = {gamma:+.4f}   p={pv}")

    # Indices de ajuste
    print("\n  INDICES DE AJUSTE:")
    try:
        stats = semopy.calc_stats(model)
        st = stats.T
        print(st.to_string())
        # extraer los clave si estan
        for k in ["CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC", "chi2", "DoF"]:
            if k in stats.columns:
                print(f"    {k} = {stats.iloc[0][k]}")
    except Exception as e:
        print(f"  [aviso] no se pudieron calcular indices: {e}")

    return beta23


def main():
    fh = open(OUT_TXT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    print("=" * 66)
    print("CFH - SEM DEFINITIVO Corpus C por compareciente (n=47)")
    print("Indicadores tuneados: y1_ebi, y10_rep(v5+norm C), y11(dialogico)")
    print("=" * 66)

    try:
        import semopy  # noqa
    except ImportError:
        print("  [ERROR] semopy no instalado. pip install semopy")
        return

    df = pd.read_csv(CSV)
    print(f"\nComparecientes: {len(df)}")

    faltan = [c for c in IND_BASE if c not in df.columns]
    if faltan:
        print(f"  [ERROR] faltan columnas: {faltan}")
        return

    # --- Diagnostico previo: correlaciones entre indicadores ---
    print("\n" + "-" * 66)
    print("MATRIZ DE CORRELACIONES (diagnostico de colinealidad)")
    print("-" * 66)
    corr = df[IND_BASE].corr()
    print(corr.round(2).to_string())
    # avisar pares muy colineales
    print("\n  Pares con |r| > 0.8 (riesgo de colinealidad):")
    altos = []
    for i in range(len(IND_BASE)):
        for j in range(i + 1, len(IND_BASE)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                altos.append((IND_BASE[i], IND_BASE[j], r))
                print(f"    {IND_BASE[i]} ~ {IND_BASE[j]}: {r:+.3f}")
    if not altos:
        print("    (ninguno) -> colinealidad controlada")

    # --- Estandarizar ---
    dfz = estandarizar(df, IND_BASE)

    # --- Correr ambas versiones ---
    b_dentro = correr_version("y12 DENTRO de eta2", SPEC_DENTRO, dfz)
    b_fuera = correr_version("y12 FUERA de eta2", SPEC_FUERA, dfz)

    # --- Comparacion final ---
    print("\n" + "=" * 66)
    print("COMPARACION DE LAS DOS VERSIONES")
    print("=" * 66)
    print(f"  beta_23 (H3) con y12 DENTRO: {b_dentro}")
    print(f"  beta_23 (H3) con y12 FUERA : {b_fuera}")
    print("\n  Recordatorio: con n=47 el SEM es EXPLORATORIO (analisis de")
    print("  senderos), no confirmatorio. Interpretar los indices con cautela.")
    print(f"\n  Log completo -> {OUT_TXT}")

    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
