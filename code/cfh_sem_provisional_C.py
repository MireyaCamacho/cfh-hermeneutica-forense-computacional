# -*- coding: utf-8 -*-
"""
CFH — SEM PROVISIONAL, Corpus C por compareciente (n=47)
========================================================
Modelo reducido con los indicadores disponibles y validos:
  xi1  (Violencia Discursiva)   =~ y2_sa + y4_nv
  eta1 (Injusticia Epistemica)  =~ y8_mafapo + y9_cidh
  eta2 (Transicion Epistemica)  =~ y10_rep + y11_conv_rest + y12_acustico
  eta1 ~ xi1        (H1)
  eta2 ~ eta1       (H3: beta_23 < 0 esperado)

PROVISIONAL: y1(EBI), y3(civil), y7(surprisal) excluidos.
Los extractores deben tunearse contra la doble anotacion (IAA) antes del
modelo definitivo. ADVERTENCIA conocida: y8/y9/y11 muy colineales (r>0.9),
lo que puede colapsar eta1 y eta2. Este SEM sirve para diagnosticar eso.

Uso:
    conda activate cfh
    python code/cfh_sem_provisional_C.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "data" / "referencias" / "indicadores_sem_compareciente.csv"

# Indicadores del modelo
IND = ["y2_sa", "y4_nv", "y8_mafapo", "y9_cidh",
       "y10_rep", "y11_conv_rest", "y12_acustico"]

SPEC = """
# Modelo de medicion
xi1  =~ y2_sa + y4_nv
eta1 =~ y8_mafapo + y9_cidh
eta2 =~ y10_rep + y11_conv_rest + y12_acustico
# Modelo estructural
eta1 ~ xi1
eta2 ~ eta1
"""


def main():
    print("=" * 64)
    print("CFH — SEM PROVISIONAL Corpus C por compareciente (n=47)")
    print("=" * 64)

    try:
        import semopy
        from semopy import Model
    except ImportError:
        print("  [ERROR] semopy no instalado. pip install semopy")
        return

    df = pd.read_csv(CSV)
    print(f"\n  Comparecientes: {len(df)}")

    # Estandarizar indicadores (z-score) — recomendado para SEM
    dfz = df.copy()
    for c in IND:
        dfz[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

    print("\n  Especificacion del modelo:")
    print(SPEC)

    # Ajustar
    model = Model(SPEC)
    try:
        model.fit(dfz, obj="MLW")
    except Exception as e:
        print(f"  [aviso] MLW fallo ({e}); reintentando con ULS...")
        model = Model(SPEC)
        model.fit(dfz, obj="ULS")

    # Parametros
    print("\n" + "=" * 64)
    print("PARAMETROS ESTIMADOS")
    print("=" * 64)
    ins = model.inspect(std_est=True)
    print(ins.to_string())

    # beta_23 (H3): eta2 ~ eta1
    print("\n" + "=" * 64)
    print("HIPOTESIS H3: eta1 -> eta2 (beta_23 esperado < 0)")
    print("=" * 64)
    b = ins[(ins["lval"] == "eta2") & (ins["rval"] == "eta1") & (ins["op"] == "~")]
    if len(b):
        est = b.iloc[0]["Estimate"]
        pval = b.iloc[0].get("p-value", np.nan)
        print(f"  beta_23 = {est:.4f}  (p={pval})")
        print(f"  {'CONSISTENTE con H3 (<0)' if est < 0 else 'OPUESTO a H3 (>0)'}")

    # Indices de ajuste
    print("\n" + "=" * 64)
    print("INDICES DE AJUSTE")
    print("=" * 64)
    try:
        stats = semopy.calc_stats(model)
        print(stats.T.to_string())
    except Exception as e:
        print(f"  [aviso] no se pudieron calcular indices: {e}")

    # Correlacion entre factores latentes (diagnostico de colinealidad)
    print("\n" + "=" * 64)
    print("DIAGNOSTICO: correlaciones observadas entre indicadores")
    print("=" * 64)
    print(df[IND].corr().round(2).to_string())
    print("\n  NOTA: si eta1 y eta2 estan casi perfectamente correlacionados,")
    print("  es por la colinealidad y8/y9/y11 (distancias al mismo embedding).")
    print("=" * 64)


if __name__ == "__main__":
    main()
