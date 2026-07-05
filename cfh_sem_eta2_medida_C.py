# -*- coding: utf-8 -*-
"""
cfh_sem_eta2_medida_C.py
=========================
Modelo de MEDIDA de eta2 (Transicion Epistemica) sobre Corpus C, usando los
tres indicadores que conviven en el mismo archivo/segmentacion:
    y11 (convergencia restaurativa), y12 (juicio), y13 (evidencialidad).

150 bloques de C (5 subcasos). NO incluye y10 porque esta en otra segmentacion
(el unificado, bloque_id incompatible). Este es un CFA de un solo factor:

    eta2 =~ y11 + y12 + y13

Responde: ¿y11, y12, y13 forman juntos un constructo coherente de transicion
epistemica? Se evalua con:
  - cargas factoriales estandarizadas (>=0.40 aceptable, >=0.70 fuerte)
  - CFI/RMSEA del modelo de medida
  - alpha de Cronbach (consistencia interna) como referencia

Uso:
    python cfh_sem_eta2_medida_C.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

Y11 = "data/indicators_y11_y12_y13.csv"


def cronbach_alpha(df):
    """Alpha de Cronbach sobre columnas estandarizadas."""
    k = df.shape[1]
    var_items = df.var(axis=0, ddof=1).sum()
    var_total = df.sum(axis=1).var(ddof=1)
    if var_total < 1e-9 or k < 2:
        return float("nan")
    return (k / (k - 1)) * (1 - var_items / var_total)


def main():
    y = pd.read_csv(Y11)
    c = y[y["corpus_type"] == "C"].copy()
    print("=" * 66)
    print(f"MODELO DE MEDIDA eta2 — CORPUS C (n={len(c)} bloques)")
    print("=" * 66)

    cols = {"y11_quotes": "y11", "y12_judgment": "y12", "y13_evidential": "y13"}
    d = c[list(cols.keys())].rename(columns=cols).dropna()
    print(f"  bloques con y11/y12/y13 completos: {len(d)}")

    print("\n  descriptivos (originales):")
    for col in ["y11", "y12", "y13"]:
        print(f"    {col}: media={d[col].mean():.3f}  sd={d[col].std():.3f}  "
              f"min={d[col].min():.3f}  max={d[col].max():.3f}")

    # correlaciones entre los 3 (clave: si no correlacionan, no forman factor)
    print("\n  CORRELACIONES entre indicadores (Pearson):")
    corr = d.corr()
    print(corr.round(3).to_string())

    # estandarizar
    dz = d.copy()
    for col in ["y11", "y12", "y13"]:
        sd = dz[col].std()
        dz[col] = (dz[col] - dz[col].mean()) / (sd if sd > 1e-9 else 1)

    # alpha de Cronbach
    alpha = cronbach_alpha(dz)
    print(f"\n  Alpha de Cronbach (3 items): {alpha:.3f}")
    if alpha >= 0.70:
        print("    -> consistencia interna ACEPTABLE (forman escala)")
    elif alpha >= 0.50:
        print("    -> consistencia MODERADA")
    else:
        print("    -> consistencia BAJA (los 3 no miden lo mismo)")

    # CFA con semopy
    print("\n" + "=" * 66)
    print("  CFA: eta2 =~ y11 + y12 + y13")
    print("=" * 66)
    try:
        import semopy
        from semopy import Model
        spec = "eta2 =~ y11 + y12 + y13"
        m = Model(spec)
        m.fit(dz, obj="MLW")
        stats = semopy.calc_stats(m)

        def g(n):
            try:
                v = stats[n]
                return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
            except Exception:
                return float("nan")

        print(f"  CFI={g('CFI'):.3f}  TLI={g('TLI'):.3f}  RMSEA={g('RMSEA'):.3f}")
        ins = m.inspect(std_est=True)
        print("\n  cargas factoriales (std):")
        for _, r in ins[ins["op"] == "=~"].iterrows():
            try:
                es = float(r.get("Est. Std", r.get("Estimate", np.nan)))
            except Exception:
                es = float("nan")
            flag = "  fuerte" if abs(es) >= 0.70 else ("  ok" if abs(es) >= 0.40 else "  DEBIL")
            print(f"    eta2 =~ {r['rval']:4s}: {es:7.3f}{flag}")
    except Exception as e:
        print(f"  [nota] CFA con 3 items es apenas identificable; {e}")
        print("  Las correlaciones y el alpha de arriba son la evidencia principal.")

    print("\n" + "=" * 66)
    print("LECTURA:")
    print("  Si las correlaciones son positivas y moderadas-altas y alpha>=0.5,")
    print("  y11/y12/y13 forman un constructo de transicion epistemica coherente.")
    print("  Si correlacionan debil o negativo, eta2 es multidimensional (cada")
    print("  indicador capta una faceta distinta) -> reportar por separado.")
    print("=" * 66)


if __name__ == "__main__":
    main()
