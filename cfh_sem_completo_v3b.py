# -*- coding: utf-8 -*-
"""
cfh_sem_completo_v3b.py
========================
SEM completo del CFH sobre Corpus A/B, usando el canónico v3b (873 filas),
que tiene TODOS los indicadores con datos verificados:

    xi1  =~ y1 + y2 + y3 + y4     (Violencia Discursiva)   [datos OK]
    xi2  =~ y5 + y6               (Contexto Institucional)  [datos OK]
    eta1 =~ y7 + y8 + y9          (DIS Score)               [datos OK]
    eta2 =~ y10                   (Transición Epistémica)   [solo y10; y11 vacío]
    eta1 ~ xi1 + xi2             (H1, H2)
    eta2 ~ eta1                  (H3: beta_23 < 0, CENTRAL)
    xi1 ~~ xi2

Nota: y8/y9 son DISTANCIAS (mayor = más lejos del polo víctima). Para que la
carga en eta1 (DIS = injusticia) sea coherente y positiva, se usan tal cual
(mayor distancia = mayor injusticia discursiva). y10 (REP) es reconocimiento;
en eta2 (transición) su carga se interpreta según polaridad estimada.

Todos los indicadores se estandarizan (z-score) antes de estimar.

Este script NO modifica datos. Solo lee el canónico y estima.

Uso:
    python cfh_sem_completo_v3b.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CANONICO = "data/features/canonico/indicators_completo_conflibert_v3b.csv"

# y1_ebi está constante (=0) en el canónico -> placeholder, se EXCLUYE.
COLMAP = {
    "y2_sa": "y2", "y3_civil": "y3", "y4_nv": "y4",
    "y5_corpus_type": "y5", "y6_period": "y6",
    "y7_surprisal": "y7", "y8_mafapo": "y8", "y9_cidh": "y9",
    "y10_rep": "y10",
}
INDIC = ["y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]

# Modelo con eta2 = y10 (y1 e y11 vacíos en A/B -> excluidos)
SPEC = """
    xi1 =~ y2 + y3 + y4
    xi2 =~ y5 + y6
    eta1 =~ y7 + y8 + y9
    eta2 =~ y10
    eta1 ~ xi1 + xi2
    eta2 ~ eta1
    xi1 ~~ xi2
"""


def main():
    print("=" * 68)
    print("SEM COMPLETO CFH — Corpus A/B (canónico v3b)")
    print("=" * 68)
    df = pd.read_csv(CANONICO)
    print(f"  filas: {len(df)}")

    d = df.rename(columns=COLMAP)
    faltan = [c for c in INDIC if c not in d.columns]
    if faltan:
        print(f"  [ERROR] faltan columnas: {faltan}")
        return
    d = d[INDIC].copy()
    antes = len(d)
    d = d.dropna(subset=INDIC)
    print(f"  filas con los 10 indicadores completos: {antes} -> {len(d)}")

    print("\n  escalas (min | media | max):")
    for c in INDIC:
        print(f"    {c:4s}: {d[c].min():8.3f} | {d[c].mean():8.3f} | {d[c].max():8.3f}")

    # z-score
    for c in INDIC:
        sd = d[c].std()
        if sd > 1e-9:
            d[c] = (d[c] - d[c].mean()) / sd
        else:
            print(f"    [AVISO] {c} es constante (sd~0)")

    print("\n  correlaciones clave (indicadores de cada latente):")
    for grupo, cols in [("xi1", ["y2", "y3", "y4"]),
                        ("eta1", ["y7", "y8", "y9"])]:
        print(f"    {grupo}: {cols}")
        print(d[cols].corr().round(2).to_string().replace("\n", "\n      "))

    print("\n" + "=" * 68)
    print("ESTIMACIÓN")
    print("=" * 68)
    import semopy
    from semopy import Model
    m = Model(SPEC)
    m.fit(d, obj="MLW")
    st = semopy.calc_stats(m)

    def g(n):
        try:
            v = st[n]
            return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
        except Exception:
            return float("nan")

    cfi, tli, rmsea = g("CFI"), g("TLI"), g("RMSEA")
    print(f"  N = {len(d)}")
    print(f"  CFI:   {cfi:7.3f}   {'OK (>=0.95)' if cfi>=0.95 else 'revisar'}")
    print(f"  TLI:   {tli:7.3f}")
    print(f"  RMSEA: {rmsea:7.3f}   {'OK (<=0.08)' if rmsea<=0.08 else 'alto'}")

    ins = m.inspect(std_est=True)

    print("\n  " + "-" * 60)
    print("  H3:  eta2 ~ eta1  (beta_23 < 0 esperado)")
    print("  " + "-" * 60)
    fila = ins[(ins["lval"] == "eta2") & (ins["op"] == "~") & (ins["rval"] == "eta1")]
    if len(fila):
        est = float(fila["Estimate"].iloc[0])
        est_std = float(fila["Est. Std"].iloc[0]) if "Est. Std" in fila.columns else float("nan")
        p = float(fila["p-value"].iloc[0]) if "p-value" in fila.columns else float("nan")
        print(f"  beta_23 crudo:         {est:8.3f}")
        print(f"  beta_23 estandarizado: {est_std:8.3f}")
        print(f"  p = {p:.4f}")
        print(f"  H3: {'APOYADA (beta<0, p<0.05)' if (est<0 and p<0.05) else 'NO apoyada en esa direccion'}")

    print("\n  " + "-" * 60)
    print("  RUTAS ESTRUCTURALES (H1, H2)")
    print("  " + "-" * 60)
    for _, r in ins[(ins["op"] == "~")].iterrows():
        try:
            es = float(r.get("Est. Std", r.get("Estimate", np.nan)))
            p = float(r.get("p-value", np.nan))
        except Exception:
            es, p = float("nan"), float("nan")
        print(f"    {r['lval']:5s} ~ {r['rval']:5s}: std={es:7.3f}  p={p:.4f}")

    print("\n  " + "-" * 60)
    print("  CARGAS FACTORIALES (std)")
    print("  " + "-" * 60)
    for _, r in ins[ins["op"] == "=~"].iterrows():
        try:
            es = float(r.get("Est. Std", r.get("Estimate", np.nan)))
        except Exception:
            es = float("nan")
        flag = "" if abs(es) >= 0.40 else "  <- debil"
        print(f"    {r['lval']:5s} =~ {r['rval']:5s}: {es:7.3f}{flag}")

    print("\n" + "=" * 68)
    print("LECTURA:")
    if cfi >= 0.95 and rmsea <= 0.08:
        print("  Modelo con BUEN ajuste. Tres latentes plenas + eta2=y10.")
        print("  Si H3 sale positiva, es un hallazgo sustantivo a interpretar")
        print("  (no necesariamente un fallo — revisar polaridad de y10/eta2).")
    else:
        print("  Ajuste a revisar. Ver cargas debiles y correlaciones por latente.")
    print("  Para robustecer eta2: rehacer y11 (convergencia por embeddings) en Colab.")
    print("=" * 68)


if __name__ == "__main__":
    main()
