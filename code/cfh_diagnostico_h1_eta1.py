# -*- coding: utf-8 -*-
r"""
cfh_diagnostico_h1_eta1.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Diagnostica DOS problemas detectados en el SEM definitivo:

  (1) COLAPSO de eta1 (caso Heywood: var(y9_cidh)=0).
      Causa probable: y8_mafapo e y9_cidh correlacionan 0.91 -> son casi el
      mismo indicador (dos distancias coseno a centroides de victimas/DDHH).
      Una latente con dos indicadores r=0.91 no esta bien identificada.

  (2) SIGNO NEGATIVO de H1 (gamma_11 = -0.33).
      Hipotesis: y8/y9 podrian estar en escala de CONVERGENCIA (alto = cerca
      del lenguaje de victimas = mejor epistemicamente), no de DISTANCIA
      (alto = lejos = peor). Si es asi, eta1 mide CERCANIA epistemica, no
      INJUSTICIA — y el signo negativo de H1 seria correcto pero la etiqueta
      de la latente estaria invertida respecto a la hipotesis.

Que hace este script (solo diagnostico, NO modifica nada):
  A. Direccion de y8/y9: correlacion con y10_rep (reconocimiento) y con la
     violencia discursiva (y1/y2/y4). Si y8/y9 correlacionan POSITIVO con
     reconocimiento y NEGATIVO con violencia -> son CONVERGENCIA (cercania).
  B. Redundancia y8~y9: cuanta varianza comparten.
  C. eta1 con UN solo indicador: como cambia H1 si eta1 = y9_cidh (o y8).
  D. Correlacion directa de cada indicador de eta1 con xi1 observado
     (promedio z de y1+y2+y4), para ver el signo sin el modelo de medicion.

Uso (raiz del repo, env cfh):
    python code\cfh_diagnostico_h1_eta1.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "data" / "referencias" / "indicadores_sem_compareciente.csv"


def z(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 1e-9 else 1.0)


def main():
    df = pd.read_csv(CSV)
    print("=" * 66)
    print("DIAGNOSTICO H1 y colapso de eta1")
    print(f"Comparecientes: {len(df)}")
    print("=" * 66)

    # ---- A. Direccion de y8/y9: distancia o convergencia? ----
    print("\n" + "-" * 66)
    print("A. DIRECCION de y8_mafapo / y9_cidh")
    print("-" * 66)
    print("Si y8/y9 correlacionan POSITIVO con reconocimiento (y10_rep) y")
    print("NEGATIVO con violencia (y1/y2/y4) -> son CONVERGENCIA (cercania),")
    print("no distancia. Eso explicaria el signo de H1.\n")
    for col in ["y8_mafapo", "y9_cidh"]:
        print(f"  {col}:")
        print(f"    vs y10_rep (reconocimiento): {df[col].corr(df['y10_rep']):+.3f}")
        print(f"    vs y11_conv_rest (dialogico):{df[col].corr(df['y11_conv_rest']):+.3f}")
        print(f"    vs y1_ebi (violencia):       {df[col].corr(df['y1_ebi']):+.3f}")
        print(f"    vs y2_sa  (violencia):       {df[col].corr(df['y2_sa']):+.3f}")
        print(f"    vs y4_nv  (violencia):       {df[col].corr(df['y4_nv']):+.3f}")
        print()
    print("  INTERPRETACION:")
    print("   - Si y8/y9 son POSITIVOS con y10/y11 y NEGATIVOS con y1/y2/y4:")
    print("     miden CERCANIA al lenguaje de victimas (convergencia). Entonces")
    print("     eta1 NO es 'injusticia epistemica' sino 'cercania epistemica',")
    print("     y H1 negativo es CORRECTO (mas violencia -> menos cercania).")
    print("   - Si son al reves: son DISTANCIA y hay otro problema.")

    # ---- B. Redundancia y8 ~ y9 ----
    print("\n" + "-" * 66)
    print("B. REDUNDANCIA y8 ~ y9")
    print("-" * 66)
    r = df["y8_mafapo"].corr(df["y9_cidh"])
    print(f"  corr(y8, y9) = {r:+.3f}   (r^2 = {r**2:.3f} varianza compartida)")
    if abs(r) > 0.85:
        print("  -> MUY colineales: eta1 con estos 2 indicadores no se identifica")
        print("     bien (caso Heywood). Son casi el mismo constructo.")

    # ---- C. eta1 con un solo indicador ----
    print("\n" + "-" * 66)
    print("C. eta1 con UN solo indicador (evita el caso Heywood)")
    print("-" * 66)
    print("  Correlacion de un xi1 observado (media z de y1+y2+y4) con y8 y con y9:")
    xi1_obs = (z(df["y1_ebi"]) + z(df["y2_sa"]) + z(df["y4_nv"])) / 3
    for col in ["y8_mafapo", "y9_cidh"]:
        c = xi1_obs.corr(z(df[col]))
        print(f"    xi1_obs vs {col}: {c:+.3f}")
    print("\n  Este es el signo REAL de H1 sin el modelo de medicion.")
    print("  (si es negativo, H1 negativo no es artefacto del SEM: es el dato)")

    # ---- D. eta2 observado vs eta1 observado (H3 sin latentes) ----
    print("\n" + "-" * 66)
    print("D. H3 con variables observadas (robustez, sin latentes)")
    print("-" * 66)
    eta1_obs = (z(df["y8_mafapo"]) + z(df["y9_cidh"])) / 2
    eta2_obs = (z(df["y10_rep"]) + z(df["y11_conv_rest"])) / 2
    c13 = xi1_obs.corr(eta1_obs)
    c_h3 = eta1_obs.corr(eta2_obs)
    print(f"  H1 observado  xi1_obs ~ eta1_obs : {c13:+.3f}")
    print(f"  H3 observado  eta1_obs ~ eta2_obs: {c_h3:+.3f}")
    print("\n  (Path analysis rapido: si el signo de H3 coincide con el SEM,")
    print("   la hipotesis central es robusta al metodo de estimacion.)")

    print("\n" + "=" * 66)
    print("Revisa la seccion A: define si eta1 es 'cercania' o 'injusticia'.")
    print("De eso depende como se interpreta y etiqueta H1 en la tesis.")
    print("=" * 66)


if __name__ == "__main__":
    main()
