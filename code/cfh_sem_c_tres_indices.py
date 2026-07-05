# -*- coding: utf-8 -*-
r"""
cfh_sem_c_tres_indices.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

SEM del Corpus C (n=47) integrando los TRES indices como observadas:
DIS (discursivo), IEI (epistemico) e ICM (multimodal). Como los tres son
casi independientes (corr<0.33), se comparan TRES estructuras teoricas:

  OPCION 1 - Cadena causal:   IEI ~ DIS ; ICM ~ DIS + IEI
  OPCION 2 - ICM predictor:   IEI ~ DIS + ICM
  OPCION 3 - Tres dimensiones paralelas: matriz de correlaciones con p-values
             (modelo saturado, DoF=0, sin indices de ajuste por definicion)

Se usan los indices ya construidos (observadas), NO latentes con indicadores
(evita colapso y8~y9; coherente con el framework CFH).

DIS/IEI de C se toman del archivo tri-corpus (filas C); ICM del CSV del SEM.
Cruce por nombre de compareciente.

Salida: consola + outputs/sem_c_tres_indices_reporte.txt

Uso (raiz del repo, env cfh):
    python code\cfh_sem_c_tres_indices.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_C = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")
OUT = os.path.join(REPO, "outputs", "sem_c_tres_indices_reporte.txt")


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def ajuste(model):
    try:
        from semopy import calc_stats
        s = calc_stats(model).T
        g = lambda k: float(s.loc[k].iloc[0]) if k in s.index else np.nan
        return g("CFI"), g("RMSEA"), g("SRMR"), g("chi2"), g("DoF")
    except Exception:
        return (np.nan,) * 5


def interpreta_cfi(cfi):
    if np.isnan(cfi):
        return "n/d"
    if cfi > 1.001:
        return "IMPOSIBLE (>1) -> modelo no identificado / saturado (n insuf.)"
    if cfi >= 0.95:
        return "bueno"
    if cfi >= 0.90:
        return "aceptable"
    return "pobre"


def corre_regresion(nombre, desc, df, cols):
    from semopy import Model
    print("\n" + "=" * 66)
    print(nombre)
    print("=" * 66)
    print("  Modelo:")
    for line in desc.strip().split("\n"):
        print("    " + line.strip())
    try:
        m = Model(desc)
        m.fit(df[cols])
        est = m.inspect()
        print("\n  Relaciones estructurales:")
        rel = est[est["op"] == "~"]
        for _, r in rel.iterrows():
            p = r.get("p-value", "-")
            try:
                p = f"{float(p):.3f}"
            except Exception:
                p = str(p)
            sig = ""
            try:
                sig = " *" if float(r["p-value"]) < 0.05 else ""
            except Exception:
                pass
            print(f"    {r['lval']:>4} ~ {r['rval']:<5} beta={r['Estimate']:+.3f}  p={p}{sig}")
        cfi, rmsea, srmr, chi2, dof = ajuste(m)
        print(f"\n  Ajuste: CFI={cfi:.3f} ({interpreta_cfi(cfi)})  "
              f"RMSEA={rmsea:.3f}  SRMR={srmr:.3f}  DoF={dof:.0f}")
        return cfi, rmsea
    except Exception as e:
        print(f"  [no converge] {e}")
        return np.nan, np.nan


def corre_correlaciones(nombre, df, cols):
    print("\n" + "=" * 66)
    print(nombre)
    print("=" * 66)
    print("  Modelo: DIS~~IEI, DIS~~ICM, IEI~~ICM (saturado, DoF=0)")
    print("\n  Correlaciones con significancia:")
    pares = [("DIS", "IEI"), ("DIS", "ICM"), ("IEI", "ICM")]
    todas_bajas = True
    for a, b in pares:
        r, p = pearsonr(df[a], df[b])
        sig = " *" if p < 0.05 else ""
        print(f"    {a} ~~ {b}: r={r:+.3f}  p={p:.3f}{sig}")
        if abs(r) >= 0.30:
            todas_bajas = False
    print("\n  Modelo saturado: sin CFI/RMSEA por definicion (DoF=0).")
    if todas_bajas:
        print("  >> Todas las correlaciones |r|<0.30: los tres indices son")
        print("     DIMENSIONES INDEPENDIENTES. Esta es la estructura que")
        print("     mejor describe los datos (coherente con el framework).")
    return todas_bajas


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    abc = pd.read_csv(F_ABC)
    c_abc = abc[abc["corpus"] == "C"][["unidad", "DIS", "IEI_A"]].copy()
    c_abc = c_abc.rename(columns={"unidad": "identidad", "IEI_A": "IEI"})
    csem = pd.read_csv(F_C)[["identidad", "icm_tricanal"]].rename(
        columns={"icm_tricanal": "ICM"})
    df = c_abc.merge(csem, on="identidad", how="inner")

    print("=" * 66)
    print("SEM de C - tres indices integrados (DIS, IEI, ICM)")
    print("=" * 66)
    print(f"Comparecientes cruzados: {len(df)}")
    print("\nDescriptivos:")
    for c in ["DIS", "IEI", "ICM"]:
        print(f"  {c}: media={df[c].mean():.3f} std={df[c].std():.3f}")

    for c in ["DIS", "IEI", "ICM"]:
        df[c] = z(df[c])
    cols = ["DIS", "IEI", "ICM"]

    r1 = corre_regresion("OPCION 1 - Cadena causal (IEI~DIS ; ICM~DIS+IEI)",
                          "IEI ~ DIS\nICM ~ DIS + IEI", df, cols)
    r2 = corre_regresion("OPCION 2 - ICM como predictor (IEI ~ DIS + ICM)",
                          "IEI ~ DIS + ICM", df, cols)
    indep = corre_correlaciones("OPCION 3 - Tres dimensiones paralelas", df, cols)

    print("\n" + "=" * 66)
    print("SINTESIS")
    print("=" * 66)
    print(f"  Opcion 1 (cadena):    CFI={r1[0]:.3f}  RMSEA={r1[1]:.3f}")
    print(f"  Opcion 2 (ICM pred):  CFI={r2[0]:.3f}  RMSEA={r2[1]:.3f}")
    print(f"  Opcion 3 (paralelas): saturado; independencia={'SI' if indep else 'NO'}")
    print("""
  LECTURA PARA LA TESIS:
  - Si las opciones causales (1,2) dan betas ~0 y ajuste pobre, CONFIRMAN
    que DIS, IEI e ICM no se predicen entre si -> son dimensiones
    independientes (refuerza el argumento dimensional del framework).
  - La opcion 3 describe directamente esa independencia via correlaciones.
  - Con n=47 es EXPLORATORIO: complementa la evidencia principal
    (DIS/IEI descriptivo entre corpus), no la sustituye.
  - Hallazgo integrador: los tres indices capturan tres dimensiones
    distintas de la injusticia (discursiva, epistemica, multimodal),
    empiricamente separables.""")

    print(f"\n  Reporte -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
