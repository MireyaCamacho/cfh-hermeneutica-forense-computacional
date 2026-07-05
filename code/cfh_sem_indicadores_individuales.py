# -*- coding: utf-8 -*-
r"""
cfh_sem_indicadores_individuales.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

CHEQUEO COMPLEMENTARIO (no reemplaza los SEM previos): corre el modelo con
las VARIABLES INDIVIDUALES (indicadores y1..y10) en vez de los indices
DIS/IEI/ICM, para verificar si algun indicador tiene una relacion que la
agregacion en indices oculta.

Se hace en DOS universos:
  - CORPUS C (n=47)
  - TRI-CORPUS (A+B+C, n=946)

Modelo (regresion, indicadores como PREDICTORES directos - opcion b):
    y10_rep ~ y1_ebi + y2_sa + y4_nv + y8_mafapo_cs + y9_cidh_cs
  (¿los indicadores de injusticia predicen la transicion epistemica y10?)

Ademas: MATRIZ DE CORRELACIONES de todos los indicadores individuales, para
ver relaciones entre indicadores que los indices pudieran diluir.

NO modifica ni sobreescribe resultados previos. Salida propia:
  outputs/sem_indicadores_individuales_reporte.txt
  outputs/matriz_correlaciones_indicadores_C.csv
  outputs/matriz_correlaciones_indicadores_ABC.csv

Uso (raiz del repo, env cfh):
    python code\cfh_sem_indicadores_individuales.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
OUT = os.path.join(REPO, "outputs", "sem_indicadores_individuales_reporte.txt")
OUT_COR_C = os.path.join(REPO, "outputs", "matriz_correlaciones_indicadores_C.csv")
OUT_COR_ABC = os.path.join(REPO, "outputs", "matriz_correlaciones_indicadores_ABC.csv")

IND = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep"]
PRED = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs"]


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
        return g("CFI"), g("RMSEA")
    except Exception:
        return np.nan, np.nan


def regresion(nombre, df):
    from semopy import Model
    print("\n" + "=" * 66)
    print(nombre)
    print("=" * 66)
    d = df.copy()
    for c in IND:
        d[c] = z(d[c].astype(float))
    desc = "y10_rep ~ " + " + ".join(PRED)
    print("  Modelo: " + desc)
    print(f"  n = {len(d)}")
    try:
        m = Model(desc)
        m.fit(d[IND])
        est = m.inspect()
        print("\n  Coeficientes (prediciendo y10_rep = transicion):")
        rel = est[est["op"] == "~"]
        for _, r in rel.iterrows():
            try:
                p = float(r["p-value"])
                pstr = f"{p:.3f}"
                sig = " *" if p < 0.05 else ""
            except Exception:
                pstr, sig = str(r["p-value"]), ""
            print(f"    y10_rep ~ {r['rval']:<14} beta={r['Estimate']:+.3f}  p={pstr}{sig}")
        cfi, rmsea = ajuste(m)
        print(f"\n  Ajuste: CFI={cfi:.3f}  RMSEA={rmsea:.3f}")
        # R2 aproximado via correlacion prediccion
        return rel
    except Exception as e:
        print(f"  [no converge] {e}")
        return None


def matriz_corr(nombre, df, path_out):
    print("\n" + "=" * 66)
    print(f"MATRIZ DE CORRELACIONES - {nombre}")
    print("=" * 66)
    cor = df[IND].corr(method="pearson").round(3)
    print(cor.to_string())
    cor.to_csv(path_out, encoding="utf-8-sig")
    # senalar correlaciones altas (posible colapso)
    print("\n  Correlaciones altas (|r|>=0.7, posible colinealidad):")
    altas = []
    for i in range(len(IND)):
        for j in range(i + 1, len(IND)):
            r = cor.iloc[i, j]
            if abs(r) >= 0.7:
                altas.append((IND[i], IND[j], r))
                print(f"    {IND[i]} ~ {IND[j]}: r={r:+.3f}")
    if not altas:
        print("    (ninguna) - no hay colinealidad fuerte entre indicadores")
    print(f"\n  Guardada -> {path_out}")


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    print("=" * 66)
    print("CHEQUEO COMPLEMENTARIO - indicadores individuales (no indices)")
    print("=" * 66)
    print("Complementa (no reemplaza) los SEM previos con DIS/IEI/ICM.")

    df = pd.read_csv(F_ABC)
    df_c = df[df["corpus"] == "C"].copy()

    # --- CORPUS C ---
    print("\n\n########## UNIVERSO 1: CORPUS C (n=47) ##########")
    regresion("Regresion indicadores -> y10 (Corpus C)", df_c)
    matriz_corr("Corpus C", df_c, OUT_COR_C)

    # --- TRI-CORPUS ---
    print("\n\n########## UNIVERSO 2: TRI-CORPUS A+B+C (n=946) ##########")
    regresion("Regresion indicadores -> y10 (A+B+C)", df)
    matriz_corr("Tri-corpus A+B+C", df, OUT_COR_ABC)

    # --- Sintesis ---
    print("\n" + "=" * 66)
    print("SINTESIS - que aporta el chequeo por indicadores")
    print("=" * 66)
    print("""  - Compara si algun indicador individual predice la transicion (y10)
    con una senal que el indice agregado (DIS/IEI) diluye.
  - La matriz de correlaciones revela colinealidades entre indicadores
    (esperado: y8~y9 alto, que motivo usar solo y8 en el SEM de C).
  - Si los coeficientes por indicador coinciden en signo/magnitud con lo
    visto por indices, CONFIRMA robustez: la conclusion no depende de la
    agregacion en indices.
  - Con n=47 (C) el chequeo es exploratorio; en tri-corpus (n=946) los
    p-values son mas potentes pero mezclan registro oral (C) y escrito (A/B).""")

    print(f"\n  Reporte -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
