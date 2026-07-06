# -*- coding: utf-8 -*-
r"""
cfh_dis_iei_por_subcaso.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Calcula DIS/IEI POR SUBCASO del Corpus C con la OPCION A (formulas auditadas),
usando la MISMA normalizacion conjunta A+B+C que el tri-corpus, para que los
valores por subcaso sean coherentes con el analisis tri-corpus.

FORMULAS (opcion A, auditadas):
  DIS   = 0.40*EBI_z + 0.30*SA_z + 0.30*(1-REP_z)
  IEI   = 0.40*MAF_z + 0.30*CIDH_z + 0.30*NV_z

Verifica que la media de C por subcaso (ponderada por n) reproduzca el C del
tri-corpus (DIS=0.394, IEI=0.370). Reporta n por subcaso con advertencia si
n<5 (media poco representativa).

Entrada: outputs/dis_iei_corpus_abc_v2.csv (ya tiene los z-scores conjuntos y
         DIS/IEI calculados por unidad, incluyendo las 47 filas de C)
         data/referencias/indicadores_sem_compareciente.csv (para el subcaso)

Salida: outputs/dis_iei_por_subcaso_C.csv
        consola con verificacion de coincidencia

Uso (raiz del repo, env cfh):
    python code\cfh_dis_iei_por_subcaso.py
"""

import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_SEM = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")
OUT = os.path.join(REPO, "outputs", "dis_iei_por_subcaso_C.csv")


def main():
    print("=" * 64)
    print("DIS/IEI por subcaso de C (opcion A, normalizacion conjunta A+B+C)")
    print("=" * 64)

    # DIS/IEI ya calculados por unidad (incluye C con la misma norma conjunta)
    abc = pd.read_csv(F_ABC)
    c = abc[abc["corpus"] == "C"][["unidad", "DIS", "IEI_A"]].copy()
    c = c.rename(columns={"unidad": "identidad", "IEI_A": "IEI"})

    # traer el subcaso
    sem = pd.read_csv(F_SEM)[["identidad", "subcaso"]]
    c = c.merge(sem, on="identidad", how="left")

    faltan = c["subcaso"].isna().sum()
    if faltan:
        print(f"[AVISO] {faltan} comparecientes sin subcaso asignado")

    # media por subcaso
    print("\nDIS/IEI por subcaso (media):")
    res = c.groupby("subcaso").agg(
        n=("DIS", "size"),
        DIS=("DIS", "mean"),
        IEI=("IEI", "mean"),
    ).round(4).sort_values("DIS", ascending=False)

    for sub, row in res.iterrows():
        adv = ""
        if row["n"] < 5:
            adv = f"  <- ATENCION n={int(row['n'])} (media poco representativa)"
        print(f"  {sub:<12} n={int(row['n']):>2}  DIS={row['DIS']:.3f}  "
              f"IEI={row['IEI']:.3f}{adv}")

    # verificacion: media global de C (ponderada por n = media simple de las 47)
    print("\n--- Verificacion de coincidencia con el tri-corpus ---")
    dis_c = c["DIS"].mean()
    iei_c = c["IEI"].mean()
    print(f"  Media C (47 comparecientes): DIS={dis_c:.3f}  IEI={iei_c:.3f}")
    print(f"  C en tri-corpus (esperado):  DIS=0.394  IEI=0.370")
    ok_dis = abs(dis_c - 0.394) < 0.005
    ok_iei = abs(iei_c - 0.370) < 0.005
    print(f"  Coincide DIS: {'SI' if ok_dis else 'NO'}   "
          f"Coincide IEI: {'SI' if ok_iei else 'NO'}")
    if ok_dis and ok_iei:
        print("  >> Los valores por subcaso son COHERENTES con el tri-corpus.")
        print("     Se pueden presentar como analisis complementarios.")
    else:
        print("  >> DISCREPANCIA: revisar antes de escribir el capitulo.")

    # guardar
    res.to_csv(OUT, encoding="utf-8-sig")
    # tambien el detalle por compareciente
    c.sort_values(["subcaso", "DIS"]).to_csv(
        OUT.replace(".csv", "_detalle.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  Por subcaso -> {OUT}")
    print(f"  Detalle     -> {OUT.replace('.csv', '_detalle.csv')}")

    print("""
  NOTA para el capitulo: Casanare (n=1) y Catatumbo (n=3) tienen n muy bajo;
  su 'media de subcaso' representa 1-3 comparecientes, no un grupo amplio.
  Reportar con transparencia. Huila (n=26) y Dabeiba (n=12) son mas solidos.""")


if __name__ == "__main__":
    main()
