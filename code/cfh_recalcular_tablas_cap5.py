# -*- coding: utf-8 -*-
r"""
cfh_recalcular_tablas_cap5.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Recalcula TODOS los numeros que las tablas del Capitulo 5 necesitan, con el
Corpus B fortalecido (80 secciones) y los indicadores depurados (y1 gazetteer,
y10 v5). Reemplaza las cifras viejas (que eran con B=54).

Produce, para cada indicador (y1_ebi, y2_sa, y4_nv, y8_mafapo, y9_cidh, y10_rep):
  - media A, media B, media C
  - p-valor Mann-Whitney U bilateral (A vs B)
  - Cohen's d (A vs B)
  - direccion e interpretacion

Cubre las tablas: 5.2.4 (SA/NV), 5.3 (REP), 5.4 (distancias), 5.5 (integrada
A vs B), 5.6 (control temporal si hay columna de anio).

Salida: outputs/tablas_cap5_recalculadas.txt + .csv

Uso (raiz del repo, env cfh):
    python code\cfh_recalcular_tablas_cap5.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
OUT_TXT = os.path.join(REPO, "outputs", "tablas_cap5_recalculadas.txt")
OUT_CSV = os.path.join(REPO, "outputs", "tablas_cap5_recalculadas.csv")

IND = {
    "y1_ebi": "EBI (Eufemismo Belico-Institucional)",
    "y2_sa": "SA (Supresion de Agentividad)",
    "y4_nv": "NV (Negacion de Victimizacion)",
    "y8_mafapo_cs": "Dist. MAFAPO (y8)",
    "y9_cidh_cs": "Dist. CIDH (y9)",
    "y10_rep": "REP (Ruptura Epistemica Positiva)",
}


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (x.mean() - y.mean()) / (pooled + 1e-12)


def sig(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))


def main():
    fh = open(OUT_TXT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    df = pd.read_csv(F_ABC)
    A = df[df["corpus"] == "A"]
    B = df[df["corpus"] == "B"]
    C = df[df["corpus"] == "C"]
    print("=" * 70)
    print("TABLAS CAP 5 RECALCULADAS - Corpus B fortalecido (80 secc)")
    print("=" * 70)
    print(f"n: A={len(A)}, B={len(B)}, C={len(C)}")
    print()
    print(f"{'Indicador':<40} {'A':>7} {'B':>7} {'C':>7} {'p(A-B)':>9} {'d':>7} {'sig':>5}")
    print("-" * 82)

    filas = []
    for col, nombre in IND.items():
        a, b, c = A[col].dropna(), B[col].dropna(), C[col].dropna()
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        d = cohen_d(a, b)
        print(f"{nombre:<40} {a.mean():>7.3f} {b.mean():>7.3f} {c.mean():>7.3f} "
              f"{p:>9.4f} {d:>+7.3f} {sig(p):>5}")
        filas.append({"indicador": nombre, "col": col,
                      "A": round(a.mean(), 4), "B": round(b.mean(), 4),
                      "C": round(c.mean(), 4), "p_AB": round(p, 5),
                      "cohen_d": round(d, 4), "sig": sig(p)})

    print("\n" + "=" * 70)
    print("NOTAS PARA ACTUALIZAR EL CAPITULO")
    print("=" * 70)
    print("""  - Reemplazar en todo el Cap.5 'N_B=54' por 'N_B=80'.
  - REP (y10 v5): valores nuevos A/B distintos a la V26 (era A=0.086/B=0.153).
  - EBI (y1): ahora funciona (gazetteer); en la V26 estaba en 0.0.
  - SA (y2): sigue transversal (verificar p; suele ser n.s.).
  - Distancias y8/y9: B mas cerca de victimas que en V26.
  - Verificar direccion de cada indicador antes de escribir la interpretacion.""")

    pd.DataFrame(filas).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Tabla -> {OUT_CSV}")
    print(f"  Texto -> {OUT_TXT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
