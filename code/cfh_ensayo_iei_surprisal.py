# -*- coding: utf-8 -*-
r"""
cfh_ensayo_iei_surprisal.py
CFH - Hermenéutica Forense Computacional | Mireya Camacho Celis

ENSAYO no destructivo: verifica que pasaria si el surprisal (y7) se integrara
al IEI (en vez del DIS). Parte del IEI AUDITADO real (columna 'IEI' del CSV),
no lo recalcula. NO modifica ningun archivo auditado.

El IEI auditado ya DISCRIMINA A vs B (A=0.513, B=0.353). La pregunta es si
agregar y7 lo refuerza, lo debilita o lo distorsiona, y si tiene sentido.

Metodo: mezcla convexa IEI_mix = (1-w)*IEI_auditado + w*y7_norm, para varios w.
Reporta A, B, C, Mann-Whitney A vs B y Cohen's d.

FUENTES (solo lectura):
  outputs/dis_iei_corpus_abc_v2.csv  (col 'IEI', 'DIS', 'corpus')
  outputs/y7_surprisal_{A,B,C}_alineado.csv

SALIDA:
  outputs/ENSAYO_iei_surprisal_reporte.txt

Uso:
    python code\cfh_ensayo_iei_surprisal.py
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_Y7 = {c: os.path.join(REPO, "outputs", f"y7_surprisal_{c}_alineado.csv") for c in "ABC"}
OUT = os.path.join(REPO, "outputs", "ENSAYO_iei_surprisal_reporte.txt")


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()
    def isatty(self): return False
    def fileno(self): return sys.__stdout__.fileno()
    @property
    def encoding(self): return getattr(sys.__stdout__, "encoding", "utf-8")


def zsig(s):
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    return 1.0 / (1.0 + np.exp(-z))


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
    return (x.mean()-y.mean())/(pooled+1e-12)


def main():
    fh = open(OUT, "w", encoding="utf-8"); sys.stdout = Tee(fh)
    print("="*66); print("ENSAYO: IEI auditado + surprisal (no destructivo)"); print("="*66)

    df = pd.read_csv(F_ABC).reset_index(drop=True)
    for col in ("IEI_A", "DIS"):
        if col not in df.columns:
            print(f"[ERROR] falta columna '{col}'"); sys.stdout=sys.__stdout__; fh.close(); return

    def stats3(serie, etiqueta):
        a = serie[df.corpus=="A"].dropna(); b = serie[df.corpus=="B"].dropna(); c = serie[df.corpus=="C"].dropna()
        p = mannwhitneyu(a,b,alternative="two-sided")[1]; d = cohen_d(a,b)
        print(f"{etiqueta:<16} A={a.mean():.4f} B={b.mean():.4f} C={c.mean():.4f} "
              f"p(A-B)={p:.4f} d={d:+.3f} [{'sig' if p<0.05 else 'n.s.'}]")
        return p

    print("\n-- Referencia auditada --")
    stats3(df["IEI_A"], "IEI auditado")
    stats3(df["DIS"], "DIS auditado")

    # cargar y7 alineado
    frames=[]
    for c in "ABC":
        d = pd.read_csv(F_Y7[c]); col = "y7_surprisal" if "y7_surprisal" in d.columns else d.columns[-1]
        frames.append(pd.DataFrame({"corpus":c,"y7":d[col].values}))
    y7 = pd.concat(frames, ignore_index=True)
    y7["y7n"] = zsig(-y7["y7"])  # mas belico -> mayor valor
    df["_y7n"] = np.nan
    for c in "ABC":
        idx = df.index[df.corpus==c].tolist()
        vals = y7.loc[y7.corpus==c,"y7n"].values
        for k,ii in enumerate(idx):
            if k < len(vals): df.at[ii,"_y7n"] = vals[k]

    print("\n" + "="*66)
    print("MEZCLA: IEI_mix = (1-w)*IEI_auditado + w*y7_norm")
    print("="*66)
    print(f"{'w (peso y7)':<14} {'A':>7} {'B':>7} {'C':>7} {'p(A-B)':>9} {'d':>7}")
    print("-"*60)
    for w in [0.0, 0.10, 0.15, 0.20, 0.30]:
        mix = (1-w)*df["IEI_A"] + w*df["_y7n"]
        a = mix[df.corpus=="A"].dropna(); b = mix[df.corpus=="B"].dropna(); c = mix[df.corpus=="C"].dropna()
        p = mannwhitneyu(a,b,alternative="two-sided")[1]; d = cohen_d(a,b)
        etiq = "IEI auditado" if w==0 else f"w={w}"
        print(f"{etiq:<14} {a.mean():>7.3f} {b.mean():>7.3f} {c.mean():>7.3f} {p:>9.4f} {d:>+7.3f}")

    # correlacion y7 con IEI y con DIS (a nivel B, donde ya sabemos que es independiente)
    print("\n" + "="*66)
    print("CORRELACION y7 con los indices (Spearman, sobre A+B+C)")
    print("="*66)
    valid = df.dropna(subset=["_y7n","IEI_A","DIS"])
    r_iei = valid["_y7n"].corr(valid["IEI_A"], method="spearman")
    r_dis = valid["_y7n"].corr(valid["DIS"], method="spearman")
    print(f"  y7 ~ IEI: rho={r_iei:+.3f}")
    print(f"  y7 ~ DIS: rho={r_dis:+.3f}")

    print("\n" + "="*66); print("LECTURA"); print("="*66)
    print("""  - El IEI auditado YA discrimina A vs B por si mismo (la brecha epistemica).
  - Agregar y7 al IEI puede reforzar la significancia, pero MEZCLA dos
    constructos distintos: el IEI mide injusticia epistemica (distancia al
    lenguaje de las victimas + credibility deficit); y7 mide el registro
    belico-institucional GLOBAL del texto. No son lo mismo.
  - Conceptualmente y7 esta MAS cerca del DIS (discursivo) que del IEI
    (epistemico). Integrarlo al IEI seria teoricamente incoherente: diluiria
    el constructo epistemico con una medida de registro discursivo.
  - Ademas, como el IEI ya discrimina, y7 no 'resuelve' ningun problema ahi;
    solo agregaria ruido conceptual.
  >> Confirma que lo apropiado es dejar y7 como DESCRIPTIVO aparte, no dentro
     del IEI ni del DIS.""")
    print(f"\n  Reporte -> {OUT}")
    print("  (Ensayo NO destructivo.)")
    sys.stdout = sys.__stdout__; fh.close()


if __name__ == "__main__":
    main()
