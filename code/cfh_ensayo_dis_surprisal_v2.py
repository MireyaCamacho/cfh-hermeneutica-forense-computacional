# -*- coding: utf-8 -*-
r"""
cfh_ensayo_dis_surprisal_v2.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

ENSAYO CORREGIDO (no destructivo): parte del DIS AUDITADO real (columna 'DIS'
del CSV) en lugar de renormalizar desde cero. Verifica si incorporar el
surprisal (y7) cambiaria la (no) discriminacion A vs B del DIS.

Base auditada confirmada: DIS col -> A=0.5102, B=0.4983, p=0.0688 (n.s.)

Metodo:
  1. Lee la columna DIS auditada (no la recalcula).
  2. Normaliza y7 con z-score+sigmoide sobre A+B+C (mas belico = mayor valor).
  3. Construye DIS_con_y7 = (1-w)*DIS_auditado + w*y7_norm, para varios w,
     que es una mezcla convexa: preserva la escala del DIS auditado y anade
     y7 con peso w. Esto NO renormaliza los componentes internos del DIS.
  4. Compara A vs B (Mann-Whitney + Cohen's d) del DIS auditado vs cada mezcla.

NO modifica ningun archivo auditado. Solo escribe un reporte nuevo.

FUENTES (solo lectura):
  outputs/dis_iei_corpus_abc_v2.csv         (col 'DIS' auditada + 'corpus')
  outputs/y7_surprisal_{A,B,C}_alineado.csv

SALIDA:
  outputs/ENSAYO_dis_surprisal_v2_reporte.txt

Uso:
    python code\cfh_ensayo_dis_surprisal_v2.py
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_Y7 = {c: os.path.join(REPO, "outputs", f"y7_surprisal_{c}_alineado.csv") for c in "ABC"}
OUT = os.path.join(REPO, "outputs", "ENSAYO_dis_surprisal_v2_reporte.txt")


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
    print("="*66); print("ENSAYO v2: DIS auditado + surprisal (no destructivo)"); print("="*66)

    df = pd.read_csv(F_ABC).reset_index(drop=True)
    if "DIS" not in df.columns:
        print("[ERROR] no hay columna 'DIS' auditada"); sys.stdout=sys.__stdout__; fh.close(); return

    # verificar base auditada
    a0 = df.loc[df.corpus=="A","DIS"].dropna(); b0 = df.loc[df.corpus=="B","DIS"].dropna()
    p0 = mannwhitneyu(a0,b0,alternative="two-sided")[1]; d0 = cohen_d(a0,b0)
    print(f"\nDIS AUDITADO (col 'DIS'): A={a0.mean():.4f} B={b0.mean():.4f} "
          f"p={p0:.4f} d={d0:+.3f}  [{'n.s.' if p0>0.05 else 'sig'}]")

    # cargar y7 alineado
    frames=[]
    for c in "ABC":
        d = pd.read_csv(F_Y7[c]); col = "y7_surprisal" if "y7_surprisal" in d.columns else d.columns[-1]
        frames.append(pd.DataFrame({"corpus":c,"y7":d[col].values}))
    y7 = pd.concat(frames, ignore_index=True)
    for c in "ABC":
        n1=sum(df.corpus==c); n2=sum(y7.corpus==c)
        print(f"  align {c}: df={n1} y7={n2} {'OK' if n1==n2 else 'DESALINEADO'}")

    # y7 normalizado (mas belico = mas negativo -> invertir -> mayor 'injusticia')
    y7["y7n"] = zsig(-y7["y7"])
    # asignar en orden por corpus
    df["_y7n"] = np.nan
    for c in "ABC":
        idx = df.index[df.corpus==c].tolist()
        vals = y7.loc[y7.corpus==c,"y7n"].values
        for k,ii in enumerate(idx):
            if k < len(vals): df.at[ii,"_y7n"] = vals[k]

    print("\n" + "="*66)
    print("MEZCLA CONVEXA: DIS_mix = (1-w)*DIS_auditado + w*y7_norm")
    print("="*66)
    print(f"{'w (peso y7)':<14} {'A':>7} {'B':>7} {'p(A-B)':>9} {'d':>7} {'disc?':>7}")
    print("-"*54)
    filas=[]
    for w in [0.0, 0.10, 0.15, 0.20, 0.30]:
        mix = (1-w)*df["DIS"] + w*df["_y7n"]
        a = mix[df.corpus=="A"].dropna(); b = mix[df.corpus=="B"].dropna()
        p = mannwhitneyu(a,b,alternative="two-sided")[1]; d = cohen_d(a,b)
        disc = "SI" if p<0.05 else "no"
        etiq = "DIS auditado" if w==0 else f"w={w}"
        print(f"{etiq:<14} {a.mean():>7.3f} {b.mean():>7.3f} {p:>9.4f} {d:>+7.3f} {disc:>7}")
        filas.append((w,p,d))

    print("\n" + "="*66); print("CONCLUSION"); print("="*66)
    # que pasa a partir de que w discrimina
    disc_ws = [w for w,p,d in filas if w>0 and p<0.05]
    if not disc_ws:
        print("  Con ningun peso razonable de y7 el DIS pasa a discriminar A vs B.")
        print("  >> CONFIRMAR y7 como DESCRIPTIVO. No tocar la auditoria.")
    else:
        wmin = min(disc_ws)
        print(f"  El DIS pasaria a discriminar (p<0.05) a partir de w={wmin} (peso de y7).")
        print("  Interpretacion:")
        print("  - y7 SI aporta senal discriminante A vs B (coherente con el gradiente).")
        print("  - PERO integrarlo cambia la narrativa: el DIS dejaria de ser transversal,")
        print("    lo que afecta el hallazgo de disociacion DIS-IEI (la brecha era epistemica).")
        print("  - Ademas y7 solo A-B da p=0.057 (limitrofe); su aporte al DIS depende del peso.")
        print("  >> DECISION DE MIREYA: descriptivo (preserva narrativa) vs integrar (gana")
        print("     discriminacion pero obliga a re-auditar y reescribir la disociacion).")

    print(f"\n  Reporte -> {OUT}")
    print("  (Ensayo NO destructivo: no se modifico ningun archivo auditado.)")
    sys.stdout = sys.__stdout__; fh.close()


if __name__ == "__main__":
    main()
