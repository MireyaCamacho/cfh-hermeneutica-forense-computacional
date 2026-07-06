# -*- coding: utf-8 -*-
r"""
cfh_ensayo_dis_surprisal.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

ENSAYO (no destructivo): calcula un DIS EXPERIMENTAL que incluye el surprisal
contrastivo (y7), SOLO para verificar si el DIS pasaria a discriminar A vs B.
NO sobrescribe ningun archivo auditado. Escribe solo un reporte nuevo.

Compara:
  DIS_actual (auditado)   = 0.40*EBI + 0.30*SA + 0.30*(1-REP)
  DIS_con_y7 (ensayo)     = varios esquemas que reparten peso hacia y7

Para cada esquema reporta media A / media B, Mann-Whitney p (A vs B) y Cohen's d.
CONCLUSION AUTOMATICA:
  - Si DIS_con_y7 sigue sin discriminar (p>0.05) -> confirmar DESCRIPTIVO
    (integrar y7 no aporta poder discriminante; no vale tocar la auditoria).
  - Si pasara a discriminar (p<0.05) -> reconsiderar (pero evaluar si rompe
    la disociacion DIS-IEI).

FUENTES (solo lectura):
  outputs/dis_iei_corpus_abc_v2.csv   (indicadores + z-scores auditados)
  outputs/y7_surprisal_A_alineado.csv
  outputs/y7_surprisal_B_alineado.csv
  outputs/y7_surprisal_C_alineado.csv

SALIDA (nueva, no pisa nada):
  outputs/ENSAYO_dis_surprisal_reporte.txt

Uso (raiz del repo, env cfh):
    python code\cfh_ensayo_dis_surprisal.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_Y7 = {
    "A": os.path.join(REPO, "outputs", "y7_surprisal_A_alineado.csv"),
    "B": os.path.join(REPO, "outputs", "y7_surprisal_B_alineado.csv"),
    "C": os.path.join(REPO, "outputs", "y7_surprisal_C_alineado.csv"),
}
OUT = os.path.join(REPO, "outputs", "ENSAYO_dis_surprisal_reporte.txt")


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()
    def isatty(self): return False
    def fileno(self): return sys.__stdout__.fileno()
    @property
    def encoding(self): return getattr(sys.__stdout__, "encoding", "utf-8")


def zscore_sigmoide(serie):
    """Normalizacion identica al pipeline: z-score sobre distribucion conjunta + sigmoide."""
    z = (serie - serie.mean()) / (serie.std(ddof=0) + 1e-12)
    return 1.0 / (1.0 + np.exp(-z))


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
    return (x.mean()-y.mean())/(pooled+1e-12)


def mw(a, b):
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        p = np.nan
    return p


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)
    print("=" * 66)
    print("ENSAYO: DIS con surprisal (y7) - NO destructivo")
    print("=" * 66)

    df = pd.read_csv(F_ABC)
    print(f"Filas base (auditado): {len(df)}  columnas y1/y2/y10: "
          f"{[c for c in df.columns if c in ('y1_ebi','y2_sa','y10_rep')]}")

    # cargar y7 alineado por corpus y concatenar en el mismo orden que df
    # el CSV abc no tiene subcaso pero si 'corpus'; alineamos por corpus + orden
    y7_frames = []
    for c in ["A", "B", "C"]:
        if not os.path.exists(F_Y7[c]):
            print(f"[ERROR] falta {F_Y7[c]}"); sys.stdout=sys.__stdout__; fh.close(); return
        d = pd.read_csv(F_Y7[c])
        col = "y7_surprisal" if "y7_surprisal" in d.columns else d.columns[-1]
        y7_frames.append(pd.DataFrame({"corpus": c, "y7": d[col].values}))
    y7all = pd.concat(y7_frames, ignore_index=True)
    print(f"y7 alineado: A={sum(y7all.corpus=='A')}, B={sum(y7all.corpus=='B')}, C={sum(y7all.corpus=='C')}")

    # Verificar que los conteos calzan con df por corpus
    for c in ["A","B","C"]:
        n_df = sum(df["corpus"]==c); n_y7 = sum(y7all["corpus"]==c)
        estado = "OK" if n_df==n_y7 else f"DESALINEADO (df={n_df}, y7={n_y7})"
        print(f"  {c}: {estado}")

    # normalizar y7 (invertir signo: mas belico = mas negativo -> mayor injusticia discursiva)
    # y7 negativo = mas belico. Para que 'mas injusticia' sea valor alto, usamos (-y7).
    y7all["y7_inv"] = -y7all["y7"]
    y7all["y7_norm"] = zscore_sigmoide(y7all["y7_inv"])

    # Reconstruir componentes normalizados del DIS actual desde el CSV auditado
    # (usar las columnas _z si existen, si no normalizar aqui igual que el pipeline)
    def norm_col(name):
        zc = name + "_z"
        if zc in df.columns:
            return 1.0/(1.0+np.exp(-df[zc]))
        return zscore_sigmoide(df[name])

    ebi_n = norm_col("y1_ebi")
    sa_n = norm_col("y2_sa")
    rep_n = norm_col("y10_rep")

    # DIS actual auditado
    dis_actual = 0.40*ebi_n + 0.30*sa_n + 0.30*(1-rep_n)

    # asignar y7_norm alineado (mismo orden por corpus)
    df = df.reset_index(drop=True)
    y7_norm_aligned = np.full(len(df), np.nan)
    for c in ["A","B","C"]:
        idx_df = df.index[df["corpus"]==c].tolist()
        vals = y7all.loc[y7all["corpus"]==c, "y7_norm"].values
        for k, ii in enumerate(idx_df):
            if k < len(vals):
                y7_norm_aligned[ii] = vals[k]
    df["_y7n"] = y7_norm_aligned
    df["_dis_actual"] = dis_actual.values

    # Esquemas de DIS con y7
    esquemas = {
        "DIS_actual (sin y7)":      None,
        "DIS+y7 (0.30/0.25/0.25/0.20)": (0.30,0.25,0.25,0.20),
        "DIS+y7 (0.35/0.25/0.20/0.20)": (0.35,0.25,0.20,0.20),
        "DIS+y7 (0.25/0.25/0.25/0.25)": (0.25,0.25,0.25,0.25),
    }

    print("\n" + "="*66)
    print("COMPARACION: DIS actual vs DIS con y7 (A vs B)")
    print("="*66)
    print(f"{'Esquema':<32} {'A':>7} {'B':>7} {'p(A-B)':>9} {'d':>7}")
    print("-"*66)

    A_mask = df["corpus"]=="A"; B_mask = df["corpus"]=="B"
    resultados = []
    for nombre, pesos in esquemas.items():
        if pesos is None:
            dis = df["_dis_actual"]
        else:
            w_ebi,w_sa,w_rep,w_y7 = pesos
            dis = w_ebi*ebi_n + w_sa*sa_n + w_rep*(1-rep_n) + w_y7*df["_y7n"]
        a = dis[A_mask].dropna(); b = dis[B_mask].dropna()
        p = mw(a,b); d = cohen_d(a,b)
        print(f"{nombre:<32} {a.mean():>7.3f} {b.mean():>7.3f} {p:>9.4f} {d:>+7.3f}")
        resultados.append((nombre, p))

    print("\n" + "="*66)
    print("CONCLUSION")
    print("="*66)
    # el actual
    p_actual = [p for n,p in resultados if "sin y7" in n][0]
    p_con = [p for n,p in resultados if "sin y7" not in n]
    todos_ns = all((pd.isna(p) or p>0.05) for p in p_con)
    print(f"  DIS actual (auditado): p={p_actual:.4f} -> {'no discrimina' if p_actual>0.05 else 'DISCRIMINA'}")
    if todos_ns:
        print("  DIS + y7 (todos los esquemas): sigue SIN discriminar A vs B (p>0.05).")
        print("  >> Integrar y7 al DIS NO aporta poder discriminante.")
        print("  >> CONFIRMAR y7 como DESCRIPTIVO. No tocar la auditoria.")
    else:
        disc = [n for n,p in zip([n for n in esquemas if 'sin y7' not in n], p_con) if not pd.isna(p) and p<0.05]
        print(f"  DIS + y7 pasaria a discriminar en: {disc}")
        print("  >> Evaluar con cuidado: podria cambiar la narrativa de disociacion DIS-IEI.")

    print(f"\n  Reporte -> {OUT}")
    print("  (Este ensayo NO modifico ningun archivo auditado.)")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
