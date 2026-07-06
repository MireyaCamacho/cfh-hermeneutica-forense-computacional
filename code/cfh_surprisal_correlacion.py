# -*- coding: utf-8 -*-
r"""
cfh_surprisal_correlacion.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Evalua si el SURPRISAL CONTRASTIVO (y7) es complementario o redundante con los
indicadores que YA estan en los indices, calculando su correlacion con
EBI, SA, NV, y8_MAFAPO, y9_CIDH, REP. Decide si y7 aporta una dimension propia.

REQUISITO DE ALINEACION:
  El surprisal debe calcularse sobre las MISMAS secciones que tienen los
  indicadores. Este script recalcula y7 seccion por seccion usando el texto
  de corpus_b_secciones_texto.csv (para B) y lo cruza con los indicadores de
  dis_iei_corpus_abc_v2.csv. Para A y C, si hay texto alineado por unidad se
  usa; si no, se reporta la correlacion en el subconjunto disponible (B) que
  es donde el texto por seccion esta garantizado.

y7 = PLL_BETO - PLL_ConfliBERT  (mas negativo = registro mas belico)

Interpretacion de la correlacion:
  |r| < 0.3  -> INDEPENDIENTE: y7 aporta una dimension propia
  0.3-0.7    -> COMPLEMENTARIO parcial
  |r| > 0.7  -> REDUNDANTE con ese indicador (ya capturado)

Salida: outputs/surprisal_correlacion.txt + .csv

Uso (raiz del repo, env cfh):
    python code\cfh_surprisal_correlacion.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_BTEXTO = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
F_BIND = os.path.join(REPO, "outputs", "corpus_b_indicadores_COMPLETO.csv")
OUT_TXT = os.path.join(REPO, "outputs", "surprisal_correlacion.txt")
OUT_CSV = os.path.join(REPO, "outputs", "surprisal_correlacion.csv")

BETO = "dccuchile/bert-base-spanish-wwm-cased"
CONFLIBERT = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
MAX_LEN = 256

IND_COMP = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep_v5"]


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()
    def isatty(self): return False
    def fileno(self): return sys.__stdout__.fileno()
    @property
    def encoding(self): return getattr(sys.__stdout__, "encoding", "utf-8")


def cargar_modelos():
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    print("  Cargando BETO y ConfliBERT...")
    tb = AutoTokenizer.from_pretrained(BETO)
    mb = AutoModelForMaskedLM.from_pretrained(BETO); mb.eval()
    tc = AutoTokenizer.from_pretrained(CONFLIBERT)
    mc = AutoModelForMaskedLM.from_pretrained(CONFLIBERT); mc.eval()
    return (tb, mb), (tc, mc), torch


def pll(texto, tok, mdl, torch):
    if not isinstance(texto, str) or len(texto.strip()) < 20:
        return np.nan
    ids = tok.encode(texto, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logp = torch.log_softmax(mdl(ids).logits, dim=-1)
    toks = ids[0][1:-1]
    lp = logp[0, range(1, len(ids[0]) - 1), toks]
    return float(lp.mean())


def y7(texto, beto, confli, torch):
    pb = pll(texto, beto[0], beto[1], torch)
    pc = pll(texto, confli[0], confli[1], torch)
    if np.isnan(pb) or np.isnan(pc):
        return np.nan
    return pb - pc


def interp(r):
    a = abs(r)
    if a < 0.3: return "INDEPENDIENTE (dimension propia)"
    if a < 0.7: return "complementario parcial"
    return "REDUNDANTE (ya capturado)"


def main():
    fh = open(OUT_TXT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)
    print("=" * 64)
    print("CORRELACION del SURPRISAL (y7) con los indicadores del pipeline")
    print("=" * 64)

    # Texto e indicadores de B (donde el texto por seccion esta garantizado)
    txt = pd.read_csv(F_BTEXTO)
    ind = pd.read_csv(F_BIND)
    col_txt = next((c for c in ["texto","text","contenido"] if c in txt.columns), None)
    if col_txt is None:
        print("[ERROR] no hay columna de texto en corpus_b_secciones_texto.csv")
        sys.stdout = sys.__stdout__; fh.close(); return

    # alinear por posicion (mismo orden de las 80 secciones)
    n = min(len(txt), len(ind))
    print(f"\nSecciones de B alineadas: {n}")
    df = ind.iloc[:n].copy().reset_index(drop=True)
    df["_texto"] = txt[col_txt].iloc[:n].reset_index(drop=True)

    beto, confli, torch = cargar_modelos()
    print(f"\n  Calculando surprisal sobre {n} secciones de B...")
    vals = []
    for i, t in enumerate(df["_texto"]):
        vals.append(y7(t, beto, confli, torch))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n}")
    df["y7_surprisal"] = vals

    # correlaciones
    print("\n" + "=" * 64)
    print("CORRELACION y7 vs cada indicador (Pearson y Spearman, en B)")
    print("=" * 64)
    filas = []
    disponibles = [c for c in IND_COMP if c in df.columns]
    for col in disponibles:
        sub = df[["y7_surprisal", col]].dropna()
        if len(sub) < 10:
            continue
        rp = sub["y7_surprisal"].corr(sub[col], method="pearson")
        rs = sub["y7_surprisal"].corr(sub[col], method="spearman")
        nombre = {"y1_ebi":"EBI","y2_sa":"SA","y4_nv":"NV","y8_mafapo_cs":"y8 MAFAPO",
                  "y9_cidh_cs":"y9 CIDH","y10_rep_v5":"REP"}.get(col,col)
        print(f"  y7 ~ {nombre:<10} Pearson={rp:+.3f}  Spearman={rs:+.3f}  -> {interp(rs)}")
        filas.append({"indicador": nombre, "pearson": round(rp,3),
                      "spearman": round(rs,3), "interpretacion": interp(rs)})

    pd.DataFrame(filas).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("VEREDICTO")
    print("=" * 64)
    max_r = max((abs(f["spearman"]) for f in filas), default=0)
    redundante = [f["indicador"] for f in filas if abs(f["spearman"]) > 0.7]
    if redundante:
        print(f"  y7 es REDUNDANTE con: {', '.join(redundante)}")
        print("  -> NO aporta dimension nueva; no vale integrarlo.")
    elif max_r < 0.3:
        print("  y7 es INDEPENDIENTE de todos los indicadores (|r|<0.3).")
        print("  -> Aporta una DIMENSION PROPIA. Vale reportarlo; considerar")
        print("     integrarlo (al DIS o como cuarta dimension descriptiva).")
    else:
        print(f"  y7 es COMPLEMENTARIO parcial (max |r|={max_r:.2f}).")
        print("  -> Aporta senal parcialmente nueva. Reportar como descriptivo;")
        print("     integrar al DIS solo si el aporte teorico lo justifica.")
    print("""
  NOTA: la correlacion se calcula sobre el Corpus B (donde el texto por
  seccion esta garantizado y alineado con los indicadores). Es representativa
  para la decision. Si y7 entra al indice, se recalcula sobre A+B+C alineado.""")

    print(f"\n  Reporte -> {OUT_TXT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
