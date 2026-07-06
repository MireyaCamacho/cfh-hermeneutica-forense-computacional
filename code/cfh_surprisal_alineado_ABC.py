# -*- coding: utf-8 -*-
r"""
cfh_surprisal_alineado_ABC.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Calcula el SURPRISAL CONTRASTIVO (y7 = PLL_BETO - PLL_ConfliBERT) ALINEADO
sobre las 946 unidades reales del pipeline:
  A: 819 secciones (via JSON de segmentacion en data/processed/corpus_a/)
  B:  80 secciones (outputs/corpus_b_secciones_texto.csv)
  C:  47 comparecientes (data/texto_por_compareciente.csv, marcacion manual)

Los valores quedan alineados por unidad, listos para (a) reportar como
descriptivo o (b) evaluar su integracion al DIS. GUARDADO INCREMENTAL: escribe
un CSV por corpus a medida que termina, para no perder avance si se interrumpe.

y7 < 0 = registro mas belico (mas predecible para ConfliBERT que para BETO).

Salida:
  outputs/y7_surprisal_A_alineado.csv
  outputs/y7_surprisal_B_alineado.csv
  outputs/y7_surprisal_C_alineado.csv
  outputs/surprisal_alineado_resumen.txt

Uso (raiz del repo, env cfh):
    python code\cfh_surprisal_alineado_ABC.py
    # opcional, un corpus a la vez:
    python code\cfh_surprisal_alineado_ABC.py A
    python code\cfh_surprisal_alineado_ABC.py C
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_A = os.path.join(REPO, "data", "processed", "corpus_a")
F_BTEXTO = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
F_C = os.path.join(REPO, "data", "texto_por_compareciente.csv")
OUT_A = os.path.join(REPO, "outputs", "y7_surprisal_A_alineado.csv")
OUT_B = os.path.join(REPO, "outputs", "y7_surprisal_B_alineado.csv")
OUT_C = os.path.join(REPO, "outputs", "y7_surprisal_C_alineado.csv")
OUT_RES = os.path.join(REPO, "outputs", "surprisal_alineado_resumen.txt")

BETO = "dccuchile/bert-base-spanish-wwm-cased"
CONFLIBERT = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
MAX_LEN = 256


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
    # para textos largos (comparecientes), promediar por ventanas de MAX_LEN
    if not isinstance(texto, str) or len(texto.strip()) < 20:
        return np.nan
    palabras = texto.split()
    if len(palabras) <= MAX_LEN:
        pb = pll(texto, beto[0], beto[1], torch)
        pc = pll(texto, confli[0], confli[1], torch)
    else:
        # ventanas de ~200 palabras, promediar
        vb, vc = [], []
        for i in range(0, len(palabras), 200):
            chunk = " ".join(palabras[i:i+200])
            b = pll(chunk, beto[0], beto[1], torch)
            c = pll(chunk, confli[0], confli[1], torch)
            if not np.isnan(b): vb.append(b)
            if not np.isnan(c): vc.append(c)
        pb = np.mean(vb) if vb else np.nan
        pc = np.mean(vc) if vc else np.nan
    if np.isnan(pb) or np.isnan(pc):
        return np.nan
    return pb - pc


def limpiar(t):
    return t if isinstance(t, str) else ""


def secciones_A():
    """Extrae secciones target de A usando los JSON de segmentacion."""
    filas = []
    for jf in glob.glob(os.path.join(DIR_A, "*.json")):
        base = jf[:-5]
        tf = base + ".txt"
        if not os.path.exists(tf):
            continue
        try:
            meta = json.load(open(jf, encoding="utf-8"))
            texto = open(tf, encoding="utf-8").read()
        except Exception:
            continue
        doc_id = os.path.basename(base)
        segs = meta.get("segmentation", {}).get("sections", []) if isinstance(meta, dict) else []
        for si, s in enumerate(segs):
            if not s.get("is_target", False):
                continue
            cr = s.get("char_range")
            if cr and len(cr) == 2:
                frag = texto[cr[0]:cr[1]]
            else:
                frag = ""
            if len(frag.strip()) >= 40:
                filas.append({"doc_id": doc_id, "section_id": si,
                              "section": s.get("type", ""), "_texto": frag})
    return pd.DataFrame(filas)


def procesar(df, col_texto, out_csv, beto, confli, torch, etiqueta):
    print(f"\n  {etiqueta}: {len(df)} unidades")
    vals = []
    for i, t in enumerate(df[col_texto]):
        vals.append(y7(limpiar(t), beto, confli, torch))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(df)}")
            # guardado incremental parcial
            tmp = df.iloc[:i+1].copy()
            tmp["y7_surprisal"] = vals
            tmp.drop(columns=[col_texto]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    df = df.copy()
    df["y7_surprisal"] = vals
    df.drop(columns=[col_texto]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    v = np.array([x for x in vals if not np.isnan(x)])
    print(f"    {etiqueta}: media={v.mean():+.4f} std={v.std():.4f} "
          f"rango[{v.min():+.3f},{v.max():+.3f}]  -> {out_csv}")
    return v


def main():
    solo = sys.argv[1].upper() if len(sys.argv) > 1 else None
    beto, confli, torch = cargar_modelos()
    resumen = {}

    if solo in (None, "A"):
        dfa = secciones_A()
        if len(dfa):
            resumen["A"] = procesar(dfa, "_texto", OUT_A, beto, confli, torch, "A")
        else:
            print("  [A] no se extrajeron secciones (revisar JSON de segmentacion)")

    if solo in (None, "B"):
        b = pd.read_csv(F_BTEXTO)
        col = next((c for c in ["texto","text","contenido"] if c in b.columns), None)
        if col:
            b = b.rename(columns={col: "_texto"})
            resumen["B"] = procesar(b[["_texto"]], "_texto", OUT_B, beto, confli, torch, "B")

    if solo in (None, "C"):
        c = pd.read_csv(F_C)
        c = c.rename(columns={"texto_completo": "_texto"})
        resumen["C"] = procesar(c[["subcaso","identidad","_texto"]], "_texto", OUT_C,
                                beto, confli, torch, "C")

    # resumen
    lines = ["="*60, "SURPRISAL CONTRASTIVO ALINEADO - resumen por corpus", "="*60]
    for k in ["A", "B", "C"]:
        if k in resumen:
            v = resumen[k]
            lines.append(f"  {k}: n={len(v)}  media={v.mean():+.4f}  std={v.std():.4f}")
    if "A" in resumen and "B" in resumen:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(resumen["A"], resumen["B"], alternative="two-sided")
        lines.append(f"\n  A vs B: p={p:.4f}")
    lines.append("\n  y7<0 = registro mas belico. Gradiente esperado A->B->C.")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(OUT_RES, "w", encoding="utf-8").write(txt)
    print(f"\n  Resumen -> {OUT_RES}")


if __name__ == "__main__":
    main()
