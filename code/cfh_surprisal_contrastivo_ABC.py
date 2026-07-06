# -*- coding: utf-8 -*-
r"""
cfh_surprisal_contrastivo_ABC.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Recalcula y7 (SURPRISAL CONTRASTIVO) sobre A, B fortalecido (80 secc) y C,
como DATO DESCRIPTIVO. No modifica ningun indice auditado.

DEFINICION (identica al calculo original de cfh_calcular_y7_surprisal.py):
    y7 = PLL_BETO - PLL_ConfliBERT   (por seccion)
    PLL = pseudo-log-likelihood promedio por token (masked language modeling).
    y7 < 0  => texto MAS predecible para ConfliBERT que para BETO
               (registro belico-institucional del conflicto).
    y7 > 0  => texto mas "civil"/general.

Modelos:
    BETO       : dccuchile/bert-base-spanish-wwm-cased
    ConfliBERT : eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1

FUENTES DE TEXTO:
    A: data/processed/corpus_a/{hash}.txt  (o el CSV con texto si existe)
    B: outputs/corpus_b_secciones_texto.csv (columna de texto)
    C: corpus_c/{audiencia}_segments.json  (campo 'text' concatenado por bloques)
       Audiencias canonicas: casanare_torres, catatumbo_audiencia_reconocimiento,
       costa_caribe, dabeiba_antioquia, huila.

SALIDA:
    outputs/surprisal_contrastivo_ABC.txt + .csv
    Reporta: media/std/rango por corpus, Mann-Whitney A vs B, Cohen's d,
    y correlacion surprisal<->EBI (para evaluar complementariedad, no redundancia).

Uso (raiz del repo, env cfh):
    python code\cfh_surprisal_contrastivo_ABC.py

NOTA: carga DOS modelos BERT; es pesado en CPU. Procesa corpus por corpus
y con muestreo configurable (MAX_POR_CORPUS) para no saturar RAM.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TXT = os.path.join(REPO, "outputs", "surprisal_contrastivo_ABC.txt")
OUT_CSV = os.path.join(REPO, "outputs", "surprisal_contrastivo_ABC.csv")

BETO = "dccuchile/bert-base-spanish-wwm-cased"
CONFLIBERT = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
MAX_LEN = 256
MAX_POR_CORPUS = None  # None = todos; pon un numero para muestrear

AUDIENCIAS_C = [
    "casanare_torres", "catatumbo_audiencia_reconocimiento",
    "costa_caribe", "dabeiba_antioquia", "huila",
]


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
    print("  Cargando BETO y ConfliBERT (puede tardar la 1a vez)...")
    tb = AutoTokenizer.from_pretrained(BETO)
    mb = AutoModelForMaskedLM.from_pretrained(BETO); mb.eval()
    tc = AutoTokenizer.from_pretrained(CONFLIBERT)
    mc = AutoModelForMaskedLM.from_pretrained(CONFLIBERT); mc.eval()
    return (tb, mb), (tc, mc), torch


def pll(texto, tok, mdl, torch):
    """Pseudo-log-likelihood promedio por token (aprox. eficiente: 1 pasada,
    log-prob del token real en cada posicion)."""
    if not isinstance(texto, str) or len(texto.strip()) < 20:
        return np.nan
    ids = tok.encode(texto, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(ids).logits
    logp = torch.log_softmax(logits, dim=-1)
    toks = ids[0][1:-1]
    lp = logp[0, range(1, len(ids[0]) - 1), toks]
    return float(lp.mean())


def y7_contrastivo(texto, beto, confli, torch):
    pb = pll(texto, beto[0], beto[1], torch)
    pc = pll(texto, confli[0], confli[1], torch)
    if np.isnan(pb) or np.isnan(pc):
        return np.nan
    return pb - pc  # PLL_BETO - PLL_ConfliBERT


def cargar_textos_A():
    # intentar CSV con texto; si no, processed/corpus_a/*.txt
    for cand in ["corpus_a_secciones_texto.csv"]:
        f = os.path.join(REPO, "outputs", cand)
        if os.path.exists(f):
            df = pd.read_csv(f)
            col = next((c for c in ["texto","text","contenido"] if c in df.columns), None)
            if col:
                return df[col].dropna().tolist()
    txts = []
    for f in glob.glob(os.path.join(REPO, "data", "processed", "corpus_a", "*.txt")):
        try:
            txts.append(open(f, encoding="utf-8").read())
        except Exception:
            pass
    return txts


def cargar_textos_B():
    f = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        col = next((c for c in ["texto","text","contenido"] if c in df.columns), None)
        if col:
            return df[col].dropna().tolist()
    return []


def cargar_textos_C():
    """Concatena el texto de cada audiencia canonica en bloques de ~2000 chars."""
    bloques = []
    for aud in AUDIENCIAS_C:
        f = os.path.join(REPO, "corpus_c", f"{aud}_segments.json")
        if not os.path.exists(f):
            print(f"    [falta] {aud}_segments.json")
            continue
        segs = json.load(open(f, encoding="utf-8"))
        texto = " ".join(s.get("text", "").strip() for s in segs if s.get("text"))
        # cortar en bloques de ~2000 chars
        for i in range(0, len(texto), 2000):
            b = texto[i:i+2000]
            if len(b) > 100:
                bloques.append(b)
    return bloques


def stats(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return {"n": len(v), "media": round(v.mean(),4), "std": round(v.std(),4),
            "min": round(v.min(),4), "max": round(v.max(),4)}, v


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
    return (x.mean()-y.mean())/(pooled+1e-12)


def main():
    fh = open(OUT_TXT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)
    print("="*64)
    print("SURPRISAL CONTRASTIVO (y7 = PLL_BETO - PLL_ConfliBERT) A/B/C")
    print("="*64)

    textos = {"A": cargar_textos_A(), "B": cargar_textos_B(), "C": cargar_textos_C()}
    for k,v in textos.items():
        print(f"  {k}: {len(v)} textos")
    if not any(textos.values()):
        print("\n[ERROR] no encontre textos. Revisar rutas."); sys.stdout=sys.__stdout__; fh.close(); return

    beto, confli, torch = cargar_modelos()

    resultados = {}
    filas = []
    for corpus, txts in textos.items():
        if not txts:
            print(f"\n  {corpus}: sin texto, omitido"); continue
        if MAX_POR_CORPUS:
            txts = txts[:MAX_POR_CORPUS]
        print(f"\n  Procesando {corpus} ({len(txts)} textos)...")
        vals = []
        for i, t in enumerate(txts):
            vals.append(y7_contrastivo(t, beto, confli, torch))
            if (i+1) % 50 == 0:
                print(f"    {i+1}/{len(txts)}")
        st, v = stats(vals)
        resultados[corpus] = v
        st["corpus"] = corpus
        filas.append(st)
        print(f"    {corpus}: media={st['media']:+.4f} std={st['std']:.4f} "
              f"rango[{st['min']:+.3f},{st['max']:+.3f}]")

    print("\n" + "="*64)
    print("RESULTADO")
    print("="*64)
    dfres = pd.DataFrame(filas)[["corpus","n","media","std","min","max"]]
    print(dfres.to_string(index=False))

    # Mann-Whitney A vs B + Cohen's d
    if "A" in resultados and "B" in resultados:
        from scipy.stats import mannwhitneyu
        a, b = resultados["A"], resultados["B"]
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        d = cohen_d(a, b)
        print(f"\n  A vs B: p={p:.4f}  Cohen's d={d:+.3f}")
        print(f"  {'B mas belico que A' if b.mean()<a.mean() else 'A mas belico que B'} "
              f"(mas negativo = mas belico)")

    # correlacion surprisal <-> EBI (complementariedad)
    try:
        base = pd.read_csv(os.path.join(REPO,"outputs","dis_iei_corpus_abc_v2.csv"))
        # solo se puede si alineamos; se reporta a nivel corpus como referencia
        print("\n  Nota: para correlacion surprisal<->EBI a nivel seccion se")
        print("  requiere alinear por seccion (pendiente si y7 entra al indice).")
    except Exception:
        pass

    dfres.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Reporte -> {OUT_TXT}")
    print("""
  DECISION:
  - Si B es claramente mas belico que A (d apreciable) y el patron es
    teoricamente coherente -> reportar como dato descriptivo de Capa 2
    (triangula con EBI). Luego decidir si entra al DIS.
  - Si es plano o erratico -> dejar como indicador explorado, no integrar.""")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
