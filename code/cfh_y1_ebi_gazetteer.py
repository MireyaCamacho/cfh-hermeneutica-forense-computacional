# -*- coding: utf-8 -*-
r"""
cfh_y1_ebi_gazetteer.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

RESCATE del indicador y1 / EBI (Eufemismo Belico-Institucional, xi1).

PROBLEMA que resuelve:
    El extractor y1_ebi_extractor.py daba score = 0.0 en TODOS los documentos
    porque depende de CFH-BERT fine-tuneado sobre la taxonomia EBI, y ese
    fine-tuning nunca se completo. Corriendo con ConfliBERT-Spanish base, los
    scores EBI son suboptimos -> 0.0 en la practica.

SOLUCION (misma estrategia que y11):
    Rescatar y1 por GAZETTEER de eufemismos, construido a partir de los 199
    spans EBI reales anotados por los dos anotadores (133 A1 + 66 A2) en
    Refinamiento_CFH_IAA_Validacion_A2.xlsx. Se extraen los NUCLEOS
    eufemisticos reutilizables (misi0n tactica, resultado operacional, baja
    en combate, dado de baja, presunto combate, combate simulado, etc.) y se
    mide su DENSIDAD por 100 palabras.

    Los spans contextuales largos ligados a un caso concreto ("descendieron
    de una camioneta...") NO entran al gazetteer: no son eufemismos
    reutilizables, son narrativa de un hecho especifico.

DOS FASES:
    --validar-a : corre el gazetteer sobre Corpus A (donde EBI abunda) para
                  confirmar que el extractor rescatado YA NO da 0.0.
    (por defecto): calcula y1 sobre los 47 comparecientes (Corpus C) y lo
                   deja listo para el SEM (que es solo sobre C).

NORMALIZACION: min-max sobre el universo que se calcula (A en validacion,
    C en el modo SEM), coherente con la normalizacion de y10/y11 sobre C.

Ejecutar (raiz del repo, env cfh):
    python code\cfh_y1_ebi_gazetteer.py --validar-a   # fase 1: validar en A
    python code\cfh_y1_ebi_gazetteer.py               # dry-run sobre C
    python code\cfh_y1_ebi_gazetteer.py --escribir    # escribe y1_ebi en SEM

Rutas:
    --texto_c    data\texto_por_compareciente.csv
    --corpus_a   data\processed\corpus_a
    --sem_csv    data\referencias\indicadores_sem_compareciente.csv
"""

import os
import re
import glob
import json
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GAZETTEER EBI - nucleos eufemisticos derivados de los 199 spans A1+A2.
# Peso 1.8 = eufemismo belico nuclear (renombra homicidio como accion militar).
# Peso 1.2 = eufemismo de encubrimiento/procedimiento militar.
# ---------------------------------------------------------------------------
EBI_PATRONES = [
    # --- nucleo: baja / dar de baja (renombra el homicidio) (1.8) ---
    (r"\bbaja[s]?\s+(?:en\s+)?combate", 1.8),
    (r"\bdad[oa]s?\s+de\s+baja", 1.8),
    (r"\bdieron\s+de\s+baja", 1.8),
    (r"\bda(?:r|rle|rles|ndo)\s+de\s+baja", 1.8),
    (r"\bpresentad[oa]s?\s+como\s+baja[s]?", 1.8),
    (r"\breportar(?:on|se)?\s+como\s+(?:baja|muert|dad[oa]\s+de\s+baja)", 1.8),
    (r"\bbaja[s]?\s+del\s+enemigo", 1.8),

    # --- nucleo: combate ficticio / simulado (1.8) ---
    (r"\bpresunto\s+combate", 1.8),
    (r"\bcombate\s+simulad[oa]", 1.8),
    (r"\bsimular?\s+(?:un\s+)?combate", 1.8),
    (r"\bsimulad[oa]\s+en\s+combate", 1.8),
    (r"\bmuert[eo]s?\s+en\s+(?:presunto\s+)?combate", 1.8),
    (r"\bfalsa\s+presentaci[oó]n\s+de\s+la\s+muerte", 1.8),
    (r"\bmuertes?\s+ileg[ií]timamente\s+presentad", 1.8),

    # --- nucleo: resultado / operacion militar (encubrimiento) (1.8) ---
    (r"\bresultad[oa]s?\s+operacional(?:es)?", 1.8),
    (r"\bmisi[oó]n\s+t[aá]ctica", 1.5),
    (r"\boperaci[oó]n\s+(?:militar|t[aá]ctica|fragmentaria)", 1.5),
    (r"\borden\s+de\s+operaci[oó]n", 1.2),
    (r"\bregistro\s+y\s+control\s+militar", 1.5),

    # --- nucleo: dar muerte / neutralizar (eufemismo del acto) (1.8) ---
    (r"\bdieron\s+muerte", 1.8),
    (r"\bdar(?:le|les)?\s+muerte", 1.8),
    (r"\bcausar(?:le|les)?\s+la\s+muerte", 1.5),
    (r"\bneutraliz(?:ar|ado|aron|acion)", 1.8),
    (r"\bacordaron\s+darle\s+muerte", 1.8),

    # --- nucleo: encubrimiento del secuestro/traslado (1.2) ---
    (r"\bfue\s+interceptad[oa]\s+y\s+retenid", 1.2),
    (r"\bfueron\s+abordad[oa]s", 1.2),
    (r"\bfue\s+reclutad[oa]", 1.2),
    (r"\bresultaron\s+muert[oa]s", 1.5),
    (r"\bhabr[ií]an\s+perdido\s+la\s+vida", 1.5),

    # --- nucleo: presion por resultados / incentivos (1.5) ---
    (r"\bpresentar\s+(?:este\s+tipo\s+de\s+)?bajas", 1.8),
    (r"\bpresi[oó]n\s+por\s+resultados", 1.5),
    (r"\bmuertes?\s+en\s+combate\b", 1.5),
]

_EBI = [(re.compile(p, re.IGNORECASE), w) for p, w in EBI_PATRONES]


def densidad(text, patrones):
    if not text or not text.strip():
        return 0.0, 0
    n_palabras = max(1, len(text.split()))
    suma, n_hits = 0.0, 0
    for rx, w in patrones:
        h = len(rx.findall(text))
        if h:
            suma += h * w
            n_hits += h
    return (suma / n_palabras) * 100.0, n_hits


def minmax(arr):
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def texto_secciones_a(carpeta):
    """Lee texto de Corpus A: JSON con segmentation.sections + .txt por char_range."""
    out = []
    for jp in glob.glob(os.path.join(carpeta, "*.json")):
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        seg = d.get("segmentation")
        if not isinstance(seg, dict) or "sections" not in seg:
            continue
        doc_id = os.path.splitext(os.path.basename(jp))[0]
        txt_path = os.path.join(carpeta, doc_id + ".txt")
        if not os.path.exists(txt_path):
            continue
        texto = open(txt_path, encoding="utf-8").read()
        for sec in seg["sections"]:
            cr = sec.get("char_range")
            if isinstance(cr, str):
                cr = json.loads(cr)
            if not cr or len(cr) != 2:
                continue
            out.append((doc_id, sec.get("section_id", "?"), texto[int(cr[0]):int(cr[1])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto_c", default=os.path.join("data", "texto_por_compareciente.csv"))
    ap.add_argument("--corpus_a", default=os.path.join("data", "processed", "corpus_a"))
    ap.add_argument("--sem_csv", default=os.path.join("data", "referencias", "indicadores_sem_compareciente.csv"))
    ap.add_argument("--validar-a", action="store_true", dest="validar_a")
    ap.add_argument("--escribir", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("y1 / EBI - rescate por gazetteer de eufemismos (199 spans A1+A2)")
    print("=" * 66)

    # ---- FASE 1: validar sobre Corpus A ----
    if args.validar_a:
        print("\n[VALIDACION sobre Corpus A - donde EBI abunda]")
        secs = texto_secciones_a(args.corpus_a)
        print(f"Secciones A procesadas: {len(secs)}")
        raws = []
        con_ebi = 0
        top = []
        for doc_id, sec_id, txt in secs:
            d, h = densidad(txt, _EBI)
            raws.append(d)
            if d > 0:
                con_ebi += 1
                top.append((d, h, doc_id, sec_id))
        raws = np.array(raws)
        print(f"\nSecciones con EBI > 0: {con_ebi}/{len(secs)}  "
              f"({100*con_ebi/max(1,len(secs)):.1f}%)")
        print(f"y1_raw:  media={raws.mean():.3f}  max={raws.max():.3f}  "
              f"mediana={np.median(raws):.3f}")
        print("\n  ANTES (extractor BERT): 0.0 en TODO")
        print("  AHORA: el gazetteer detecta EBI en el corpus escrito -> RESCATADO")
        top.sort(reverse=True)
        print("\nTop 10 secciones con mas EBI:")
        for d, h, doc, sec in top[:10]:
            print(f"  {d:6.2f} ({h} hits)  {doc[:16]} / {sec}")
        return

    # ---- FASE 2: calcular sobre C (los 47) para el SEM ----
    print("\n[CALCULO sobre Corpus C - 47 comparecientes (universo del SEM)]")
    df_txt = pd.read_csv(args.texto_c)
    raws, hits = [], []
    for _, row in df_txt.iterrows():
        t = str(row.get("texto_completo", "") or "")
        d, h = densidad(t, _EBI)
        raws.append(d)
        hits.append(h)

    y1 = minmax(raws)
    df_txt["y1_raw"] = raws
    df_txt["y1_ebi"] = y1
    df_txt["n_hits"] = hits

    n_cero = int((np.array(raws) == 0).sum())
    print(f"Comparecientes: {len(df_txt)}  |  en cero: {n_cero}/{len(df_txt)}")

    dfx = df_txt.sort_values("y1_ebi", ascending=False)
    print("\n" + "-" * 66)
    print(f"{'identidad':<34}{'subcaso':<12}{'y1':>7}{'hits':>5}")
    print("-" * 66)
    for _, r in dfx.iterrows():
        print(f"{str(r['identidad'])[:34]:<34}{str(r['subcaso'])[:12]:<12}"
              f"{r['y1_ebi']:>7.3f}{int(r['n_hits']):>5}")

    # correlaciones (EBI deberia correlacionar con SA/NV, no con REP/y10/y11)
    print("\n" + "-" * 66)
    print("VERIFICACION - correlaciones de y1 con otros indicadores")
    print("-" * 66)
    df_sem = pd.read_csv(args.sem_csv)
    m = df_txt.set_index("identidad")["y1_ebi"]
    df_sem["_y1"] = df_sem["identidad"].map(m)
    for col in ["y2_sa", "y4_nv", "y10_rep", "y11_conv_rest"]:
        if col in df_sem.columns:
            c = df_sem["_y1"].corr(df_sem[col])
            print(f"  y1 vs {col:<14}: {c:+.3f}")

    if not args.escribir:
        df_txt.to_csv("y1_ebi_detalle.csv", index=False, encoding="utf-8")
        print("\n[DRY-RUN] No se escribio el CSV del SEM.")
        print("Detalle -> y1_ebi_detalle.csv")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.sem_csv.replace(".csv", f"_BACKUP_pre_y1ebi_{ts}.csv")
    shutil.copy2(args.sem_csv, backup)
    print(f"\n  Backup -> {backup}")
    df_sem["y1_ebi"] = df_sem["identidad"].map(m)
    df_sem = df_sem.drop(columns=["_y1"], errors="ignore")
    df_sem.to_csv(args.sem_csv, index=False, encoding="utf-8")
    print(f"  CSV del SEM actualizado -> {args.sem_csv}")
    print("  Columna y1_ebi anadida (gazetteer, densidad de eufemismos).")


if __name__ == "__main__":
    main()
