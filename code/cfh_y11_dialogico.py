# -*- coding: utf-8 -*-
r"""
cfh_y11_dialogico.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

REDISENO FINAL del indicador y11 (Transicion Epistemica, eta2).

y11 = DENSIDAD DE ORIENTACION DIALOGICA / TESTIMONIAL hacia la victima.

Que mide (y como se distingue de los otros indicadores):
  - y8/y9  -> distancia semantica a MAFAPO/CIDH (geometria de embeddings).
  - y10    -> el ACTO reparador en si (reconozco, pido perdon, acepto
              responsabilidad) del compareciente.
  - y11    -> la ORIENTACION DIALOGICA: el gesto relacional de dirigirse a
              las victimas y reconocerlas como INTERLOCUTORAS epistemicas
              validas — dar la cara, hablarles directamente, reconocer que
              merecen la verdad, nombrar el dano que se les causo a ELLAS.
              Es la correccion de la injusticia TESTIMONIAL de Fricker: no
              el acto de reparar, sino el reconocimiento del otro como sujeto
              de conocimiento con derecho a ser escuchado y a saber.

Por que solo esta dimension:
  El componente restaurativo INSTITUCIONAL (TOAR, reparacion integral,
  garantias de no repeticion, SIVJRNR) NO aparece en el habla oral de los
  comparecientes — es vocabulario de los autos escritos JEP (Corpus B).
  El analisis de los 47 textos mostro que la senal restaurativa oral es
  dialogica/testimonial, no institucional. Se descarta y11a.

METODO: densidad = (marcadores ponderados / n_palabras) * 100.
NORMALIZACION: min-max sobre los 47 comparecientes (universo del SEM),
  coherente con la normalizacion de y10 sobre C.

Ejecutar (raiz del repo, env cfh):
  python code\cfh_y11_dialogico.py             # dry-run (por defecto)
  python code\cfh_y11_dialogico.py --escribir  # escribe y11_conv_rest en el SEM

Rutas:
  --texto_c   data\texto_por_compareciente.csv
  --sem_csv   data\referencias\indicadores_sem_compareciente.csv
"""

import os
import re
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# VOCABULARIO y11 - ORIENTACION DIALOGICA / TESTIMONIAL hacia la victima
# Derivado del habla oral REAL de los 47 comparecientes (Corpus C).
# Peso alto (1.8) = gesto dialogico fuerte y personal.
# Peso medio (1.2-1.5) = saludo/orientacion a la victima mas formular.
# ---------------------------------------------------------------------------
Y11_PATRONES = [
    # --- dar la cara / comparecer ante las victimas (gesto fuerte, 1.8) ---
    (r"\b(?:vengo|vine|estoy\s+aqu[ií]|hoy\s+estoy)\s+(?:a\s+)?"
     r"(?:dar(?:le|les)?\s+la\s+cara|responder(?:le|les)?|dar(?:le|les)?\s+"
     r"respuesta)", 1.8),
    (r"\bdar(?:le|les)?\s+la\s+cara\s+(?:a\s+)?(?:las?\s+v[ií]ctimas|"
     r"los?\s+familiares|ustedes|la\s+familia)", 1.8),
    (r"\bdarles?\s+la\s+cara\b", 1.6),

    # --- dirigirse directamente a las victimas como interlocutoras (1.8) ---
    (r"\b(?:a\s+)?ustedes\s+(?:las?\s+)?v[ií]ctimas", 1.8),
    (r"\bme\s+dirijo\s+a\s+(?:usted|ustedes|las?\s+v[ií]ctimas|la\s+familia|"
     r"los?\s+familiares)", 1.8),
    (r"\b(?:a\s+)?ustedes\s+(?:los?\s+)?familiares", 1.6),
    (r"\bdirigir(?:me|le|les)?\s+(?:un\s+saludo\s+)?(?:muy\s+)?"
     r"(?:respetuoso\s+|especial\s+)?a\s+(?:las?\s+v[ií]ctimas|ustedes)", 1.5),

    # --- saludo con reconocimiento explicito a las victimas (1.2) ---
    (r"\bsaludo\s+(?:muy\s+)?(?:especial|respetuoso|cordial)\s+"
     r"(?:para\s+|a\s+)(?:todas?\s+)?(?:las?\s+v[ií]ctimas|ustedes\s+"
     r"v[ií]ctimas)", 1.2),
    (r"\b(?:especialmente|principalmente|en\s+especial)\s+(?:a\s+|para\s+)"
     r"(?:las?\s+v[ií]ctimas|los?\s+familiares)", 1.2),
    (r"\bpor\s+tener\s+el\s+valor\s+de\s+estar\s+aqu[ií]", 1.8),
    (r"\bgracias\s+(?:a\s+)?(?:las?\s+v[ií]ctimas|ustedes)\s+por", 1.5),
    (r"\b(?:que\s+)?(?:nos\s+)?dan\s+la\s+oportunidad\s+(?:hoy\s+)?"
     r"de\s+(?:presentarles|contarles|darles)", 1.6),

    # --- nombrar el dano causado A ELLAS / a sus seres queridos (1.8) ---
    (r"\b(?:el\s+)?dolor\s+(?:tan\s+grande\s+)?que\s+(?:ustedes\s+|"
     r"les?\s+)?(?:tienen|sienten|cargan|viven)", 1.8),
    (r"\b(?:sus\s+)?seres\s+queridos", 1.2),
    (r"\bnosotros\s+(?:como\s+victimarios\s+)?(?:las?\s+|les?\s+|los?\s+)?"
     r"(?:arrebat\w*|quitamos|matamos|asesinamos)", 1.8),
    (r"\bcomo\s+victimarios\b", 1.5),
    (r"\bcontra\s+sus\s+(?:familiares|seres\s+queridos|hijos|madres)", 1.5),
    (r"\bel\s+dolor\s+(?:tan\s+grande\s+)?(?:que\s+)?"
     r"(?:tienen|cargan)\s+(?:todos\s+)?(?:los\s+)?(?:familiares|ustedes)", 1.8),
    (r"\bdarle(?:s)?\s+(?:un\s+poco\s+de\s+)?alivio", 1.6),

    # --- pedir perdon DIRIGIDO a persona/familia concreta (1.8) ---
    # (el acto de perdon lo capta y10; aqui capta la ORIENTACION al otro)
    (r"\b(?:le|les)\s+pido\s+perd[oó]n\s+(?:de\s+nuevo\s+)?a\s+"
     r"[A-ZÁÉÍÓÚÑ]", 1.8),
    (r"\bperd[oó]n\s+a\s+(?:la\s+familia\s+de\s+|los?\s+familiares\s+de\s+)"
     r"[A-ZÁÉÍÓÚÑ]", 1.8),

    # --- reconocer a la victima como quien merece/tiene derecho a la verdad (1.5) ---
    (r"\b(?:ustedes|las?\s+v[ií]ctimas)\s+(?:tienen\s+|merecen\s+)"
     r"(?:todo\s+el\s+)?derecho\s+a\s+(?:saber|la\s+verdad|conocer)", 1.5),
    (r"\bpresentarles\s+esta\s+verdad", 1.6),
    (r"\bcuando\s+ustedes\s+conocen\s+la\s+verdad", 1.5),
    (r"\ble(?:s)?\s+debo\s+(?:la\s+)?verdad", 1.5),
    (r"\bcon\s+(?:el\s+)?coraz[oó]n\s+arrugado", 1.5),
    (r"\b(?:presentar|comparecer)(?:me|nos)?\s+ante\s+ustedes", 1.5),
    (r"\bante\s+ustedes\s*,?\s+(?:ante\s+)?(?:las?\s+v[ií]ctimas|"
     r"colombia|mi\s+familia)", 1.4),
    (r"\bvengamos\s+a\s+contarles", 1.4),
]

_Y11 = [(re.compile(p, re.IGNORECASE), w) for p, w in Y11_PATRONES]


def densidad(text, patrones):
    if not text or not text.strip():
        return 0.0, 0
    n_palabras = max(1, len(text.split()))
    suma = 0.0
    n_hits = 0
    for rx, w in patrones:
        hits = len(rx.findall(text))
        if hits:
            suma += hits * w
            n_hits += hits
    return (suma / n_palabras) * 100.0, n_hits


def minmax(arr):
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto_c", default=os.path.join("data", "texto_por_compareciente.csv"))
    ap.add_argument("--sem_csv", default=os.path.join("data", "referencias", "indicadores_sem_compareciente.csv"))
    ap.add_argument("--escribir", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.texto_c):
        raise SystemExit(f"No encuentro {args.texto_c}")
    if not os.path.exists(args.sem_csv):
        raise SystemExit(f"No encuentro {args.sem_csv}")

    print("=" * 68)
    print("y11 REDISENADO - orientacion dialogica/testimonial hacia la victima")
    print("=" * 68)

    df_txt = pd.read_csv(args.texto_c)
    print(f"\nComparecientes: {len(df_txt)}")

    raws, hits = [], []
    for _, row in df_txt.iterrows():
        t = str(row.get("texto_completo", "") or "")
        d, h = densidad(t, _Y11)
        raws.append(d)
        hits.append(h)

    y11 = minmax(raws)
    df_txt["y11_raw"] = raws
    df_txt["y11_dialogico"] = y11
    df_txt["n_hits"] = hits

    n_cero = int((np.array(raws) == 0).sum())
    print(f"En cero: {n_cero}/{len(df_txt)}  (antes con y11a+b: 28)")

    # --- Reporte ordenado ---
    print("\n" + "-" * 68)
    dfx = df_txt.sort_values("y11_dialogico", ascending=False)
    print(f"{'identidad':<34}{'subcaso':<12}{'y11':>7}{'hits':>5}")
    print("-" * 68)
    for _, r in dfx.iterrows():
        print(f"{str(r['identidad'])[:34]:<34}{str(r['subcaso'])[:12]:<12}"
              f"{r['y11_dialogico']:>7.3f}{int(r['n_hits']):>5}")

    # --- Verificacion de correlaciones ---
    print("\n" + "-" * 68)
    print("VERIFICACION - correlacion del y11 nuevo con y8/y9/y10")
    print("-" * 68)
    df_sem = pd.read_csv(args.sem_csv)
    m = df_txt.set_index("identidad")["y11_dialogico"]
    df_sem["_y11"] = df_sem["identidad"].map(m)
    for col in ["y8_mafapo", "y9_cidh", "y10_rep"]:
        if col in df_sem.columns:
            c = df_sem["_y11"].corr(df_sem[col])
            print(f"  y11 vs {col:<12}: {c:+.3f}")
    print("\n  y11 viejo (centroide): y8=0.952 y9=0.972 -> duplicaba")
    print("  Objetivo: |corr| < ~0.6 con todos (no duplica y8/y9 ni y10)")

    if not args.escribir:
        df_txt.to_csv("y11_dialogico_detalle.csv", index=False, encoding="utf-8")
        print("\n[DRY-RUN] No se escribio el CSV del SEM.")
        print("Detalle -> y11_dialogico_detalle.csv")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.sem_csv.replace(".csv", f"_BACKUP_pre_y11dial_{ts}.csv")
    shutil.copy2(args.sem_csv, backup)
    print(f"\n  Backup -> {backup}")

    df_sem["y11_conv_rest_old"] = df_sem["y11_conv_rest"] if "y11_conv_rest" in df_sem.columns else np.nan
    df_sem["y11_conv_rest"] = df_sem["identidad"].map(m)
    df_sem = df_sem.drop(columns=["_y11"], errors="ignore")
    df_sem.to_csv(args.sem_csv, index=False, encoding="utf-8")
    print(f"  CSV del SEM actualizado -> {args.sem_csv}")
    print("  y11_conv_rest = dialogico nuevo | y11_conv_rest_old = viejo (centroide)")


if __name__ == "__main__":
    main()
