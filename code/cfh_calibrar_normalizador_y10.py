# -*- coding: utf-8 -*-
r"""
cfh_calibrar_normalizador_y10.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Calibra el NORMALIZADOR CONTINUO de y10 (REP) para el SEM.

Metodo de normalizacion (documento maestro CFH §6.3.1):
    z-score + sigmoide sobre la distribucion de los 47 COMPARECIENTES (Corpus C).
    y10_norm = 1 / (1 + exp(-(z)))  con  z = (raw - mu) / sigma
    donde mu y sigma se estiman sobre los scores raw de los 47
    comparecientes (Corpus C), que es el UNIVERSO de este SEM.

    NOTA: este SEM es por compareciente (n=47, todos Corpus C). Por eso la
    normalizacion se hace sobre C y no sobre A+B+C: incluir el Corpus A
    escrito (1658 secciones, casi todas sin REP) aplastaria la distribucion
    y mandaria a todos los comparecientes al techo de la sigmoide, matando
    la varianza discriminante que el SEM necesita. La normalizacion conjunta
    A+B+C queda reservada para el MG-SEM (analisis multigrupo entre corpus).

Flujo (3 pasos, todos verificables en pantalla):
  PASO 1 - Recalcular y10_raw con el extractor v4 sobre:
             * Corpus A: data\processed\corpus_a\*.txt  (secciones por char_range)
             * Corpus B: data\processed\corpus_b\*.txt  (secciones por char_range)
             * Corpus C: data\texto_por_compareciente.csv (47 comparecientes)
  PASO 2 - Estimar mu, sigma sobre la distribucion conjunta A+B+C.
           Guardar parametros en data\referencias\y10_normalizador_params.json
  PASO 3 - Aplicar la normalizacion a los 47 comparecientes y escribir
           y10_rep normalizado en indicadores_sem_compareciente.csv
           (con BACKUP del original antes de tocar nada).

NO sobreescribe nada sin backup. Imprime todo para verificacion.

UBICACION SUGERIDA:  code\cfh_calibrar_normalizador_y10.py

COMO CORRERLO (raiz del repo, env cfh):
  python code\cfh_calibrar_normalizador_y10.py

Rutas por defecto (ajustables por CLI):
  --extractor   code\src\features
  --corpus_a    data\processed\corpus_a
  --corpus_b    data\processed\corpus_b
  --texto_c     data\texto_por_compareciente.csv
  --sem_csv     data\referencias\indicadores_sem_compareciente.csv
  --params_out  data\referencias\y10_normalizador_params.json
  --dry-run     (si se pasa, NO escribe el CSV del SEM; solo reporta)
"""

import os
import sys
import json
import glob
import math
import shutil
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Carga del extractor v4
# ---------------------------------------------------------------------------
def cargar_extractor(carpeta):
    carpeta = os.path.abspath(carpeta)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    try:
        import y10_rep_extractor as y10
    except ImportError as e:
        print(f"ERROR importando y10_rep_extractor desde {carpeta}: {e}")
        sys.exit(1)
    # Verificar que sea la v4
    ruta = os.path.join(carpeta, "y10_rep_extractor.py")
    txt = open(ruta, encoding="utf-8").read()
    if "y10-v5" not in txt:
        print("ADVERTENCIA: el extractor cargado NO parece ser la v5 (Opcion B).")
        print("             Verifica el sello CFH_VERSION antes de continuar.")
    else:
        print("[extractor] y10-v5 confirmado (Opcion B activa)")
    return y10


def build_extractor(y10):
    try:
        ext = y10.REPExtractor(model_name="es_core_news_lg")
        print("[nlp] es_core_news_lg OK")
        return ext
    except Exception as e:
        print(f"ERROR: no pude cargar es_core_news_lg ({e})")
        print("       El mecanismo nominal REQUIERE NER. Instala el modelo:")
        print("       python -m spacy download es_core_news_lg")
        sys.exit(1)


# ---------------------------------------------------------------------------
# PASO 1 - Recalcular y10_raw sobre A+B+C con el extractor v4
# ---------------------------------------------------------------------------
def secciones_de_documento(json_path, txt_dir):
    """Devuelve lista de (section_id, texto) para un documento.

    Soporta las DOS estructuras del corpus CFH:
      - Corpus A: JSON con 'segmentation'->'sections' (char_range) + .txt aparte.
      - Corpus B: JSON con 'secciones' (dict) donde cada seccion trae 'texto'.
    """
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)

    # --- Estructura Corpus B: 'secciones' dict con texto embebido ---
    if isinstance(d.get("secciones"), dict):
        out = []
        for sec_id, sec in d["secciones"].items():
            if isinstance(sec, dict) and sec.get("texto"):
                out.append((sec_id, sec["texto"]))
        return out
    if isinstance(d.get("secciones"), list):
        out = []
        for i, sec in enumerate(d["secciones"]):
            if isinstance(sec, dict) and sec.get("texto"):
                out.append((sec.get("section_id", f"sec_{i}"), sec["texto"]))
        return out

    # --- Estructura Corpus A: segmentation.sections + .txt por char_range ---
    seg = d.get("segmentation")
    if not isinstance(seg, dict) or "sections" not in seg:
        return []
    doc_id = os.path.splitext(os.path.basename(json_path))[0]
    txt_path = os.path.join(txt_dir, doc_id + ".txt")
    if not os.path.exists(txt_path):
        return []
    texto = open(txt_path, encoding="utf-8").read()
    out = []
    for sec in seg["sections"]:
        cr = sec.get("char_range")
        if isinstance(cr, str):
            cr = json.loads(cr)
        if not cr or len(cr) != 2:
            continue
        ini, fin = int(cr[0]), int(cr[1])
        out.append((sec.get("section_id", "?"), texto[ini:fin]))
    return out


def procesar_corpus_escrito(ext, carpeta, corpus_type, y10):
    """Corre el extractor v4 sobre todas las secciones de un corpus A o B."""
    jsons = glob.glob(os.path.join(carpeta, "*.json"))
    raws = []
    n_sec = 0
    print(f"  {corpus_type}: {len(jsons)} documentos...")
    for jp in jsons:
        for section_id, txt in secciones_de_documento(jp, carpeta):
            if not txt or len(txt.strip()) < 20:
                continue
            res = ext.extract(
                text=txt,
                doc_id=os.path.splitext(os.path.basename(jp))[0],
                section_id=section_id,
                corpus_type=corpus_type,
            )
            raws.append(res.score_raw)
            n_sec += 1
    print(f"     -> {n_sec} secciones procesadas")
    return raws


def procesar_corpus_c(ext, texto_csv):
    """Corre el extractor v4 sobre los 47 comparecientes (Corpus C)."""
    df = pd.read_csv(texto_csv)
    raws = []
    detalle = []
    print(f"  C: {len(df)} comparecientes...")
    for _, row in df.iterrows():
        txt = str(row.get("texto_completo", "") or "")
        res = ext.extract(
            text=txt,
            doc_id=str(row.get("identidad", "?")),
            section_id="compareciente",
            corpus_type="C",
        )
        raws.append(res.score_raw)
        detalle.append({
            "subcaso": row.get("subcaso"),
            "identidad": row.get("identidad"),
            "y10_raw_v4": res.score_raw,
            "n_nominal": res.n_nominal,
            "n_recon": res.n_reconocimiento,
            "n_restit": res.n_restitución,
            "n_repar": res.n_reparación,
        })
    print(f"     -> {len(raws)} comparecientes procesados")
    return raws, detalle


# ---------------------------------------------------------------------------
# PASO 2 - Normalizador z-score + sigmoide
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def normalizar(raw, mu, sigma):
    if sigma <= 1e-12:
        return 0.5
    return sigmoide((raw - mu) / sigma)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default=os.path.join("code", "src", "features"))
    ap.add_argument("--corpus_a", default=os.path.join("data", "processed", "corpus_a"))
    ap.add_argument("--corpus_b", default=os.path.join("data", "processed", "corpus_b"))
    ap.add_argument("--texto_c", default=os.path.join("data", "texto_por_compareciente.csv"))
    ap.add_argument("--sem_csv", default=os.path.join("data", "referencias", "indicadores_sem_compareciente.csv"))
    ap.add_argument("--params_out", default=os.path.join("data", "referencias", "y10_normalizador_params.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solo-c", action="store_true",
                    help="Salta A y B (no entran en la normalizacion de C). Mas rapido.")
    args = ap.parse_args()

    print("=" * 66)
    print("CALIBRACION NORMALIZADOR y10 (z-score+sigmoide sobre C / SEM comparecientes)")
    print("=" * 66)

    y10 = cargar_extractor(args.extractor)
    ext = build_extractor(y10)

    # ---- PASO 1 ----
    print("\n" + "-" * 66)
    print("PASO 1 - Recalcular y10_raw con extractor v4 sobre A+B+C")
    print("-" * 66)
    if args.solo_c:
        print("  [--solo-c] Saltando A y B (no entran en la normalizacion de C).")
        raws_a, raws_b = [], []
    else:
        raws_a = procesar_corpus_escrito(ext, args.corpus_a, "A", y10) if os.path.isdir(args.corpus_a) else []
        raws_b = procesar_corpus_escrito(ext, args.corpus_b, "B", y10) if os.path.isdir(args.corpus_b) else []
    raws_c, detalle_c = procesar_corpus_c(ext, args.texto_c)

    dist_c = np.array(raws_c, dtype=float)
    # A y B se calculan solo como REFERENCIA/diagnostico (no entran en mu/sigma)
    print(f"\n  Scores raw C (universo del SEM): {len(dist_c)}")
    print(f"    (referencia: A={len(raws_a)}  B={len(raws_b)}  C={len(raws_c)})")

    # ---- PASO 2 ----
    print("\n" + "-" * 66)
    print("PASO 2 - Estimar mu, sigma sobre los 47 comparecientes (Corpus C)")
    print("-" * 66)
    mu = float(dist_c.mean())
    sigma = float(dist_c.std())
    print(f"  mu    = {mu:.6f}")
    print(f"  sigma = {sigma:.6f}")
    print(f"  min={dist_c.min():.4f}  max={dist_c.max():.4f}  "
          f"mediana={np.median(dist_c):.4f}")

    params = {
        "metodo": "zscore_sigmoide",
        "distribucion": "corpus_C_47_comparecientes",
        "extractor_version": "y10-v4",
        "mu": mu,
        "sigma": sigma,
        "n_total": int(len(dist_c)),
        "n_a": len(raws_a),
        "n_b": len(raws_b),
        "n_c": len(raws_c),
        "generado": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.params_out), exist_ok=True)
    with open(args.params_out, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"  Parametros guardados en: {args.params_out}")

    # ---- PASO 3 ----
    print("\n" + "-" * 66)
    print("PASO 3 - Aplicar normalizacion a los 47 comparecientes (SEM)")
    print("-" * 66)

    df_sem = pd.read_csv(args.sem_csv)
    # Mapear y10 normalizado por identidad
    norm_por_id = {}
    for row in detalle_c:
        norm_por_id[str(row["identidad"])] = normalizar(row["y10_raw_v4"], mu, sigma)

    # Emparejar por identidad (con aviso si algo no matchea)
    y10_viejo = df_sem["y10_rep"].copy()
    nuevos = []
    no_match = []
    for _, row in df_sem.iterrows():
        ident = str(row["identidad"])
        if ident in norm_por_id:
            nuevos.append(norm_por_id[ident])
        else:
            nuevos.append(np.nan)
            no_match.append(ident)

    df_sem["y10_rep_v4_norm"] = nuevos

    print(f"  Comparecientes emparejados: {len(df_sem) - len(no_match)}/{len(df_sem)}")
    if no_match:
        print(f"  SIN EMPAREJAR ({len(no_match)}): {no_match}")
        print("  (revisa que 'identidad' coincida entre los dos CSV)")

    # Comparativa viejo vs nuevo
    print("\n  Comparativa y10 (viejo -> v4_norm) por compareciente:")
    print(f"  {'subcaso':<12} {'identidad':<32} {'viejo':>7} {'v4norm':>7}")
    for _, row in df_sem.iterrows():
        print(f"  {str(row['subcaso'])[:12]:<12} {str(row['identidad'])[:32]:<32} "
              f"{row['y10_rep']:>7.3f} {row['y10_rep_v4_norm']:>7.3f}"
              if not pd.isna(row['y10_rep_v4_norm'])
              else f"  {str(row['subcaso'])[:12]:<12} {str(row['identidad'])[:32]:<32} "
                   f"{row['y10_rep']:>7.3f} {'NaN':>7}")

    if args.dry_run:
        print("\n  [DRY-RUN] No se escribio el CSV del SEM. Revisa la comparativa.")
        # Guardar detalle C aparte para inspeccion
        pd.DataFrame(detalle_c).to_csv("y10_raw_v4_corpus_c_detalle.csv",
                                       index=False, encoding="utf-8")
        print("  Detalle C -> y10_raw_v4_corpus_c_detalle.csv")
        return

    # BACKUP antes de escribir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.sem_csv.replace(".csv", f"_BACKUP_pre_y10v4_{ts}.csv")
    shutil.copy2(args.sem_csv, backup)
    print(f"\n  Backup del SEM original -> {backup}")

    # Reemplazar y10_rep con la version v4 normalizada (conservando la vieja aparte)
    df_sem["y10_rep_old"] = y10_viejo
    df_sem["y10_rep"] = df_sem["y10_rep_v4_norm"]
    df_sem.to_csv(args.sem_csv, index=False, encoding="utf-8")
    print(f"  CSV del SEM actualizado -> {args.sem_csv}")
    print("  Columnas: y10_rep (=v4_norm ahora), y10_rep_old (viejo), y10_rep_v4_norm")
    print("\nListo. y10 normalizado (z-score+sigmoide A+B+C) integrado al SEM.")


if __name__ == "__main__":
    main()
