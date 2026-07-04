# -*- coding: utf-8 -*-
r"""
cfh_diagnostico_fp_rep.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

DIAGNOSTICO DE FALSOS POSITIVOS del extractor y10 (REP) contra el gold
reconciliado (kappa = 0.841).

Un FALSO POSITIVO = fragmento con REP_gold=0 pero que el extractor detecta
(score_raw > 0). Este script te muestra, para cada FP, QUE mecanismo y QUE
span exacto lo dispararon — para decidir a ojo si el gold estaba bien (y hay
que ajustar el extractor) o si el fragmento si era REP (y el gold se queda).

Tambien lista los FALSOS NEGATIVOS (REP_gold=1 no detectados) con su texto,
para el otro lado del error.

UBICACION SUGERIDA:  code\cfh_diagnostico_fp_rep.py

COMO CORRERLO (raiz del repo, env cfh):
  python code\cfh_diagnostico_fp_rep.py

Rutas por defecto (ajustables por CLI):
  --gold       data\referencias\gold_REP_reconciliado.json
  --extractor  code\src\features
  --out        cfh_diagnostico_fp_rep.csv
"""

import json
import csv
import argparse
import os
import sys


def cargar_extractor(carpeta):
    carpeta = os.path.abspath(carpeta)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    try:
        import y10_rep_extractor as y10
        return y10
    except ImportError as e:
        print(f"ERROR importando y10_rep_extractor desde {carpeta}: {e}")
        sys.exit(1)


def build_extractor(y10):
    try:
        ext = y10.REPExtractor(model_name="es_core_news_lg")
        print("[nlp] usando es_core_news_lg")
        return ext
    except Exception as e:
        print(f"[nlp] es_core_news_lg no disponible ({e}) -> sentencizer liviano")
        import spacy
        ext = y10.REPExtractor.__new__(y10.REPExtractor)
        ext.model_name = "blank_es_sentencizer"
        ext.normalizer = y10.REPScoreNormalizer()
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 3_000_000
        ext._nlp = nlp
        return ext


def corpus_type_from_field(v):
    if not v:
        return "B"
    c = v.strip().upper()
    if c.startswith("A"):
        return "A"
    if c.startswith("C"):
        return "C"
    return "B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join("data", "referencias", "gold_REP_reconciliado.json"))
    ap.add_argument("--extractor", default=os.path.join("code", "src", "features"))
    ap.add_argument("--out", default="cfh_diagnostico_fp_rep.csv")
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        print(f"ERROR: no encuentro el gold en {args.gold}")
        sys.exit(1)

    y10 = cargar_extractor(args.extractor)
    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    ext = build_extractor(y10)

    print(f"\nProcesando {len(gold)} fragmentos...\n")
    detalle = []
    fps, fns, tps = [], [], []

    for r in gold:
        ct = corpus_type_from_field(r.get("corpus"))
        res = ext.extract(
            text=r.get("text", ""),
            doc_id=str(r.get("id", "?")),
            section_id="gold",
            corpus_type=ct,
        )
        gold_lab = int(r.get("REP_gold", 0))
        pred = 1 if res.score_raw > 0 else 0

        # spans que dispararon, ordenados por peso
        spans = sorted(res.instances, key=lambda x: -x.weight)
        spans_txt = " || ".join(
            f"[{i.mechanism}:{i.weight:.2f}] {i.text_span[:50]}"
            for i in spans[:4]
        )

        fila = {
            "id": r.get("id"),
            "corpus": r.get("corpus"),
            "corpus_type": ct,
            "rep_gold": gold_lab,
            "pred": pred,
            "score_raw": round(res.score_raw, 4),
            "n_nominal": res.n_nominal,
            "n_recon": res.n_reconocimiento,
            "n_restit": res.n_restitución,
            "n_repar": res.n_reparación,
            "spans_disparadores": spans_txt,
            "texto": r.get("text", "").strip().replace("\n", " ")[:200],
        }
        detalle.append(fila)

        if gold_lab == 0 and pred == 1:
            fps.append(fila)
        elif gold_lab == 1 and pred == 0:
            fns.append(fila)
        elif gold_lab == 1 and pred == 1:
            tps.append(fila)

    # ---- Resumen ----
    print("=" * 70)
    print(f"TP={len(tps)}  FP={len(fps)}  FN={len(fns)}")
    print("=" * 70)

    # ---- FALSOS POSITIVOS ----
    print("\n" + "#" * 70)
    print(f"# FALSOS POSITIVOS ({len(fps)}) — gold=0 pero el extractor detecta")
    print("#" * 70)
    for f in fps:
        print(f"\n[id {f['id']} | {f['corpus']}] score={f['score_raw']} "
              f"(nom={f['n_nominal']} rec={f['n_recon']} res={f['n_restit']} rep={f['n_repar']})")
        print(f"  DISPARO: {f['spans_disparadores']}")
        print(f"  TEXTO:   {f['texto'][:150]}")

    # ---- FALSOS NEGATIVOS ----
    print("\n" + "#" * 70)
    print(f"# FALSOS NEGATIVOS ({len(fns)}) — gold=1 pero el extractor NO detecta")
    print("#" * 70)
    for f in fns:
        print(f"\n[id {f['id']} | {f['corpus']}]")
        print(f"  TEXTO: {f['texto'][:150]}")

    # ---- CSV completo ----
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(detalle[0].keys()))
        w.writeheader()
        w.writerows(detalle)
    print(f"\nDetalle completo (100 filas) -> {args.out}")


if __name__ == "__main__":
    main()
