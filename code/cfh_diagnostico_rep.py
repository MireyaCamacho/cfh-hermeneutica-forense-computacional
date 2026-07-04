# -*- coding: utf-8 -*-
r"""
cfh_diagnostico_rep.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

DIAGNOSTICO del extractor y10 (REP) contra el gold REP reconciliado
(consenso estricto, kappa = 0.841).

Que hace:
  1. Carga gold_REP_reconciliado.json (100 fragmentos: id, text, corpus, REP_gold).
  2. Corre el extractor y10 actual sobre cada fragmento con su corpus_type.
  3. Imprime:
       - score por fragmento (raw y normalizado)
       - que detecto y que no
       - desglose por corpus (A / B)
       - barrido de umbral con precision / recall / F1
       - lista de positivos gold NO detectados (para revisar a ojo)
  4. Escribe cfh_diagnostico_rep_resultados.csv con todo el detalle.

NO modifica el extractor. Solo mide y reporta.

UBICACION SUGERIDA:  code\cfh_diagnostico_rep.py

COMO CORRERLO (desde la raiz del repo, en el env cfh):
  python code\cfh_diagnostico_rep.py

Rutas por defecto (ajustables por CLI):
  --gold       data\referencias\gold_REP_reconciliado.json
  --extractor  code\src\features   (carpeta donde esta y10_rep_extractor.py)
  --out        cfh_diagnostico_rep_resultados.csv
"""

import json
import csv
import argparse
import os
import sys


def cargar_extractor(carpeta_extractor):
    """Importa y10_rep_extractor desde la carpeta indicada."""
    carpeta_extractor = os.path.abspath(carpeta_extractor)
    if carpeta_extractor not in sys.path:
        sys.path.insert(0, carpeta_extractor)
    try:
        import y10_rep_extractor as y10
        return y10
    except ImportError as e:
        print(f"ERROR: no pude importar y10_rep_extractor desde {carpeta_extractor}")
        print(f"       {e}")
        print("       Ajusta --extractor a la carpeta donde esta el .py")
        sys.exit(1)


def build_extractor(y10):
    """
    Construye el REPExtractor. Intenta es_core_news_lg (tu entorno cfh
    lo tiene). Si por lo que sea no esta, cae a un sentencizer liviano
    para que el diagnostico corra igual (la deteccion REP es regex pura;
    spaCy solo parte oraciones).
    """
    try:
        ext = y10.REPExtractor(model_name="es_core_news_lg")
        print("[nlp] usando es_core_news_lg")
        return ext
    except Exception as e:
        print(f"[nlp] es_core_news_lg no disponible ({e})")
        print("[nlp] -> sentencizer liviano (blank es + sentencizer)")
        import spacy
        ext = y10.REPExtractor.__new__(y10.REPExtractor)
        ext.model_name = "blank_es_sentencizer"
        ext.normalizer = y10.REPScoreNormalizer()
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 3_000_000
        ext._nlp = nlp
        return ext


def corpus_type_from_field(corpus_value):
    """Mapea 'A-CE','A-CSJ','B','C' -> 'A'/'B'/'C'."""
    if not corpus_value:
        return "B"
    c = corpus_value.strip().upper()
    if c.startswith("A"):
        return "A"
    if c.startswith("C"):
        return "C"
    return "B"


def prf(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, (tp, fp, fn, tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join("data", "referencias", "gold_REP_reconciliado.json"))
    ap.add_argument("--extractor", default=os.path.join("code", "src", "features"))
    ap.add_argument("--out", default="cfh_diagnostico_rep_resultados.csv")
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        print(f"ERROR: no encuentro el gold en {args.gold}")
        print("       Ajusta --gold a la ruta correcta.")
        sys.exit(1)

    y10 = cargar_extractor(args.extractor)
    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    ext = build_extractor(y10)

    print(f"\nProcesando {len(gold)} fragmentos...\n")
    rows = []
    for r in gold:
        ct = corpus_type_from_field(r.get("corpus"))
        res = ext.extract(
            text=r.get("text", ""),
            doc_id=str(r.get("id", "?")),
            section_id="gold",
            corpus_type=ct,
        )
        rows.append({
            "id": r.get("id"),
            "corpus": r.get("corpus"),
            "corpus_type": ct,
            "score_raw": round(res.score_raw, 4),
            "score_norm": round(res.score, 4),
            "n_instances": res.n_instances,
            "n_reconoc": res.n_reconocimiento,
            "n_restit": res.n_restitución,
            "n_repar": res.n_reparación,
            "rep_gold": int(r.get("REP_gold", 0)),
        })

    y_true = [x["rep_gold"] for x in rows]
    scores = [x["score_raw"] for x in rows]

    # ---- Resumen general ----
    print("=" * 64)
    print("RESUMEN GENERAL")
    print("=" * 64)
    print(f"Positivos gold: {sum(y_true)}/{len(y_true)}")
    print(f"Fragmentos con score_raw>0 (extractor dispara): {sum(1 for s in scores if s>0)}")
    print(f"Rango score_raw: [{min(scores):.4f}, {max(scores):.4f}]")

    # ---- Desglose por corpus ----
    print("\n" + "=" * 64)
    print("DESGLOSE POR CORPUS")
    print("=" * 64)
    print(f"{'ct':>3} | {'n':>3} | {'pos_gold':>8} | {'dispara':>7} | {'pos_detect':>10}")
    print("-" * 44)
    for ct in ["A", "B", "C"]:
        sub = [x for x in rows if x["corpus_type"] == ct]
        if not sub:
            continue
        n = len(sub)
        pos = sum(x["rep_gold"] for x in sub)
        disp = sum(1 for x in sub if x["score_raw"] > 0)
        posd = sum(1 for x in sub if x["score_raw"] > 0 and x["rep_gold"] == 1)
        print(f"{ct:>3} | {n:>3} | {pos:>8} | {disp:>7} | {posd:>10}")

    # ---- Barrido de umbral ----
    print("\n" + "=" * 64)
    print("BARRIDO DE UMBRAL (sobre score_raw)")
    print("=" * 64)
    print(f"{'umbral':>7} | {'prec':>6} | {'recall':>6} | {'F1':>6} | (tp,fp,fn,tn)")
    print("-" * 56)
    mejor = (-1.0, None, None)
    for i in range(0, 51):
        u = round(0.01 * i, 3)
        y_pred = [1 if s > u else 0 for s in scores]
        prec, rec, f1, cm = prf(y_true, y_pred)
        if f1 > mejor[0]:
            mejor = (f1, u, (prec, rec, cm))
        if i % 5 == 0:
            print(f"{u:>7.2f} | {prec:>6.3f} | {rec:>6.3f} | {f1:>6.3f} | {cm}")
    f1b, ub, (pb, rb, cmb) = mejor
    print("-" * 56)
    print(f"UMBRAL OPTIMO: {ub:.3f}  ->  F1={f1b:.3f}  prec={pb:.3f}  recall={rb:.3f}")
    print(f"  matriz (tp,fp,fn,tn) = {cmb}")

    # ---- Positivos gold NO detectados ----
    print("\n" + "=" * 64)
    print("POSITIVOS GOLD NO DETECTADOS (score_raw = 0)")
    print("=" * 64)
    gmap = {str(r["id"]): r for r in gold}
    no_det = [x for x in rows if x["rep_gold"] == 1 and x["score_raw"] == 0]
    print(f"Total no detectados: {len(no_det)}\n")
    for x in no_det:
        t = gmap[str(x["id"])]["text"].strip().replace("\n", " ")
        print(f"[id {x['id']:>3} | {x['corpus']:>5}] {t[:110]}")

    # ---- CSV ----
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDetalle por fragmento -> {args.out}")


if __name__ == "__main__":
    main()
