# -*- coding: utf-8 -*-
"""
CFH — Calibracion de extractores y2 (SA) y y4 (NV) contra gold standard
========================================================================
Compara la salida de los extractores automaticos con la anotacion humana
(A1 = Mireya) sobre los 100 fragmentos, para diagnosticar y calibrar.

Metrica a nivel FRAGMENTO (presencia/ausencia de la categoria):
  - El extractor da un score continuo [0,1]; se binariza con umbral.
  - Se compara con: ¿el fragmento tiene al menos un span de esa categoria?
  - Se reportan precision, recall, F1 y la curva por umbral.

Uso:
    conda activate cfh
    python code/cfh_calibrar_y2_y4.py

Requiere: annotations_mireya_v1.json en data/referencias/ (o ajustar ruta).
"""
import sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

GOLD = REPO / "data" / "referencias" / "annotations_mireya_v1.json"
SPACY_MODEL = "es_core_news_lg"


def cargar_gold():
    with open(GOLD, encoding="utf-8") as f:
        data = json.load(f)
    frags = []
    for d in data:
        cats = set()
        for span in d.get("label", []):
            for lab in span.get("labels", []):
                cats.add(lab)
        frags.append({
            "id": d["id"],
            "text": d["text"],
            "corpus": d.get("corpus_type", "?"),
            "tiene_SA": "SA" in cats,
            "tiene_NV": "NV" in cats,
        })
    return frags


def metricas(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1, (tp, fp, fn, tn)


def main():
    print("=" * 64)
    print("CFH — Calibracion y2 (SA) y y4 (NV) vs gold standard")
    print("=" * 64)

    if not GOLD.exists():
        print(f"  [ERROR] no existe {GOLD}")
        print("  Copia annotations_mireya_v1.json a data/referencias/")
        return

    frags = cargar_gold()
    print(f"\n  Fragmentos gold: {len(frags)}")
    print(f"    con SA: {sum(f['tiene_SA'] for f in frags)}")
    print(f"    con NV: {sum(f['tiene_NV'] for f in frags)}")

    from features.y2_sa_extractor import SAExtractor
    from features.y4_nv_extractor import NVExtractor
    print("\n  Cargando extractores...")
    sa = SAExtractor(model_name=SPACY_MODEL)
    nv = NVExtractor(model_name=SPACY_MODEL)
    for ext in [sa, nv]:
        if hasattr(ext, "_nlp"):
            ext._nlp.max_length = 3_000_000

    # Correr extractores
    print("  Corriendo extractores sobre los 100 fragmentos...")
    sa_scores, nv_scores = [], []
    for f in frags:
        r_sa = sa.extract(f["text"], doc_id=str(f["id"]), section_id="frag",
                          corpus_type=f["corpus"][0] if f["corpus"] else "A")
        r_nv = nv.extract(f["text"], doc_id=str(f["id"]), section_id="frag",
                          corpus_type=f["corpus"][0] if f["corpus"] else "A")
        sa_scores.append(r_sa.score)
        nv_scores.append(r_nv.score)

    # Evaluar por umbral
    for cat, scores, key in [("SA (y2)", sa_scores, "tiene_SA"),
                             ("NV (y4)", nv_scores, "tiene_NV")]:
        y_true = [f[key] for f in frags]
        print(f"\n  --- {cat} ---")
        print(f"    score: min={min(scores):.3f} max={max(scores):.3f} "
              f"media={np.mean(scores):.3f}")
        print(f"    umbral | prec  recall  F1    (tp,fp,fn,tn)")
        best_f1, best_thr = 0, 0
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred = [s >= thr for s in scores]
            prec, rec, f1, conf = metricas(y_true, y_pred)
            marca = ""
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
                marca = " <-- mejor F1"
            print(f"     {thr:.1f}   | {prec:.2f}  {rec:.2f}   {f1:.2f}  {conf}{marca}")
        print(f"    >> Mejor umbral: {best_thr:.1f} (F1={best_f1:.2f})")

    print("\n" + "=" * 64)
    print("Interpretacion:")
    print("  - recall bajo  = el extractor NO detecta casos que el humano marco")
    print("  - precision baja = el extractor marca casos que el humano NO anoto")
    print("  - el umbral optimo calibra el score continuo del extractor")
    print("=" * 64)


if __name__ == "__main__":
    main()
