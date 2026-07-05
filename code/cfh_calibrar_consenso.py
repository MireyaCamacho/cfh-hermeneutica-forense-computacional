# -*- coding: utf-8 -*-
"""
CFH — Calibracion y2 (SA) y y4 (NV) contra CONSENSO A1&A2
=========================================================
Calibra los extractores contra el gold consolidado (interseccion de ambos
anotadores) = casos de alta confianza donde A1 y A2 coinciden.
Compara consenso vs union vs A1-solo.

Uso:
    conda activate cfh
    python code/cfh_calibrar_consenso.py

Requiere: gold_consolidado_A1A2.json en data/referencias/
"""
import sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

GOLD = REPO / "data" / "referencias" / "gold_consolidado_A1A2.json"
SPACY_MODEL = "es_core_news_lg"


def metricas(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1, (tp, fp, fn, tn)


def evaluar(nombre, scores, gold, key):
    y_true = [g[key] for g in gold]
    print(f"\n  --- {nombre} vs {key} ---")
    best_f1, best_thr = 0, 0
    res = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = [s >= thr for s in scores]
        prec, rec, f1, conf = metricas(y_true, y_pred)
        res.append((thr, prec, rec, f1, conf))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    for thr, prec, rec, f1, conf in res:
        marca = " <-- mejor" if thr == best_thr else ""
        print(f"     umbral {thr:.1f} | prec {prec:.2f} rec {rec:.2f} F1 {f1:.2f} {conf}{marca}")
    return best_thr, best_f1


def main():
    print("=" * 64)
    print("CFH — Calibracion y2/y4 contra CONSENSO A1&A2")
    print("=" * 64)

    if not GOLD.exists():
        print(f"  [ERROR] no existe {GOLD}")
        print("  Copia gold_consolidado_A1A2.json a data/referencias/")
        return

    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    print(f"\n  Fragmentos: {len(gold)}")

    from features.y2_sa_extractor import SAExtractor
    from features.y4_nv_extractor import NVExtractor
    print("  Cargando extractores...")
    sa = SAExtractor(model_name=SPACY_MODEL)
    nv = NVExtractor(model_name=SPACY_MODEL)
    for ext in [sa, nv]:
        if hasattr(ext, "_nlp"):
            ext._nlp.max_length = 3_000_000

    print("  Corriendo extractores...")
    sa_scores, nv_scores = [], []
    for g in gold:
        ct = g["corpus"][0] if g.get("corpus") else "A"
        r_sa = sa.extract(g["text"], doc_id=str(g["id"]), section_id="frag", corpus_type=ct)
        r_nv = nv.extract(g["text"], doc_id=str(g["id"]), section_id="frag", corpus_type=ct)
        sa_scores.append(r_sa.score)
        nv_scores.append(r_nv.score)

    # Comparar los 3 golds para SA
    print("\n" + "#" * 64)
    print("# y2 (SA)")
    print("#" * 64)
    evaluar("y2 vs A1-solo", sa_scores, gold, "SA_A1")
    evaluar("y2 vs CONSENSO", sa_scores, gold, "SA_consenso")
    evaluar("y2 vs UNION", sa_scores, gold, "SA_union")

    print("\n" + "#" * 64)
    print("# y4 (NV)")
    print("#" * 64)
    evaluar("y4 vs A1-solo", nv_scores, gold, "NV_A1")
    evaluar("y4 vs CONSENSO", nv_scores, gold, "NV_consenso")
    evaluar("y4 vs UNION", nv_scores, gold, "NV_union")

    print("\n" + "=" * 64)
    print("El CONSENSO (A1&A2) es el gold mas robusto: casos donde ambos")
    print("anotadores coinciden. Si F1 sube vs A1-solo, confirma que parte")
    print("del error era ruido de anotacion individual, no del extractor.")
    print("=" * 64)


if __name__ == "__main__":
    main()
