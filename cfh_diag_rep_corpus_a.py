# -*- coding: utf-8 -*-
"""
cfh_diag_rep_corpus_a.py
=========================
Corpus A es justicia ordinaria (Consejo de Estado, Corte Suprema). NO tiene
comparecientes reconociendo responsabilidad. Si el REP subio en 284 secciones
tras ampliar patrones, hay que ver QUE se esta matcheando: reconocimiento real
(citado en la sentencia) o FALSOS POSITIVOS (patrones demasiado laxos).

Este script recorre las secciones de Corpus A, corre el extractor, y para las
secciones con REP>0 muestra EXACTAMENTE que span disparo cada match y con que
patron. Asi vemos si "yo fui", "reconozco", "pido perdon" estan matcheando
contextos que no son reconocimiento del compareciente.

Uso:
    python cfh_diag_rep_corpus_a.py > diag_rep_a.txt 2>&1
    type diag_rep_a.txt
"""

import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(".")
sys.path.insert(0, str(BASE / "code" / "src"))

PROC_A = BASE / "data" / "processed" / "corpus_a"


def iter_secciones():
    for jp in sorted(PROC_A.glob("*.json")):
        if jp.name.startswith("batch_summary"):
            continue
        tp = jp.with_suffix(".txt")
        if not tp.exists():
            continue
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        doc_id = d.get("sha256_clean") or d.get("metadata", {}).get("doc_id")
        if not doc_id:
            continue
        clean = tp.read_text(encoding="utf-8", errors="replace")
        for sec in d.get("segmentation", {}).get("sections", []):
            if not sec.get("is_target", False):
                continue
            sid, rng = sec.get("section_id"), sec.get("char_range")
            if not sid or not rng or len(rng) != 2:
                continue
            seg = clean[rng[0]:rng[1]].strip()
            if seg:
                yield doc_id, sid, seg


def main():
    from features.y10_rep_extractor import REPExtractor
    ext = REPExtractor()

    print("=" * 72)
    print("DIAGNOSTICO REP EN CORPUS A (no deberia haber reconocimiento)")
    print("=" * 72)

    spans_por_mecanismo = Counter()
    spans_texto = Counter()
    n_con_rep = 0
    n_total = 0
    ejemplos = []

    for doc_id, sid, texto in iter_secciones():
        n_total += 1
        res = ext.extract(texto, doc_id=doc_id, section_id=sid, corpus_type="A")
        if res.n_instances > 0:
            n_con_rep += 1
            for inst in res.instances:
                spans_por_mecanismo[inst.mechanism] += 1
                # normalizar el span a minusculas para agrupar
                span_norm = inst.text_span.lower().strip()[:40]
                spans_texto[span_norm] += 1
                if len(ejemplos) < 40:
                    ejemplos.append((doc_id[:10], sid, inst.mechanism, inst.text_span[:60]))

    print(f"\n  secciones totales A: {n_total}")
    print(f"  secciones con REP>0: {n_con_rep}")
    print(f"  total instancias:    {sum(spans_por_mecanismo.values())}")

    print("\n  INSTANCIAS POR MECANISMO:")
    for mec, n in spans_por_mecanismo.most_common():
        print(f"    {mec:35s}: {n}")

    print("\n  SPANS MAS FRECUENTES (que texto dispara el match):")
    for span, n in spans_texto.most_common(30):
        print(f"    [{n:3d}x]  '{span}'")

    print("\n  EJEMPLOS (doc / seccion / mecanismo / span):")
    for doc, sid, mec, span in ejemplos:
        print(f"    {doc} / {sid:16s} / {mec[:20]:20s} / '{span}'")

    print("\n" + "=" * 72)
    print("LECTURA: si los spans frecuentes son cosas como 'reconoce que',")
    print("'verdad', citas normativas, o frases fuera de un reconocimiento real")
    print("del compareciente, son FALSOS POSITIVOS y hay que restringir el patron.")
    print("Si son reconocimientos citados textualmente en la sentencia, es")
    print("legitimo (la sentencia cita al militar reconociendo).")


if __name__ == "__main__":
    main()
