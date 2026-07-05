# -*- coding: utf-8 -*-
"""
CFH — Diagnostico de REFERENTE en y4 (NV): a quien modifica el termino
=======================================================================
Para cada FP, muestra la ORACION completa donde aparece el termino NV, para
decidir manualmente si el termino:
  (a) reproduce el marco NV contra la VICTIMA  -> NV real (mantener)
  (b) se refiere al VICTIMARIO/perpetrador     -> NO es NV (excluir)
  (c) es uso neutro/comun (falso amigo lexico) -> NO es NV (excluir)

No modifica nada. Diagnostico para tomar la decision con el texto delante.

Uso:
    conda activate cfh
    python code/cfh_diagnostico_y4_referente.py
"""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

GOLD = REPO / "data" / "referencias" / "annotations_mireya_v1.json"
SPACY_MODEL = "es_core_news_lg"

FP_IDS = [21, 58, 20, 17, 16, 13, 70, 94, 6, 98]


def cargar_gold():
    with open(GOLD, encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d for d in data}


def main():
    print("=" * 70)
    print("CFH — Diagnostico de REFERENTE de los terminos NV (FP)")
    print("=" * 70)

    if not GOLD.exists():
        print(f"  [ERROR] no existe {GOLD}")
        return

    gold = cargar_gold()
    import spacy
    print("  Cargando spaCy...")
    nlp = spacy.load(SPACY_MODEL)
    nlp.max_length = 3_000_000

    from features.y4_nv_extractor import NVExtractor, NV_COMBATIENTE_LEMMAS, NV_DESHUMANIZACION_DIRECTA
    nv = NVExtractor(model_name=SPACY_MODEL)
    nv._nlp.max_length = 3_000_000

    terminos_interes = set(NV_COMBATIENTE_LEMMAS) | set(NV_DESHUMANIZACION_DIRECTA)

    for fid in FP_IDS:
        if fid not in gold:
            continue
        frag = gold[fid]
        ct = frag.get("corpus_type", "A")
        ct = ct[0] if ct else "A"
        r = nv.extract(frag["text"], doc_id=str(fid), section_id="frag", corpus_type=ct)
        print(f"\n{'='*70}")
        print(f"Fragmento #{fid} (corpus={ct}, score={r.score:.3f})")
        print(f"{'='*70}")

        # Para cada instancia detectada, mostrar la oracion completa
        doc = nlp(frag["text"])
        for inst in sorted(r.instances, key=lambda x: -x.weight):
            span_txt = inst.text_span
            # localizar la oracion que contiene el span
            oracion = None
            for sent in doc.sents:
                if sent.start_char <= inst.char_start < sent.end_char:
                    oracion = sent.text.strip()
                    break
            oracion = (oracion or "")[:220]
            print(f"\n  · [{inst.mechanism}] span=\"{span_txt}\"")
            print(f"    Oracion: \"{oracion}\"")

    print("\n" + "=" * 70)
    print("DECISION por fragmento: (a) NV real reproducido | (b) victimario | (c) neutro")
    print("=" * 70)


if __name__ == "__main__":
    main()
