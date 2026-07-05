# -*- coding: utf-8 -*-
"""
CFH — Diagnostico fino de y4 (NV): que span/mecanismo dispara en cada error
============================================================================
Para los FALSOS POSITIVOS: imprime cada instancia NV detectada (mecanismo +
texto exacto + peso) para saber POR QUE el extractor marca de mas.
Para los FALSOS NEGATIVOS: confirma que el score es 0 y muestra el texto,
para disenar los patrones que faltan.

No modifica nada. Solo diagnostica sobre el extractor ACTUAL (ya reescrito).

Uso:
    conda activate cfh
    python code/cfh_diagnostico_y4_spans.py
"""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

GOLD = REPO / "data" / "referencias" / "annotations_mireya_v1.json"
SPACY_MODEL = "es_core_news_lg"

# IDs de los errores detectados en la calibracion (umbral 0.4)
FALSOS_POSITIVOS = [21, 58, 20, 17, 16, 13, 70, 94, 6, 98]
FALSOS_NEGATIVOS = [1, 95, 92, 64, 59, 55, 41, 44, 35, 12, 39]  # los que tienen span textual


def cargar_gold():
    with open(GOLD, encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d for d in data}


def nv_del_gold(frag):
    spans = []
    for span in frag.get("label", []):
        if "NV" in span.get("labels", []):
            spans.append(span.get("text", "").strip())
    return spans


def main():
    print("=" * 68)
    print("CFH — Diagnostico fino de spans y4 (NV)")
    print("=" * 68)

    if not GOLD.exists():
        print(f"  [ERROR] no existe {GOLD}")
        return

    gold = cargar_gold()

    from features.y4_nv_extractor import NVExtractor
    print("  Cargando extractor NV (reescrito)...")
    nv = NVExtractor(model_name=SPACY_MODEL)
    if hasattr(nv, "_nlp"):
        nv._nlp.max_length = 3_000_000

    # ---- FALSOS POSITIVOS: por que dispara ----
    print("\n" + "#" * 68)
    print("# FALSOS POSITIVOS — que mecanismo/span dispara de mas")
    print("#" * 68)
    for fid in FALSOS_POSITIVOS:
        if fid not in gold:
            continue
        frag = gold[fid]
        ct = frag.get("corpus_type", "A")
        ct = ct[0] if ct else "A"
        r = nv.extract(frag["text"], doc_id=str(fid), section_id="frag", corpus_type=ct)
        print(f"\n  --- Fragmento #{fid} (score={r.score:.3f}, corpus={ct}) ---")
        print(f"      NV en gold: {nv_del_gold(frag) or '(nada — es FP real)'}")
        if not r.instances:
            print("      [sin instancias — el score viene del normalizador]")
        for inst in sorted(r.instances, key=lambda x: -x.weight):
            q = " [cuestionado]" if inst.is_questioned else ""
            print(f"      · {inst.mechanism:28s} peso={inst.weight:.2f} "
                  f"span=\"{inst.text_span}\"{q}")

    # ---- FALSOS NEGATIVOS: confirmar que no dispara nada ----
    print("\n" + "#" * 68)
    print("# FALSOS NEGATIVOS — NV real que el extractor no ve (score 0)")
    print("#" * 68)
    for fid in FALSOS_NEGATIVOS:
        if fid not in gold:
            continue
        frag = gold[fid]
        ct = frag.get("corpus_type", "A")
        ct = ct[0] if ct else "A"
        r = nv.extract(frag["text"], doc_id=str(fid), section_id="frag", corpus_type=ct)
        print(f"\n  --- Fragmento #{fid} (score={r.score:.3f}) ---")
        print(f"      NV en gold: {nv_del_gold(frag)}")
        print(f"      instancias detectadas: {len(r.instances)}")
        # mostrar un pedazo del texto alrededor del primer span del gold
        nvspans = nv_del_gold(frag)
        if nvspans and nvspans[0]:
            objetivo = nvspans[0][:40]
            pos = frag["text"].find(objetivo)
            if pos >= 0:
                ini = max(0, pos - 30)
                fin = min(len(frag["text"]), pos + len(objetivo) + 40)
                print(f"      contexto: ...{frag['text'][ini:fin]}...")

    print("\n" + "=" * 68)
    print("Con esto sabemos exactamente que patron anadir (FN) y que quitar (FP).")
    print("=" * 68)


if __name__ == "__main__":
    main()
