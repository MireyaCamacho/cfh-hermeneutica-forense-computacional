import json
from pathlib import Path

for corpus_dir in ["data/processed/corpus_b_json", "data/processed/corpus_a"]:
    files = list(Path(corpus_dir).glob("*.json"))
    if not files:
        print(f"[{corpus_dir}] — sin JSONs\n")
        continue

    p = files[0]
    d = json.loads(p.read_text(encoding="utf-8"))
    secs = d.get("segmentation", {}).get("sections", [])

    print(f"\n{'='*55}")
    print(f"Corpus: {corpus_dir}")
    print(f"Doc: {d.get('doc_id')}")
    print(f"Total secciones: {len(secs)}")
    for s in secs[:5]:
        print(
            f"  - {s.get('section_id', '?'):35s} "
            f"target={s.get('is_target', '?')}  "
            f"chars={len(s.get('text', ''))}"
        )
    print(f"(mostrando {min(5, len(secs))} de {len(secs)})")
