import json
from pathlib import Path

for corpus_dir in ["data/processed/corpus_b_json", "data/processed/corpus_a"]:
    files = list(Path(corpus_dir).glob("*.json"))
    if not files:
        print(f"[{corpus_dir}] — sin JSONs\n")
        continue

    p = files[0]
    d = json.loads(p.read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"Corpus: {corpus_dir} | Archivo: {p.name}")
    print(f"Claves raíz: {list(d.keys())}")

    # Mostrar estructura completa de la primera sección
    secs = d.get("segmentation", {}).get("sections", [])
    if secs:
        print(f"\nPrimera sección — claves: {list(secs[0].keys())}")
        # Mostrar primeros 200 chars de cada campo
        for k, v in secs[0].items():
            if isinstance(v, str):
                print(f"  {k}: '{v[:100]}'")
            else:
                print(f"  {k}: {v}")

    # Mostrar claves de segmentation
    seg = d.get("segmentation", {})
    print(f"\nClaves de segmentation: {list(seg.keys())}")
