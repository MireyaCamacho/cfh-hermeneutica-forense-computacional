import json
from pathlib import Path

for corpus_dir, raw_dir in [
    ("data/processed/corpus_b_json", "data/raw/corpus_b"),
    ("data/processed/corpus_a", "data/raw/corpus_a"),
]:
    files = list(Path(corpus_dir).glob("*.json"))
    if not files:
        print(f"[{corpus_dir}] — sin JSONs\n")
        continue

    p = files[0]
    d = json.loads(p.read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"Corpus: {corpus_dir} | Archivo: {p.name}")
    print(f"source_file: {d.get('source_file')}")
    print(f"corpus_type: {d.get('corpus_type')}")

    # metadata
    meta = d.get("metadata", {})
    print(f"metadata claves: {list(meta.keys())}")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # primera sección target con char_range
    secs = d.get("segmentation", {}).get("sections", [])
    target = next((s for s in secs if s.get("is_target")), None)
    if target:
        print(f"\nPrimera sección target: {target}")

    # Ver si hay txt correspondiente
    raw_path = Path(raw_dir)
    print(f"\nArchivos en {raw_dir}: {len(list(raw_path.glob('*')))}")
    for f in list(raw_path.glob("*"))[:3]:
        print(f"  {f.name}")
