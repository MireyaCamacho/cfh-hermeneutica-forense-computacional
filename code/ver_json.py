import json
from pathlib import Path

# Ver estructura del primer JSON de corpus_a
p = Path("data/processed/corpus_a")
primer_json = sorted(p.glob("*.json"))[0]
print(f"Archivo: {primer_json.name}")
with open(primer_json, encoding="utf-8") as f:
    data = json.load(f)
print(f"Tipo: {type(data)}")
if isinstance(data, dict):
    print(f"Claves: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: lista de {len(v)} elementos")
            if v:
                print(f"    primer elemento: {str(v[0])[:200]}")
        else:
            print(f"  {k}: {str(v)[:200]}")
elif isinstance(data, list):
    print(f"Lista de {len(data)} elementos")
    print(f"Primer elemento: {str(data[0])[:300]}")
