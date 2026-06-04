import json
from pathlib import Path

p = Path("data/processed/corpus_a")
primer_json = sorted(p.glob("*.json"))[0]
with open(primer_json, encoding="utf-8") as f:
    data = json.load(f)

secciones = data["segmentation"]["sections"]
print(f"Total secciones: {len(secciones)}")
for s in secciones:
    print(f"\n--- Sección: {s.get('section_id')} ---")
    print(f"  Claves: {list(s.keys())}")
    # Mostrar texto truncado si existe
    for k in s:
        v = s[k]
        if isinstance(v, str) and len(v) > 50:
            print(f"  {k}: {v[:150]}...")
        elif isinstance(v, list):
            print(f"  {k}: lista de {len(v)} items")
            if v and isinstance(v[0], dict):
                print(f"    primer item claves: {list(v[0].keys())}")
                for kk, vv in v[0].items():
                    if isinstance(vv, str) and len(vv) > 30:
                        print(f"      {kk}: {vv[:120]}...")
                    else:
                        print(f"      {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
