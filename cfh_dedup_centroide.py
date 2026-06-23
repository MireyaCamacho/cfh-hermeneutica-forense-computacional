"""
CFH — Deduplicación y limpieza de candidatos para centroide v5
Elimina duplicados entre v4 y v5, genera JSON limpio y recalcula centroide.
"""
import json
import numpy as np
from pathlib import Path

REPO    = Path(__file__).resolve().parent
REF_DIR = REPO / "data" / "referencias"
JSON_V4 = Path(r"G:\Mi unidad\CHF_Corpus\referencias\corpus_c_victimas_ampliado.json")
JSON_V5 = REF_DIR / "corpus_c_victimas_v5.json"
OUT_CLEAN = REF_DIR / "corpus_c_victimas_v5_clean.json"

print("=" * 60)
print("CFH — Deduplicación candidatos centroide v5")
print("=" * 60)

# Cargar v4
v4_data = json.load(open(JSON_V4, encoding="utf-8"))
segs_v4 = v4_data.get("segmentos", [])
claves_v4 = {s["texto"].strip()[:100] for s in segs_v4}
print(f"\nv4: {len(segs_v4)} segmentos")

# Cargar v5
v5_data = json.load(open(JSON_V5, encoding="utf-8"))
segs_v5 = v5_data.get("segmentos", [])
print(f"v5: {len(segs_v5)} segmentos")

# Deduplicar: solo los que NO están en v4
vistos = set(claves_v4)
unicos_v5 = []
duplicados = 0
for s in segs_v5:
    clave = s["texto"].strip()[:100]
    if clave not in vistos and len(s["texto"].strip()) >= 10:
        vistos.add(clave)
        unicos_v5.append(s)
    else:
        duplicados += 1

print(f"\nDuplicados eliminados: {duplicados}")
print(f"Únicos v5 limpios: {len(unicos_v5)}")

# Guardar v5 limpio
output = {
    "version": "corpus_c_victimas_v5_clean",
    "n_segmentos": len(unicos_v5),
    "segmentos": unicos_v5
}
json.dump(output, open(OUT_CLEAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n✓ Guardado: {OUT_CLEAN}")

# Resumen por nivel
from collections import Counter
niveles = Counter(s.get("nivel", 3) for s in unicos_v5)
print(f"\nDistribución por nivel:")
for niv, cnt in sorted(niveles.items()):
    print(f"  Nivel {niv}: {cnt} segmentos")

print(f"\n→ Ahora corre: python code/cfh_centroide_mafapo_v5_final.py")
print("[CFH] Deduplicación completada.")
