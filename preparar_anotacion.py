"""
CFH · Preparación de muestras para anotación en Label Studio
=============================================================
Genera un archivo JSON con 100 fragmentos estratificados del corpus
listos para importar en Label Studio.

Estrategia de muestreo:
- 50 fragmentos del Corpus A (25 CE + 25 CSJ)
  - Priorizando secciones HECHOS y CARGOS (mayor densidad EBI/NV)
- 50 fragmentos del Corpus B (JEP)
  - Priorizando secciones RECONOCIMIENTO (mayor densidad REP)

Cada fragmento: 500-1000 caracteres (tamaño óptimo para anotación manual)
"""
import json
import random
from pathlib import Path

random.seed(42)

# Rutas
CORPUS_A_DIR = Path("data/processed/corpus_a")
CORPUS_B_DIR = Path("data/processed/corpus_b_json")
OUTPUT_PATH  = Path("data/annotations/label_studio_input.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Texto fuente desde char_range
def get_text(json_data, section):
    source_file = json_data.get("source_file", "")
    char_range = section.get("char_range", [])
    if not source_file or len(char_range) != 2:
        return ""
    # Intentar leer el archivo fuente
    for candidate in [Path(source_file), Path(source_file.replace("\\", "/"))]:
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = candidate.read_text(encoding="latin-1")
            return text[char_range[0]:char_range[1]].strip()
    return ""

def extract_fragment(text, max_chars=800, min_chars=200):
    """Extrae un fragmento representativo del texto."""
    # Limpiar saltos de línea internos del OCR
    import re
    text = re.sub(r'\n+', ' ', text)          # colapsar múltiples newlines
    text = re.sub(r' {2,}', ' ', text)         # colapsar espacios múltiples
    text = text.strip()

    if len(text) <= max_chars:
        return text if len(text) >= min_chars else ""
    # Tomar los primeros max_chars caracteres, cortando en el último punto
    fragment = text[:max_chars]
    last_period = max(fragment.rfind(". "), fragment.rfind(".\n"))
    if last_period > min_chars:
        fragment = fragment[:last_period + 1]
    return fragment.strip()

samples = []
sample_id = 0

# ── Corpus A — 50 fragmentos ──────────────────────────────────────────────────
print("Procesando Corpus A...")
json_files_a = sorted(CORPUS_A_DIR.glob("*.json"))
random.shuffle(json_files_a)

# Priorizar HECHOS y CARGOS (mayor densidad EBI/NV)
priority_sections_a = {"HECHOS", "CARGOS", "CARGOS_UNICO", "HECHOS_JURIDICAMENTE_RELEVANTES"}
other_sections_a = {"CONSIDERACIONES", "DECISIÓN", "DECISION"}

priority_fragments = []
other_fragments = []

for json_path in json_files_a:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        doc_id = metadata.get("doc_id", json_path.stem)[:16] + "..."
        tribunal = metadata.get("tribunal", "")
        corpus_subtype = "A-CSJ" if "Suprema" in tribunal or "Penal" in tribunal else "A-CE"
        sections = data.get("segmentation", {}).get("sections", [])

        for sec in sections:
            if not sec.get("is_target"):
                continue
            section_id = sec.get("section_id", "").upper()
            text = get_text(data, sec)
            fragment = extract_fragment(text)
            if not fragment:
                continue

            item = {
                "id": sample_id,
                "data": {
                    "text": fragment,
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "corpus_type": corpus_subtype,
                }
            }

            if section_id in priority_sections_a:
                priority_fragments.append(item)
            elif section_id in other_sections_a:
                other_fragments.append(item)

            sample_id += 1

    except Exception as e:
        continue

# Seleccionar 50: 35 de prioridad + 15 otros
random.shuffle(priority_fragments)
random.shuffle(other_fragments)
samples_a = priority_fragments[:35] + other_fragments[:15]
random.shuffle(samples_a)
samples.extend(samples_a[:50])
print(f"  Corpus A: {len(samples_a[:50])} fragmentos")

# ── Corpus B — 50 fragmentos ──────────────────────────────────────────────────
print("Procesando Corpus B...")
json_files_b = sorted(CORPUS_B_DIR.glob("*.json"))

priority_sections_b = {"RECONOCIMIENTO", "CALIFICACION_JURIDICA"}
other_sections_b = {"CONSIDERACIONES", "RESUELVE", "HECHOS_Y_CONDUCTAS", "PATRONES_MACROCRIMINALES"}

priority_b = []
other_b = []

for json_path in json_files_b:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        doc_id = metadata.get("doc_id", json_path.stem)[:16] + "..."
        sections = data.get("segmentation", {}).get("sections", [])

        for sec in sections:
            if not sec.get("is_target"):
                continue
            section_id = sec.get("section_id", "").upper()
            text = get_text(data, sec)

            # Para fragmentos largos del corpus B, tomar diferentes partes
            # para maximizar la diversidad de anotación
            if len(text) > 2000:
                # Tomar 3 fragmentos de diferentes partes
                chunks = [
                    text[:800],
                    text[len(text)//2 - 400: len(text)//2 + 400],
                    text[-800:]
                ]
                for i, chunk in enumerate(chunks):
                    fragment = extract_fragment(chunk)
                    if not fragment:
                        continue
                    item = {
                        "id": sample_id,
                        "data": {
                            "text": fragment,
                            "doc_id": doc_id,
                            "section_id": f"{section_id}_p{i+1}",
                            "corpus_type": "B",
                        }
                    }
                    if section_id in priority_sections_b:
                        priority_b.append(item)
                    else:
                        other_b.append(item)
                    sample_id += 1
            else:
                fragment = extract_fragment(text)
                if not fragment:
                    continue
                item = {
                    "id": sample_id,
                    "data": {
                        "text": fragment,
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "corpus_type": "B",
                    }
                }
                if section_id in priority_sections_b:
                    priority_b.append(item)
                else:
                    other_b.append(item)
                sample_id += 1

    except Exception as e:
        continue

random.shuffle(priority_b)
random.shuffle(other_b)
samples_b = priority_b[:35] + other_b[:15]
random.shuffle(samples_b)
samples.extend(samples_b[:50])
print(f"  Corpus B: {len(samples_b[:50])} fragmentos")

# ── Guardar ───────────────────────────────────────────────────────────────────
# Re-numerar IDs
for i, s in enumerate(samples):
    s["id"] = i

OUTPUT_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n✓ {len(samples)} fragmentos guardados en {OUTPUT_PATH}")
print(f"  Corpus A: {sum(1 for s in samples if s['data']['corpus_type'] != 'B')}")
print(f"  Corpus B: {sum(1 for s in samples if s['data']['corpus_type'] == 'B')}")
print(f"\nPróximo paso: importar {OUTPUT_PATH} en Label Studio")
print("  1. Abrir el proyecto CFH en Label Studio")
print("  2. Click 'Import' → subir label_studio_input.json")
