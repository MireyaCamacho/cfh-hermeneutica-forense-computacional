import sys
sys.path.insert(0, 'code/src')
from ingestion.pipeline import CFHIngestionPipeline
from pathlib import Path
import logging
logging.disable(logging.CRITICAL)

CORPUS = Path('data/processed/corpus_b')
OUTPUT = Path('data/processed/corpus_b_json')
OUTPUT.mkdir(parents=True, exist_ok=True)

pipeline = CFHIngestionPipeline(corpus_type='B')
documentos = list(CORPUS.glob('*.txt'))

print(f"Procesando {len(documentos)} documentos JEP...")
print()

for i, archivo in enumerate(documentos, 1):
    result = pipeline.process_file(archivo)
    if result.success:
        pipeline.save_result(result, OUTPUT)
        conf = result.metadata.get('extraction_confidence', 0)
        secciones = result.segmentation.get('total_sections', 0)
        target = len([s for s in result.segmentation.get('sections', []) if s.get('is_target')])
        simbolo = '✓' if conf >= 0.5 else '⚠'
        print(f"[{i:02d}/{len(documentos)}] {simbolo} {archivo.stem[:50]}")
        print(f"         confianza={conf:.0%} | secciones={secciones} | target={target}")
    else:
        print(f"[{i:02d}/{len(documentos)}] ✗ {archivo.name}: {result.error_message}")

jsons = list(OUTPUT.glob('*.json'))
print()
print(f"Completado: {len(jsons)} JSONs en {OUTPUT}")