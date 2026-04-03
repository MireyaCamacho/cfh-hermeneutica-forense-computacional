import sys
sys.path.insert(0, 'code/src')
from ingestion.pipeline import CFHIngestionPipeline
from pathlib import Path
import logging
logging.disable(logging.CRITICAL)

CORPUS = Path('data/raw/corpus_a')
OUTPUT = Path('data/processed/corpus_a')
OUTPUT.mkdir(parents=True, exist_ok=True)

pipeline = CFHIngestionPipeline(corpus_type='A')
documentos = list(CORPUS.iterdir())

print(f"Procesando {len(documentos)} documentos...")
print()

for i, archivo in enumerate(documentos, 1):
    result = pipeline.process_file(archivo)
    if result.success:
        pipeline.save_result(result, OUTPUT)
        conf = result.metadata.get('extraction_confidence', 0)
        simbolo = '✓' if conf >= 0.5 else '⚠'
        print(f"[{i:03d}/{len(documentos)}] {simbolo} {archivo.name[:45]}")
    else:
        print(f"[{i:03d}/{len(documentos)}] ✗ {archivo.name[:45]}")

archivos_generados = list(OUTPUT.glob('*.json'))
print()
print(f"Completado: {len(archivos_generados)} JSONs guardados en {OUTPUT}")