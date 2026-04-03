import sys
sys.path.insert(0, 'code/src')
from ingestion.pipeline import CFHIngestionPipeline
from pathlib import Path
import logging
logging.disable(logging.CRITICAL)

pipeline = CFHIngestionPipeline(corpus_type='A')
corpus = list(Path('data/raw/corpus_a').iterdir())

exitosos = 0
baja_confianza = 0

for f in corpus:
    result = pipeline.process_file(f)
    if result.success:
        exitosos += 1
        conf = result.metadata.get("extraction_confidence", 0)
        if conf < 0.5:
            baja_confianza += 1

print(f"Total documentos: {len(corpus)}")
print(f"Procesados exitosamente: {exitosos}")
print(f"Con baja confianza: {baja_confianza}")
print(f"Confianza alta (>=50%): {exitosos - baja_confianza}")