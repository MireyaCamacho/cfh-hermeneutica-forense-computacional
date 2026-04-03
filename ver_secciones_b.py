import sys, logging
sys.path.insert(0, 'code/src')
logging.disable(logging.CRITICAL)
from ingestion.pipeline import CFHIngestionPipeline
from pathlib import Path

pipeline = CFHIngestionPipeline(corpus_type='B')
result = pipeline.process_file(
    Path('data/processed/corpus_b/Auto_125_2021_Norte_Santander.txt')
)
seg = result.segmentation
print(f"Total secciones: {seg.get('total_sections')}")
print()
for s in seg.get('sections', []):
    t = '★' if s.get('is_target') else ' '
    print(f"  {t} [{s.get('section_id')}] {s.get('word_count')} palabras")

targets = [s for s in seg.get('sections', []) if s.get('is_target')]
print(f"\nTarget sections: {len(targets)}")