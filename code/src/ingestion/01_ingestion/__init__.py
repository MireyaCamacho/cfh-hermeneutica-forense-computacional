# CFH · Módulo de Ingesta
# Hermenéutica Forense Computacional

from .cleaner import JudicialTextCleaner, CleanedDocument, CleaningReport
from .metadata import JudicialMetadataExtractor, JudicialMetadata
from .segmenter import JudicialDocumentSegmenter, SegmentedDocument, DocumentSegment
from .pipeline import CFHIngestionPipeline, IngestionResult

__all__ = [
    "JudicialTextCleaner",
    "CleanedDocument",
    "CleaningReport",
    "JudicialMetadataExtractor",
    "JudicialMetadata",
    "JudicialDocumentSegmenter",
    "SegmentedDocument",
    "DocumentSegment",
    "CFHIngestionPipeline",
    "IngestionResult",
]
