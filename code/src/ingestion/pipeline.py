"""
CFH · Pipeline de Ingesta — Orquestador Principal
==================================================
Proyecto: Hermenéutica Forense Computacional

Este módulo conecta los tres sub-módulos (cleaner → metadata → segmenter)
en un pipeline lineal, determinístico y completamente auditable.

Cada documento procesado genera un registro de auditoría que incluye:
- SHA-256 del archivo original (antes de cualquier transformación)
- SHA-256 del texto limpio
- Timestamp ISO 8601 del procesamiento
- Versión del pipeline
- Reporte completo de transformaciones

Este registro es la "cadena de custodia digital" del corpus CFH.
Es inmutable: se escribe una vez y nunca se sobreescribe.

Uso típico con Antigravity:
--------------------------
    from ingestion.pipeline import CFHIngestionPipeline

    pipeline = CFHIngestionPipeline(corpus_type="A")
    result = pipeline.process_file("path/to/sentence.txt")

    # Guardar resultado
    pipeline.save_result(result, output_dir="data/processed/corpus_a/")
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Union

from .cleaner import JudicialTextCleaner, CleanedDocument
from .metadata import JudicialMetadataExtractor, JudicialMetadata
from .segmenter import JudicialDocumentSegmenter, SegmentedDocument

logger = logging.getLogger("cfh.ingestion.pipeline")

# Versión del pipeline — incrementar cuando cambien los módulos de limpieza
PIPELINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Tipo de resultado del pipeline
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """
    Resultado completo del pipeline para un documento.
    Contiene todo lo necesario para reproducibilidad y auditoría.
    """
    # Identificación
    pipeline_version: str
    corpus_type: str
    source_file: str
    processed_at: str              # ISO 8601 UTC

    # Cadena de custodia
    sha256_source: str             # Hash del archivo ANTES de procesar
    sha256_clean: str              # Hash del texto limpio

    # Resultados
    clean_text: str
    metadata: dict                 # JudicialMetadata.to_dict()
    segmentation: dict             # SegmentedDocument.to_dict()
    cleaning_report: dict

    # Estado del proceso
    success: bool = True
    error_message: Optional[str] = None

    def to_json(self, indent: int = 2) -> str:
        """Serializa el resultado a JSON. El texto limpio no se incluye aquí."""
        d = {
            "pipeline_version": self.pipeline_version,
            "corpus_type": self.corpus_type,
            "source_file": self.source_file,
            "processed_at": self.processed_at,
            "sha256_source": self.sha256_source,
            "sha256_clean": self.sha256_clean,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "segmentation": self.segmentation,
            "cleaning_report": self.cleaning_report,
        }
        return json.dumps(d, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class CFHIngestionPipeline:
    """
    Orquestador del pipeline de ingesta para el proyecto CFH.

    Ejecuta en orden:
    1. Lectura y hash del archivo fuente
    2. Limpieza (JudicialTextCleaner)
    3. Extracción de metadatos (JudicialMetadataExtractor)
    4. Segmentación (JudicialDocumentSegmenter)
    5. Generación del registro de auditoría

    Parámetros
    ----------
    corpus_type : str
        "A" | "B" | "C" — determina estrategias de limpieza y segmentación.
    apply_ocr_corrections : bool
        True para corpus A y B (documentos escaneados).
        False para corpus C (transcripciones nativas digitales).
    min_confidence_threshold : float
        Documentos con confianza de metadatos inferior a este umbral
        se procesan igual pero se marcan como low_confidence en el log.
    """

    def __init__(
        self,
        corpus_type: str,
        apply_ocr_corrections: Optional[bool] = None,
        min_confidence_threshold: float = 0.5,
    ):
        assert corpus_type in ("A", "B", "C")
        self.corpus_type = corpus_type
        self.min_confidence_threshold = min_confidence_threshold

        # Configuración por defecto según corpus
        if apply_ocr_corrections is None:
            apply_ocr_corrections = corpus_type in ("A", "B")

        self._cleaner = JudicialTextCleaner(
            apply_ocr_corrections=apply_ocr_corrections,
            preserve_case=True,  # Preservar case para extracción de metadatos
        )
        self._metadata_extractor = JudicialMetadataExtractor(corpus_type=corpus_type)
        self._segmenter = JudicialDocumentSegmenter(corpus_type=corpus_type)

        logger.info(
            f"CFHIngestionPipeline inicializado — "
            f"corpus={corpus_type} | OCR={apply_ocr_corrections} | "
            f"version={PIPELINE_VERSION}"
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Procesa un archivo de texto y retorna el resultado de ingesta completo.

        Parámetros
        ----------
        file_path : Union[str, Path]
            Ruta al archivo de texto (output de extractor PDF/DOCX).
            Encoding esperado: UTF-8. Si falla, intenta latin-1.

        Retorna
        -------
        IngestionResult con todos los artefactos del pipeline.
        Si ocurre un error no recuperable, retorna IngestionResult
        con success=False y error_message descriptivo.
        """
        file_path = Path(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"Procesando: {file_path.name} [corpus {self.corpus_type}]")

        try:
            # Paso 1: Leer archivo y calcular hash de fuente
            raw_text = self._read_file(file_path)
            sha256_source = self._sha256_text(raw_text)

            # Paso 2: Limpieza
            cleaned: CleanedDocument = self._cleaner.clean(
                raw_text, doc_id=file_path.stem
            )

            # Paso 3: Metadatos
            metadata: JudicialMetadata = self._metadata_extractor.extract(
                cleaned.text, filename=file_path.name
            )

            # Alerta de baja confianza
            if metadata.extraction_confidence < self.min_confidence_threshold:
                logger.warning(
                    f"[{metadata.doc_id[:8]}] Baja confianza de metadatos: "
                    f"{metadata.extraction_confidence:.0%} < "
                    f"{self.min_confidence_threshold:.0%}"
                )

            # Paso 4: Segmentación
            segmentation: SegmentedDocument = self._segmenter.segment(cleaned.text)

            # Paso 5: Registro de auditoría
            result = IngestionResult(
                pipeline_version=PIPELINE_VERSION,
                corpus_type=self.corpus_type,
                source_file=str(file_path),
                processed_at=timestamp,
                sha256_source=sha256_source,
                sha256_clean=metadata.doc_id,  # El doc_id ES el SHA-256 del texto limpio
                clean_text=cleaned.text,
                metadata=metadata.to_dict(),
                segmentation=segmentation.to_dict(),
                cleaning_report={
                    "original_length": cleaned.report.original_length,
                    "final_length": cleaned.report.final_length,
                    "ocr_corrections": cleaned.report.ocr_corrections,
                    "encoding_fixes": cleaned.report.encoding_fixes,
                    "chars_removed": cleaned.report.chars_removed,
                    "compression_ratio": cleaned.report.compression_ratio,
                    "operations_applied": cleaned.report.operations_applied,
                },
                success=True,
            )

            logger.info(
                f"✓ {file_path.name} — "
                f"doc_id={metadata.doc_id[:12]}... | "
                f"secciones={segmentation.total_sections} | "
                f"confianza_meta={metadata.extraction_confidence:.0%}"
            )
            return result

        except Exception as e:
            logger.error(f"✗ Error procesando {file_path.name}: {e}", exc_info=True)
            return IngestionResult(
                pipeline_version=PIPELINE_VERSION,
                corpus_type=self.corpus_type,
                source_file=str(file_path),
                processed_at=timestamp,
                sha256_source="",
                sha256_clean="",
                clean_text="",
                metadata={},
                segmentation={},
                cleaning_report={},
                success=False,
                error_message=str(e),
            )

    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".txt", ".text"),
        stop_on_error: bool = False,
    ) -> dict:
        """
        Procesa todos los archivos de un directorio en lote.

        Parámetros
        ----------
        input_dir : Union[str, Path]
            Directorio con los archivos de texto extraídos.
        output_dir : Union[str, Path]
            Directorio de salida para JSON de resultados.
        extensions : tuple
            Extensiones de archivo a procesar.
        stop_on_error : bool
            Si True, detiene el batch al primer error.
            Si False (default), continúa y reporta errores al final.

        Retorna
        -------
        dict con estadísticas del batch:
        {
            "total": int,
            "success": int,
            "failed": int,
            "low_confidence": int,
            "failed_files": list[str],
        }
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in input_dir.iterdir() if f.suffix in extensions]
        files.sort()

        stats = {"total": len(files), "success": 0, "failed": 0,
                 "low_confidence": 0, "failed_files": []}

        logger.info(f"Batch iniciado — {len(files)} archivos en {input_dir}")

        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {file_path.name}")
            result = self.process_file(file_path)

            if not result.success:
                stats["failed"] += 1
                stats["failed_files"].append(file_path.name)
                if stop_on_error:
                    raise RuntimeError(
                        f"Batch detenido en {file_path.name}: {result.error_message}"
                    )
                continue

            # Guardar resultado
            self.save_result(result, output_dir)
            stats["success"] += 1

            confidence = result.metadata.get("extraction_confidence", 0)
            if confidence < self.min_confidence_threshold:
                stats["low_confidence"] += 1

        # Guardar resumen del batch
        summary_path = output_dir / f"batch_summary_{self.corpus_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Batch completado — "
            f"✓ {stats['success']} | ✗ {stats['failed']} | "
            f"⚠ low_conf: {stats['low_confidence']}"
        )
        return stats

    def save_result(self, result: IngestionResult, output_dir: Union[str, Path]) -> Path:
        """
        Guarda el resultado de ingesta en disco.
        Estructura de salida:
          output_dir/
            {doc_id[:16]}.json    ← metadatos + segmentación + audit
            {doc_id[:16]}.txt     ← texto limpio
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Usar los primeros 16 chars del doc_id como nombre de archivo
        doc_id_short = result.sha256_clean[:16] if result.sha256_clean else "error"

        # JSON: todo excepto el texto limpio
        json_path = output_dir / f"{doc_id_short}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        # TXT: texto limpio separado
        txt_path = output_dir / f"{doc_id_short}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result.clean_text)

        logger.debug(f"Guardado: {json_path.name} + {txt_path.name}")
        return json_path

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _read_file(self, file_path: Path) -> str:
        """Lee archivo de texto con fallback de encoding."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"No se pudo decodificar {file_path.name} con ningún encoding conocido")

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
