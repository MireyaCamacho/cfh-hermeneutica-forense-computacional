"""
CFH · Pipeline de Integración de Features
==========================================
Proyecto: Hermenéutica Forense Computacional

Orquesta todos los extractores sin GPU sobre el corpus procesado y
produce una tabla de indicadores lista para el módulo SEM.

Indicadores producidos por este módulo (sin GPU):
    y₁  EBI Score           — requiere ConfliBERT (placeholder 0.0 si no disponible)
    y₂  SA Score            — spaCy
    y₃  Distancia civil     — TF-IDF + Jaccard
    y₄  NV Score            — spaCy + léxico
    y₅  Tipo de corpus      — metadatos JSON
    y₆  Período normativo   — metadatos JSON
    y₁₀ REP Score           — spaCy + léxico

Indicadores pendientes (requieren Colab Pro / ConfliBERT):
    y₇  Surprisal           — ConfliBERT vs Peace-LM
    y₈  Distancia MAFAPO    — embeddings ConfliBERT
    y₉  Distancia CIDH      — embeddings ConfliBERT
    y₁₁ Convergencia rest.  — embeddings ConfliBERT

Salida:
    data/features/indicators_corpus_a.csv  — una fila por sección target
    data/features/indicators_corpus_b.csv
    data/features/indicators_all.csv       — corpus completo

Uso:
    python -m src.features.feature_pipeline --corpus A
    python -m src.features.feature_pipeline --corpus B
    python -m src.features.feature_pipeline --corpus all

    O desde Python:
        from src.features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = pipeline.run_corpus_a()
        df.to_csv("features_a.csv", index=False)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Importar extractores
from features.y2_sa_extractor import SAExtractor
from features.y3_civil_extractor import CivilLexiconExtractor
from features.y4_nv_extractor import NVExtractor
from features.y10_rep_extractor import REPExtractor
from features.context_extractor import ContextExtractor

# y₁ EBI — lazy import (requiere ConfliBERT/GPU)
try:
    from features.y1_ebi_extractor import EBIExtractor
    _EBI_AVAILABLE = True
except Exception:
    _EBI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cfh.pipeline")


# ---------------------------------------------------------------------------
# Secciones target por subsistema
# ---------------------------------------------------------------------------

# Solo estas secciones producen features — el resto del texto no entra al SEM
SECCIONES_TARGET_CE = {
    "HECHOS", "CONSIDERACIONES", "DECISION",
    "HECHOS_PROBADOS",
}

SECCIONES_TARGET_CSJ = {
    "HECHOS_JURIDICAMENTE_RELEVANTES", "HECHOS_JR",
    "CARGOS", "CARGO_UNICO",
    "CONSIDERACIONES_DE_LA_CORTE", "CONSIDERACIONES_CORTE",
    "DECISION",
}

SECCIONES_TARGET_JEP = {
    "HECHOS_Y_CONDUCTAS", "PATRONES_MACROCRIMINALES",
    "CALIFICACION_JURIDICA", "RECONOCIMIENTO", "RESUELVE",
}

ALL_TARGET_SECTIONS = (
    SECCIONES_TARGET_CE |
    SECCIONES_TARGET_CSJ |
    SECCIONES_TARGET_JEP
)


# ---------------------------------------------------------------------------
# Estructura del resultado por sección
# ---------------------------------------------------------------------------

@dataclass
class SectionFeatures:
    """Features completos de una sección judicial."""
    # ── Identificadores ──────────────────────────────────────────────────────
    doc_id: str
    section_id: str
    corpus_type: str          # "A-CE" | "A-CSJ" | "B" | "C"
    year: Optional[int]

    # ── Indicadores SEM ──────────────────────────────────────────────────────
    y1_ebi:  float = 0.0     # placeholder hasta ConfliBERT
    y2_sa:   float = 0.0
    y3_civil: float = 0.0
    y4_nv:   float = 0.0
    y5_corpus_type: int = 0
    y6_period: float = 0.0
    y10_rep: float = 0.0

    # Pendientes GPU
    y7_surprisal:  float = float("nan")
    y8_mafapo:     float = float("nan")
    y9_cidh:       float = float("nan")
    y11_conv_rest: float = float("nan")

    # ── Metadatos de calidad ─────────────────────────────────────────────────
    text_length_chars: int = 0
    n_sa_instances: int = 0
    n_nv_instances: int = 0
    n_rep_instances: int = 0
    n_nv_questioned: int = 0
    processing_time_s: float = 0.0
    has_warning: bool = False

    def to_dict(self) -> dict:
        return {
            "doc_id":           self.doc_id,
            "section_id":       self.section_id,
            "corpus_type":      self.corpus_type,
            "year":             self.year,
            "y1_ebi":           round(self.y1_ebi, 4),
            "y2_sa":            round(self.y2_sa, 4),
            "y3_civil":         round(self.y3_civil, 4),
            "y4_nv":            round(self.y4_nv, 4),
            "y5_corpus_type":   self.y5_corpus_type,
            "y6_period":        round(self.y6_period, 4),
            "y10_rep":          round(self.y10_rep, 4),
            "y7_surprisal":     self.y7_surprisal,
            "y8_mafapo":        self.y8_mafapo,
            "y9_cidh":          self.y9_cidh,
            "y11_conv_rest":    self.y11_conv_rest,
            "text_length_chars":   self.text_length_chars,
            "n_sa_instances":      self.n_sa_instances,
            "n_nv_instances":      self.n_nv_instances,
            "n_rep_instances":     self.n_rep_instances,
            "n_nv_questioned":     self.n_nv_questioned,
            "processing_time_s":   round(self.processing_time_s, 3),
            "has_warning":         self.has_warning,
        }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Orquestador de extracción de features CFH.

    Carga todos los extractores sin GPU una sola vez y los aplica
    sobre cada sección target del corpus procesado.

    Parámetros
    ----------
    spacy_model : str
        Modelo spaCy a usar en todos los extractores.
    use_ebi : bool
        Si True, intenta cargar el extractor y₁ EBI (requiere ConfliBERT).
        Default: False (produce y₁ = 0.0 como placeholder).
    """

    def __init__(
        self,
        spacy_model: str = "es_core_news_lg",
        use_ebi: bool = False,
    ):
        logger.info("Inicializando FeaturePipeline CFH...")
        t0 = time.perf_counter()

        # Cargar extractores (spaCy se carga una sola vez y se comparte)
        logger.info("Cargando extractor y₂ SA...")
        self.sa  = SAExtractor(model_name=spacy_model)

        logger.info("Cargando extractor y₃ Civil...")
        self.civ = CivilLexiconExtractor(spacy_model=spacy_model)

        logger.info("Cargando extractor y₄ NV...")
        self.nv  = NVExtractor(model_name=spacy_model)

        logger.info("Cargando extractor y₁₀ REP...")
        self.rep = REPExtractor(model_name=spacy_model)

        logger.info("Cargando extractor contexto y₅/y₆...")
        self.ctx = ContextExtractor()

        # EBI — opcional
        self.ebi = None
        if use_ebi and _EBI_AVAILABLE:
            try:
                logger.info("Cargando extractor y₁ EBI (ConfliBERT)...")
                self.ebi = EBIExtractor()
            except Exception as e:
                logger.warning(f"y₁ EBI no disponible: {e}. Usando placeholder 0.0.")

        elapsed = time.perf_counter() - t0
        # Aumentar límite de longitud de texto para spaCy
        # Los documentos JEP pueden superar 1M caracteres
        for extractor in [self.sa, self.nv, self.rep]:
            if hasattr(extractor, '_nlp'):
                extractor._nlp.max_length = 3_000_000
        if hasattr(self.civ, '_nlp') and self.civ._use_spacy:
            self.civ._nlp.max_length = 3_000_000

        logger.info(f"Pipeline inicializado en {elapsed:.1f}s")

    # ── Métodos públicos ──────────────────────────────────────────────────────

    def run_corpus_a(
        self,
        corpus_dir: Path,
        output_path: Optional[Path] = None,
        max_docs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extrae features de todo el Corpus A (CE + CSJ).

        Parámetros
        ----------
        corpus_dir : Path
            Directorio con los JSONs procesados del Corpus A.
            Ejemplo: data/processed/corpus_a/
        output_path : Path, opcional
            Si se especifica, guarda el CSV en esta ruta.
        max_docs : int, opcional
            Límite de documentos para pruebas rápidas.

        Retorna
        -------
        pd.DataFrame con una fila por sección target.
        """
        return self._run_corpus(
            corpus_dir=corpus_dir,
            corpus_label="A",
            target_sections=SECCIONES_TARGET_CE | SECCIONES_TARGET_CSJ,
            output_path=output_path,
            max_docs=max_docs,
        )

    def run_corpus_b(
        self,
        corpus_dir: Path,
        output_path: Optional[Path] = None,
        max_docs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Extrae features de todo el Corpus B (JEP)."""
        return self._run_corpus(
            corpus_dir=corpus_dir,
            corpus_label="B",
            target_sections=SECCIONES_TARGET_JEP,
            output_path=output_path,
            max_docs=max_docs,
        )

    def run_all(
        self,
        corpus_a_dir: Path,
        corpus_b_dir: Path,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Extrae features de ambos corpus y los concatena."""
        df_a = self.run_corpus_a(corpus_a_dir)
        df_b = self.run_corpus_b(corpus_b_dir)
        df = pd.concat([df_a, df_b], ignore_index=True)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"Features completos guardados: {output_path} ({len(df)} filas)")
        return df

    # ── Motor interno ─────────────────────────────────────────────────────────

    def _run_corpus(
        self,
        corpus_dir: Path,
        corpus_label: str,
        target_sections: set[str],
        output_path: Optional[Path],
        max_docs: Optional[int],
    ) -> pd.DataFrame:
        """Procesa todos los JSONs de un directorio de corpus."""
        corpus_dir = Path(corpus_dir)
        json_files = sorted(corpus_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No se encontraron JSONs en {corpus_dir}")
            return pd.DataFrame()

        if max_docs:
            json_files = json_files[:max_docs]

        logger.info(
            f"Procesando Corpus {corpus_label}: "
            f"{len(json_files)} documentos en {corpus_dir}"
        )

        all_features: list[SectionFeatures] = []
        n_sections_total = 0
        n_sections_processed = 0
        errors = 0

        for i, json_path in enumerate(json_files, 1):
            try:
                features_list = self._process_document(
                    json_path, target_sections, corpus_label
                )
                all_features.extend(features_list)
                n_sections_total += len(features_list)
                n_sections_processed += sum(
                    1 for f in features_list if not f.has_warning
                )

                if i % 50 == 0 or i == len(json_files):
                    logger.info(
                        f"  [{i}/{len(json_files)}] "
                        f"secciones procesadas: {n_sections_total}"
                    )
            except Exception as e:
                logger.error(f"Error en {json_path.name}: {e}")
                errors += 1

        logger.info(
            f"Corpus {corpus_label} completado: "
            f"{len(json_files)} docs | "
            f"{n_sections_total} secciones target | "
            f"{errors} errores"
        )

        if not all_features:
            return pd.DataFrame()

        df = pd.DataFrame([f.to_dict() for f in all_features])

        # Resumen estadístico
        self._log_summary(df, corpus_label)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"CSV guardado: {output_path} ({len(df)} filas)")

        return df

    def _process_document(
        self,
        json_path: Path,
        target_sections: set[str],
        corpus_label: str = "A",
    ) -> list[SectionFeatures]:
        """
        Extrae features de todas las secciones target de un documento.

        Lee el JSON, extrae el contexto institucional (y₅, y₆) una sola vez,
        y luego aplica los extractores de texto a cada sección target.
        """
        t0 = time.perf_counter()
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # doc_id está en metadata, no en la raíz
        metadata = data.get("metadata", {})
        doc_id = metadata.get("doc_id", json_path.stem)

        segmentation = data.get("segmentation", {})
        sections = segmentation.get("sections", [])

        # Cargar texto fuente desde source_file + char_range
        source_file = data.get("source_file", "")
        source_text = ""
        if source_file:
            candidates = [
                Path(source_file),
                Path(source_file.replace("\\", "/")),
                json_path.parent.parent.parent / source_file.replace("\\", "/"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    try:
                        source_text = candidate.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        source_text = candidate.read_text(encoding="latin-1")
                    break
            if not source_text:
                logger.warning(f"Texto fuente no encontrado: {source_file}")

        # Contexto institucional — una sola vez por documento
        ctx_result = self.ctx.extract_from_dict(data, doc_id=doc_id)

        # Corregir corpus_type si el JSON está mal etiquetado
        # (algunos JSONs del corpus A tienen corpus_type=B por error de ingesta)
        if corpus_label == "A" and ctx_result.corpus_type_raw == "B":
            import dataclasses
            ctx_result = dataclasses.replace(
                ctx_result,
                corpus_type_raw="A-CE" if "CE" in str(data.get("metadata",{}).get("tribunal","")) else "A-CSJ",
                y5_corpus_type=0,
            )

        features_list = []

        for section in sections:
            section_id = section.get("section_id", "")
            is_target = section.get("is_target", False)

            # Normalizar el section_id para comparación
            section_id_norm = section_id.upper().replace(" ", "_").replace("-", "_")

            # Filtrar solo secciones target
            if not is_target and section_id_norm not in target_sections:
                continue
            if section_id_norm not in target_sections and not is_target:
                continue

            # Extraer texto usando char_range sobre el texto fuente
            char_range = section.get("char_range", [])
            if source_text and len(char_range) == 2:
                text = source_text[char_range[0]:char_range[1]].strip()
            else:
                text = section.get("text", "").strip()

            if len(text) < 30:
                continue

            # Truncar secciones muy largas para spaCy en CPU
            # 8000 chars (~1500 palabras) es suficiente para capturar señal discursiva
            MAX_CHARS = 8000
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]

            # Extraer features de texto
            t_sec = time.perf_counter()

            r_sa  = self.sa.extract(text, doc_id=doc_id, section_id=section_id,
                                    corpus_type=ctx_result.corpus_type_raw)
            r_civ = self.civ.extract(text, doc_id=doc_id, section_id=section_id,
                                     corpus_type=ctx_result.corpus_type_raw)
            r_nv  = self.nv.extract(text, doc_id=doc_id, section_id=section_id,
                                    corpus_type=ctx_result.corpus_type_raw)
            r_rep = self.rep.extract(text, doc_id=doc_id, section_id=section_id,
                                     corpus_type=ctx_result.corpus_type_raw)

            # y₁ EBI — placeholder si no hay ConfliBERT
            y1_score = 0.0
            if self.ebi:
                try:
                    r_ebi = self.ebi.extract(text, doc_id=doc_id,
                                             section_id=section_id)
                    y1_score = r_ebi.score
                except Exception:
                    y1_score = 0.0

            sec_time = time.perf_counter() - t_sec

            has_warning = any([
                r_sa.warning, r_civ.warning, r_nv.warning, r_rep.warning
            ])

            features_list.append(SectionFeatures(
                doc_id=doc_id,
                section_id=section_id,
                corpus_type=ctx_result.corpus_type_raw,
                year=ctx_result.year,
                y1_ebi=y1_score,
                y2_sa=r_sa.score,
                y3_civil=r_civ.score,
                y4_nv=r_nv.score,
                y5_corpus_type=ctx_result.y5_corpus_type,
                y6_period=ctx_result.y6_period,
                y10_rep=r_rep.score,
                text_length_chars=len(text),
                n_sa_instances=r_sa.n_instances,
                n_nv_instances=r_nv.n_instances,
                n_rep_instances=r_rep.n_instances,
                n_nv_questioned=r_nv.n_questioned,
                processing_time_s=sec_time,
                has_warning=has_warning,
            ))

        return features_list

    def _log_summary(self, df: pd.DataFrame, corpus_label: str) -> None:
        """Imprime un resumen estadístico de los features extraídos."""
        if df.empty:
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMEN FEATURES — Corpus {corpus_label}")
        logger.info(f"{'='*60}")
        logger.info(f"Total secciones:    {len(df)}")
        logger.info(f"Total documentos:   {df['doc_id'].nunique()}")
        logger.info(f"Tipos de corpus:    {df['corpus_type'].value_counts().to_dict()}")
        logger.info(f"Secciones target:   {df['section_id'].value_counts().to_dict()}")
        logger.info(f"\nIndicadores (media ± std):")
        for col in ["y2_sa", "y3_civil", "y4_nv", "y10_rep", "y6_period"]:
            if col in df.columns:
                logger.info(
                    f"  {col:12s}: {df[col].mean():.3f} ± {df[col].std():.3f} "
                    f"[min={df[col].min():.3f}, max={df[col].max():.3f}]"
                )
        logger.info(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CFH Feature Pipeline — extrae indicadores del corpus"
    )
    parser.add_argument(
        "--corpus", choices=["A", "B", "all"], default="all",
        help="Corpus a procesar (default: all)"
    )
    parser.add_argument(
        "--corpus-a-dir", type=Path,
        default=Path("data/processed/corpus_a"),
        help="Directorio JSONs Corpus A"
    )
    parser.add_argument(
        "--corpus-b-dir", type=Path,
        default=Path("data/processed/corpus_b_json"),
        help="Directorio JSONs Corpus B"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/features"),
        help="Directorio de salida para CSVs"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Límite de documentos (para pruebas rápidas)"
    )
    parser.add_argument(
        "--use-ebi", action="store_true",
        help="Intentar usar extractor EBI con ConfliBERT"
    )
    args = parser.parse_args()

    pipeline = FeaturePipeline(use_ebi=args.use_ebi)

    if args.corpus == "A":
        df = pipeline.run_corpus_a(
            corpus_dir=args.corpus_a_dir,
            output_path=args.output_dir / "indicators_corpus_a.csv",
            max_docs=args.max_docs,
        )
        print(f"\n✓ Corpus A: {len(df)} secciones procesadas")

    elif args.corpus == "B":
        df = pipeline.run_corpus_b(
            corpus_dir=args.corpus_b_dir,
            output_path=args.output_dir / "indicators_corpus_b.csv",
            max_docs=args.max_docs,
        )
        print(f"\n✓ Corpus B: {len(df)} secciones procesadas")

    else:  # all
        df = pipeline.run_all(
            corpus_a_dir=args.corpus_a_dir,
            corpus_b_dir=args.corpus_b_dir,
            output_path=args.output_dir / "indicators_all.csv",
        )
        print(f"\n✓ Corpus completo: {len(df)} secciones procesadas")
        print(df[["corpus_type", "section_id", "y2_sa", "y4_nv", "y10_rep"]].head(10).to_string())
