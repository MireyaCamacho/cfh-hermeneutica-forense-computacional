"""
CFH · Extractor de Indicadores y₅ e y₆ — Contexto Institucional
================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₂ (Contexto Institucional)

Qué miden y₅ e y₆:
    Estos dos indicadores capturan la influencia del sistema de justicia
    y el período histórico como fuentes de variación exógena en el modelo
    SEM. Son las únicas variables del modelo que no requieren análisis
    de texto — se extraen directamente de los metadatos del documento.

    y₅ — Tipo de corpus (variable categórica ordinal):
        Codifica el sistema de justicia que produjo el documento.
        La escala ordinal refleja la distancia esperada al polo de
        reconocimiento: CE (más distante) → CSJ → JEP escrita → JEP oral.

        0: Corpus A-CE  — Consejo de Estado (justicia admin. ordinaria)
        1: Corpus A-CSJ — Corte Suprema de Justicia Sala de Casación Penal
        2: Corpus B-JEP — JEP autos y resoluciones (justicia transicional escrita)
        3: Corpus C-JEP — JEP audiencias (justicia transicional oral)

        La escala no es arbitraria: refleja la hipótesis teórica de que
        la justicia transicional (2, 3) produce mayor REP y menor EBI/NV
        que la justicia ordinaria (0, 1), y que la dimensión oral (3) es
        la más rica en REP por el carácter dialógico y performativo del
        reconocimiento público.

    y₆ — Período normativo (variable continua normalizada):
        Codifica el año de expedición del documento como variable continua
        en el rango [0, 1], donde 0 = 1994 (primer año del corpus A-CE)
        y 1 = 2023 (último año del corpus B).

        La variable temporal opera como indicador de contexto institucional
        porque el marco normativo y la jurisprudencia sobre DIH y derechos
        humanos evolucionó significativamente entre 1994 y 2023, con puntos
        de inflexión en 2008 (destitución de generales), 2016 (Acuerdo de
        Paz) y 2021-2022 (primeros autos y resoluciones JEP).

        Normalización: y₆ = (año - AÑO_MIN) / (AÑO_MAX - AÑO_MIN)

Diseño de implementación:
    El ContextExtractor lee los JSONs procesados por el pipeline de ingesta
    y extrae y₅ e y₆ de los campos de metadatos. Es el módulo más simple
    del sistema de extracción de features pero metodológicamente importante:
    garantiza que las variables exógenas del SEM reflejen los metadatos
    reales del corpus, no valores asignados manualmente.

Dependencias:
    - Solo biblioteca estándar de Python + numpy
    - Lee JSONs producidos por pipeline.py
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cfh.features.context")


# ---------------------------------------------------------------------------
# Constantes de codificación
# ---------------------------------------------------------------------------

# Codificación ordinal de y₅ — tipo de corpus
CORPUS_TYPE_ENCODING = {
    # Corpus A — Justicia ordinaria
    "A":     0,
    "A-CE":  0,
    "A-CSJ": 1,
    # Corpus B — JEP escrita
    "B":     2,
    "B-JEP": 2,
    # Corpus C — JEP oral
    "C":     3,
    "C-JEP": 3,
}

# Identificación del subsistema desde el campo 'tribunal' del JSON
TRIBUNAL_TO_CORPUS_SUBTYPE = {
    # Consejo de Estado y sus secciones → A-CE
    "Consejo de Estado":                         "A-CE",
    "Consejo de Estado · Sección Tercera":       "A-CE",
    "Consejo de Estado · Sección Primera":       "A-CE",
    "Consejo de Estado · Sección Segunda":       "A-CE",
    "Consejo de Estado · Sección Cuarta":        "A-CE",
    "Consejo de Estado · Sección Quinta":        "A-CE",
    "Consejo de Estado · Sala Plena":            "A-CE",
    # Tribunales administrativos → A-CE (instancia inferior del CE)
    "Tribunal Administrativo":                   "A-CE",
    # Corte Suprema de Justicia → A-CSJ
    "Corte Suprema de Justicia":                 "A-CSJ",
    "Corte Suprema · Sala Penal":                "A-CSJ",
    "Corte Suprema · Sala Civil":                "A-CSJ",
    "Corte Suprema · Sala Laboral":              "A-CSJ",
    # JEP → B
    "JEP":                                       "B",
    "JEP · Sala de Reconocimiento":             "B",
    "Jurisdicción Especial para la Paz":         "B",
    "JEP · SRVR":                                "B",
}

# Rango temporal del corpus CFH para normalización de y₆
AÑO_MIN = 1994   # primera sentencia CE del corpus
AÑO_MAX = 2023   # último auto JEP del corpus
AÑO_RANGE = AÑO_MAX - AÑO_MIN  # = 29

# Puntos de inflexión históricos (para análisis exploratorio, no para y₆)
INFLEXION_POINTS = {
    2008: "destitución_generales_barreto",
    2016: "acuerdo_paz_la_habana",
    2018: "apertura_macrocaso_003",
    2021: "primer_auto_determinacion_hechos",
    2022: "primera_resolucion_conclusiones",
}


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class ContextExtractionResult:
    """
    Resultado de la extracción de y₅ e y₆ para un documento.
    Contiene los valores codificados listos para el SEM.
    """
    # ── Indicadores del SEM ──────────────────────────────────────────────────
    y5_corpus_type: int          # codificación ordinal [0, 1, 2, 3]
    y6_period: float             # período normalizado [0, 1]

    # ── Metadatos originales ─────────────────────────────────────────────────
    doc_id: str
    corpus_type_raw: str         # "A-CE" | "A-CSJ" | "B" | "C"
    tribunal: Optional[str]
    date_issued: Optional[str]   # "AAAA-MM-DD"
    year: Optional[int]
    extraction_confidence: float

    # ── Flags ────────────────────────────────────────────────────────────────
    year_inferred: bool = False  # True si el año se infirió del nombre del archivo
    warning: Optional[str] = None

    @property
    def period_label(self) -> str:
        """Etiqueta descriptiva del período para logging y visualización."""
        if self.year is None:
            return "desconocido"
        if self.year <= 2008:
            return "pre-escándalo (≤2008)"
        elif self.year <= 2016:
            return "post-escándalo (2009-2016)"
        elif self.year <= 2020:
            return "post-acuerdo (2017-2020)"
        else:
            return "JEP activa (2021+)"

    def to_dict(self) -> dict:
        return {
            "y5_corpus_type":       self.y5_corpus_type,
            "y6_period":            round(self.y6_period, 4),
            "doc_id":               self.doc_id,
            "corpus_type_raw":      self.corpus_type_raw,
            "tribunal":             self.tribunal,
            "year":                 self.year,
            "period_label":         self.period_label,
            "extraction_confidence": self.extraction_confidence,
            "year_inferred":        self.year_inferred,
            "warning":              self.warning,
        }


# ---------------------------------------------------------------------------
# Extractor de contexto institucional
# ---------------------------------------------------------------------------

class ContextExtractor:
    """
    Extractor de los indicadores y₅ (tipo de corpus) e y₆ (período normativo).

    Lee los metadatos de los JSONs producidos por el pipeline de ingesta y
    codifica y₅ e y₆ de acuerdo con el esquema del modelo SEM.

    Parámetros
    ----------
    año_min : int
        Año mínimo del corpus para normalización de y₆. Default: 1994.
    año_max : int
        Año máximo del corpus para normalización de y₆. Default: 2023.
    """

    def __init__(self, año_min: int = AÑO_MIN, año_max: int = AÑO_MAX):
        self.año_min = año_min
        self.año_max = año_max
        self.año_range = año_max - año_min
        if self.año_range <= 0:
            raise ValueError(f"año_max ({año_max}) debe ser mayor que año_min ({año_min})")

    # ── Método principal ─────────────────────────────────────────────────────

    def extract_from_json(self, json_path: Path) -> ContextExtractionResult:
        """
        Extrae y₅ e y₆ de un JSON del pipeline de ingesta CFH.

        El JSON debe tener la estructura producida por pipeline.py:
        {
            "doc_id": "...",
            "metadata": {
                "tribunal": "...",
                "case_number": "...",
                "date_issued": "AAAA-MM-DD",
                "extraction_confidence": 0.90
            },
            "segmentation": {
                "corpus_type": "A" | "B" | "C"
            }
        }

        Parámetros
        ----------
        json_path : Path
            Ruta al archivo JSON procesado.

        Retorna
        -------
        ContextExtractionResult con y₅, y₆ y metadatos.
        """
        try:
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error leyendo {json_path}: {e}")
            return self._fallback_result(str(json_path.stem))

        doc_id = data.get("doc_id", json_path.stem)
        metadata = data.get("metadata", {})
        segmentation = data.get("segmentation", {})

        tribunal = metadata.get("tribunal")
        date_issued = metadata.get("date_issued")
        confidence = metadata.get("extraction_confidence", 0.0)
        corpus_type_raw_seg = segmentation.get("corpus_type", "A")

        # ── Determinar y₅ (tipo de corpus + subsistema) ───────────────────
        corpus_subtype, y5 = self._encode_corpus_type(
            corpus_type_raw_seg, tribunal, doc_id
        )

        # ── Determinar y₆ (período normalizado) ──────────────────────────
        year, year_inferred, warning_year = self._extract_year(
            date_issued, doc_id
        )
        y6 = self._normalize_year(year)

        warning = warning_year

        logger.debug(
            f"Contexto [{doc_id}]: y₅={y5} ({corpus_subtype}) | "
            f"y₆={y6:.3f} (año={year}) | conf={confidence:.0%}"
        )

        return ContextExtractionResult(
            y5_corpus_type=y5,
            y6_period=y6,
            doc_id=doc_id,
            corpus_type_raw=corpus_subtype,
            tribunal=tribunal,
            date_issued=date_issued,
            year=year,
            extraction_confidence=confidence,
            year_inferred=year_inferred,
            warning=warning,
        )

    def extract_from_dict(
        self,
        data: dict,
        doc_id: str = "unknown"
    ) -> ContextExtractionResult:
        """
        Extrae y₅ e y₆ de un diccionario de metadatos ya cargado.
        Útil cuando el JSON ya está en memoria.
        """
        metadata = data.get("metadata", {})
        segmentation = data.get("segmentation", {})
        doc_id = data.get("doc_id", doc_id)

        tribunal = metadata.get("tribunal")
        date_issued = metadata.get("date_issued")
        confidence = metadata.get("extraction_confidence", 0.0)
        corpus_type_raw_seg = segmentation.get("corpus_type", "A")

        corpus_subtype, y5 = self._encode_corpus_type(
            corpus_type_raw_seg, tribunal, doc_id
        )
        year, year_inferred, warning = self._extract_year(date_issued, doc_id)
        y6 = self._normalize_year(year)

        return ContextExtractionResult(
            y5_corpus_type=y5,
            y6_period=y6,
            doc_id=doc_id,
            corpus_type_raw=corpus_subtype,
            tribunal=tribunal,
            date_issued=date_issued,
            year=year,
            extraction_confidence=confidence,
            year_inferred=year_inferred,
            warning=warning,
        )

    def extract_batch(
        self, json_dir: Path, glob_pattern: str = "*.json"
    ) -> list[ContextExtractionResult]:
        """
        Procesa todos los JSONs de un directorio.

        Retorna lista de resultados ordenada por doc_id.
        """
        json_files = sorted(Path(json_dir).glob(glob_pattern))
        if not json_files:
            logger.warning(f"No se encontraron JSONs en {json_dir}")
            return []

        results = []
        for f in json_files:
            r = self.extract_from_json(f)
            results.append(r)

        logger.info(
            f"Contexto extraído: {len(results)} documentos | "
            f"y₅ distribución: "
            f"CE={sum(1 for r in results if r.y5_corpus_type == 0)}, "
            f"CSJ={sum(1 for r in results if r.y5_corpus_type == 1)}, "
            f"JEP={sum(1 for r in results if r.y5_corpus_type == 2)}, "
            f"Oral={sum(1 for r in results if r.y5_corpus_type == 3)}"
        )
        return results

    def to_feature_matrix(
        self, results: list[ContextExtractionResult]
    ) -> np.ndarray:
        """
        Convierte una lista de resultados en una matriz NumPy [n_docs, 2].
        Columna 0 = y₅, Columna 1 = y₆.
        Lista para alimentar directamente al módulo SEM.
        """
        matrix = np.array(
            [[r.y5_corpus_type, r.y6_period] for r in results],
            dtype=np.float64
        )
        return matrix

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _encode_corpus_type(
        self,
        corpus_type_seg: str,
        tribunal: Optional[str],
        doc_id: str,
    ) -> tuple[str, int]:
        """
        Determina el subtipo de corpus y lo codifica como y₅.

        Lógica de prioridad:
        1. Si el corpus_type del segmentador es B o C → directo
        2. Si es A, usar el tribunal para distinguir CE vs CSJ
        3. Si el tribunal no está disponible, inferir del doc_id
        """
        corpus_type_seg = corpus_type_seg.upper().strip()

        # Corpus B (JEP escrita) y C (JEP oral) — directo
        if corpus_type_seg in {"B", "C"}:
            subtype = "B" if corpus_type_seg == "B" else "C"
            return subtype, CORPUS_TYPE_ENCODING[subtype]

        # Corpus A — distinguir CE vs CSJ
        if tribunal:
            for t_key, subtype in TRIBUNAL_TO_CORPUS_SUBTYPE.items():
                if t_key.lower() in tribunal.lower():
                    return subtype, CORPUS_TYPE_ENCODING[subtype]

        # Fallback: inferir del doc_id (SP/AP → CSJ, radicado con guiones → CE)
        doc_id_upper = doc_id.upper()
        if re.match(r"^[SA]P\d", doc_id_upper):
            return "A-CSJ", CORPUS_TYPE_ENCODING["A-CSJ"]
        if re.match(r"^\d{5}-\d{2}-\d{2}", doc_id_upper):
            return "A-CE", CORPUS_TYPE_ENCODING["A-CE"]

        # Default: A-CE (subsistema más frecuente del corpus A)
        logger.debug(f"y₅ inferido como A-CE por defecto para {doc_id}")
        return "A-CE", CORPUS_TYPE_ENCODING["A-CE"]

    def _extract_year(
        self,
        date_issued: Optional[str],
        doc_id: str,
    ) -> tuple[Optional[int], bool, Optional[str]]:
        """
        Extrae el año de expedición del documento.

        Retorna (año, fue_inferido, advertencia).
        """
        # Desde date_issued "AAAA-MM-DD"
        if date_issued:
            m = re.match(r"(\d{4})", date_issued)
            if m:
                year = int(m.group(1))
                if 1990 <= year <= 2030:
                    return year, False, None
                else:
                    logger.warning(f"Año fuera de rango ({year}) para {doc_id}")

        # Fallback: inferir del doc_id
        # Formatos: SP036-2018(42374), AP4064-2016(46318), 42770(11-12-13)
        patterns = [
            r"[A-Z]{2}\d+-(\d{4})",          # SP036-2018, AP4064-2016
            r"\d+-(\d{4})-",                  # radicado CE: ...-2006-...
            r"\((\d{2})-\d{2}-(\d{2})\)",    # (11-12-13) → 2013
        ]
        for i, pattern in enumerate(patterns):
            m = re.search(pattern, doc_id)
            if m:
                if i == 2:  # formato (DD-MM-AA)
                    year_2d = int(m.group(2))
                    year = 2000 + year_2d if year_2d < 50 else 1900 + year_2d
                else:
                    year = int(m.group(1))
                if 1990 <= year <= 2030:
                    return year, True, f"year_from_filename: {doc_id}"

        return None, False, "year_not_found: usando valor medio del corpus"

    def _normalize_year(self, year: Optional[int]) -> float:
        """
        Normaliza el año al rango [0, 1].
        Años fuera del rango del corpus se clampean.
        """
        if year is None:
            # Usar la media del corpus como valor por defecto
            return 0.5
        clamped = max(self.año_min, min(self.año_max, year))
        return (clamped - self.año_min) / self.año_range

    def _fallback_result(self, doc_id: str) -> ContextExtractionResult:
        """Resultado de fallback cuando el JSON no se puede leer."""
        return ContextExtractionResult(
            y5_corpus_type=0,
            y6_period=0.5,
            doc_id=doc_id,
            corpus_type_raw="desconocido",
            tribunal=None,
            date_issued=None,
            year=None,
            extraction_confidence=0.0,
            warning="json_read_error",
        )
