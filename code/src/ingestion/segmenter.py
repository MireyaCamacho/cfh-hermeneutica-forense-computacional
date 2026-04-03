"""
CFH · Módulo de Segmentación por Secciones
===========================================
Proyecto: Hermenéutica Forense Computacional

Divide un documento judicial colombiano en sus secciones semánticamente
significativas para el análisis de injusticia discursiva.

La segmentación es el puente entre la ingesta bruta y el análisis NLP:
cada sección es tratada como una unidad discursiva independiente
sobre la que se calculará el DIS Score.

Secciones objetivo por corpus
------------------------------
Corpus A (Justicia Ordinaria):
  ENCABEZADO → HECHOS → CONSIDERACIONES → PRUEBAS → DECISIÓN

Corpus B (Autos JEP):
  ENCABEZADO → ANTECEDENTES → CONSIDERACIONES → RESUELVE

Corpus C (Audiencias JEP):
  ENCABEZADO → APERTURA → TESTIMONIO_COMPARECIENTE →
  TESTIMONIO_VICTIMA → DELIBERACION → CIERRE

Criterio de diseño:
  La segmentación es conservadora: ante la duda, una sección
  se asigna a "CUERPO" genérico. Nunca se pierde texto.
  La suma de todos los segmentos == texto original completo.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("cfh.ingestion.segmenter")


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class DocumentSegment:
    """Una sección semántica de un documento judicial."""
    section_id: str          # Etiqueta semántica: HECHOS, CONSIDERACIONES, etc.
    section_index: int       # Posición ordinal en el documento (0-based)
    text: str                # Texto de la sección
    char_start: int          # Posición de inicio en el documento completo
    char_end: int            # Posición de fin
    word_count: int = 0
    is_target_section: bool = False  # Secciones relevantes para DIS Score

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class SegmentedDocument:
    """Documento dividido en segmentos semánticos."""
    segments: list[DocumentSegment]
    corpus_type: str
    total_sections: int = 0
    coverage: float = 0.0        # % del texto asignado a sección nombrada
    segmentation_warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.total_sections = len(self.segments)

    def get_section(self, section_id: str) -> Optional[DocumentSegment]:
        """Retorna la primera sección con el section_id dado."""
        for seg in self.segments:
            if seg.section_id == section_id:
                return seg
        return None

    def get_target_sections(self) -> list[DocumentSegment]:
        """Secciones marcadas como relevantes para DIS Score."""
        return [s for s in self.segments if s.is_target_section]

    def to_dict(self) -> dict:
        return {
            "corpus_type": self.corpus_type,
            "total_sections": self.total_sections,
            "coverage": round(self.coverage, 4),
            "warnings": self.segmentation_warnings,
            "sections": [
                {
                    "section_id": s.section_id,
                    "index": s.section_index,
                    "word_count": s.word_count,
                    "is_target": s.is_target_section,
                    "char_range": [s.char_start, s.char_end],
                }
                for s in self.segments
            ],
        }


# ---------------------------------------------------------------------------
# Patrones de encabezado de sección por corpus
# ---------------------------------------------------------------------------

# Formato: (etiqueta_semántica, [patrones_regex], es_sección_target)
SECTIONS_CORPUS_A = [
    # Encabezado / identificación del caso
    ("ENCABEZADO", [
        r"^(?:TRIBUNAL|JUZGADO|CORTE|FISCALÍA|REPÚBLICA\s+DE\s+COLOMBIA)",
        r"^(?:EXPEDIENTE|RADICADO|PROCESO)\s+N",
    ], False),

    # ── Secciones específicas CSJ Sala de Casación Penal ─────────────────────
    # HECHOS JURÍDICAMENTE RELEVANTES — narración fáctica en casación penal
    # Alta densidad EBI/NV: aquí se describe cómo fue presentado el homicidio
    ("HECHOS", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?(?:LOS\s+)?HECHOS?\s+JURI[DÍ]DICAMENTE\s+RELEVANTES?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?(?:LOS\s+)?HECHOS?(?:\s+PROBADOS?)?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?RELACI[ÓO]N\s+DE\s+HECHOS",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?CIRCUNSTANCIAS\s+F[ÁA]CTICAS",
        r"(?:^|\n)PLANTEAMIENTO\s+DEL\s+CASO",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?SITUACI[ÓO]N\s+F[ÁA]CTICA",
    ], True),

    # CARGOS — formulación de cargos en casación: donde se cita el lenguaje
    # militar original para cuestionarlo jurídicamente (zona mixta EBI + NV)
    ("CARGOS", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?CARGOS?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?CARGO\s+[ÚU]NICO(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?PRIMER\s+CARGO(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?SEGUNDO\s+CARGO(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?DEMANDA\s+DE\s+CASACI[ÓO]N(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?FUNDAMENTOS?\s+DE\s+LA\s+CASACI[ÓO]N(?:\s*:|\s*\n)",
    ], True),

    # CONSIDERACIONES DE LA CORTE — análisis jurídico de la CSJ
    # Alta densidad de calificación jurídica: homicidio en persona protegida
    # vs. "baja en combate" — zona de máxima tensión discursiva
    ("CONSIDERACIONES", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?CONSIDERACIONES?\s+DE\s+LA\s+CORTE(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?CONSIDERACIONES?(?:\s+DEL?\s+(?:TRIBUNAL|JUZGADO|DESPACHO|SALA))?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?FUNDAMENTOS?\s+(?:JURI[DÍ]DICOS?|DE\s+DERECHO)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?MARCO\s+JUR[IÍ]DICO",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?AN[ÁA]LISIS\s+(?:JURI[DÍ]DICO|DEL?\s+CASO)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?PROBLEMA\s+JUR[IÍ]DICO(?:\s*:|\s*\n)",
    ], True),

    # ANTECEDENTES — en autos AP de segunda instancia
    ("ANTECEDENTES", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?ANTECEDENTES?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?RESUMEN\s+(?:DEL?\s+CASO|PROCESAL)(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?ACTUACI[ÓO]N\s+PROCESAL(?:\s*:|\s*\n)",
    ], False),

    # Pruebas
    ("PRUEBAS", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?(?:VALORACI[ÓO]N\s+)?PRUEBAS?(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?MATERIAL\s+PROBATORIO",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?M[ÉE]RITOS\s+(?:Y\s+)?PRUEBAS?",
    ], False),

    # Decisión / fallo — terminología de calificación final del hecho
    ("DECISIÓN", [
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?DECISI[ÓO]N(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?RESUELVE(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?FALLA(?:\s*:|\s*\n)",
        r"(?:^|\n)EN\s+M[ÉE]RITO\s+DE\s+LO\s+EXPUESTO",
        r"(?:^|\n)POR\s+LO\s+EXPUESTO[,\s]+(?:LA\s+)?(?:CORTE|SALA|JUZGADO|TRIBUNAL|DESPACHO)",
        r"(?:^|\n)(?:[IVXLC]+\.?\s+)?PARTE\s+RESOLUTIVA(?:\s*:|\s*\n)",
    ], True),
]

SECTIONS_CORPUS_B = [
    # ENCABEZADO: patrones muy restrictivos — solo al inicio del documento
    # Removido SALA DE RECONOCIMIENTO y JURISDICCIÓN ESPECIAL: aparecen
    # al inicio de párrafos en el cuerpo y crean falsos límites de sección
    ("ENCABEZADO", [
        r"^REPÚBLICA\s+DE\s+COLOMBIA\s*$",
        r"^AUTO\s+(?:CDG|N[Oº]\.?|No\.?)\s*[-–]?\s*(?:No\.?\s*)?\d",
    ], False),

    # ASUNTO — sección I de los autos JEP
    ("ASUNTO", [
        r"(?:^|\n)I\.\s+ASUNTO\s*$",
        r"(?:^|\n)I\.\s+OBJETO\s+(?:DEL?\s+AUTO|DE\s+LA\s+PROVIDENCIA)\s*$",
        r"(?:^|\n)ASUNTO\s*$",
    ], False),

    ("ANTECEDENTES", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?ANTECEDENTES?\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?ANTECEDENTES?\s+(?:PROCESALES?|DEL?\s+CASO)\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?HECHOS?\s+RELEVANTES?\s*$",
    ], False),

    # ZONA CRÍTICA: hechos y conductas — alta densidad EBI/SA/NV
    ("HECHOS_Y_CONDUCTAS", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?DETERMINACI[ÓO]N\s+DE\s+(?:LOS\s+)?HECHOS\s+Y\s+CONDUCTAS\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?HECHOS\s+Y\s+CONDUCTAS\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?DESCRIPCI[ÓO]N\s+DE\s+(?:LOS\s+)?HECHOS\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?UNIVERSO\s+DE\s+(?:LAS\s+)?V[IÍ]CTIMAS\s*$",
    ], True),

    # ZONA CRÍTICA: patrones macrocriminales
    ("PATRONES_MACROCRIMINALES", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?PATRONES?\s+(?:MACRO)?CRIMINALES?\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?MODALIDADES?\s+(?:DE\s+)?(?:LA\s+)?CONDUCTA\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?MODUS\s+OPERANDI\s*$",
    ], True),

    # ZONA CRÍTICA: calificación jurídica
    ("CALIFICACION_JURIDICA", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?CALIFICACI[ÓO]N\s+JUR[IÍ]DICA\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?TIPIFICACI[ÓO]N\s+(?:DE\s+(?:LAS?\s+)?CONDUCTAS?)?\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?CR[IÍ]MENES?\s+INTERNACIONALES?\s*$",
    ], True),

    # ZONA TARGET: reconocimiento — indicador REP
    ("RECONOCIMIENTO", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?RECONOCIMIENTO\s+DE\s+(?:VERDAD|RESPONSABILIDAD)\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?DECLARACI[ÓO]N\s+DE\s+RECONOCIMIENTO\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?APORTE\s+A\s+LA\s+VERDAD\s*$",
    ], True),

    ("CONSIDERACIONES", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?CONSIDERACIONES?\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?CONSIDERACIONES?\s+(?:DE\s+LA\s+SALA|JUR[IÍ]DICAS?)\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?AN[ÁA]LISIS\s+JUR[IÍ]DICO\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?MARCO\s+(?:JUR[IÍ]DICO|NORMATIVO)\s*$",
    ], True),

    ("RESUELVE", [
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?RESUELVE\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?PARTE\s+RESOLUTIVA\s*$",
        r"(?:^|\n)(?:[IVXLC]+\.\s+)?EN\s+VIRTUD\s+DE\s+LO\s+EXPUESTO\s*[,:]?\s*$",
    ], True),
]

SECTIONS_CORPUS_C = [
    ("ENCABEZADO", [
        r"^(?:ACTA|TRANSCRIPCI[ÓO]N)\s+(?:DE\s+)?AUDIENCIA",
        r"^JEP[,\s]",
    ], False),

    ("APERTURA", [
        r"(?:^|\n)(?:APERTURA|INSTALACI[ÓO]N)\s+(?:DE\s+LA\s+)?AUDIENCIA(?:\s*:|\s*\n)",
        r"(?:^|\n)PRESIDENTE[:\s]+(?:Se\s+instala|Declaramos)",
    ], False),

    ("TESTIMONIO_COMPARECIENTE", [
        r"(?:^|\n)(?:DECLARACI[ÓO]N|INTERVENCI[ÓO]N)\s+(?:DEL?\s+)?COMPARECIENTE(?:\s*:|\s*\n)",
        r"(?:^|\n)COMPARECIENTE[:\s]+",
        r"(?:^|\n)RECONOCIMIENTO\s+P[ÚU]BLICO(?:\s*:|\s*\n)",
    ], True),  # TARGET: voz del perpetrador — análisis de arrepentimiento

    ("TESTIMONIO_VICTIMA", [
        r"(?:^|\n)(?:DECLARACI[ÓO]N|INTERVENCI[ÓO]N)\s+(?:DE\s+(?:LA\s+)?)?V[ÍI]CTIMA(?:\s*:|\s*\n)",
        r"(?:^|\n)V[ÍI]CTIMA[:\s]+",
        r"(?:^|\n)FAMILIAR(?:\s+DE\s+V[ÍI]CTIMA)?[:\s]+",
        r"(?:^|\n)REPRESENTANTE\s+(?:DE\s+)?(?:LAS?\s+)?V[ÍI]CTIMAS?[:\s]+",
    ], True),  # TARGET: voz de la víctima — polo del ground truth

    ("DELIBERACION", [
        r"(?:^|\n)(?:DELIBERACI[ÓO]N|DEBATE)(?:\s*:|\s*\n)",
        r"(?:^|\n)PREGUNTAS?\s+(?:DEL?\s+)?(?:MAGISTRADO|JUEZ|SALA)(?:\s*:|\s*\n)",
    ], True),

    ("CIERRE", [
        r"(?:^|\n)CIERRE\s+(?:DE\s+LA\s+)?AUDIENCIA(?:\s*:|\s*\n)",
        r"(?:^|\n)(?:SE\s+)?(?:LEVANTA|SUSPENDE)\s+(?:LA\s+)?AUDIENCIA",
        r"(?:^|\n)(?:PRESIDENTE|MAGISTRADO)[:\s]+(?:Se\s+)?(?:levanta|declara\s+terminada)",
    ], False),
]

CORPUS_SECTION_MAP = {
    "A": SECTIONS_CORPUS_A,
    "B": SECTIONS_CORPUS_B,
    "C": SECTIONS_CORPUS_C,
}


# ---------------------------------------------------------------------------
# Clase principal de segmentación
# ---------------------------------------------------------------------------

class JudicialDocumentSegmenter:
    """
    Divide un texto judicial colombiano en secciones semánticamente
    etiquetadas.

    Estrategia: búsqueda de marcadores de sección por regex,
    con fallback a sección CUERPO para texto no clasificado.
    Garantiza cobertura total: todo el texto queda en algún segmento.

    Parámetros
    ----------
    corpus_type : str
        "A" | "B" | "C" — determina el esquema de secciones.
    min_section_words : int
        Secciones con menos palabras que este umbral se fusionan
        con la siguiente. Evita secciones de una sola línea.
    """

    def __init__(self, corpus_type: str, min_section_words: int = 20):
        assert corpus_type in ("A", "B", "C")
        self.corpus_type = corpus_type
        self.min_section_words = min_section_words
        self._section_defs = CORPUS_SECTION_MAP[corpus_type]

        # Compilar patrones: lista de (etiqueta, [compiled_patterns], is_target)
        self._compiled_sections = [
            (label, [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns], is_target)
            for label, patterns, is_target in self._section_defs
        ]

    def segment(self, clean_text: str) -> SegmentedDocument:
        """
        Segmenta el texto en secciones etiquetadas.

        Parámetros
        ----------
        clean_text : str
            Texto ya procesado por JudicialTextCleaner.

        Retorna
        -------
        SegmentedDocument con lista de DocumentSegment ordenados.
        """
        warnings = []

        # Paso 1: Encontrar todas las posiciones de inicio de sección
        boundaries = self._find_section_boundaries(clean_text)

        # Paso 2: Si no se encontró ninguna sección, tratar como CUERPO
        if not boundaries:
            warnings.append("no_sections_found: documento tratado como CUERPO único")
            seg = DocumentSegment(
                section_id="CUERPO",
                section_index=0,
                text=clean_text,
                char_start=0,
                char_end=len(clean_text),
                is_target_section=False,
            )
            return SegmentedDocument(
                segments=[seg],
                corpus_type=self.corpus_type,
                coverage=0.0,
                segmentation_warnings=warnings,
            )

        # Paso 3: Añadir límite de inicio si el primer marcador no está al principio
        if boundaries[0][0] > 100:
            boundaries.insert(0, (0, "ENCABEZADO", False))
            warnings.append("encabezado_inferred: inicio del documento asignado a ENCABEZADO")

        # Paso 4: Construir segmentos con sus rangos de texto
        segments = []
        for i, (start, label, is_target) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(clean_text)
            seg_text = clean_text[start:end].strip()

            # Saltear secciones vacías o muy cortas
            if len(seg_text.split()) < self.min_section_words and i < len(boundaries) - 1:
                warnings.append(
                    f"short_section_merged: {label} ({len(seg_text.split())} words) fusionada con siguiente"
                )
                # Extender el segmento previo si existe
                if segments:
                    prev = segments[-1]
                    merged_text = clean_text[prev.char_start:end].strip()
                    segments[-1] = DocumentSegment(
                        section_id=prev.section_id,
                        section_index=prev.section_index,
                        text=merged_text,
                        char_start=prev.char_start,
                        char_end=end,
                        is_target_section=prev.is_target_section,
                    )
                continue

            seg = DocumentSegment(
                section_id=label,
                section_index=len(segments),
                text=seg_text,
                char_start=start,
                char_end=end,
                is_target_section=is_target,
            )
            segments.append(seg)

        # Paso 5: Calcular cobertura
        named_chars = sum(
            s.char_end - s.char_start
            for s in segments
            if s.section_id != "CUERPO"
        )
        coverage = named_chars / len(clean_text) if clean_text else 0.0

        doc = SegmentedDocument(
            segments=segments,
            corpus_type=self.corpus_type,
            coverage=coverage,
            segmentation_warnings=warnings,
        )

        logger.info(
            f"Segmentación completada — {len(segments)} secciones | "
            f"cobertura: {coverage:.0%} | "
            f"target sections: {len(doc.get_target_sections())}"
        )
        return doc

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _find_section_boundaries(
        self, text: str
    ) -> list[tuple[int, str, bool]]:
        """
        Encuentra posiciones de inicio de sección.
        Retorna lista de (char_position, label, is_target) ordenada por posición.
        """
        found = []

        for label, patterns, is_target in self._compiled_sections:
            for pattern in patterns:
                for match in pattern.finditer(text):
                    pos = match.start()
                    # Evitar duplicados: no agregar si ya hay un marcador
                    # de la misma etiqueta a menos de 200 chars
                    duplicate = any(
                        existing_label == label and abs(existing_pos - pos) < 200
                        for existing_pos, existing_label, _ in found
                    )
                    if not duplicate:
                        found.append((pos, label, is_target))

        # Ordenar por posición en el documento
        found.sort(key=lambda x: x[0])

        # Resolver conflictos: si dos etiquetas distintas se solapan
        # dentro de 50 chars, conservar la más específica (la más tardía)
        deduplicated = []
        for i, (pos, label, is_target) in enumerate(found):
            if deduplicated and pos - deduplicated[-1][0] < 50:
                # Reemplazar: la más tardía en el doc es más específica
                deduplicated[-1] = (pos, label, is_target)
            else:
                deduplicated.append((pos, label, is_target))

        return deduplicated
