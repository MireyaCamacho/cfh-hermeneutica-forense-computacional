"""
CFH · Módulo de Extracción de Metadatos Judiciales
===================================================
Proyecto: Hermenéutica Forense Computacional

Extrae de forma determinística los metadatos estructurados de los tres corpus:

Corpus A — Justicia Ordinaria (Consejo Superior de la Judicatura, CSJ, CE, CC)
Corpus B — Autos escritos JEP
Corpus C — Transcripciones de audiencias orales JEP

Los metadatos son la llave de trazabilidad: cada documento procesado
debe poder vincularse a su fuente original.

Metadatos extraídos
-------------------
- doc_id          : hash SHA-256 del texto limpio (garantiza unicidad)
- corpus_type     : A | B | C
- tribunal        : nombre normalizado del tribunal emisor
- case_number     : número de radicado / auto / expediente
- date_issued     : fecha de emisión (ISO 8601)
- date_text_raw   : cadena de fecha tal como aparece en el documento
- parties         : diccionario {acusado, victima, fiscal, defensor}
- region          : departamento/municipio (para análisis geográfico)
- doc_type        : sentencia | auto | providencia | audiencia | acta
- page_count      : número de páginas (cuando aplica)
- word_count      : palabras en texto limpio
- extraction_confidence : float [0, 1] — confianza global de extracción
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("cfh.ingestion.metadata")


# ---------------------------------------------------------------------------
# Tipo de datos de metadatos
# ---------------------------------------------------------------------------

@dataclass
class JudicialMetadata:
    """
    Metadatos estructurados de un documento judicial colombiano.
    Inmutable tras creación. Serializable a dict para almacenamiento.
    """
    doc_id: str                          # SHA-256 del texto limpio
    corpus_type: str                     # "A" | "B" | "C"
    tribunal: Optional[str] = None
    tribunal_raw: Optional[str] = None  # Texto tal como aparece
    case_number: Optional[str] = None
    date_issued: Optional[str] = None   # ISO 8601: YYYY-MM-DD
    date_text_raw: Optional[str] = None
    parties: dict = field(default_factory=dict)
    region: Optional[str] = None
    doc_type: Optional[str] = None
    page_count: Optional[int] = None
    word_count: int = 0
    extraction_confidence: float = 0.0
    extraction_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "corpus_type": self.corpus_type,
            "tribunal": self.tribunal,
            "tribunal_raw": self.tribunal_raw,
            "case_number": self.case_number,
            "date_issued": self.date_issued,
            "date_text_raw": self.date_text_raw,
            "parties": self.parties,
            "region": self.region,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "extraction_confidence": round(self.extraction_confidence, 4),
            "extraction_warnings": self.extraction_warnings,
        }


# ---------------------------------------------------------------------------
# Tablas de referencia — Tribunales colombianos
# ---------------------------------------------------------------------------

# Mapa: fragmento de texto (regex) → nombre canónico normalizado
TRIBUNAL_PATTERNS = [
    # Cortes nacionales — primero las más específicas
    (r"Corte\s+Suprema\s+de\s+Justicia\s*[\r\n]+\s*SALA\s+DE\s+CASACI[ÓO]N\s+PENAL",
     "Corte Suprema · Sala Penal"),
    (r"Corte\s+Suprema\s+de\s+Justicia\s*[\r\n]+\s*SALA\s+DE\s+CASACI[ÓO]N\s+CIVIL",
     "Corte Suprema · Sala Civil"),
    (r"Corte\s+Suprema\s+de\s+Justicia\s*[\r\n]+\s*SALA\s+DE\s+CASACI[ÓO]N\s+LABORAL",
     "Corte Suprema · Sala Laboral"),
    (r"SALA\s+DE\s+CASACI[ÓO]N\s+PENAL", "Corte Suprema · Sala Penal"),
    (r"Corte\s+Suprema\s+de\s+Justicia", "Corte Suprema de Justicia"),
    (r"Corte\s+Constitucional", "Corte Constitucional"),
    # Consejo de Estado — secciones específicas (antes del patrón general)
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO\s+SECCI[OÓ]N\s+TERCERA",
     "Consejo de Estado · Sección Tercera"),
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO\s+SECCI[OÓ]N\s+SEGUNDA",
     "Consejo de Estado · Sección Segunda"),
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO\s+SECCI[OÓ]N\s+PRIMERA",
     "Consejo de Estado · Sección Primera"),
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO\s+SECCI[OÓ]N\s+CUARTA",
     "Consejo de Estado · Sección Cuarta"),
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO\s+SECCI[OÓ]N\s+QUINTA",
     "Consejo de Estado · Sección Quinta"),
    (r"SALA\s+DE\s+LO\s+CONTENCIOSO\s+ADMINISTRATIVO",
     "Consejo de Estado · Sala Contencioso"),
    (r"Consejo\s+de\s+Estado", "Consejo de Estado"),
    # Tribunales administrativos departamentales (corpus CE)
    (r"TRIBUNAL\s+ADMINISTRATIVO\s+DE\s+(?:CUNDINAMARCA|BOGOT[AÁ])",
     "Tribunal Administrativo de Cundinamarca"),
    (r"TRIBUNAL\s+ADMINISTRATIVO\s+DE\s+ANTIOQUIA",
     "Tribunal Administrativo de Antioquia"),
    (r"TRIBUNAL\s+ADMINISTRATIVO\s+DE\s+(\w+)",
     "Tribunal Administrativo · \\1"),
    (r"Tribunal\s+Administrativo\s+de\s+(\w+)",
     "Tribunal Administrativo · \\1"),
    # Rama judicial genérica — detectar cuando solo aparece el encabezado
    (r"RAMA\s+JUDICIAL\s+DEL\s+PODER\s+P[UÚ]BLICO", "Rama Judicial"),
    (r"RAMA\s+JUDICIAL", "Rama Judicial"),
    # JEP
    (r"Jurisdicci[oó]n\s+Especial\s+para\s+la\s+Paz|JEP", "JEP"),
    (r"Secci[oó]n\s+de\s+Reconocimiento\s+de\s+Verdad", "JEP · Sección Reconocimiento"),
    (r"Secci[oó]n\s+de\s+Ausencia\s+de\s+Reconocimiento", "JEP · Sección Ausencia"),
    (r"Tribunal\s+para\s+la\s+Paz", "JEP · Tribunal para la Paz"),
    (r"Sala\s+de\s+Amnist[ií]a\s+e\s+Ind[uú]lto", "JEP · Sala Amnistía"),
    # Fiscalía
    (r"Fiscal[ií]a\s+General\s+de\s+la\s+Naci[oó]n", "Fiscalía General de la Nación"),
    (r"Unidad\s+Nacional\s+de\s+Fiscal[ií]as", "Fiscalía · UNASE"),
    # Justicia ordinaria departamental
    (r"Tribunal\s+Superior\s+(?:del?\s+)?(?:Distrito\s+Judicial\s+de\s+)?(\w+)",
     "Tribunal Superior · \\1"),
    (r"Juzgado\s+(\d+[°º]?)\s+(?:Penal\s+)?(?:del?\s+Circuito|Municipal)(?:\s+de\s+(\w+))?",
     "Juzgado \\1 · \\2"),
    # Procuraduría
    (r"Procuradur[ií]a\s+General\s+de\s+la\s+Naci[oó]n", "Procuraduría General"),
]

# Tipos documentales por corpus
DOC_TYPE_PATTERNS = {
    "sentencia": r"\b[Ss]entencia\b",
    "auto": r"\b[Aa]uto\b(?!\s+de\s+fondo)",
    "providencia": r"\b[Pp]rovidencia\b",
    "audiencia": r"\b[Aa]udiencia\b",
    "acta": r"\b[Aa]cta\s+de\s+[Aa]udiencia\b",
    "resolucion": r"\b[Rr]esoluci[oó]n\b",
    "fallo": r"\b[Ff]allo\b",
}

# Departamentos de Colombia — para extracción geográfica
COLOMBIA_REGIONS = [
    "Antioquia", "Atlántico", "Bogotá", "Bolívar", "Boyacá",
    "Caldas", "Caquetá", "Casanare", "Cauca", "Cesar",
    "Chocó", "Córdoba", "Cundinamarca", "Guajira", "Huila",
    "Magdalena", "Meta", "Nariño", "Norte de Santander",
    "Putumayo", "Quindío", "Risaralda", "Santander", "Sucre",
    "Tolima", "Valle del Cauca", "Arauca", "Vichada", "Vaupés",
    # Ciudades relevantes para falsos positivos
    "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga",
    "Cúcuta", "Montería", "Villavicencio", "Sincelejo", "Pasto",
    "Soacha", "La Guajira",
]

# Meses en español para parseo de fechas
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    # Abreviaciones
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}


# ---------------------------------------------------------------------------
# Clase principal de extracción
# ---------------------------------------------------------------------------

class JudicialMetadataExtractor:
    """
    Extrae metadatos estructurados de texto judicial colombiano normalizado.

    Diseño: extrae por campo de forma independiente.
    Si un campo falla, reporta warning en lugar de lanzar excepción.
    Esto garantiza que un documento con metadatos parciales
    siga siendo procesado (nunca se pierde información).

    Parámetros
    ----------
    corpus_type : str
        "A" (Justicia Ordinaria), "B" (Autos JEP), "C" (Audiencias JEP)
    head_chars : int
        Cuántos caracteres del inicio del documento usar para extracción
        de metadatos de cabecera. Por defecto 3000 (suficiente para
        el encabezado judicial estándar colombiano).
    """

    def __init__(self, corpus_type: str, head_chars: int = 3000):
        assert corpus_type in ("A", "B", "C"), \
            f"corpus_type debe ser A, B o C — recibido: {corpus_type}"
        self.corpus_type = corpus_type
        self.head_chars = head_chars

        # Compilar patrones de tribunal
        self._tribunal_patterns = [
            (re.compile(p, re.IGNORECASE), norm)
            for p, norm in TRIBUNAL_PATTERNS
        ]
        self._doc_type_patterns = {
            dtype: re.compile(p, re.IGNORECASE)
            for dtype, p in DOC_TYPE_PATTERNS.items()
        }
        self._region_pattern = re.compile(
            r"\b(" + "|".join(re.escape(r) for r in COLOMBIA_REGIONS) + r")\b",
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def extract(self, clean_text: str, filename: Optional[str] = None) -> JudicialMetadata:
        """
        Extrae todos los metadatos de un documento limpio.

        Parámetros
        ----------
        clean_text : str
            Texto ya procesado por JudicialTextCleaner.
        filename : str, optional
            Nombre del archivo origen (se usa como hint para metadatos).

        Retorna
        -------
        JudicialMetadata con campos extraídos y score de confianza.
        """
        warnings_log = []
        head = clean_text[:self.head_chars]

        # Generar doc_id determinístico
        doc_id = self._compute_doc_id(clean_text)

        # Extraer cada campo de forma independiente
        tribunal, tribunal_raw = self._extract_tribunal(head, warnings_log)
        case_number = self._extract_case_number(head, filename, warnings_log)
        date_iso, date_raw = self._extract_date(head, warnings_log)
        parties = self._extract_parties(head, warnings_log)
        region = self._extract_region(clean_text)
        doc_type = self._extract_doc_type(head)
        word_count = len(clean_text.split())

        # Calcular confianza global
        confidence = self._compute_confidence(
            tribunal, case_number, date_iso, parties, region, doc_type
        )

        meta = JudicialMetadata(
            doc_id=doc_id,
            corpus_type=self.corpus_type,
            tribunal=tribunal,
            tribunal_raw=tribunal_raw,
            case_number=case_number,
            date_issued=date_iso,
            date_text_raw=date_raw,
            parties=parties,
            region=region,
            doc_type=doc_type,
            word_count=word_count,
            extraction_confidence=confidence,
            extraction_warnings=warnings_log,
        )

        logger.info(
            f"[{doc_id[:8]}] Metadatos extraídos — "
            f"tribunal={tribunal!r} | radicado={case_number!r} | "
            f"fecha={date_iso!r} | confianza={confidence:.0%}"
        )
        if warnings_log:
            for w in warnings_log:
                logger.warning(f"[{doc_id[:8]}] ⚠ {w}")

        return meta

    # ------------------------------------------------------------------
    # Extracción por campo
    # ------------------------------------------------------------------

    def _compute_doc_id(self, text: str) -> str:
        """SHA-256 del texto limpio — identificador determinístico y único."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_tribunal(
        self, head: str, warnings: list
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Identifica el tribunal emisor y devuelve
        (nombre_canónico, texto_bruto_encontrado).
        """
        for pattern, canonical in self._tribunal_patterns:
            match = pattern.search(head)
            if match:
                raw = match.group(0)
                # Si el canonical contiene backreferences, resolverlas
                if r"\1" in canonical:
                    try:
                        resolved = pattern.sub(canonical, raw)
                        # Limpiar None que aparece cuando grupo no capturó
                        resolved = re.sub(r"\s*·\s*None", "", resolved)
                        return resolved.strip(), raw
                    except Exception:
                        return canonical.replace(r"\1", "").strip(), raw
                return canonical, raw

        warnings.append("tribunal_not_found: no se identificó tribunal emisor en los primeros 3000 chars")
        return None, None

    def _extract_case_number(
        self, head: str, filename: Optional[str], warnings: list
    ) -> Optional[str]:
        """
        Extrae número de radicado o expediente.

        Formatos colombianos soportados:
        - Radicado SIREJ: 11001310300120100005400
        - Radicado JEP: 20221510248001CE
        - Expediente: 2005-00123-01
        - Auto JEP: Auto No. 019 de 2021
        - Proceso penal: rad. 2007-0015
        """
        patterns = [
            # ── Corte Suprema de Justicia — Sala de Casación Penal ────────────
            # Formato con etiqueta: "Radicación SP036-2018(42374)"
            # o "Radicación núm. AP4064-2016(46318)"
            (r"[Rr]adicaci[oó]n\s*(?:n[uú]m\.?\s*)?:?\s*"
             r"([AS]P\d{1,6}-\d{4}(?:\(\d+\))?)",
             "CSJ_SP_AP_etiqueta"),
            # Radicado CSJ sin etiqueta — SP/AP seguido de número y año
            # Ej: SP036-2018(42374), AP4064-2016(46318)
            (r"\b([AS]P\d{1,6}-\d{4}(?:\(\d+\))?)\b",
             "CSJ_SP_AP_raw"),
            # CSJ: radicado numérico solo (NO seguido de guión — para no capturar CE)
            (r"[Rr]adicaci[oó]n\s*(?:n[uú]m\.?\s*)?:?\s*(\d{4,6})\b(?!-)",
             "CSJ_radicacion_numerica"),
            (r"[Rr]adicado\s*[Nn][oº]?\.?\s*:?\s*(\d{4,6})\b(?!-)",
             "CSJ_radicado_numerico"),
            # AC (autos de constitucionalidad CSJ)
            (r"\b(AC\d{4}-\d{4}(?:\(\d{4}-\d{5}-\d{2}\))?)\b",
             "CSJ_AC"),

            # ── Consejo de Estado — formato con guiones ───────────────────────
            (r"[Rr]adicaci[oó]n\s*(?:[Nn]o?\.?\s*)?:?\s*"
             r"(\d{5}-\d{2}-\d{2}-\d{3}-\d{4}-\d{5}-\d{2}(?:\(\d+\))?)",
             "CE_radicacion_guiones"),
            (r"(\d{5}-\d{2}-\d{2}-\d{3}-\d{4}-\d{5}-\d{2})(?:\((\d+)\))?",
             "CE_radicado_raw"),

            # ── Otros formatos ────────────────────────────────────────────────
            (r"[Rr]adicado\s*[Nn]o?\.?\s*:?\s*(\d{22,23})", "SIREJ_22d"),
            (r"[Rr]adicado\s*[Nn]o?\.?\s*:?\s*(\d{16}[A-Z]{2})", "JEP_radicado"),
            (r"Auto\s+[Nn]o?\.\s*(\d{3,4})\s+de\s+(\d{4})", "JEP_auto"),
            (r"[Ee]xpediente\s*[Nn]o?\.?\s*:?\s*(\d{4}-\d{5}-\d{2})", "expediente"),
            (r"[Pp]roceso\s+(?:[Nn]o?\.?\s*)?(\d{4}-\d{4,6})", "proceso_penal"),
            (r"[Mm]acrocaso\s+(?:N[oº]\.?\s*)?(\d{3})", "JEP_macrocaso"),
            (r"(?:[Rr]adicado|[Ee]xpediente|[Pp]roceso)\s*:?\s*(\d{10,})", "fallback_numeric"),
        ]

        for pattern, label in patterns:
            m = re.search(pattern, head)
            if m:
                groups = [g for g in m.groups() if g]
                if label == "CE_radicado_raw" and len(groups) == 2:
                    value = f"{groups[0]}({groups[1]})"
                else:
                    value = groups[0] if len(groups) == 1 else "-".join(groups)
                logger.debug(f"Radicado encontrado [{label}]: {value}")
                return value

        # ── Fallbacks desde nombre de archivo ─────────────────────────────────
        if filename:
            # CSJ: SP/AP + número + año (ej. SP036-2018, AP4064-2016)
            csj_fn = re.search(r"([AS]P\d{1,6}-\d{4}(?:\(\d+\))?)", filename)
            if csj_fn:
                warnings.append(
                    f"case_number_from_filename: radicado CSJ del nombre ({filename})"
                )
                return csj_fn.group(1)

            # CSJ: radicado numérico en paréntesis al final (ej. SP036-2018(42374))
            csj_exp = re.search(r"\((\d{4,6})\)", filename)
            if csj_exp:
                warnings.append(
                    f"case_number_from_filename: expediente CSJ del nombre ({filename})"
                )
                return csj_exp.group(1)

            # CE: radicado con guiones
            ce_fn = re.search(
                r"(\d{5}-\d{2}-\d{2}-\d{3}-\d{4}-\d{5}-\d{2}(?:\(\d+\))?)",
                filename
            )
            if ce_fn:
                warnings.append(
                    f"case_number_from_filename: radicado CE del nombre ({filename})"
                )
                return ce_fn.group(1)

            # Último fallback: secuencia numérica larga
            fn_match = re.search(r"(\d{10,23})", filename)
            if fn_match:
                warnings.append(
                    f"case_number_from_filename: numérico del nombre ({filename})"
                )
                return fn_match.group(1)

        warnings.append("case_number_not_found: no se encontró número de radicado")
        return None

    def _extract_date(
        self, head: str, warnings: list
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extrae la fecha de emisión del documento.

        Formatos soportados:
        - "quince (15) de marzo de dos mil ocho (2008)"  ← Colombia JEP
        - "(15) de marzo de 2008"
        - "15 de marzo de 2007"
        - "15/03/2007"
        - "2007-03-15"

        Estrategia: buscar PRIMERO en zona de cabecera (primeros 600 chars),
        luego en el resto del encabezado. Evita capturar fechas del cuerpo.
        """
        zones = [head[:600], head]  # preferir zona de cabecera

        YEAR_WORDS_MAP = {
            # Siglo XXI
            "dos mil": 2000, "dos mil uno": 2001, "dos mil dos": 2002,
            "dos mil tres": 2003, "dos mil cuatro": 2004, "dos mil cinco": 2005,
            "dos mil seis": 2006, "dos mil siete": 2007, "dos mil ocho": 2008,
            "dos mil nueve": 2009, "dos mil diez": 2010,
            "dos mil once": 2011, "dos mil doce": 2012,
            "dos mil trece": 2013, "dos mil catorce": 2014,
            "dos mil quince": 2015, "dos mil dieciseis": 2016,
            "dos mil diecisiete": 2017, "dos mil dieciocho": 2018,
            "dos mil diecinueve": 2019, "dos mil veinte": 2020,
            "dos mil veintiuno": 2021, "dos mil veintidos": 2022,
            "dos mil veintitres": 2023, "dos mil veinticuatro": 2024,
            # Siglo XX — muy frecuentes en corpus CE 1993-2000
            "mil novecientos noventa": 1990,
            "mil novecientos noventa y uno": 1991,
            "mil novecientos noventa y dos": 1992,
            "mil novecientos noventa y tres": 1993,
            "mil novecientos noventa y cuatro": 1994,
            "mil novecientos noventa y cinco": 1995,
            "mil novecientos noventa y seis": 1996,
            "mil novecientos noventa y siete": 1997,
            "mil novecientos noventa y ocho": 1998,
            "mil novecientos noventa y nueve": 1999,
            "dos mil": 2000,
        }

        # Patrón flexible para año en letras siglo XX y XXI
        YEAR_WORDS_PATTERN = (
            r"(?:mil\s+novecientos\s+\w+(?:\s+y\s+\w+)?|dos\s+mil(?:\s+\w+)?)"
        )
        MES_GROUP = "|".join(MESES_ES.keys())

        for zone in zones:
            # P1: día en letras + (dígito) + mes + año en letras siglo XXI + (año dígito)
            # "quince (15) de marzo de dos mil ocho (2008)"
            p1 = re.compile(
                r"\(\s*(\d{1,2})\s*\)\s+de\s+(" + MES_GROUP + r")\s+de\s+"
                r"(dos\s+mil(?:\s+\w+)?)\s*\(\s*(\d{4})\s*\)",
                re.IGNORECASE,
            )
            m = p1.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year = int(m.group(4))
                if self._valid_date(day, month, year):
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

            # P1-bis: (dígito) + mes + año en letras siglo XX + (año dígito)
            # "catorce (14) de marzo de mil novecientos noventa y cinco (1995)"
            p1b = re.compile(
                r"\(\s*(\d{1,2})\s*\)\s+de\s+(" + MES_GROUP + r")\s+de\s+"
                r"(mil\s+novecientos\s+\w+(?:\s+y\s+\w+)?)\s*\(\s*(\d{4})\s*\)",
                re.IGNORECASE,
            )
            m = p1b.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year = int(m.group(4))
                if self._valid_date(day, month, year):
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

            # P2: dígito directo + mes + año dígito — "15 de marzo de 2007"
            p2 = re.compile(
                r"(\d{1,2})\s+de\s+(" + MES_GROUP + r")\s+de\s+(\d{4})",
                re.IGNORECASE,
            )
            m = p2.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year = int(m.group(3))
                # Filtrar años fuera del rango del corpus
                if self._valid_date(day, month, year) and 2000 <= year <= 2025:
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

            # P3: dígito + mes + año en letras siglo XXI
            p3 = re.compile(
                r"(\d{1,2})\s+de\s+(" + MES_GROUP + r")\s+de\s+"
                r"(dos\s+mil(?:\s+\w+)?)",
                re.IGNORECASE,
            )
            m = p3.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year_str = m.group(3).lower().strip()
                year = YEAR_WORDS_MAP.get(year_str)
                if year and self._valid_date(day, month, year):
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

            # P3-bis: dígito + mes + año en letras siglo XX
            # "quince de marzo de mil novecientos noventa y cinco"
            p3b = re.compile(
                r"(\d{1,2})\s+de\s+(" + MES_GROUP + r")\s+de\s+"
                r"(mil\s+novecientos\s+\w+(?:\s+y\s+\w+)?)",
                re.IGNORECASE,
            )
            m = p3b.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year_str = m.group(3).lower().strip()
                year = YEAR_WORDS_MAP.get(year_str)
                if year and self._valid_date(day, month, year):
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

            # P3-ter: año con dígito entre paréntesis (común en CE)
            # "de mil novecientos noventa y cinco (1995)"
            p3t = re.compile(
                r"(\d{1,2})\s+de\s+(" + MES_GROUP + r")\s+de\s+"
                r"(?:mil\s+novecientos\s+\w+(?:\s+y\s+\w+)?|dos\s+mil(?:\s+\w+)?)"
                r"\s*\(\s*(\d{4})\s*\)",
                re.IGNORECASE,
            )
            m = p3t.search(zone)
            if m:
                day = int(m.group(1))
                month = MESES_ES.get(m.group(2).lower(), 0)
                year = int(m.group(3))
                if self._valid_date(day, month, year):
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

        # P4: formato numérico DD/MM/AAAA en cabecera — ampliar rango a 1990-2025
        for zone in zones:
            p4 = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})")
            m = p4.search(zone)
            if m:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if self._valid_date(day, month, year) and 1990 <= year <= 2025:
                    return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

        # P5: ISO AAAA-MM-DD
        p5 = re.compile(r"(20\d{2})-(0[1-9]|1[0-2])-(\d{2})")
        m = p5.search(head)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if self._valid_date(day, month, year):
                return f"{year:04d}-{month:02d}-{day:02d}", m.group(0)

        warnings.append("date_not_found: no se encontró fecha de emisión")
        return None, None

    def _extract_parties(self, head: str, warnings: list) -> dict:
        """
        Extrae partes del proceso: procesado/acusado, víctima/afectado,
        fiscal, defensor.

        Nota: por ética de investigación, los nombres se anonimizar_an
        en la versión final. Este módulo extrae para trazabilidad interna.
        """
        parties = {}

        # Rangos militares / títulos que pueden preceder al nombre
        RANK_PREFIX = (
            r"(?:(?:Sargento\s+(?:Primero|Segundo|Mayor)|Cabo\s+(?:Primero|Segundo)|"
            r"Suboficial|Teniente\s+(?:Coronel)?|Mayor|Coronel\s*(?:\(R\))?|General|"
            r"Brigadier|Capitán|Subteniente|Soldado|Dr\.?|Dra\.?|Señor|Señora)\s+)*"
        )

        # Procesado / imputado / acusado
        accused_patterns = [
            r"(?:[Pp]rocesado|[Aa]cusado|[Ii]mputado|[Cc]ompareciente)[:\s]+" + RANK_PREFIX +
            r"([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+){1,4})",
            r"[Cc]ontra\s+" + RANK_PREFIX +
            r"([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+){1,4})",
        ]
        for p in accused_patterns:
            m = re.search(p, head)
            if m:
                parties["procesado"] = m.group(1).strip()
                break

        # Fiscal asignado
        m = re.search(
            r"[Ff]iscal\s+(?:\d+[°º]?\s+)?(?:delegado\s+)?([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,2})",
            head,
        )
        if m:
            parties["fiscal"] = m.group(1).strip()

        # Defensor
        m = re.search(
            r"[Dd]efensor(?:a)?\s+(?:de\s+oficio\s+)?([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,2})",
            head,
        )
        if m:
            parties["defensor"] = m.group(1).strip()

        # Víctima o familiar (contexto falsos positivos)
        m = re.search(
            r"(?:v[ií]ctima|familiar|MAFAPO)[:\s]+([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,3})",
            head,
        )
        if m:
            parties["victima"] = m.group(1).strip()

        if not parties:
            warnings.append("parties_not_found: no se identificaron partes del proceso")

        return parties

    def _extract_region(self, text: str) -> Optional[str]:
        """
        Identifica el departamento/ciudad de Colombia más mencionado.
        Útil para el análisis geográfico de patrones de falsos positivos.
        """
        mentions = {}
        for m in self._region_pattern.finditer(text):
            region = m.group(1).capitalize()
            mentions[region] = mentions.get(region, 0) + 1

        if not mentions:
            return None
        # Retorna la región más frecuente
        return max(mentions, key=mentions.get)

    def _extract_doc_type(self, head: str) -> Optional[str]:
        """Clasifica el tipo de documento judicial."""
        for dtype, pattern in self._doc_type_patterns.items():
            if pattern.search(head):
                return dtype
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid_date(self, day: int, month: int, year: int) -> bool:
        """Valida que la fecha sea plausible para el corpus (1990–2030)."""
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return False
        if not (1990 <= year <= 2030):
            return False
        try:
            datetime(year, month, day)
            return True
        except ValueError:
            return False

    def _compute_confidence(
        self,
        tribunal: Optional[str],
        case_number: Optional[str],
        date_iso: Optional[str],
        parties: dict,
        region: Optional[str],
        doc_type: Optional[str],
    ) -> float:
        """
        Calcula score de confianza [0, 1] basado en campos extraídos.
        Pesos calibrados según criticidad para el proyecto CFH.
        """
        weights = {
            "tribunal": 0.25,
            "case_number": 0.30,  # más crítico: identifica el documento en la cadena de custodia
            "date_iso": 0.20,
            "parties": 0.10,
            "region": 0.05,
            "doc_type": 0.10,
        }
        score = 0.0
        if tribunal:
            score += weights["tribunal"]
        if case_number:
            score += weights["case_number"]
        if date_iso and len(date_iso) == 10:  # ISO completa YYYY-MM-DD
            score += weights["date_iso"]
        elif date_iso:  # Solo año
            score += weights["date_iso"] * 0.3
        if parties:
            score += weights["parties"]
        if region:
            score += weights["region"]
        if doc_type:
            score += weights["doc_type"]
        return round(score, 4)
