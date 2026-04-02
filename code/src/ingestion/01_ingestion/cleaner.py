"""
CFH · Módulo de Limpieza de Texto Judicial
==========================================
Proyecto: Hermenéutica Forense Computacional
Corpus objetivo: Justicia Ordinaria (A), Autos JEP (B), Audiencias JEP (C)

Responsabilidades:
- Normalización de encoding (UTF-8 canónico)
- Corrección de errores OCR comunes en documentos judiciales colombianos
- Limpieza de artefactos de digitalización
- Normalización tipográfica preservando terminología jurídica

Principio de auditoría: toda transformación es trazable.
Cada método retorna el texto transformado + un log de operaciones aplicadas.
"""

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("cfh.ingestion.cleaner")


# ---------------------------------------------------------------------------
# Tipos de datos de resultado
# ---------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Registro inmutable de las transformaciones aplicadas a un documento."""
    original_length: int
    final_length: int
    operations_applied: list[str] = field(default_factory=list)
    ocr_corrections: int = 0
    encoding_fixes: int = 0
    whitespace_fixes: int = 0
    chars_removed: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return round(1 - (self.final_length / self.original_length), 4)


@dataclass
class CleanedDocument:
    text: str
    report: CleaningReport


# ---------------------------------------------------------------------------
# Tablas de corrección OCR para documentos judiciales colombianos
# ---------------------------------------------------------------------------

# Errores OCR frecuentes en digitalizaciones del Consejo Superior de la Judicatura
# y corpus JEP. Pares (patrón_regex, reemplazo, descripción).
OCR_CORRECTIONS_JUDICIAL = [
    # Confusiones de caracteres individuales
    (r"\bI\b(?=[a-záéíóúñ])", "l", "I mayúscula → l minúscula al inicio de token"),
    (r"(?<=[a-záéíóúñ])0(?=[a-záéíóúñ])", "o", "cero → letra 'o' entre letras"),
    (r"(?<=[a-záéíóúñ])1(?=[a-záéíóúñ])", "l", "uno → letra 'l' entre letras"),
    (r"\brn\b", "m", "rn → m (error OCR clásico)"),
    (r"(?<=[a-z])rn(?=[a-z])", "m", "rn → m dentro de palabra"),
    (r"vv(?=[aeiouáéíóú])", "w", "vv → w antes de vocal"),

    # Terminología jurídica colombiana específica — errores frecuentes de OCR
    (r"\bSenten[c|ç]ia\b", "Sentencia", "OCR en 'Sentencia'"),
    (r"\bJurisdicci[o0]n\b", "Jurisdicción", "OCR en 'Jurisdicción'"),
    (r"\bEjecu[c|ç]i[o0]n\b", "Ejecución", "OCR en 'Ejecución'"),
    (r"\bFiscal[i|í]a\b", "Fiscalía", "OCR en 'Fiscalía'"),
    (r"\bProcurad[u|ú]r[i|í]a\b", "Procuraduría", "OCR en 'Procuraduría'"),
    (r"\bContralorí[a|@]\b", "Contraloría", "OCR en 'Contraloría'"),
    (r"\bdefen[s|5]or\b", "defensor", "OCR 5→s en 'defensor'"),
    (r"\bprinci[p|r]io\b", "principio", "OCR r→p en 'principio'"),
    (r"\bha[b|6]eas\s+corpus\b", "habeas corpus", "OCR en 'habeas corpus'"),
    (r"\bJEP\b", "JEP", "normalizar sigla JEP"),
    (r"\bMAFAPO\b", "MAFAPO", "normalizar sigla MAFAPO"),

    # Fragmentación de palabras por salto de línea (guion OCR)
    (r"(\w+)-\n(\w+)", r"\1\2", "reunir palabras partidas por guion de fin de línea"),

    # Números de artículo / norma
    (r"\bArt[i|í][c|ç]ulo\b", "Artículo", "OCR en 'Artículo'"),
    (r"\bNum[e|3]ral\b", "Numeral", "OCR en 'Numeral'"),

    # Artefactos de escaneo
    (r"[|]{2,}", "ll", "|| → ll (OCR en doble ele)"),
    (r"\b[li]{2}(?=[aeiouáéíóú])", "ll", "li/il → ll ante vocal"),
]

# Caracteres que parecen espacios pero no lo son (error OCR/PDF export)
SPACE_LOOKALIKES = [
    "\u00a0",  # non-breaking space
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2009",  # thin space
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM
    "\u00ad",  # soft hyphen
]

# Comillas y guiones que difieren en fuentes escaneadas
TYPOGRAPHY_MAP = {
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u2013": "-",  # en dash → hyphen
    "\u2014": "-",  # em dash → hyphen
    "\u2026": "...",  # horizontal ellipsis
    "\u00b7": "·",  # middle dot (preserved as separator)
}


# ---------------------------------------------------------------------------
# Clase principal de limpieza
# ---------------------------------------------------------------------------

class JudicialTextCleaner:
    """
    Limpia y normaliza texto de documentos judiciales colombianos.

    Diseñado para ser determinístico: misma entrada → misma salida siempre.
    Cada instancia puede configurarse para distintos tipos de corpus.

    Parámetros
    ----------
    apply_ocr_corrections : bool
        Activar correcciones OCR. True para corpus escaneados (A, B).
        False para transcripciones digitales nativas (C oral).
    preserve_case : bool
        Si False, normaliza a minúsculas. Generalmente False para NLP.
    min_token_length : int
        Longitud mínima de tokens válidos tras limpieza.
    """

    def __init__(
        self,
        apply_ocr_corrections: bool = True,
        preserve_case: bool = True,
        min_token_length: int = 2,
    ):
        self.apply_ocr_corrections = apply_ocr_corrections
        self.preserve_case = preserve_case
        self.min_token_length = min_token_length

        # Compilar patrones OCR una sola vez (performance)
        self._compiled_ocr = [
            (re.compile(pattern, re.IGNORECASE), replacement, desc)
            for pattern, replacement, desc in OCR_CORRECTIONS_JUDICIAL
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def clean(self, raw_text: str, doc_id: Optional[str] = None) -> CleanedDocument:
        """
        Pipeline completo de limpieza. Punto de entrada principal.

        Parámetros
        ----------
        raw_text : str
            Texto bruto del documento (salida de extractor PDF/DOCX).
        doc_id : str, optional
            Identificador del documento para logging.

        Retorna
        -------
        CleanedDocument con texto limpio + CleaningReport auditable.
        """
        doc_id = doc_id or "UNKNOWN"
        logger.info(f"[{doc_id}] Iniciando limpieza — {len(raw_text)} chars")

        report = CleaningReport(original_length=len(raw_text), final_length=0)
        text = raw_text

        # Paso 1: Normalización de encoding
        text, enc_count = self._normalize_encoding(text)
        report.encoding_fixes = enc_count
        if enc_count:
            report.operations_applied.append(f"encoding_normalization({enc_count} fixes)")

        # Paso 2: Eliminar caracteres de control y artefactos de PDF
        text = self._remove_control_chars(text)
        report.operations_applied.append("control_chars_removed")

        # Paso 3: Normalizar tipografía (comillas, guiones, ellipsis)
        text = self._normalize_typography(text)
        report.operations_applied.append("typography_normalized")

        # Paso 4: Correcciones OCR (solo si aplica al tipo de corpus)
        if self.apply_ocr_corrections:
            text, ocr_count = self._apply_ocr_corrections(text)
            report.ocr_corrections = ocr_count
            report.operations_applied.append(f"ocr_corrections({ocr_count} applied)")

        # Paso 5: Normalización de espacios y saltos de línea
        text, ws_count = self._normalize_whitespace(text)
        report.whitespace_fixes = ws_count
        report.operations_applied.append(f"whitespace_normalized({ws_count} fixes)")

        # Paso 6: Normalización Unicode (NFC para compatibilidad con tokenizadores)
        text = unicodedata.normalize("NFC", text)
        report.operations_applied.append("unicode_NFC")

        # Paso 7: Lowercase opcional
        if not self.preserve_case:
            text = text.lower()
            report.operations_applied.append("lowercased")

        report.final_length = len(text)
        report.chars_removed = report.original_length - report.final_length

        logger.info(
            f"[{doc_id}] Limpieza completa — "
            f"{report.final_length} chars | "
            f"OCR fixes: {report.ocr_corrections} | "
            f"ratio: {report.compression_ratio:.2%}"
        )

        return CleanedDocument(text=text, report=report)

    # ------------------------------------------------------------------
    # Pasos internos del pipeline
    # ------------------------------------------------------------------

    def _normalize_encoding(self, text: str) -> tuple[str, int]:
        """
        Elimina caracteres que no son UTF-8 válido y espacios especiales.
        Retorna (texto_limpio, número_de_fixes).
        """
        count = 0

        # Reemplazar lookalikes de espacio
        for char in SPACE_LOOKALIKES:
            occurrences = text.count(char)
            if occurrences:
                text = text.replace(char, " ")
                count += occurrences

        # Forzar NFC antes de cualquier otra operación
        text = unicodedata.normalize("NFC", text)

        # Eliminar caracteres que no son imprimibles ni espaciado
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            # Mantener: letras (L*), números (N*), puntuación (P*),
            #            símbolos (S*), separadores de espacio (Zs), saltos (\n \t)
            if cat.startswith(("L", "N", "P", "S", "Z")) or ch in "\n\t\r":
                cleaned.append(ch)
            else:
                count += 1
        return "".join(cleaned), count

    def _remove_control_chars(self, text: str) -> str:
        """
        Elimina caracteres de control (U+0000–U+001F, U+007F)
        excepto \n, \t y \r que se preservan para segmentación.
        """
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _normalize_typography(self, text: str) -> str:
        """Unifica comillas, guiones y puntuación tipográfica variada."""
        for original, replacement in TYPOGRAPHY_MAP.items():
            text = text.replace(original, replacement)
        return text

    def _apply_ocr_corrections(self, text: str) -> tuple[str, int]:
        """
        Aplica las correcciones OCR compiladas.
        Retorna (texto_corregido, total_de_sustituciones).
        """
        total_subs = 0
        for pattern, replacement, desc in self._compiled_ocr:
            new_text, n = pattern.subn(replacement, text)
            if n:
                logger.debug(f"  OCR: {desc} → {n} sustitución(es)")
                total_subs += n
                text = new_text
        return text, total_subs

    def _normalize_whitespace(self, text: str) -> tuple[str, int]:
        """
        Normaliza espacios múltiples y saltos de línea redundantes.
        Preserva separación de párrafos (doble salto) como señal de segmentación.
        """
        original_length = len(text)

        # Normalizar saltos de línea Windows/Mac → Unix
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Eliminar espacios al final de cada línea
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Colapsar espacios múltiples (preservar saltos de línea)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Máximo 2 saltos de línea consecutivos (separador de párrafo)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Eliminar espacio al inicio de documento
        text = text.strip()

        fixes = original_length - len(text)
        return text, max(0, fixes)
