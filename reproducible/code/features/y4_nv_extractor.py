"""
CFH · Extractor de Indicador y₄ — Score NV (Negación de Victimización)
=======================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₁ (Violencia Discursiva)

Qué mide y₄:
    La densidad de expresiones que niegan la condición de víctima civil
    de las personas asesinadas en el marco del Macrocaso 003, reencuadrando
    su identidad dentro del marco semántico del enemigo combatiente o del
    sujeto ilegítimo merecedor de la muerte.

    Mecanismos detectados (en orden de peso):

    1. RECATEGORIZACION_COMBATIENTE (peso 1.0)
       Uso de vocabulario que reclasifica a la víctima como miembro de
       un grupo armado ilegal o como objetivo militar legítimo.
       Ejemplos: "guerrillero", "delincuente", "terrorista", "narcoterrorista",
       "miembro de grupo al margen de la ley", "objetivo de alto valor".

    2. ATRIBUCION_ARMAMENTO (peso 0.90)
       Atribución post-mortem de armas, uniformes o pertenencias militares
       que legitiman la recategorización como combatiente.
       Ejemplos: "portaba fusil Galil", "vestía prendas de uso privativo",
       "tenía en su poder material de guerra".

    3. DESHUMANIZACION (peso 0.70)
       Uso de términos que niegan la identidad personal y humana de la víctima,
       reduciendo a la persona a una categoría abstracta o cosificada.
       Ejemplos: "el individuo", "el sujeto", "el occiso", "el cuerpo",
       "el elemento", "el blanco".

    4. DESCALIFICACION_MORAL (peso 0.65)
       Atribución de características morales negativas que justifican
       implícitamente la muerte sin necesidad de recategorizar formalmente.
       Ejemplos: "de baja trayectoria social", "vinculado a actividades ilícitas",
       "con antecedentes penales", "habitante de calle", "drogadicto".

Estrategia de implementación:
    Al igual que y₂, NV combina análisis léxico con spaCy para el
    análisis de contexto sintáctico. La detección es primariamente léxica
    (basada en diccionarios y patrones regex) porque los mecanismos NV
    operan principalmente a nivel de vocabulario, no de estructura gramatical.

    spaCy se usa para:
    - Verificar el contexto gramatical (¿"guerrillero" es sujeto o predicativo?)
    - Detectar negaciones que anulan el NV ("NO era guerrillero")
    - Identificar citas que reproducen NV para cuestionarlo (→ peso reducido)

Nota metodológica sobre NV en corpus JEP:
    En el corpus B (JEP), las secciones HECHOS_Y_CONDUCTAS frecuentemente
    citan el vocabulario NV del expediente original para recalificarlo.
    Ejemplo: "presentado fraudulentamente como guerrillero" — aquí
    "guerrillero" es NV pero el contexto ("presentado fraudulentamente")
    es REP. El extractor detecta el NV léxico pero registra el contexto
    de cuestionamiento para que el SEM pueda ponderar correctamente.

Dependencias:
    - spacy >= 3.7.0 + es_core_news_lg
    - numpy >= 1.26.0

Referencia teórica:
    Galtung, J. (1990). Cultural violence. JPR, 27(3), 291-305.
    Van Dijk, T. A. (2008). Discourse and Power. Palgrave Macmillan.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

logger = logging.getLogger("cfh.features.y4_nv")


# ---------------------------------------------------------------------------
# Diccionarios de la taxonomía NV
# ---------------------------------------------------------------------------

# ── Mecanismo 1: Recategorización como combatiente ────────────────────────
NV_COMBATIENTE_LEMMAS = {
    # Guerrilla y grupos armados
    "guerrillero", "guerrilla", "insurgente", "subversivo",
    "terrorista", "narcoterrorista", "narco",
    # Delincuencia
    "delincuente", "criminal", "bandido", "forajido",
    "maleante", "hampón", "antisocial",
    # Vocabulario militar de objetivo
    "objetivo", "blanco", "enemigo", "combatiente",
    "integrante", "miembro",  # solo cuando van con "de grupo armado"
    # Expresiones del expediente
    "dado de baja", "abatido", "neutralizado", "eliminado",
}

# Frases nominales de recategorización (requieren análisis de n-gramas)
NV_COMBATIENTE_FRASES = [
    r"\bgrupo(?:s)?\s+(?:al\s+margen\s+de\s+la\s+ley|armado(?:s)?(?:\s+ilegal(?:es)?)?)",
    r"\bintegrante(?:s)?\s+de\s+(?:las?\s+)?(?:farc|eln|auc|bacrim|guerrilla)",
    r"\bmiembro(?:s)?\s+de\s+(?:grupo(?:s)?\s+)?(?:armado|ilegal|irregular)",
    r"\bpresunto(?:s)?\s+(?:guerrillero|terrorista|delincuente|integrante)",
    r"\bobjetivo(?:s)?\s+de\s+(?:alto\s+valor|la\s+operación|interés)",
    r"\bestado\s+(?:mayor|de\s+guerra)",
    r"\bvinculado(?:s)?\s+(?:a|con)\s+(?:grupo(?:s)?\s+)?(?:armado|ilegal|al\s+margen)",
]

# ── Mecanismo 2: Atribución de armamento ─────────────────────────────────
NV_ARMAMENTO_LEMMAS = {
    "fusil", "pistola", "revólver", "arma", "armamento", "munición",
    "explosivo", "granada", "caleta", "material",
    "uniforme", "camuflado", "prendas",
}

NV_ARMAMENTO_FRASES = [
    r"\bportaba(?:n)?\s+(?:un\s+)?(?:fusil|pistola|arma|revólver|armamento)",
    r"\btenía(?:n)?\s+en\s+su\s+(?:poder|haber|posesión)\s+(?:material|arma|fusil|munición)",
    r"\bvestía(?:n)?\s+prendas?\s+de\s+uso\s+(?:privativo|militar|castrense)",
    r"\bcon\s+prendas?\s+(?:de\s+uso\s+)?(?:privativo|militar|camuflado)",
    r"\bequipado(?:s?)?\s+con\s+(?:armamento|fusil|armas?)",
    r"\bhalló\s+(?:en\s+su\s+poder\s+)?(?:un\s+)?(?:fusil|arma|pistola|material)",
    r"\bencont(?:ró|raron)\s+(?:en\s+(?:su\s+)?poder\s+)?(?:armamento|armas?|fusil)",
    r"\bkit\s+del?\s+falso\s+positivo",  # referencia directa al modus operandi
]

# ── Mecanismo 3: Deshumanización ─────────────────────────────────────────
NV_DESHUMANIZACION_LEMMAS = {
    "individuo", "sujeto", "elemento", "tipo", "fulano",
    "occiso", "cuerpo", "cadáver", "fallecido",
    "persona",  # solo como "sin identificar" → se controla por contexto
}

# Términos de deshumanización que requieren contexto (sin contexto = falso positivo)
NV_DESHUMANIZACION_CON_CONTEXTO = {
    "occiso", "cadáver", "cuerpo",  # siempre deshumanizan al referirse a la víctima
    "fallecido",  # puede ser neutral, pero en contexto judicial de FP → NV
}

# Términos que por sí solos son NV sin contexto adicional
NV_DESHUMANIZACION_DIRECTA = {
    "individuo", "sujeto", "elemento", "tipo",
}

# ── Mecanismo 4: Descalificación moral ───────────────────────────────────
NV_DESCALIFICACION_FRASES = [
    r"\bde\s+baja\s+(?:trayectoria|extracción|condición)\s+(?:social|moral)?",
    r"\bvinculado(?:s)?\s+(?:a|con)\s+(?:actividades?\s+ilícitas?|el\s+crimen|el\s+hampa)",
    r"\bcon\s+antecedentes?\s+(?:penales?|judiciales?|delictivos?)",
    r"\bhabitante(?:s)?\s+de\s+(?:la\s+)?calle",
    r"\bconsumidor(?:es)?\s+de\s+(?:sustancias?\s+(?:psicoactivas?|alucinógenas?))",
    r"\b(?:drogadicto|farmacodependiente|adicto)",
    r"\bsin\s+oficio\s+(?:conocido|lícito)",
    r"\bde\s+(?:dudosa|mala|poca)\s+(?:reputación|fama|conducta)",
    r"\bpróximo\s+a\s+(?:grupos?|organizaciones?)\s+(?:criminales?|delincuenciales?)",
]

# ── Contextos de cuestionamiento del NV (→ reducen el peso) ───────────────
# Cuando el NV aparece en un contexto de cuestionamiento, el peso se reduce
# porque está siendo citado para ser refutado (fenómeno del metalenguaje)
NV_CUESTIONAMIENTO_TRIGGERS = [
    r"\bpresentado(?:s)?\s+(?:falsamente|fraudulentamente|ilegítimamente)\s+como",
    r"\bfalsamente\s+(?:presentado|reportado|identificado)",
    r"\berróneamente\s+(?:identificado|catalogado|clasificado)",
    r"\bque\s+(?:en\s+realidad|realmente)\s+(?:era|eran|no\s+era|no\s+eran)",
    r"\bno\s+era(?:n)?\s+(?:guerrillero|combatiente|delincuente|integrante)",
    r"\b(?:civil|civiles)\s+(?:inocente|inermé|desarmado)",
    r"\bpresentado\s+como\s+(?:guerrillero|baja|resultado)",
]

# Patrones compilados
_NV_COMBATIENTE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_COMBATIENTE_FRASES]
_NV_ARMAMENTO_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_ARMAMENTO_FRASES]
_NV_DESCALIFICACION_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_DESCALIFICACION_FRASES]
_NV_CUESTIONAMIENTO_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_CUESTIONAMIENTO_TRIGGERS]

# Pesos por mecanismo
NV_MECHANISM_WEIGHTS = {
    "recategorizacion_combatiente": 1.00,
    "atribucion_armamento":         0.90,
    "deshumanizacion":              0.70,
    "descalificacion_moral":        0.65,
}

# Factor de reducción cuando el NV aparece en contexto de cuestionamiento
NV_CUESTIONAMIENTO_FACTOR = 0.25


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class NVInstance:
    """Una instancia detectada de Negación de Victimización."""
    mechanism: str
    text_span: str
    char_start: int
    char_end: int
    weight: float           # peso efectivo (puede reducirse por contexto)
    weight_base: float      # peso base del mecanismo
    is_questioned: bool     # True si el NV aparece en contexto de cuestionamiento
    sent_index: int
    details: dict = field(default_factory=dict)


@dataclass
class NVExtractionResult:
    """Resultado completo de la extracción del indicador y₄."""
    score: float
    score_raw: float
    doc_id: str
    section_id: str
    corpus_type: str
    text_length_chars: int
    n_sentences: int
    n_instances: int
    n_questioned: int           # instancias NV en contexto de cuestionamiento
    n_recategorizacion: int = 0
    n_atribucion_armamento: int = 0
    n_deshumanizacion: int = 0
    n_descalificacion: int = 0
    instances: list[NVInstance] = field(default_factory=list)
    processing_time_s: float = 0.0
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.n_sentences > 0 and self.text_length_chars > 20

    def to_dict(self) -> dict:
        return {
            "y4_nv_score": self.score,
            "y4_nv_score_raw": self.score_raw,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "corpus_type": self.corpus_type,
            "n_sentences": self.n_sentences,
            "n_nv_instances": self.n_instances,
            "n_questioned": self.n_questioned,
            "nv_by_mechanism": {
                "recategorizacion_combatiente": self.n_recategorizacion,
                "atribucion_armamento":         self.n_atribucion_armamento,
                "deshumanizacion":              self.n_deshumanizacion,
                "descalificacion_moral":        self.n_descalificacion,
            },
            "top_nv_spans": [
                {
                    "text": i.text_span,
                    "mechanism": i.mechanism,
                    "weight": i.weight,
                    "questioned": i.is_questioned,
                }
                for i in sorted(self.instances, key=lambda x: -x.weight)[:5]
            ],
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Normalizador
# ---------------------------------------------------------------------------

class NVScoreNormalizer:
    """Normaliza el score NV bruto al rango [0, 1]."""

    def __init__(
        self,
        method: str = "percentile",
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
    ):
        self.method = method
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self._fitted = False
        # Valores por defecto empíricos — actualizar con calibración
        self._p_low: float = 0.0
        self._p_high: float = 0.6
        self._mean: float = 0.15
        self._std: float = 0.18

    def fit(self, raw_scores: list[float]) -> "NVScoreNormalizer":
        arr = np.array(raw_scores)
        self._p_low = float(np.percentile(arr, self.low_percentile))
        self._p_high = float(np.percentile(arr, self.high_percentile))
        self._mean = float(arr.mean())
        self._std = float(arr.std()) or 1e-8
        self._fitted = True
        return self

    def normalize(self, raw_score: float) -> float:
        if self.method == "percentile":
            denom = self._p_high - self._p_low
            normalized = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5
        elif self.method == "zscore":
            normalized = (raw_score - self._mean) / self._std
            normalized = (normalized + 3) / 6
        else:
            denom = self._p_high - self._p_low
            normalized = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5
        return float(np.clip(normalized, 0.0, 1.0))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "method": self.method,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "p_low": self._p_low, "p_high": self._p_high,
            "mean": self._mean, "std": self._std,
            "fitted": self._fitted,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NVScoreNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(method=data["method"])
        obj._p_low = data["p_low"]
        obj._p_high = data["p_high"]
        obj._mean = data["mean"]
        obj._std = data["std"]
        obj._fitted = data["fitted"]
        return obj


# ---------------------------------------------------------------------------
# Extractor principal
# ---------------------------------------------------------------------------

class NVExtractor:
    """
    Extractor del indicador y₄ (Score NV — Negación de Victimización).

    Detecta cuatro mecanismos de NV combinando análisis léxico con
    verificación de contexto mediante spaCy. Maneja correctamente el
    caso del NV citado para ser cuestionado (frecuente en corpus JEP).

    Parámetros
    ----------
    model_name : str
        Modelo spaCy para análisis contextual.
    normalizer : NVScoreNormalizer
        Normalizador del score. Si None, se usa uno con defaults.
    context_window_chars : int
        Ventana de caracteres antes/después de un span NV para buscar
        triggers de cuestionamiento.
    """

    def __init__(
        self,
        model_name: str = "es_core_news_lg",
        normalizer: Optional[NVScoreNormalizer] = None,
        context_window_chars: int = 150,
    ):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spaCy no instalado. "
                "Ejecuta: pip install spacy && python -m spacy download es_core_news_lg"
            )
        self.model_name = model_name
        self.context_window_chars = context_window_chars
        self.normalizer = normalizer or NVScoreNormalizer()

        logger.info(f"Cargando modelo spaCy: {model_name}")
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Modelo '{model_name}' no encontrado. "
                f"Instala: python -m spacy download {model_name}"
            )

    def extract(
        self,
        text: str,
        doc_id: str = "unknown",
        section_id: str = "unknown",
        corpus_type: str = "A",
    ) -> NVExtractionResult:
        """Extrae el score NV de un segmento textual judicial."""
        t0 = time.perf_counter()

        if not text or len(text.strip()) < 20:
            return self._empty_result(doc_id, section_id, corpus_type)

        doc = self._nlp(text)
        sentences = list(doc.sents)
        sent_count = len([s for s in sentences if len(list(s)) >= 3])

        all_instances: list[NVInstance] = []

        # Análisis por oración para mecanismos que requieren contexto sintáctico
        for sent_idx, sent in enumerate(sentences):
            if len(list(sent)) < 3:
                continue
            instances = (
                self._detect_recategorizacion(sent, sent_idx, text) +
                self._detect_atribucion_armamento(sent, sent_idx, text) +
                self._detect_deshumanizacion(sent, sent_idx, text)
            )
            all_instances.extend(instances)

        # Descalificación moral — análisis sobre texto completo
        all_instances.extend(self._detect_descalificacion(text, sentences))

        # Calcular score bruto: suma ponderada / oraciones
        if sent_count == 0:
            score_raw = 0.0
        else:
            peso_total = sum(i.weight for i in all_instances)
            score_raw = peso_total / sent_count

        score_normalized = self.normalizer.normalize(score_raw)
        elapsed = time.perf_counter() - t0

        n_rec = sum(1 for i in all_instances if i.mechanism == "recategorizacion_combatiente")
        n_arm = sum(1 for i in all_instances if i.mechanism == "atribucion_armamento")
        n_des = sum(1 for i in all_instances if i.mechanism == "deshumanizacion")
        n_desc = sum(1 for i in all_instances if i.mechanism == "descalificacion_moral")
        n_quest = sum(1 for i in all_instances if i.is_questioned)

        logger.debug(
            f"y₄ NV [{doc_id}/{section_id}]: score={score_normalized:.3f} "
            f"raw={score_raw:.3f} instances={len(all_instances)} "
            f"(rec={n_rec}, arm={n_arm}, des={n_des}, desc={n_desc}, "
            f"questioned={n_quest}) t={elapsed:.2f}s"
        )

        return NVExtractionResult(
            score=score_normalized,
            score_raw=score_raw,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=len(text),
            n_sentences=sent_count,
            n_instances=len(all_instances),
            n_questioned=n_quest,
            n_recategorizacion=n_rec,
            n_atribucion_armamento=n_arm,
            n_deshumanizacion=n_des,
            n_descalificacion=n_desc,
            instances=all_instances,
            processing_time_s=elapsed,
        )

    # ── Detectores ────────────────────────────────────────────────────────

    def _detect_recategorizacion(
        self, sent: "Span", sent_idx: int, full_text: str
    ) -> list[NVInstance]:
        """Detecta recategorización de la víctima como combatiente."""
        instances = []

        # Análisis léxico token por token
        for token in sent:
            lemma = token.lemma_.lower()
            if lemma not in NV_COMBATIENTE_LEMMAS:
                continue
            # Excluir si el token es el verbo principal (no el sujeto/predicativo)
            if token.pos_ == "VERB":
                continue
            # Verificar si hay negación directa ("NO era guerrillero")
            if self._has_direct_negation(token):
                continue

            weight_base = NV_MECHANISM_WEIGHTS["recategorizacion_combatiente"]
            is_questioned = self._is_in_questioning_context(
                token.idx, token.idx + len(token.text), full_text
            )
            weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)

            instances.append(NVInstance(
                mechanism="recategorizacion_combatiente",
                text_span=token.text,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
                weight=weight,
                weight_base=weight_base,
                is_questioned=is_questioned,
                sent_index=sent_idx,
                details={"lemma": lemma, "pos": token.pos_}
            ))

        # Frases nominales de recategorización
        for pattern in _NV_COMBATIENTE_COMPILED:
            for match in pattern.finditer(sent.text):
                char_start = sent.start_char + match.start()
                char_end = sent.start_char + match.end()
                is_questioned = self._is_in_questioning_context(
                    char_start, char_end, full_text
                )
                weight_base = NV_MECHANISM_WEIGHTS["recategorizacion_combatiente"]
                weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)
                instances.append(NVInstance(
                    mechanism="recategorizacion_combatiente",
                    text_span=match.group(),
                    char_start=char_start,
                    char_end=char_end,
                    weight=weight,
                    weight_base=weight_base,
                    is_questioned=is_questioned,
                    sent_index=sent_idx,
                    details={"is_phrase": True}
                ))

        return instances

    def _detect_atribucion_armamento(
        self, sent: "Span", sent_idx: int, full_text: str
    ) -> list[NVInstance]:
        """Detecta atribución post-mortem de armamento o prendas militares."""
        instances = []

        for pattern in _NV_ARMAMENTO_COMPILED:
            for match in pattern.finditer(sent.text):
                char_start = sent.start_char + match.start()
                char_end = sent.start_char + match.end()
                is_questioned = self._is_in_questioning_context(
                    char_start, char_end, full_text
                )
                weight_base = NV_MECHANISM_WEIGHTS["atribucion_armamento"]
                weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)
                instances.append(NVInstance(
                    mechanism="atribucion_armamento",
                    text_span=match.group(),
                    char_start=char_start,
                    char_end=char_end,
                    weight=weight,
                    weight_base=weight_base,
                    is_questioned=is_questioned,
                    sent_index=sent_idx,
                ))

        # Términos de armamento individuales con contexto
        for token in sent:
            if token.lemma_.lower() not in NV_ARMAMENTO_LEMMAS:
                continue
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            # Solo cuenta como NV si está en contexto de posesión atribuida
            # ("portaba X", "tenía X", "con X")
            has_possession_context = any(
                child.lemma_.lower() in {"portar", "tener", "llevar", "cargar"}
                or child.text.lower() in {"con", "sin"}
                for child in token.head.children
                if child != token
            )
            if not has_possession_context:
                continue

            is_questioned = self._is_in_questioning_context(
                token.idx, token.idx + len(token.text), full_text
            )
            weight_base = NV_MECHANISM_WEIGHTS["atribucion_armamento"] * 0.7
            weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)
            instances.append(NVInstance(
                mechanism="atribucion_armamento",
                text_span=token.text,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
                weight=weight,
                weight_base=weight_base,
                is_questioned=is_questioned,
                sent_index=sent_idx,
                details={"individual_token": True}
            ))

        return instances

    def _detect_deshumanizacion(
        self, sent: "Span", sent_idx: int, full_text: str
    ) -> list[NVInstance]:
        """
        Detecta términos que niegan la identidad personal de la víctima.

        Los términos de deshumanización directa (individuo, sujeto, elemento)
        siempre son NV cuando aparecen como sustantivos que refieren a personas.
        Los términos neutros (occiso, cuerpo, cadáver) son NV por defecto
        en el contexto del corpus CFH (referencia a víctimas de FP).
        """
        instances = []

        for token in sent:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()

            is_direct = lemma in NV_DESHUMANIZACION_DIRECTA or text_lower in NV_DESHUMANIZACION_DIRECTA
            is_contextual = lemma in NV_DESHUMANIZACION_CON_CONTEXTO or text_lower in NV_DESHUMANIZACION_CON_CONTEXTO

            if not (is_direct or is_contextual):
                continue

            # Solo sustantivos o pronombres
            if token.pos_ not in {"NOUN", "PRON", "PROPN"}:
                continue

            # Para términos contextuales, verificar que referencian a la víctima
            # Heurística: el token es sujeto o complemento de verbo de acción
            if is_contextual and not is_direct:
                is_referencing_victim = token.dep_ in {"nsubj", "nsubjpass", "obj", "dobj"}
                if not is_referencing_victim:
                    continue

            # Excluir si hay negación directa
            if self._has_direct_negation(token):
                continue

            is_questioned = self._is_in_questioning_context(
                token.idx, token.idx + len(token.text), full_text
            )
            weight_base = NV_MECHANISM_WEIGHTS["deshumanizacion"]
            weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)

            instances.append(NVInstance(
                mechanism="deshumanizacion",
                text_span=token.text,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
                weight=weight,
                weight_base=weight_base,
                is_questioned=is_questioned,
                sent_index=sent_idx,
                details={"is_direct": is_direct, "lemma": lemma}
            ))

        return instances

    def _detect_descalificacion(
        self, text: str, sentences: list
    ) -> list[NVInstance]:
        """Detecta descalificación moral de la víctima mediante patrones regex."""
        instances = []

        for pattern in _NV_DESCALIFICACION_COMPILED:
            for match in pattern.finditer(text):
                is_questioned = self._is_in_questioning_context(
                    match.start(), match.end(), text
                )
                weight_base = NV_MECHANISM_WEIGHTS["descalificacion_moral"]
                weight = weight_base * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)
                instances.append(NVInstance(
                    mechanism="descalificacion_moral",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    weight_base=weight_base,
                    is_questioned=is_questioned,
                    sent_index=-1,
                ))

        return instances

    # ── Helpers ──────────────────────────────────────────────────────────

    def _has_direct_negation(self, token: "Token") -> bool:
        """Verifica si el token tiene una negación directa en su contexto."""
        for child in token.head.children:
            if child.dep_ == "neg":
                return True
        for child in token.children:
            if child.dep_ == "neg":
                return True
        return False

    def _is_in_questioning_context(
        self, char_start: int, char_end: int, full_text: str
    ) -> bool:
        """
        Verifica si un span NV aparece en un contexto de cuestionamiento.

        Busca triggers de cuestionamiento en la ventana de contexto
        inmediata alrededor del span.
        """
        window_start = max(0, char_start - self.context_window_chars)
        window_end = min(len(full_text), char_end + self.context_window_chars)
        context = full_text[window_start:window_end]

        return any(
            pattern.search(context) is not None
            for pattern in _NV_CUESTIONAMIENTO_COMPILED
        )

    def _empty_result(
        self, doc_id: str, section_id: str, corpus_type: str
    ) -> NVExtractionResult:
        return NVExtractionResult(
            score=0.0, score_raw=0.0,
            doc_id=doc_id, section_id=section_id, corpus_type=corpus_type,
            text_length_chars=0, n_sentences=0, n_instances=0, n_questioned=0,
            warning="texto_vacio_o_muy_corto",
        )
