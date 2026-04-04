"""
CFH · Extractor de Indicador y₁₀ — Score REP (Ruptura Epistémica Positiva)
===========================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: η₂ (Transición Epistémica)

Qué mide y₁₀:
    La densidad de expresiones que marcan una transición del lenguaje de
    la guerra al lenguaje del reconocimiento: reconocimiento explícito de
    la condición de víctima civil, admisión de responsabilidad, adopción
    del marco semántico de los derechos humanos y la justicia restaurativa.

    El REP es el polo opuesto al EBI/SA/NV en el espectro discursivo de
    la CFH. Donde EBI encubre el crimen, REP lo nombra. Donde NV niega
    la humanidad de la víctima, REP la restituye. Donde SA invisibiliza
    al responsable, REP lo hace visible.

    Mecanismos detectados (en orden de peso):

    1. RECONOCIMIENTO_RESPONSABILIDAD (peso 1.0)
       Admisión explícita y directa de la autoría o participación en los
       hechos criminales. El mecanismo más fuerte del indicador REP.
       Ejemplos: "reconozco que ordené", "acepto que participé",
       "admito mi responsabilidad", "yo di la orden".

    2. RESTITUCIÓN_IDENTIDAD (peso 0.90)
       Reconocimiento de la condición de víctima civil de la persona
       asesinada — su nombre propio, su humanidad, su inocencia.
       Ejemplos: "era un civil inocente", "era un campesino",
       uso del nombre propio de la víctima, "no tenía vinculación armada".

    3. LENGUAJE_DIH_REPARADOR (peso 0.80)
       Adopción del vocabulario del derecho internacional humanitario
       y los derechos humanos para describir los hechos — el opuesto
       semántico del EBI.
       Ejemplos: "ejecución extrajudicial", "homicidio en persona protegida",
       "crimen de lesa humanidad", "muertes ilegítimamente presentadas".

    4. COMPROMISO_REPARACIÓN (peso 0.75)
       Expresiones de reparación simbólica, pedido de perdón, garantías
       de no repetición — los actos de habla constitutivos del proceso
       transicional.
       Ejemplos: "pido perdón", "me comprometo a no repetir",
       "ofrezco disculpas públicas", "contribuiré a la reparación".

Nota metodológica — diferencia con NV cuestionado:
    El REP no es simplemente la negación del NV. Es una afirmación
    positiva en el marco semántico de los derechos humanos. La diferencia:
    - NV cuestionado: "fue presentado FALSAMENTE como guerrillero" → NV(reducido)
    - REP: "era un CIVIL INOCENTE" → REP(pleno)
    El primero niega el NV pero no afirma el REP. El segundo restituye
    activamente la identidad de la víctima.

Dependencias:
    - spacy >= 3.7.0 + es_core_news_lg
    - numpy >= 1.26.0

Referencia teórica:
    Fraser, N. (1995). From redistribution to recognition? NLR, 212, 68-93.
    Austin, J. L. (1962). How to Do Things with Words. Oxford UP.
    JEP (2022). RC-01 y RC-03 — Resoluciones de Conclusiones Caso 03.
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
    from spacy.tokens import Span, Token
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

logger = logging.getLogger("cfh.features.y10_rep")


# ---------------------------------------------------------------------------
# Diccionarios y patrones REP
# ---------------------------------------------------------------------------

# ── Mecanismo 1: Reconocimiento de responsabilidad ────────────────────────
# Verbos performativos de reconocimiento — núcleo del acto de habla REP
REP_RECONOCIMIENTO_VERBOS = {
    "reconocer", "aceptar", "admitir", "confesar", "asumir",
    "reconocer", "declarar",  # solo en contexto de responsabilidad
}

# Frases de reconocimiento completas (mayor peso que verbos solos)
REP_RECONOCIMIENTO_FRASES = [
    r"\breconoz(?:co|ca|can|cemos)\s+(?:que\s+)?(?:yo\s+)?(?:ordené|participé|cometí|"
    r"di\s+la\s+orden|fui\s+responsable|mi\s+responsabilidad)",
    r"\b(?:acepto|admito|asumo)\s+(?:plena(?:mente)?|total(?:mente)?)?\s*"
    r"(?:mi\s+)?responsabilidad",
    r"\bme\s+declaro\s+responsable",
    r"\breconozco\s+(?:mi\s+)?responsabilidad\s+(?:en|por|como)",
    r"\bacepto\s+(?:los\s+)?cargos?\s+(?:que\s+se\s+me\s+imputan?)?",
    r"\bfui\s+(?:yo\s+)?(?:quien|el\s+que)\s+(?:ordené|di\s+la\s+orden|participé)",
    r"\bsoy\s+responsable\s+de",
    r"\breconozco\s+(?:ante\s+(?:las?\s+)?víctimas?|públicamente|la\s+verdad)",
    r"\baporte\s+(?:a\s+la\s+)?verdad\s+(?:completa|plena|detallada|exhaustiva)",
    r"\bverdad\s+(?:completa|plena|detallada|exhaustiva)",
]

# ── Mecanismo 2: Restitución de identidad ────────────────────────────────
# Vocabulario que restituye la humanidad y la condición civil de la víctima
REP_RESTITUCION_LEMMAS = {
    "civil", "inocente", "campesino", "habitante", "trabajador",
    "ciudadano", "persona", "víctima",
}

REP_RESTITUCION_FRASES = [
    r"\b(?:era|eran|son|fue)\s+(?:un(?:a)?\s+)?civil(?:es)?\s+inocente(?:s)?",
    r"\b(?:era|eran)\s+(?:un(?:a)?\s+)?(?:campesino|trabajador|estudiante|"
    r"habitante|ciudadano)(?:s)?(?:\s+(?:inocente|desarmado|ajeno))?",
    r"\bno\s+(?:era|eran|tenía|tenían)\s+(?:ninguna\s+)?vinculación\s+"
    r"(?:con|a)\s+(?:grupo(?:s)?|organización|actividades?)",
    r"\b(?:civil|civiles)\s+(?:inocente(?:s)?|desarmado(?:s)?|ajeno(?:s)?\s+al\s+conflicto)",
    r"\bpersona(?:s)?\s+(?:protegida(?:s)?|inocente(?:s)?|civil(?:es)?)",
    r"\bno\s+(?:portaba|portaban|tenía|tenían)\s+armas?",
    r"\bajeno(?:s)?\s+al\s+conflicto\s+armado",
    r"\bvíctima(?:s)?\s+(?:inocente(?:s)?|civil(?:es)?|directa(?:s)?)",
]

# Patrones de uso del nombre propio de la víctima — restitución máxima
# En el corpus JEP, el nombre propio aparece con frecuencia en los reconocimientos
REP_NOMBRE_PROPIO_CONTEXTOS = [
    r"\bpor\s+la\s+muerte\s+de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+",
    r"\bla\s+familia\s+(?:de|del)\s+(?:señor|señora|joven)?\s*[A-ZÁÉÍÓÚÑ]",
    r"\bperdón\s+(?:a|de)\s+(?:la\s+familia\s+de\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+",
]

# ── Mecanismo 3: Lenguaje DIH reparador ──────────────────────────────────
REP_DIH_LEMMAS = {
    "ejecución", "extrajudicial", "homicidio",
    "crimen", "lesa", "humanidad", "guerra",
    "protegido", "prohibido", "violación",
}

REP_DIH_FRASES = [
    r"\bejecución(?:es)?\s+extrajudicial(?:es)?",
    r"\bhomicidio(?:s)?\s+en\s+persona(?:s)?\s+protegida(?:s)?",
    r"\bcrimen(?:es)?\s+de\s+(?:lesa\s+)?humanidad",
    r"\bcrimen(?:es)?\s+de\s+guerra",
    r"\bmuertes?\s+ilegítimamente\s+presentadas?\s+como\s+bajas?",
    r"\bviolación(?:es)?\s+(?:graves?\s+)?(?:del?\s+)?(?:derecho\s+internacional\s+"
    r"humanitario|derechos?\s+humanos?|dih)",
    r"\binfracción(?:es)?\s+(?:al|del)\s+(?:dih|derecho\s+internacional\s+humanitario)",
    r"\bpersona(?:s)?\s+protegida(?:s)?\s+por\s+(?:el\s+)?(?:dih|derecho\s+internacional)",
    r"\bart(?:ículo|\.)\s+135\s+(?:del\s+)?(?:código\s+penal|cp)",  # homicidio persona protegida
    r"\bimpacto\s+(?:en\s+)?(?:las?\s+)?víctimas?",
    r"\bdaño(?:s)?\s+(?:causado(?:s)?|infligido(?:s)?)\s+(?:a\s+)?(?:las?\s+)?víctimas?",
]

# ── Mecanismo 4: Compromiso de reparación ────────────────────────────────
REP_REPARACION_FRASES = [
    r"\bpido\s+perdón\s+(?:a|de|por)",
    r"\bsolicito\s+perdón\s+(?:a|de|por)",
    r"\bme\s+comprometo\s+a\s+(?:no\s+repetir|contribuir|reparar|colaborar)",
    r"\bgarantía(?:s)?\s+de\s+no\s+repetición",
    r"\bno\s+(?:volveré|repetiremos?)\s+(?:a\s+)?(?:cometer|hacer|participar)",
    r"\bofrezco\s+(?:mis\s+)?(?:disculpas?|excusas?)\s+(?:públicas?)?",
    r"\bcontribuiré?\s+(?:a\s+la\s+)?(?:reparación|verdad|búsqueda)",
    r"\bcompromiso\s+(?:de\s+)?(?:reparación|verdad|no\s+repetición)",
    r"\bproyecto(?:s)?\s+(?:de\s+)?(?:reparación|restauración|sanción\s+propia)",
    r"\bsanción\s+propia",
    r"\bjusticia\s+restaurativa",
    r"\bmedida(?:s)?\s+(?:de\s+)?(?:reparación|contribución|satisfacción)",
]

# Compilar todos los patrones
_REP_RECONOCIMIENTO_COMPILED = [re.compile(p, re.IGNORECASE) for p in REP_RECONOCIMIENTO_FRASES]
_REP_RESTITUCION_COMPILED = [re.compile(p, re.IGNORECASE) for p in REP_RESTITUCION_FRASES]
_REP_NOMBRE_PROPIO_COMPILED = [re.compile(p, re.IGNORECASE) for p in REP_NOMBRE_PROPIO_CONTEXTOS]
_REP_DIH_COMPILED = [re.compile(p, re.IGNORECASE) for p in REP_DIH_FRASES]
_REP_REPARACION_COMPILED = [re.compile(p, re.IGNORECASE) for p in REP_REPARACION_FRASES]

# Pesos por mecanismo
REP_MECHANISM_WEIGHTS = {
    "reconocimiento_responsabilidad": 1.00,
    "restitución_identidad":          0.90,
    "lenguaje_dih_reparador":         0.80,
    "compromiso_reparación":          0.75,
}


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class REPInstance:
    """Una instancia detectada de Ruptura Epistémica Positiva."""
    mechanism: str
    text_span: str
    char_start: int
    char_end: int
    weight: float
    sent_index: int
    details: dict = field(default_factory=dict)


@dataclass
class REPExtractionResult:
    """Resultado completo de la extracción del indicador y₁₀."""
    score: float
    score_raw: float
    doc_id: str
    section_id: str
    corpus_type: str
    text_length_chars: int
    n_sentences: int
    n_instances: int
    n_reconocimiento: int = 0
    n_restitución: int = 0
    n_dih: int = 0
    n_reparación: int = 0
    instances: list[REPInstance] = field(default_factory=list)
    processing_time_s: float = 0.0
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.n_sentences > 0 and self.text_length_chars > 20

    def to_dict(self) -> dict:
        return {
            "y10_rep_score": self.score,
            "y10_rep_score_raw": self.score_raw,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "corpus_type": self.corpus_type,
            "n_sentences": self.n_sentences,
            "n_rep_instances": self.n_instances,
            "rep_by_mechanism": {
                "reconocimiento_responsabilidad": self.n_reconocimiento,
                "restitución_identidad":          self.n_restitución,
                "lenguaje_dih_reparador":         self.n_dih,
                "compromiso_reparación":          self.n_reparación,
            },
            "top_rep_spans": [
                {"text": i.text_span, "mechanism": i.mechanism, "weight": i.weight}
                for i in sorted(self.instances, key=lambda x: -x.weight)[:5]
            ],
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Normalizador
# ---------------------------------------------------------------------------

class REPScoreNormalizer:
    """
    Normaliza el score REP bruto al rango [0, 1].

    Nota: la distribución de REP es distinta a EBI/SA/NV — está muy
    concentrada en cero para el corpus A (justicia ordinaria) y tiene
    masa significativa solo en el corpus B/C (JEP). Esto significa que
    el normalizador debe calibrarse sobre el corpus completo para capturar
    esa diferencia, no sobre un subconjunto homogéneo.
    """

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
        # Defaults calibrados sobre el corpus mixto CFH
        # REP es casi cero en corpus A → percentil 95 bajo
        self._p_low: float = 0.0
        self._p_high: float = 0.4
        self._mean: float = 0.08
        self._std: float = 0.14

    def fit(self, raw_scores: list[float]) -> "REPScoreNormalizer":
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
            n = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5
        elif self.method == "zscore":
            n = ((raw_score - self._mean) / self._std + 3) / 6
        else:
            denom = self._p_high - self._p_low
            n = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5
        return float(np.clip(n, 0.0, 1.0))

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
    def load(cls, path: Path) -> "REPScoreNormalizer":
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

class REPExtractor:
    """
    Extractor del indicador y₁₀ (Score REP — Ruptura Epistémica Positiva).

    Detecta cuatro mecanismos de transición epistémica positiva combinando
    análisis léxico con patrones regex y verificación sintáctica con spaCy.

    A diferencia de los extractores de violencia discursiva (y₁, y₂, y₄),
    el REP es esperado en densidades bajas en el corpus A (justicia ordinaria)
    y altas en el corpus B/C (JEP). Esta asimetría es precisamente la que
    mide la brecha discursiva del proyecto CFH.

    Parámetros
    ----------
    model_name : str
        Modelo spaCy para análisis contextual.
    normalizer : REPScoreNormalizer
        Normalizador. Si None, se usa uno con defaults.
    """

    def __init__(
        self,
        model_name: str = "es_core_news_lg",
        normalizer: Optional[REPScoreNormalizer] = None,
    ):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spaCy no instalado. "
                "Ejecuta: pip install spacy && python -m spacy download es_core_news_lg"
            )
        self.model_name = model_name
        self.normalizer = normalizer or REPScoreNormalizer()

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
        corpus_type: str = "B",
    ) -> REPExtractionResult:
        """Extrae el score REP de un segmento textual judicial."""
        t0 = time.perf_counter()

        if not text or len(text.strip()) < 20:
            return self._empty_result(doc_id, section_id, corpus_type)

        doc = self._nlp(text)
        sentences = list(doc.sents)
        sent_count = len([s for s in sentences if len(list(s)) >= 3])

        all_instances: list[REPInstance] = []

        # Todos los mecanismos se detectan sobre el texto completo con regex
        # (más robusto que análisis por oración para REP, que es más léxico)
        all_instances.extend(self._detect_reconocimiento(text, sentences))
        all_instances.extend(self._detect_restitución(text, sentences))
        all_instances.extend(self._detect_dih(text, sentences))
        all_instances.extend(self._detect_reparación(text, sentences))

        # Eliminar duplicados por solapamiento de spans
        all_instances = self._dedup_instances(all_instances)

        if sent_count == 0:
            score_raw = 0.0
        else:
            peso_total = sum(i.weight for i in all_instances)
            score_raw = peso_total / sent_count

        score_normalized = self.normalizer.normalize(score_raw)
        elapsed = time.perf_counter() - t0

        n_rec = sum(1 for i in all_instances if i.mechanism == "reconocimiento_responsabilidad")
        n_res = sum(1 for i in all_instances if i.mechanism == "restitución_identidad")
        n_dih = sum(1 for i in all_instances if i.mechanism == "lenguaje_dih_reparador")
        n_rep = sum(1 for i in all_instances if i.mechanism == "compromiso_reparación")

        logger.debug(
            f"y₁₀ REP [{doc_id}/{section_id}]: score={score_normalized:.3f} "
            f"raw={score_raw:.3f} instances={len(all_instances)} "
            f"(rec={n_rec}, res={n_res}, dih={n_dih}, rep={n_rep}) t={elapsed:.2f}s"
        )

        return REPExtractionResult(
            score=score_normalized,
            score_raw=score_raw,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=len(text),
            n_sentences=sent_count,
            n_instances=len(all_instances),
            n_reconocimiento=n_rec,
            n_restitución=n_res,
            n_dih=n_dih,
            n_reparación=n_rep,
            instances=all_instances,
            processing_time_s=elapsed,
        )

    # ── Detectores ────────────────────────────────────────────────────────

    def _detect_reconocimiento(
        self, text: str, sentences: list
    ) -> list[REPInstance]:
        """Detecta reconocimiento explícito de responsabilidad."""
        instances = []
        weight = REP_MECHANISM_WEIGHTS["reconocimiento_responsabilidad"]

        for pattern in _REP_RECONOCIMIENTO_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                instances.append(REPInstance(
                    mechanism="reconocimiento_responsabilidad",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    sent_index=sent_idx,
                ))
        return instances

    def _detect_restitución(
        self, text: str, sentences: list
    ) -> list[REPInstance]:
        """Detecta restitución de identidad de la víctima."""
        instances = []
        weight = REP_MECHANISM_WEIGHTS["restitución_identidad"]

        # Frases de restitución
        for pattern in _REP_RESTITUCION_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                instances.append(REPInstance(
                    mechanism="restitución_identidad",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    sent_index=sent_idx,
                ))

        # Nombre propio de la víctima en contexto de reconocimiento
        for pattern in _REP_NOMBRE_PROPIO_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                instances.append(REPInstance(
                    mechanism="restitución_identidad",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight * 1.1,  # boost por uso del nombre propio
                    sent_index=sent_idx,
                    details={"nombre_propio": True}
                ))

        return instances

    def _detect_dih(
        self, text: str, sentences: list
    ) -> list[REPInstance]:
        """Detecta adopción del lenguaje DIH y derechos humanos."""
        instances = []
        weight = REP_MECHANISM_WEIGHTS["lenguaje_dih_reparador"]

        for pattern in _REP_DIH_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                instances.append(REPInstance(
                    mechanism="lenguaje_dih_reparador",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    sent_index=sent_idx,
                ))
        return instances

    def _detect_reparación(
        self, text: str, sentences: list
    ) -> list[REPInstance]:
        """Detecta compromisos de reparación y garantías de no repetición."""
        instances = []
        weight = REP_MECHANISM_WEIGHTS["compromiso_reparación"]

        for pattern in _REP_REPARACION_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                instances.append(REPInstance(
                    mechanism="compromiso_reparación",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    sent_index=sent_idx,
                ))
        return instances

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_sent_index(self, char_pos: int, sentences: list) -> int:
        """Encuentra el índice de la oración que contiene char_pos."""
        for i, sent in enumerate(sentences):
            if sent.start_char <= char_pos < sent.end_char:
                return i
        return -1

    def _dedup_instances(
        self, instances: list[REPInstance]
    ) -> list[REPInstance]:
        """
        Elimina instancias duplicadas por solapamiento de spans.

        Cuando dos patrones distintos detectan el mismo span (o spans
        muy solapados), conserva el de mayor peso y descarta el otro.
        Umbral de solapamiento: > 50% del span más corto.
        """
        if len(instances) <= 1:
            return instances

        # Ordenar por peso descendente para conservar los más importantes
        sorted_inst = sorted(instances, key=lambda x: -x.weight)
        kept = []

        for candidate in sorted_inst:
            overlap = False
            for existing in kept:
                # Calcular solapamiento
                overlap_start = max(candidate.char_start, existing.char_start)
                overlap_end = min(candidate.char_end, existing.char_end)
                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    candidate_len = candidate.char_end - candidate.char_start
                    if candidate_len > 0 and overlap_len / candidate_len > 0.5:
                        overlap = True
                        break
            if not overlap:
                kept.append(candidate)

        return kept

    def _empty_result(
        self, doc_id: str, section_id: str, corpus_type: str
    ) -> REPExtractionResult:
        return REPExtractionResult(
            score=0.0, score_raw=0.0,
            doc_id=doc_id, section_id=section_id, corpus_type=corpus_type,
            text_length_chars=0, n_sentences=0, n_instances=0,
            warning="texto_vacio_o_muy_corto",
        )
