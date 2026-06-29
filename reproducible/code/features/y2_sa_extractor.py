"""
CFH · Extractor de Indicador y₂ — Score SA (Supresión de Agentividad)
======================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₁ (Violencia Discursiva)

Qué mide y₂:
    La densidad de construcciones gramaticales que ocultan o borran al
    agente responsable de una acción violenta, presentando el evento
    como si ocurriera sin agente identificable, de forma pasiva, o como
    resultado de circunstancias abstractas.

    Mecanismos detectados (en orden de peso):
    1. PASIVA_SIN_AGENTE   — voz pasiva sin complemento agente explícito
                             "fue reportado como baja"  (sin "por X")
    2. SE_IMPERSONAL       — construcciones con 'se' impersonal que encubren
                             al responsable directo de la acción violenta
                             "se presentó como resultado operacional"
    3. NOMINALIZACION      — nominalización de verbos de acción que elimina
                             al sujeto: "la presentación de la baja"
                             en lugar de "el comandante presentó la baja"
    4. SUJETO_INSTITUCIONAL — sujeto colectivo que diluye la responsabilidad
                             individual: "la unidad procedió", "el Ejército reportó"

    La SA es particularmente insidiosa porque opera a nivel gramatical,
    no léxico: no basta con detectar palabras clave, hay que analizar
    la estructura sintáctica de la oración.

Estrategia de implementación:
    A diferencia de y₁ (que requiere ConfliBERT para clasificación de tokens),
    y₂ usa spaCy (es_core_news_lg) para análisis sintáctico de dependencias.
    Esto permite:
    - Detección de voz pasiva mediante arco de dependencia "nsubjpass"
    - Detección de 'se' impersonal mediante patrones POS + DEP
    - Detección de nominalizaciones mediante sufijos nominales conocidos
    - Detección de sujetos institucionales mediante lista de entidades

    spaCy no requiere GPU — corre en CPU con velocidad adecuada para
    el corpus CFH (~100 tokens/seg en CPU estándar).

Diseño de ventana deslizante:
    Mismo patrón que y₁: ventana de oraciones (no tokens) para preservar
    la estructura sintáctica. El análisis de dependencias requiere oraciones
    completas — no se puede partir una oración por la mitad.

Dependencias:
    - spacy >= 3.7.0
    - es_core_news_lg (modelo spaCy para español)
    - numpy >= 1.26.0

    Instalación:
        pip install spacy
        python -m spacy download es_core_news_lg

Referencia teórica:
    Van Dijk, T. A. (2008). Discourse and Power. Palgrave Macmillan.
    Galtung, J. (1990). Cultural violence. JPR, 27(3), 291-305.
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

# Lazy import de spaCy — el módulo es importable sin el modelo instalado
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

logger = logging.getLogger("cfh.features.y2_sa")


# ---------------------------------------------------------------------------
# Constantes del dominio CFH
# ---------------------------------------------------------------------------

# Sufijos nominales frecuentes en nominalizaciones judiciales colombianas
# que ocultan al agente de una acción violenta
NOMINALIZACION_SUFIJOS = [
    "ación", "ición", "sión", "ción",   # presentación, disposición, misión
    "miento", "amiento", "imiento",      # cumplimiento, enfrentamiento
    "aje",                               # reclutaje, patrullaje
    "tura", "ura",                       # captura, baja (figurado)
    "nte",                               # compareciente, compareciente
]

# Verbos de acción violenta cuya nominalización suprime agentividad
# Cuando aparecen como sustantivos (no verbos), se detecta como SA
VERBOS_ACCION_NOMINALS = {
    # Verbo → forma nominal
    "presentar": ["presentación", "presentamiento"],
    "reportar": ["reporte", "reporto", "reporte"],
    "ejecutar": ["ejecución", "ejecutoria"],
    "capturar": ["captura"],
    "abatir": ["abatimiento"],
    "neutralizar": ["neutralización"],
    "dar de baja": ["baja", "bajas"],
    "proceder": ["procedimiento", "proceder"],
    "realizar": ["realización"],
    "llevar a cabo": ["llevamiento"],
    "efectuar": ["efectuación"],
    "desarrollar": ["desarrollo"],
    "cumplir": ["cumplimiento"],
}

# Lista plana de nominales que son SA cuando aparecen como sustantivos
# sin mención explícita del agente en la misma oración
SA_NOMINAL_TRIGGERS = set()
for verbales in VERBOS_ACCION_NOMINALS.values():
    SA_NOMINAL_TRIGGERS.update(verbales)
SA_NOMINAL_TRIGGERS.update([
    "baja", "bajas", "resultado", "resultados", "contacto", "operación",
    "misión", "tarea", "acción", "actividad", "procedimiento",
])

# Sujetos institucionales que diluyen responsabilidad individual
# (son SA cuando son el único sujeto de una acción violenta)
SUJETOS_INSTITUCIONALES = {
    # Unidades militares
    "ejército", "fuerzas militares", "fuerza pública",
    "batallón", "brigada", "unidad", "tropa", "tropas",
    "patrulla", "personal", "efectivos",
    # Sujetos impersonales institucionales
    "la institución", "el mando", "las fuerzas", "el operativo",
    "la misión", "el procedimiento", "el protocolo",
}

# Sustantivos que NO son nominalizaciones SA aunque tengan sufijos típicos
# Rangos militares, cargos, personas — son agentes, no nominalizaciones de acción
SA_NOMINAL_EXCLUSIONS = {
    # Rangos militares
    "comandante", "teniente", "capitán", "general", "coronel", "mayor",
    "sargento", "cabo", "soldado", "subteniente", "brigadier", "almirante",
    "suboficial", "oficial", "agente",
    # Cargos judiciales
    "magistrado", "magistrada", "juez", "fiscal", "defensor", "representante",
    "compareciente", "imputado", "procesado", "acusado", "testigo",
    # Personas genéricas
    "estudiante", "habitante", "participante", "integrante", "miembro",
    "presidente", "vicepresidente", "gobernante", "dirigente",
    # Otros sustantivos con sufijos típicos que no son SA
    "ambiente", "accidente", "incidente", "expediente", "antecedente",
    "componente", "continente", "paciente", "cliente", "gerente",
}

# Preposiciones que introducen el complemento agente en pasiva
# Si hay "por" + SN después del participio, hay agente explícito → no es SA
PREPOSICIONES_AGENTE = {"por", "mediante"}

# Verbos de reporte que en voz pasiva sin agente son SA prototípico
# "fue reportado" (sin "por X") = SA fuerte
VERBOS_REPORTE_PASIVA = {
    "reportar", "presentar", "registrar", "documentar", "anotar",
    "consignar", "informar", "comunicar", "notificar", "certificar",
}

# Patrones léxicos de SA que no requieren análisis sintáctico
# (complementan el análisis de dependencias con cobertura léxica rápida)
SA_LEXICAL_PATTERNS = [
    # Pasivas estereotipadas sin agente
    r"\bfue\s+(?:dado\s+de\s+baja|reportado|presentado|registrado|"
    r"encontrado|hallado|identificado|certificado)\b(?!\s+por\b)",
    r"\bfueron\s+(?:dados\s+de\s+baja|reportados|presentados|registrados|"
    r"encontrados|hallados|certificados)\b(?!\s+por\b)",
    # Se impersonal de encubrimiento
    r"\bse\s+(?:presentó|reportó|registró|procedió|realizó|llevó\s+a\s+cabo|"
    r"efectuó|desarrolló|ejecutó|verificó)\b",
    # Nominalizaciones de acción sin sujeto
    r"\b(?:la|el|los|las)\s+(?:presentación|reporte|registro|procedimiento|"
    r"verificación|identificación|certificación)\s+(?:de|del|como)\b",
    # Sujeto colectivo difuso
    r"\b(?:personal|tropas|efectivos|miembros|integrantes)\s+(?:del?\s+)?"
    r"(?:ejército|batallón|brigada|unidad|fuerza)",
]

# Compilar patrones léxicos
_SA_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in SA_LEXICAL_PATTERNS
]

# Pesos de cada mecanismo SA para el score compuesto
SA_MECHANISM_WEIGHTS = {
    "pasiva_sin_agente":    1.0,   # el mecanismo más fuerte: pasiva + sin "por X"
    "se_impersonal":        0.85,  # muy frecuente en textos judiciales militares
    "nominalizacion":       0.65,  # más sutil — nominaliza el evento violento
    "sujeto_institucional": 0.50,  # el más débil — puede ser legítimo
    "patron_lexical":       0.75,  # patrones compilados con regex
}


# ---------------------------------------------------------------------------
# Tipos de datos de resultado
# ---------------------------------------------------------------------------

@dataclass
class SAInstance:
    """Una instancia detectada de Supresión de Agentividad."""
    mechanism: str          # "pasiva_sin_agente" | "se_impersonal" | etc.
    text_span: str          # texto del span SA
    char_start: int
    char_end: int
    weight: float           # peso del mecanismo para el score
    sent_index: int         # índice de la oración en el segmento
    details: dict = field(default_factory=dict)  # info de debug


@dataclass
class SAExtractionResult:
    """
    Resultado completo de la extracción del indicador y₂.
    Misma estructura que EBIExtractionResult para consistencia del pipeline.
    """
    # ── Score principal ──────────────────────────────────────────────────────
    score: float            # y₂ normalizado [0, 1] — entra al SEM
    score_raw: float        # densidad SA bruta sin normalizar

    # ── Metadatos del segmento ───────────────────────────────────────────────
    doc_id: str
    section_id: str
    corpus_type: str
    text_length_chars: int
    n_sentences: int

    # ── Descomposición por mecanismo ─────────────────────────────────────────
    n_instances: int                            # total de instancias SA
    n_pasiva_sin_agente: int = 0
    n_se_impersonal: int = 0
    n_nominalizacion: int = 0
    n_sujeto_institucional: int = 0
    n_patron_lexical: int = 0

    # ── Instancias detectadas ────────────────────────────────────────────────
    instances: list[SAInstance] = field(default_factory=list)

    # ── Metadatos de procesamiento ───────────────────────────────────────────
    processing_time_s: float = 0.0
    spacy_model: str = "es_core_news_lg"

    # ── Flags de calidad ─────────────────────────────────────────────────────
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.n_sentences > 0 and self.text_length_chars > 20

    def to_dict(self) -> dict:
        return {
            "y2_sa_score": self.score,
            "y2_sa_score_raw": self.score_raw,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "corpus_type": self.corpus_type,
            "n_sentences": self.n_sentences,
            "n_sa_instances": self.n_instances,
            "sa_by_mechanism": {
                "pasiva_sin_agente":    self.n_pasiva_sin_agente,
                "se_impersonal":        self.n_se_impersonal,
                "nominalizacion":       self.n_nominalizacion,
                "sujeto_institucional": self.n_sujeto_institucional,
                "patron_lexical":       self.n_patron_lexical,
            },
            "top_sa_spans": [
                {"text": i.text_span, "mechanism": i.mechanism, "weight": i.weight}
                for i in sorted(self.instances, key=lambda x: -x.weight)[:5]
            ],
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Normalizador — misma interfaz que EBIScoreNormalizer
# ---------------------------------------------------------------------------

class SAScoreNormalizer:
    """
    Normaliza el score SA bruto al rango [0, 1].

    El score bruto es la densidad de instancias SA ponderadas por
    mecanismo, dividida por el número de oraciones del segmento.
    Esta densidad tiene una distribución sesgada a la derecha:
    la mayoría de los segmentos tienen pocas instancias SA, pero
    las secciones de hechos del corpus A pueden tener muchas.

    Parámetros por defecto calibrados sobre muestra piloto del corpus CFH.
    Actualizar con fit() después de procesar el corpus completo.
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
        # Valores por defecto — actualizar con calibración real
        self._p_low: float = 0.0
        self._p_high: float = 0.5
        self._mean: float = 0.12
        self._std: float = 0.15

    def fit(self, raw_scores: list[float]) -> "SAScoreNormalizer":
        arr = np.array(raw_scores)
        self._p_low = float(np.percentile(arr, self.low_percentile))
        self._p_high = float(np.percentile(arr, self.high_percentile))
        self._mean = float(arr.mean())
        self._std = float(arr.std()) or 1e-8
        self._fitted = True
        logger.info(
            f"SANormalizer calibrado: "
            f"p{self.low_percentile:.0f}={self._p_low:.4f}, "
            f"p{self.high_percentile:.0f}={self._p_high:.4f}"
        )
        return self

    def normalize(self, raw_score: float) -> float:
        if self.method == "percentile":
            denom = self._p_high - self._p_low
            if denom < 1e-10:
                return 0.5
            normalized = (raw_score - self._p_low) / denom
        elif self.method == "zscore":
            normalized = (raw_score - self._mean) / self._std
            # Mapear [-3, 3] → [0, 1]
            normalized = (normalized + 3) / 6
        else:  # minmax
            denom = self._p_high - self._p_low
            normalized = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5

        return float(np.clip(normalized, 0.0, 1.0))

    def save(self, path: Path) -> None:
        data = {
            "method": self.method,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "p_low": self._p_low,
            "p_high": self._p_high,
            "mean": self._mean,
            "std": self._std,
            "fitted": self._fitted,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SAScoreNormalizer":
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

class SAExtractor:
    """
    Extractor del indicador y₂ (Score SA — Supresión de Agentividad).

    Combina análisis sintáctico de dependencias con spaCy y patrones
    léxicos compilados para detectar los cuatro mecanismos SA:
    pasiva sin agente, se impersonal, nominalización, sujeto institucional.

    Parámetros
    ----------
    model_name : str
        Modelo spaCy a usar. Debe ser el modelo grande (lg) para mejor
        calidad en el análisis de dependencias.
    normalizer : SAScoreNormalizer
        Instancia del normalizador. Si es None, se crea uno con defaults.
    min_sentence_tokens : int
        Oraciones con menos tokens que este umbral se omiten del análisis
        (evitan falsos positivos en oraciones fragmentarias por OCR).
    """

    def __init__(
        self,
        model_name: str = "es_core_news_lg",
        normalizer: Optional[SAScoreNormalizer] = None,
        min_sentence_tokens: int = 5,
    ):
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spaCy no está instalado. "
                "Ejecuta: pip install spacy && python -m spacy download es_core_news_lg"
            )

        self.model_name = model_name
        self.min_sentence_tokens = min_sentence_tokens
        self.normalizer = normalizer or SAScoreNormalizer()

        logger.info(f"Cargando modelo spaCy: {model_name}")
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Modelo spaCy '{model_name}' no encontrado. "
                f"Instala con: python -m spacy download {model_name}"
            )
        logger.info("Modelo spaCy cargado correctamente.")

    # ── Método principal ─────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        doc_id: str = "unknown",
        section_id: str = "unknown",
        corpus_type: str = "A",
    ) -> SAExtractionResult:
        """
        Extrae el score SA de un segmento textual judicial.

        El texto debe ser la sección limpia ya procesada por cleaner.py.
        No es necesario segmentar por oraciones previamente — spaCy
        lo hace automáticamente.

        Parámetros
        ----------
        text : str
            Texto de la sección judicial a analizar.
        doc_id : str
            Identificador del documento para trazabilidad.
        section_id : str
            Identificador de la sección (HECHOS, CONSIDERACIONES, etc.)
        corpus_type : str
            "A" | "B" | "C"

        Retorna
        -------
        SAExtractionResult con score normalizado y todas las instancias.
        """
        t0 = time.perf_counter()

        if not text or len(text.strip()) < 20:
            return self._empty_result(doc_id, section_id, corpus_type)

        # Procesar con spaCy (incluye tokenización, POS, dependencias)
        doc = self._nlp(text)
        sentences = list(doc.sents)

        all_instances: list[SAInstance] = []
        sent_count = 0

        for sent_idx, sent in enumerate(sentences):
            tokens = [t for t in sent if not t.is_space]
            if len(tokens) < self.min_sentence_tokens:
                continue
            sent_count += 1

            # Detectar los cuatro mecanismos SA
            instances = (
                self._detect_pasiva_sin_agente(sent, sent_idx) +
                self._detect_se_impersonal(sent, sent_idx) +
                self._detect_nominalizacion(sent, sent_idx) +
                self._detect_sujeto_institucional(sent, sent_idx)
            )
            all_instances.extend(instances)

        # Detectar patrones léxicos sobre el texto completo
        lexical_instances = self._detect_patrones_lexicales(text)
        all_instances.extend(lexical_instances)

        # Calcular score bruto: suma ponderada / número de oraciones
        if sent_count == 0:
            score_raw = 0.0
        else:
            peso_total = sum(i.weight for i in all_instances)
            score_raw = peso_total / sent_count

        score_normalized = self.normalizer.normalize(score_raw)

        elapsed = time.perf_counter() - t0

        # Contar por mecanismo
        n_pasiva = sum(1 for i in all_instances if i.mechanism == "pasiva_sin_agente")
        n_se = sum(1 for i in all_instances if i.mechanism == "se_impersonal")
        n_nom = sum(1 for i in all_instances if i.mechanism == "nominalizacion")
        n_suj = sum(1 for i in all_instances if i.mechanism == "sujeto_institucional")
        n_lex = sum(1 for i in all_instances if i.mechanism == "patron_lexical")

        logger.debug(
            f"y₂ SA [{doc_id}/{section_id}]: score={score_normalized:.3f} "
            f"raw={score_raw:.3f} instances={len(all_instances)} "
            f"(pasiva={n_pasiva}, se={n_se}, nom={n_nom}, "
            f"suj={n_suj}, lex={n_lex}) t={elapsed:.2f}s"
        )

        return SAExtractionResult(
            score=score_normalized,
            score_raw=score_raw,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=len(text),
            n_sentences=sent_count,
            n_instances=len(all_instances),
            n_pasiva_sin_agente=n_pasiva,
            n_se_impersonal=n_se,
            n_nominalizacion=n_nom,
            n_sujeto_institucional=n_suj,
            n_patron_lexical=n_lex,
            instances=all_instances,
            processing_time_s=elapsed,
            spacy_model=self.model_name,
        )

    # ── Detectores por mecanismo ─────────────────────────────────────────────

    def _detect_pasiva_sin_agente(
        self, sent: "Span", sent_idx: int
    ) -> list[SAInstance]:
        """
        Detecta construcciones pasivas sin complemento agente explícito.

        En español, la voz pasiva se marca en spaCy con:
        - El sujeto paciente tiene dep_ == "nsubjpass"
        - El verbo auxiliar es "ser" o "estar" + participio

        Una pasiva tiene agente explícito si hay un token con dep_ == "agent"
        o si hay "por" + SN inmediatamente después del participio.

        Solo cuenta como SA si:
        1. Hay voz pasiva (nsubjpass en algún token)
        2. No hay complemento agente ("por" + SN que responda "por quién")
        3. El verbo principal está relacionado con acción sobre personas
        """
        instances = []

        # Buscar sujetos pasivos en la oración
        passive_subjects = [t for t in sent if t.dep_ == "nsubjpass"]

        for subj in passive_subjects:
            # Encontrar el verbo del que depende el sujeto pasivo
            verb = subj.head
            verb_lemma = verb.lemma_.lower()

            # Verificar si hay complemento agente explícito
            has_agent = self._has_explicit_agent(verb, sent)
            if has_agent:
                continue  # No es SA si hay agente

            # Calcular peso: mayor si el verbo es de reporte o acción directa
            weight = SA_MECHANISM_WEIGHTS["pasiva_sin_agente"]
            if verb_lemma in VERBOS_REPORTE_PASIVA:
                weight *= 1.3  # boost para verbos de reporte sin agente

            # Extraer el span de la construcción pasiva
            span_start = min(subj.idx, verb.idx)
            span_end = max(subj.idx + len(subj.text), verb.idx + len(verb.text))
            span_text = sent.text[
                max(0, span_start - sent.start_char):
                min(len(sent.text), span_end - sent.start_char + 20)
            ]

            instances.append(SAInstance(
                mechanism="pasiva_sin_agente",
                text_span=span_text.strip(),
                char_start=span_start,
                char_end=span_end,
                weight=weight,
                sent_index=sent_idx,
                details={
                    "verb_lemma": verb_lemma,
                    "subject": subj.text,
                    "has_agent": False,
                }
            ))

        return instances

    def _detect_se_impersonal(
        self, sent: "Span", sent_idx: int
    ) -> list[SAInstance]:
        """
        Detecta construcciones con 'se' impersonal que encubren al responsable.

        En spaCy, el 'se' impersonal/pasivo reflejo típicamente tiene:
        - token.text.lower() == "se"
        - token.dep_ en {"expl", "expl:pv", "nsubj"} (varía por modelo)
        - El verbo al que se adjunta es intransitivo-izado

        Se excluye el 'se' reflexivo genuino (cuando hay un agente claro
        y el sujeto coincide con el objeto).

        Adicionalmente, se usa análisis léxico para capturar casos que
        spaCy puede no etiquetar correctamente.
        """
        instances = []

        for token in sent:
            if token.text.lower() != "se":
                continue
            if token.dep_ not in {"expl", "expl:pv", "nsubj", "obj", "iobj"}:
                continue

            head = token.head
            head_lemma = head.lemma_.lower()

            # Excluir reflexivos verdaderos (sujeto == objeto semántico)
            # Heurística: si el sujeto nominal es una persona específica → no SA
            has_person_subject = any(
                t.dep_ == "nsubj" and t.ent_type_ in {"PER", "ORG"}
                for t in head.children
            )
            if has_person_subject:
                continue

            # Verificar que el verbo sea de acción relevante para el dominio CFH
            is_relevant = (
                head_lemma in {
                    "presentar", "reportar", "registrar", "proceder",
                    "realizar", "efectuar", "llevar", "desarrollar",
                    "ejecutar", "verificar", "certificar", "identificar",
                    "dar", "establecer",
                }
            )
            weight = SA_MECHANISM_WEIGHTS["se_impersonal"]
            if not is_relevant:
                weight *= 0.5  # peso reducido para verbos genéricos

            span_start = min(token.idx, head.idx)
            span_end = max(token.idx + len(token.text), head.idx + len(head.text))
            span_text = sent.text[
                max(0, span_start - sent.start_char):
                min(len(sent.text), span_end - sent.start_char + 20)
            ]

            instances.append(SAInstance(
                mechanism="se_impersonal",
                text_span=span_text.strip(),
                char_start=span_start,
                char_end=span_end,
                weight=weight,
                sent_index=sent_idx,
                details={"verb_lemma": head_lemma, "relevant": is_relevant}
            ))

        return instances

    def _detect_nominalizacion(
        self, sent: "Span", sent_idx: int
    ) -> list[SAInstance]:
        """
        Detecta nominalizaciones de acción que eliminan al sujeto.

        Una nominalización SA es cuando un sustantivo derivado de un verbo
        de acción violenta aparece sin mención del agente en la misma oración.

        Ejemplo:
        - SA:    "La presentación de la baja se realizó conforme al protocolo"
                 (¿quién presentó? no se dice)
        - No SA: "El comandante procedió a la presentación de la baja"
                 (hay agente: el comandante)
        """
        instances = []

        for token in sent:
            # Solo sustantivos
            if token.pos_ not in {"NOUN"}:
                continue

            lemma = token.lemma_.lower()
            text_lower = token.text.lower()

            # Excluir rangos militares, cargos y personas — no son SA
            if lemma in SA_NOMINAL_EXCLUSIONS or text_lower in SA_NOMINAL_EXCLUSIONS:
                continue

            # Verificar si es un nominal de acción SA
            is_sa_nominal = (
                lemma in SA_NOMINAL_TRIGGERS or
                text_lower in SA_NOMINAL_TRIGGERS or
                any(text_lower.endswith(suf) for suf in NOMINALIZACION_SUFIJOS
                    if len(text_lower) > len(suf) + 3)
            )

            if not is_sa_nominal:
                continue

            # Verificar si hay un poseedor/agente explícito (gen. objetivo)
            # "la presentación de <agente>" — el agente va en genitivo subjetivo
            has_subjective_genitive = any(
                child.dep_ in {"nmod", "nmod:poss"} and
                child.ent_type_ in {"PER", "ORG"}
                for child in token.children
            )
            if has_subjective_genitive:
                continue

            # Verificar si el nominal está en la posición de sujeto de la oración
            # (nominalizaciones en posición de sujeto son SA más fuertes)
            is_subject = token.dep_ in {"nsubj", "nsubjpass"}
            weight = SA_MECHANISM_WEIGHTS["nominalizacion"]
            if is_subject:
                weight *= 1.2

            instances.append(SAInstance(
                mechanism="nominalizacion",
                text_span=token.text,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
                weight=weight,
                sent_index=sent_idx,
                details={
                    "lemma": lemma,
                    "is_subject": is_subject,
                    "has_subjective_genitive": has_subjective_genitive,
                }
            ))

        return instances

    def _detect_sujeto_institucional(
        self, sent: "Span", sent_idx: int
    ) -> list[SAInstance]:
        """
        Detecta sujetos institucionales que diluyen la responsabilidad individual.

        Cuando el sujeto gramatical de una acción violenta es una entidad
        colectiva institucional (el Ejército, la unidad, el batallón) sin
        especificar al individuo responsable, se detecta como SA de peso menor.
        """
        instances = []

        # Buscar sujetos nominales del verbo principal
        for token in sent:
            if token.dep_ not in {"nsubj", "nsubjpass"}:
                continue

            token_text = token.text.lower()
            token_lemma = token.lemma_.lower()

            is_institutional = (
                token_lemma in SUJETOS_INSTITUCIONALES or
                token_text in SUJETOS_INSTITUCIONALES or
                # Detectar "la unidad X", "el batallón Y" como sujeto
                any(
                    child.lemma_.lower() in SUJETOS_INSTITUCIONALES
                    for child in token.children
                    if child.dep_ in {"det", "amod"}
                )
            )

            if not is_institutional:
                continue

            # El verbo del que depende el sujeto institucional
            verb = token.head
            verb_lemma = verb.lemma_.lower()

            # Solo contar si el verbo es de acción relevante
            is_action_verb = verb.pos_ == "VERB" and verb_lemma in {
                "proceder", "realizar", "reportar", "presentar", "efectuar",
                "ejecutar", "dar", "llevar", "identificar", "verificar",
                "encontrar", "hallar",
            }
            if not is_action_verb:
                continue

            weight = SA_MECHANISM_WEIGHTS["sujeto_institucional"]

            instances.append(SAInstance(
                mechanism="sujeto_institucional",
                text_span=f"{token.text} {verb.text}",
                char_start=min(token.idx, verb.idx),
                char_end=max(token.idx + len(token.text), verb.idx + len(verb.text)),
                weight=weight,
                sent_index=sent_idx,
                details={"subject": token.text, "verb": verb.text}
            ))

        return instances

    def _detect_patrones_lexicales(self, text: str) -> list[SAInstance]:
        """
        Detecta patrones SA mediante expresiones regulares sobre el texto completo.

        Complementa el análisis sintáctico con cobertura léxica rápida para
        casos que spaCy puede no etiquetar correctamente (OCR imperfecto,
        oraciones fragmentadas, jerga judicial específica).
        """
        instances = []
        weight = SA_MECHANISM_WEIGHTS["patron_lexical"]

        for pattern in _SA_PATTERNS_COMPILED:
            for match in pattern.finditer(text):
                instances.append(SAInstance(
                    mechanism="patron_lexical",
                    text_span=match.group(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=weight,
                    sent_index=-1,  # no aplica para patrones de texto completo
                    details={"pattern": pattern.pattern[:50]}
                ))

        return instances

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _has_explicit_agent(self, verb_token: "Token", sent: "Span") -> bool:
        """
        Determina si un verbo tiene un complemento agente explícito.

        En español, el complemento agente de la pasiva se introduce con "por".
        Busca hijos directos del verbo con dep_ == "agent" o
        preposición "por" seguida de SN.
        """
        for child in verb_token.children:
            if child.dep_ == "agent":
                return True
            if child.text.lower() in PREPOSICIONES_AGENTE and child.dep_ == "prep":
                return True

        # Búsqueda adicional: "por" en los 5 tokens siguientes al verbo
        verb_pos_in_sent = verb_token.i - sent.start
        for offset in range(1, 6):
            idx = verb_pos_in_sent + offset
            if idx >= len(list(sent)):
                break
            tok = list(sent)[idx]
            if tok.text.lower() == "por":
                return True

        return False

    def _empty_result(
        self, doc_id: str, section_id: str, corpus_type: str
    ) -> SAExtractionResult:
        return SAExtractionResult(
            score=0.0,
            score_raw=0.0,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=0,
            n_sentences=0,
            n_instances=0,
            warning="texto_vacio_o_muy_corto",
        )
