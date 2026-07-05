"""
CFH · Extractor de Indicador y₄ — Score NV (Negación de Victimización)
=======================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₁ (Violencia Discursiva)

Qué mide y₄:
    La densidad de expresiones que niegan la condición de víctima civil
    de las personas asesinadas en el marco del Macrocaso 003, reencuadrando
    su identidad dentro del marco semántico del enemigo combatiente, del
    sujeto ilegítimo merecedor de la muerte, o de la cifra anónima sin
    identidad.

    La NV se reproduce tanto en el habla del compareciente como en el
    lenguaje institucional del expediente (fiscalía, tribunal), que puede
    reproducir el marco de negación al citarlo, cuantificar a las víctimas
    como cifras sin nombre o recodificar la muerte como baja en combate.
    Por ello el indicador opera en los tres corpus (A, B, C).

    Mecanismos detectados (en orden de peso):

    1. RECATEGORIZACION_COMBATIENTE (peso 1.0)
       Vocabulario que reclasifica a la víctima como miembro de un grupo
       armado ilegal, objetivo militar legítimo, o baja en combate.
       Incluye alias criminal ("alias X") y la recodificación de la muerte
       como resultado operacional ("dado de baja en combate", "presentado
       como baja").
       CLAVE: solo cuenta cuando el término reproduce el marco de negación
       aplicado a la VÍCTIMA (concreta o abstracta). NO cuenta cuando el
       término designa al VICTIMARIO/institución perpetradora ("organización
       criminal", "miembros del batallón") ni en usos neutros ("objetivo de
       la investigación", "elementos probatorios", "tipo penal").

    2. DESPERSONALIZACION_CUANTIFICADA (peso 0.85)
       Reducción de la víctima a una cifra anónima o a un colectivo sin
       identidad: "N personas no identificadas", "N sin información etaria",
       "otras personas", "otros dos asesinatos". Es la forma administrativa
       de la negación: la víctima deja de tener nombre y se vuelve número.

    3. DESHUMANIZACION (peso 0.70)
       Términos que niegan la identidad personal de la víctima, reduciéndola
       a una categoría abstracta o cosificada: "el individuo", "el sujeto",
       "el occiso", "el cuerpo", "el elemento". Sujeta a los mismos filtros
       de referente y neutralidad que la recategorización.

    4. DESCALIFICACION_MORAL (peso 0.65)
       Atribución de características morales negativas que justifican
       implícitamente la muerte: "de baja trayectoria social", "con
       antecedentes penales", "habitante de calle", "drogadicto".

Filtros de precisión (auditoría conceptual, jul-2026):
    - FILTRO DE REFERENTE-VICTIMARIO: excluye el término cuando modifica o
      se refiere a la fuerza pública o a la estructura perpetradora
      (batallón, brigada, pelotón, ejército, "organización/plan/estructura
      criminal", "adscrito a", "anidada en"). En esos casos el término
      designa al victimario, no niega a la víctima.
    - FILTRO DE NEUTRALIDAD: excluye colocaciones donde el término es un
      falso amigo léxico ("elementos descritos/probatorios", "tipo penal/de
      bajas", "objetivo de denunciar/de la investigación", "integrantes de
      la representación/de la Sala").
    Ambos filtros preservan la NV legítima: "presentar a la víctima como
    miembro insurgente" (recae sobre la víctima) y "mostrar bajas del
    enemigo" (reproduce el marco en abstracto) SÍ cuentan como NV.

Nota metodológica (auditoría conceptual, jul-2026):
    Se eliminó el mecanismo previo "ATRIBUCION_ARMAMENTO" (atribución
    post-mortem de armas/uniformes). La atribución de armamento pertenece
    conceptualmente al indicador y₁ (EBI — Eufemismo Bélico-Institucional)
    y/o al montaje operacional, no a la Negación de Victimización en sentido
    estricto.

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
    r"\bmiembro(?:s)?\s+de\s+(?:grupo(?:s)?\s+)?(?:armado|ilegal|irregular|insurgente)",
    r"\bpresunto(?:s)?\s+(?:guerrillero|terrorista|delincuente|integrante)",
    r"\bobjetivo(?:s)?\s+de\s+(?:alto\s+valor|la\s+operación\s+militar)",
    r"\bvinculado(?:s)?\s+(?:a|con)\s+(?:grupo(?:s)?\s+)?(?:armado|ilegal|al\s+margen)",
    # Recodificación de la muerte como baja/resultado operacional
    r"\b(?:dado(?:s)?|dar)\s+de\s+baja\b",
    r"\bbaja(?:s)?\s+(?:en|de)\s+combate",
    r"\bpresentad[oa](?:s)?\s+como\s+(?:baja|muert[oa]|result[a]?|dad[oa]\s+de\s+baja)",
    r"\bmuert[oa](?:s)?\s+en\s+(?:combate|enfrentamiento)",
    r"\ben\s+combate\s+con\s+(?:el\s+|las?\s+)?(?:frente|farc|eln|auc|guerrilla|grupo)",
    # Alias criminal (mote que recategoriza)
    r"\balias\s+[«\"']?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+",
]

# ── Mecanismo 2: Despersonalización cuantificada / anonimización ──────────
# Reduce a la víctima a cifra anónima o colectivo sin identidad.
# _NUM cubre dígitos ("18") y numerales en palabra ("cuatro", "dos"...),
# frecuentes en el expediente ("cuatro personas no identificadas").
_NUM = (r"(?:\d+|un[oa]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
        r"once|doce|trece|catorce|quince|dieciséis|diecisiete|dieciocho|"
        r"diecinueve|veinte|varias?|muchas?|numerosas?|algun[oa]s?)")

NV_DESPERSONALIZACION_FRASES = [
    # cifra + persona(s)/hombre(s)/mujer(es) + no identificad*
    _NUM + r"\s+(?:persona|hombre|mujer|joven|víctima|cuerpo|cadáver)s?,?\s*"
    r"(?:no\s+)?(?:identificad|reconocid)[oa]s?",
    # N (personas) no identificad* (cifra o palabra, sin sustantivo intermedio)
    _NUM + r"\s+no\s+(?:estaban\s+|fueron\s+|han\s+sido\s+|se\s+han\s+)?"
    r"(?:plenamente\s+)?identificad[oa]s?",
    # N sin información / sin identificar / sin identidad
    _NUM + r"\s+sin\s+(?:información|identificación|identificar|identidad|datos)",
    # colectivo anónimo sin cifra
    r"\botr[oa]s\s+(?:dos\s+|tres\s+|cuatro\s+|varias?\s+|muchas?\s+)?"
    r"(?:persona|víctima|asesinato|homicidio|muerte)s?",
    r"\bvari[oa]s\s+(?:hombre|persona|individuo)s?\b",
    # "N personas" como cifra que sustituye el nombre (en contexto de muerte)
    _NUM + r"\s+(?:persona|hombre|mujer|civil|víctima)s?\b(?![\w\s]{0,15}"
    r"(?:identificad|llamad|de\s+nombre|conocid))",
    # "sin identificar" / "no identificado" aplicado a la víctima
    r"\b(?:persona|hombre|mujer|joven|individuo|cuerpo|cadáver)s?,?\s*"
    r"(?:no\s+identificad|sin\s+identificar)[oa]?s?",
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

# ── FILTRO DE REFERENTE-VICTIMARIO ────────────────────────────────────────
# Cuando el término NV modifica/refiere a la fuerza pública o a la estructura
# perpetradora, NO es NV (designa al victimario, no niega a la víctima).
# Señales léxicas de que el referente es el victimario/institución:
NV_REFERENTE_VICTIMARIO = {
    "batallón", "brigada", "pelotón", "compañía", "ejército", "militar",
    "militares", "tropa", "tropas", "soldado", "soldados", "oficial",
    "comandante", "unidad", "contraguerrilla", "fuerza",
    "organización", "estructura", "aparato", "plan", "patrón",
    "grupo especial", "banda", "red",
    "representación",  # "integrantes de la representación de víctimas"
    "sala", "despacho", "fiscalía", "tribunal", "magistratura",
}

# Frases que marcan referente-victimario de forma inequívoca
NV_VICTIMARIO_FRASES = [
    r"organización\s+criminal",
    r"estructura\s+criminal",
    r"aparato\s+criminal",
    r"plan\s+criminal",
    r"patrón\s+(?:macro)?criminal",
    r"empresa\s+criminal",
    r"anidad[oa]\s+en",
    r"adscrit[oa]s?\s+(?:al?|a\s+la)\s+(?:batallón|brigada|pelotón|ejército|unidad)",
    r"integrante(?:s)?\s+de(?:l)?\s+(?:pelotón|batallón|brigada|ejército|"
    r"grupo\s+especial|la\s+representación|la\s+sala|la\s+fiscalía)",
    r"miembro(?:s)?\s+adscrit[oa]s?",
    r"personal\s+integrante",
]

# ── FILTRO DE NEUTRALIDAD (falsos amigos léxicos) ────────────────────────
# Colocaciones donde el término NO es NV sino uso común/procesal.
NV_NEUTRALIDAD_FRASES = [
    r"elemento(?:s)?\s+(?:descrit[oa]s?|probatori[oa]s?|de\s+(?:juicio|prueba|"
    r"convicción)|material(?:es)?|del?\s+(?:plan|tipo|delito))",
    r"tipo\s+(?:penal|de\s+baja|de\s+conducta|delictivo|de\s+hecho)",
    r"este\s+tipo\s+de",
    r"objetivo(?:s)?\s+de\s+(?:denunciar|la\s+(?:investigación|casación|demanda|"
    r"norma|disposición|sala)|el\s+(?:proceso|recurso|cargo))",
    r"con\s+el\s+objetivo\s+de",
    r"integrante(?:s)?\s+de\s+la\s+representación\s+de\s+(?:las?\s+)?víctimas?",
    r"miembro(?:s)?\s+de\s+la\s+(?:sala|fiscalía|magistratura|comisión)",
]

# ── Contextos de cuestionamiento del NV (→ reducen el peso) ───────────────
# Cuando el NV aparece en un contexto de cuestionamiento, el peso se reduce
# porque está siendo citado para ser refutado (fenómeno del metalenguaje)
NV_CUESTIONAMIENTO_TRIGGERS = [
    r"\bpresentad[oa](?:s)?\s+(?:falsa|fraudulenta|ilegítima)mente\s+como",
    r"\bfalsamente\s+(?:presentad|reportad|identificad)[oa]s?",
    r"\berróneamente\s+(?:identificad|catalogad|clasificad)[oa]s?",
    r"\bque\s+(?:en\s+realidad|realmente)\s+(?:era|eran|no\s+era|no\s+eran)",
    r"\bno\s+era(?:n)?\s+(?:guerrillero|combatiente|delincuente|integrante)",
    r"\b(?:civil|civiles)\s+(?:inocente|inerme|desarmad[oa])",
]

# NOTA: "presentar a la víctima como X" NO es cuestionamiento — es el vehículo
# de la reproducción del marco NV (recae sobre la víctima). Se detecta como
# recategorización plena, no como cuestionamiento.

# Patrones compilados
_NV_COMBATIENTE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_COMBATIENTE_FRASES]
_NV_DESPERSONALIZACION_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_DESPERSONALIZACION_FRASES]
_NV_DESCALIFICACION_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_DESCALIFICACION_FRASES]
_NV_VICTIMARIO_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_VICTIMARIO_FRASES]
_NV_NEUTRALIDAD_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_NEUTRALIDAD_FRASES]
_NV_CUESTIONAMIENTO_COMPILED = [re.compile(p, re.IGNORECASE) for p in NV_CUESTIONAMIENTO_TRIGGERS]

# Pesos por mecanismo
NV_MECHANISM_WEIGHTS = {
    "recategorizacion_combatiente":   1.00,
    "despersonalizacion_cuantificada": 0.85,
    "deshumanizacion":                0.70,
    "descalificacion_moral":          0.65,
}

# Factor de reducción cuando el NV aparece en contexto de cuestionamiento
NV_CUESTIONAMIENTO_FACTOR = 0.25

# Ventana de caracteres para evaluar referente/neutralidad alrededor del término
NV_REFERENTE_WINDOW = 60


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
    n_despersonalizacion: int = 0
    n_deshumanizacion: int = 0
    n_descalificacion: int = 0
    n_excluidas_victimario: int = 0  # términos descartados por referente-victimario
    n_excluidas_neutralidad: int = 0  # términos descartados por falso amigo léxico
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
                "recategorizacion_combatiente":    self.n_recategorizacion,
                "despersonalizacion_cuantificada": self.n_despersonalizacion,
                "deshumanizacion":                 self.n_deshumanizacion,
                "descalificacion_moral":           self.n_descalificacion,
            },
            "n_excluidas": {
                "victimario":  self.n_excluidas_victimario,
                "neutralidad": self.n_excluidas_neutralidad,
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
    verificación de contexto mediante spaCy, aplicando dos filtros de
    precisión (referente-victimario y neutralidad) que distinguen la
    reproducción del marco de negación contra la víctima del uso del término
    para designar al victimario o de sus falsos amigos léxicos.

    Mecanismos: recategorizacion_combatiente, despersonalizacion_cuantificada,
    deshumanizacion, descalificacion_moral.

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
        n_excl_vict = 0
        n_excl_neut = 0

        # Análisis por oración para mecanismos que requieren contexto sintáctico
        for sent_idx, sent in enumerate(sentences):
            if len(list(sent)) < 3:
                continue
            rec, ev, en = self._detect_recategorizacion(sent, sent_idx, text)
            all_instances.extend(rec)
            n_excl_vict += ev
            n_excl_neut += en

            des, ev2, en2 = self._detect_deshumanizacion(sent, sent_idx, text)
            all_instances.extend(des)
            n_excl_vict += ev2
            n_excl_neut += en2

        # Despersonalización cuantificada — sobre texto completo (regex)
        all_instances.extend(self._detect_despersonalizacion(text, sentences))

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
        n_dep = sum(1 for i in all_instances if i.mechanism == "despersonalizacion_cuantificada")
        n_des = sum(1 for i in all_instances if i.mechanism == "deshumanizacion")
        n_desc = sum(1 for i in all_instances if i.mechanism == "descalificacion_moral")
        n_quest = sum(1 for i in all_instances if i.is_questioned)

        logger.debug(
            f"y₄ NV [{doc_id}/{section_id}]: score={score_normalized:.3f} "
            f"raw={score_raw:.3f} instances={len(all_instances)} "
            f"(rec={n_rec}, dep={n_dep}, des={n_des}, desc={n_desc}, "
            f"quest={n_quest}, excl_vict={n_excl_vict}, excl_neut={n_excl_neut}) "
            f"t={elapsed:.2f}s"
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
            n_despersonalizacion=n_dep,
            n_deshumanizacion=n_des,
            n_descalificacion=n_desc,
            n_excluidas_victimario=n_excl_vict,
            n_excluidas_neutralidad=n_excl_neut,
            instances=all_instances,
            processing_time_s=elapsed,
        )

    # ── Detectores ────────────────────────────────────────────────────────

    def _detect_recategorizacion(
        self, sent: "Span", sent_idx: int, full_text: str
    ) -> tuple[list[NVInstance], int, int]:
        """
        Detecta recategorización de la víctima como combatiente.

        Devuelve (instancias, n_excluidas_victimario, n_excluidas_neutralidad).
        Aplica los filtros de referente y neutralidad antes de aceptar cada
        término léxico.
        """
        instances = []
        n_excl_vict = 0
        n_excl_neut = 0

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

            # FILTRO DE NEUTRALIDAD (falso amigo léxico)
            if self._es_uso_neutro(token, full_text):
                n_excl_neut += 1
                continue

            # FILTRO DE REFERENTE-VICTIMARIO
            if self._refiere_victimario(token, full_text):
                n_excl_vict += 1
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

        # Frases nominales de recategorización (alias, recodificación-combate,
        # grupos armados) — estas ya son inequívocas, no requieren filtro de
        # referente (una frase como "dado de baja en combate" siempre recae
        # sobre la víctima presentada como baja).
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

        return instances, n_excl_vict, n_excl_neut

    def _detect_despersonalizacion(
        self, text: str, sentences: list
    ) -> list[NVInstance]:
        """
        Detecta despersonalización cuantificada / anonimización de la víctima.

        Reduce a la víctima a cifra o colectivo sin identidad. Se detecta con
        regex sobre el texto completo porque el patrón es primariamente léxico
        ("N no identificados", "otras personas").
        """
        instances = []
        weight = NV_MECHANISM_WEIGHTS["despersonalizacion_cuantificada"]

        for pattern in _NV_DESPERSONALIZACION_COMPILED:
            for match in pattern.finditer(text):
                sent_idx = self._find_sent_index(match.start(), sentences)
                is_questioned = self._is_in_questioning_context(
                    match.start(), match.end(), text
                )
                w = weight * (NV_CUESTIONAMIENTO_FACTOR if is_questioned else 1.0)
                instances.append(NVInstance(
                    mechanism="despersonalizacion_cuantificada",
                    text_span=match.group().strip(),
                    char_start=match.start(),
                    char_end=match.end(),
                    weight=w,
                    weight_base=weight,
                    is_questioned=is_questioned,
                    sent_index=sent_idx,
                ))

        return instances

    def _detect_deshumanizacion(
        self, sent: "Span", sent_idx: int, full_text: str
    ) -> tuple[list[NVInstance], int, int]:
        """
        Detecta términos que niegan la identidad personal de la víctima.

        Devuelve (instancias, n_excluidas_victimario, n_excluidas_neutralidad).
        Los términos de deshumanización directa (individuo, sujeto, elemento)
        son NV cuando refieren a personas; se filtran los usos neutros
        ("elementos probatorios") y los que refieren al victimario.
        """
        instances = []
        n_excl_vict = 0
        n_excl_neut = 0

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

            # FILTRO DE NEUTRALIDAD (elementos probatorios, tipo penal, etc.)
            if self._es_uso_neutro(token, full_text):
                n_excl_neut += 1
                continue

            # FILTRO DE REFERENTE-VICTIMARIO (elementos del plan criminal, etc.)
            if self._refiere_victimario(token, full_text):
                n_excl_vict += 1
                continue

            # Para términos contextuales, verificar que referencian a la víctima
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

        return instances, n_excl_vict, n_excl_neut

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
                    sent_index=self._find_sent_index(match.start(), sentences),
                ))

        return instances

    # ── Filtros de referente y neutralidad ────────────────────────────────

    def _refiere_victimario(self, token: "Token", full_text: str) -> bool:
        """
        Determina si el término NV se refiere al victimario/institución
        perpetradora (y por tanto NO es NV contra la víctima).

        Estrategia híbrida:
        1. Sintáctica: revisa si el token modifica o depende de un núcleo
           que denota fuerza pública o estructura perpetradora.
        2. Léxica (respaldo): busca señales de victimario en la ventana
           inmediata alrededor del token, robusto cuando el árbol de
           dependencias sale sucio (OCR, numeración judicial).
        """
        # 1. Sintáctica — el head o los hijos denotan victimario/institución
        candidatos = [token.head]
        candidatos.extend(token.children)
        # también el "abuelo" (para "integrantes de la representación")
        if token.head is not None and token.head.head is not None:
            candidatos.append(token.head.head)
        for c in candidatos:
            if c is None:
                continue
            if c.lemma_.lower() in NV_REFERENTE_VICTIMARIO or c.text.lower() in NV_REFERENTE_VICTIMARIO:
                return True

        # 2. Léxica de respaldo — señales en ventana + frases inequívocas
        start = max(0, token.idx - NV_REFERENTE_WINDOW)
        end = min(len(full_text), token.idx + len(token.text) + NV_REFERENTE_WINDOW)
        ventana = full_text[start:end].lower()

        for pattern in _NV_VICTIMARIO_COMPILED:
            if pattern.search(ventana):
                return True

        return False

    def _es_uso_neutro(self, token: "Token", full_text: str) -> bool:
        """
        Determina si el término es un falso amigo léxico (uso común/procesal)
        y no una expresión de NV. Ej.: "elementos probatorios", "tipo penal",
        "objetivo de la investigación".
        """
        start = max(0, token.idx - NV_REFERENTE_WINDOW)
        end = min(len(full_text), token.idx + len(token.text) + NV_REFERENTE_WINDOW)
        ventana = full_text[start:end].lower()

        for pattern in _NV_NEUTRALIDAD_COMPILED:
            if pattern.search(ventana):
                return True
        return False

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_sent_index(self, char_pos: int, sentences: list) -> int:
        """Encuentra el índice de la oración que contiene char_pos."""
        for i, sent in enumerate(sentences):
            if sent.start_char <= char_pos < sent.end_char:
                return i
        return -1

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
