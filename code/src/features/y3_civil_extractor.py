"""
CFH · Extractor de Indicador y₃ — Distancia Léxico Civil
=========================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₁ (Violencia Discursiva)

Qué mide y₃:
    La distancia semántica entre el lenguaje de un segmento judicial y
    un léxico de referencia civil — el vocabulario con que se describe
    la vida cotidiana, la identidad, las relaciones y el trabajo de las
    personas civiles ajenas al conflicto armado.

    Un score alto indica que el texto usa vocabulario muy distante del
    léxico civil: habla de operaciones, bajas y resultados en lugar de
    personas, familias y trabajo. Un score bajo indica que el texto usa
    vocabulario próximo al de las víctimas: sus nombres, sus oficios,
    sus vínculos familiares.

    Este indicador operacionaliza la dimensión de "distancia del mundo
    civil" de la violencia discursiva: los documentos que nunca hablan
    de personas sino de resultados operacionales están discursivamente
    más alejados de la humanidad de las víctimas.

Implementación sin GPU (versión de producción con ConfliBERT pendiente):
    Esta versión usa representación TF-IDF + similitud coseno contra
    un léxico de referencia civil construido a partir de:
    1. Testimonios conceptuales de MAFAPO (vocabulario de víctimas)
    2. Léxico de la vida cotidiana y el trabajo campesino colombiano
    3. Términos de identidad personal (nombre, familia, oficio)

    Cuando ConfliBERT-Spanish esté disponible en Colab Pro, reemplazar
    el vectorizador TF-IDF por embeddings de ConfliBERT. La arquitectura
    de normalización y el resultado permanecen idénticos.

    La distancia se calcula como: y₃ = 1 - similitud_coseno(texto, léxico_civil)
    De modo que y₃ ∈ [0, 1]:
    - y₃ ≈ 0: el texto usa vocabulario muy similar al léxico civil (REP)
    - y₃ ≈ 1: el texto usa vocabulario muy distante del léxico civil (EBI/NV)

Léxico civil de referencia (basado en testimonios MAFAPO conceptuales):
    El léxico se construye sobre tres dimensiones del "mundo civil":
    - Identidad: nombre, apellido, joven, hijo, padre, madre, familia
    - Trabajo: carpintero, campesino, agricultor, obrero, empleado
    - Cotidianidad: casa, vereda, finca, trabajo, comida, escuela

Dependencias:
    - scikit-learn (TfidfVectorizer, cosine_similarity)
    - numpy
    - spacy (para tokenización y lematización en español)

Referencia teórica:
    Fraser, N. (1995). Reconocimiento e injusticia.
    Galtung, J. (1990). Violencia cultural. JPR, 27(3).
    MAFAPO (2019). Relatos de las Madres de Falsos Positivos.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

logger = logging.getLogger("cfh.features.y3_civil")


# ---------------------------------------------------------------------------
# Léxico civil de referencia — basado en testimonios MAFAPO
# ---------------------------------------------------------------------------

# El léxico se organiza en cinco dimensiones del "mundo civil"
# que contrastan con el "mundo militar-institucional" del EBI/NV

LEXICO_CIVIL_DIMENSIONES = {

    # Dimensión 1: Identidad personal
    # Términos que humanizanVíctimas — sus nombres, relaciones, etapas de vida
    "identidad": [
        "joven", "muchacho", "hijo", "hija", "padre", "madre", "hermano",
        "hermana", "niño", "niña", "esposo", "esposa", "abuelo", "abuela",
        "familiar", "pariente", "compañero", "amigo", "vecino", "persona",
        "ciudadano", "colombiano", "hombre", "mujer",
        # Nombres propios frecuentes en testimonios MAFAPO (genéricos)
        "fair", "leonardo", "jaime", "carlos", "luis", "jorge", "diego",
        "andrés", "camilo", "mario", "pedro", "juan", "manuel",
    ],

    # Dimensión 2: Trabajo y sustento
    # El trabajo como marcador de vida civil — opuesto a "resultado operacional"
    "trabajo": [
        "carpintero", "campesino", "agricultor", "obrero", "trabajador",
        "empleado", "comerciante", "vendedor", "conductor", "constructor",
        "mecánico", "pintor", "albañil", "jornalero", "cultivador",
        "recolector", "cultivar", "sembrar", "cosechar", "trabajar",
        "oficio", "labor", "empleo", "jornal", "salario", "sustento",
        "trabajo", "ocupación",
    ],

    # Dimensión 3: Vida cotidiana y espacio doméstico
    # La cotidianidad como opuesto al vocabulario de "operaciones"
    "cotidianidad": [
        "casa", "hogar", "barrio", "vereda", "finca", "pueblo", "municipio",
        "escuela", "colegio", "universidad", "calle", "parque", "plaza",
        "mercado", "tienda", "iglesia", "cancha",
        "desayuno", "almuerzo", "comida", "dormir", "salir", "llegar",
        "regresar", "visitar", "celebrar", "reunir",
    ],

    # Dimensión 4: Vulnerabilidad y desprotección
    # Términos que describen la condición de indefensión de las víctimas
    "vulnerabilidad": [
        "desarmado", "indefenso", "inocente", "civil", "ajeno",
        "sin armas", "desprotegido", "vulnerable", "pobre", "humilde",
        "buscando trabajo", "engañado", "reclutado con engaño",
        "desaparecido", "buscado", "extrañado", "llorado",
    ],

    # Dimensión 5: Memoria y duelo
    # Vocabulario del proceso de duelo y búsqueda de verdad de las familias
    "memoria": [
        "memoria", "recuerdo", "ausencia", "dolor", "sufrimiento",
        "lágrimas", "luto", "duelo", "pérdida", "extrañar",
        "verdad", "justicia", "reparación", "reconocimiento",
        "madres", "MAFAPO", "soacha", "víctimas",
        "buscar", "identificar", "cuerpo", "restos", "tumba",
    ],
}

# Lista plana del léxico civil para vectorización
LEXICO_CIVIL_PLANO = [
    token
    for tokens in LEXICO_CIVIL_DIMENSIONES.values()
    for token in tokens
]

# Documento de referencia civil: texto construido con el léxico civil
# Este es el "polo civil" contra el que se mide la distancia de cada segmento
TEXTO_REFERENCIA_CIVIL = " ".join(LEXICO_CIVIL_PLANO * 3)  # repetido para peso

# Vocabulario anti-civil (vocabulario del mundo militar-institucional)
# Se usa para verificar que el vectorizador los captura correctamente
VOCABULARIO_ANTI_CIVIL = [
    "baja", "bajas", "operacional", "táctica", "misión",
    "resultado", "combate", "contacto", "neutralizado",
    "guerrillero", "individuo", "sujeto", "elemento",
]


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class CivilDistanceResult:
    """Resultado de la extracción del indicador y₃."""
    score: float            # y₃ normalizado [0, 1] — distancia al léxico civil
    score_raw: float        # similitud coseno bruta [0, 1]

    doc_id: str
    section_id: str
    corpus_type: str
    text_length_chars: int
    text_length_tokens: int

    # Descomposición por dimensión (útil para análisis exploratorio)
    similarity_by_dimension: dict[str, float] = field(default_factory=dict)

    # Top tokens civiles detectados en el texto
    top_civil_tokens: list[str] = field(default_factory=list)

    processing_time_s: float = 0.0
    vectorizer_type: str = "tfidf"  # "tfidf" | "conliBERT" (futuro)
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.text_length_tokens > 5

    def to_dict(self) -> dict:
        return {
            "y3_civil_distance": self.score,
            "y3_civil_similarity_raw": self.score_raw,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "corpus_type": self.corpus_type,
            "text_length_tokens": self.text_length_tokens,
            "similarity_by_dimension": self.similarity_by_dimension,
            "top_civil_tokens": self.top_civil_tokens[:5],
            "vectorizer_type": self.vectorizer_type,
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Normalizador
# ---------------------------------------------------------------------------

class CivilDistanceNormalizer:
    """
    Normaliza y₃.

    Nota: y₃ es 1 - similitud_coseno, por lo que naturalmente está en [0,1].
    La normalización ajusta la distribución para que la diferencia entre
    corpus A y corpus B sea más pronunciada en la escala del SEM.
    """

    def __init__(self, method: str = "passthrough"):
        self.method = method
        self._fitted = False
        self._p_low: float = 0.0
        self._p_high: float = 1.0

    def fit(self, raw_scores: list[float]) -> "CivilDistanceNormalizer":
        arr = np.array(raw_scores)
        self._p_low = float(np.percentile(arr, 5))
        self._p_high = float(np.percentile(arr, 95))
        self._fitted = True
        return self

    def normalize(self, raw_score: float) -> float:
        if self.method == "passthrough":
            return float(np.clip(raw_score, 0.0, 1.0))
        denom = self._p_high - self._p_low
        n = (raw_score - self._p_low) / denom if denom > 1e-10 else 0.5
        return float(np.clip(n, 0.0, 1.0))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "method": self.method, "p_low": self._p_low,
            "p_high": self._p_high, "fitted": self._fitted,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CivilDistanceNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(method=data["method"])
        obj._p_low = data["p_low"]
        obj._p_high = data["p_high"]
        obj._fitted = data["fitted"]
        return obj


# ---------------------------------------------------------------------------
# Extractor principal
# ---------------------------------------------------------------------------

class CivilLexiconExtractor:
    """
    Extractor del indicador y₃ (Distancia Léxico Civil).

    Versión TF-IDF (sin GPU) para desarrollo y validación.
    Reemplazar el vectorizador por ConfliBERT cuando esté disponible.

    El extractor construye un espacio vectorial TF-IDF a partir del
    texto de entrada + el texto de referencia civil, y calcula la
    similitud coseno entre ambos. La distancia y₃ = 1 - similitud.

    Parámetros
    ----------
    spacy_model : str
        Modelo spaCy para lematización. Si None, se usa tokenización simple.
    normalizer : CivilDistanceNormalizer
        Normalizador del score.
    min_df : int
        Frecuencia mínima de documento para TF-IDF.
    """

    def __init__(
        self,
        spacy_model: str = "es_core_news_lg",
        normalizer: Optional[CivilDistanceNormalizer] = None,
        min_df: int = 1,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn no instalado. Ejecuta: pip install scikit-learn"
            )

        self.normalizer = normalizer or CivilDistanceNormalizer()
        self._use_spacy = False

        # Intentar cargar spaCy para lematización
        if _SPACY_AVAILABLE and spacy_model:
            try:
                self._nlp = spacy.load(spacy_model)
                self._use_spacy = True
                logger.info(f"Usando spaCy ({spacy_model}) para lematización.")
            except OSError:
                logger.warning(
                    f"Modelo spaCy '{spacy_model}' no disponible. "
                    "Usando tokenización simple."
                )

        # Vectorizador TF-IDF configurado para el dominio judicial español
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=self._tokenize,
            min_df=min_df,
            sublinear_tf=True,   # log(1 + tf) — reduce el efecto de términos muy frecuentes
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),  # unigramas + bigramas para capturar "baja en combate"
        )

        # Precomputar el vector de referencia civil
        self._civil_reference = TEXTO_REFERENCIA_CIVIL
        self._vectorizer_fitted = False

        # Vectores por dimensión para análisis exploratorio
        self._dim_refs = {
            dim: " ".join(tokens * 3)
            for dim, tokens in LEXICO_CIVIL_DIMENSIONES.items()
        }

    def extract(
        self,
        text: str,
        doc_id: str = "unknown",
        section_id: str = "unknown",
        corpus_type: str = "A",
    ) -> CivilDistanceResult:
        """
        Calcula la distancia léxico-civil de un segmento textual.

        Parámetros
        ----------
        text : str
            Texto del segmento judicial a analizar.
        doc_id : str
            Identificador del documento.
        section_id : str
            Identificador de la sección.
        corpus_type : str
            "A" | "B" | "C"

        Retorna
        -------
        CivilDistanceResult con y₃ y metadatos de análisis.
        """
        import time
        t0 = time.perf_counter()

        if not text or len(text.strip()) < 20:
            return self._empty_result(doc_id, section_id, corpus_type)

        tokens = self._tokenize(text)
        if len(tokens) < 5:
            return self._empty_result(doc_id, section_id, corpus_type,
                                      warning="texto_muy_corto")

        # Ajustar el vectorizador si es la primera llamada
        if not self._vectorizer_fitted:
            self._fit_vectorizer(text)
        else:
            # Re-fit incremental: agregar el texto actual al corpus del vectorizador
            try:
                self._fit_vectorizer(text)
            except Exception:
                pass

        # Calcular similitud coseno texto vs. referencia civil
        try:
            corpus_to_fit = [self._civil_reference, text]
            tfidf_matrix = self._vectorizer.fit_transform(corpus_to_fit)
            similarity = float(cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )[0][0])
        except Exception as e:
            logger.warning(f"Error en vectorización [{doc_id}]: {e}")
            return self._empty_result(doc_id, section_id, corpus_type,
                                      warning=f"vectorization_error: {e}")

        # y₃ = distancia = 1 - similitud
        score_raw = similarity
        y3 = self.normalizer.normalize(1.0 - similarity)

        # Calcular similitud por dimensión usando Jaccard sobre tokens
        # (más robusto que TF-IDF por dimensión con sklearn 1.8+)
        text_tokens_set = set(tokens)
        sim_by_dim = {}
        for dim, dim_tokens in LEXICO_CIVIL_DIMENSIONES.items():
            dim_set = set(t.lower() for t in dim_tokens)
            intersection = len(text_tokens_set & dim_set)
            union = len(text_tokens_set | dim_set)
            sim_by_dim[dim] = round(intersection / union, 4) if union > 0 else 0.0

        # Tokens civiles detectados en el texto
        top_civil = self._find_civil_tokens(text)

        elapsed = time.perf_counter() - t0

        logger.debug(
            f"y₃ Civil [{doc_id}/{section_id}]: "
            f"distancia={y3:.3f} similitud={similarity:.3f} "
            f"civil_tokens={len(top_civil)} t={elapsed:.2f}s"
        )

        return CivilDistanceResult(
            score=y3,
            score_raw=score_raw,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=len(text),
            text_length_tokens=len(tokens),
            similarity_by_dimension=sim_by_dim,
            top_civil_tokens=top_civil,
            processing_time_s=elapsed,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokeniza y lematiza el texto.
        Usa spaCy si está disponible, tokenización simple si no.
        """
        if self._use_spacy:
            doc = self._nlp(text.lower())
            return [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct
                and not token.is_space and len(token.lemma_) > 2
            ]
        else:
            # Tokenización simple: split + limpieza básica
            text_clean = re.sub(r"[^\w\sáéíóúñü]", " ", text.lower())
            return [t for t in text_clean.split() if len(t) > 2]

    def _fit_vectorizer(self, sample_text: str) -> None:
        """Ajusta el vectorizador sobre el texto de referencia + muestra."""
        self._vectorizer.fit([self._civil_reference, sample_text])
        self._vectorizer_fitted = True

    def _similarity_by_dimension(self, text: str) -> dict[str, float]:
        """Calcula la similitud coseno por dimensión del léxico civil."""
        results = {}
        for dim, ref_text in self._dim_refs.items():
            try:
                mat = self._vectorizer.transform([ref_text, text])
                sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
                results[dim] = round(sim, 4)
            except Exception:
                results[dim] = 0.0
        return results

    def _find_civil_tokens(self, text: str) -> list[str]:
        """Encuentra tokens del léxico civil presentes en el texto."""
        text_lower = text.lower()
        civil_set = set(LEXICO_CIVIL_PLANO)
        found = []
        tokens = self._tokenize(text)
        for t in tokens:
            if t in civil_set and t not in found:
                found.append(t)
        return found[:10]

    def _empty_result(
        self,
        doc_id: str,
        section_id: str,
        corpus_type: str,
        warning: str = "texto_vacio_o_muy_corto",
    ) -> CivilDistanceResult:
        # y₃ = 1.0 cuando no hay texto (máxima distancia al léxico civil)
        return CivilDistanceResult(
            score=1.0,
            score_raw=0.0,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=0,
            text_length_tokens=0,
            warning=warning,
        )
