"""
CFH · Tests unitarios — y₂ SA Extractor (Supresión de Agentividad)
===================================================================
Ejecutar: pytest code/tests/test_y2_sa.py -v

Los tests están diseñados para correr sin GPU y sin ConfliBERT.
Solo requieren spaCy + es_core_news_lg instalados.

Si es_core_news_lg no está instalado, los tests que lo requieren
se omiten automáticamente con pytest.mark.skipif.
"""

import sys
import pytest
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Verificar disponibilidad de spaCy
try:
    import spacy
    spacy.load("es_core_news_lg")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False

from features.y2_sa_extractor import (
    SAExtractor,
    SAScoreNormalizer,
    SAInstance,
    SAExtractionResult,
    SA_MECHANISM_WEIGHTS,
)

requires_spacy = pytest.mark.skipif(
    not SPACY_AVAILABLE,
    reason="es_core_news_lg no instalado — ejecutar: python -m spacy download es_core_news_lg"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    """Instancia del extractor — carga spaCy una sola vez para todos los tests."""
    if not SPACY_AVAILABLE:
        pytest.skip("spaCy no disponible")
    return SAExtractor()


# ─────────────────────────────────────────────────────────────────────────────
# Tests del normalizador — no requieren spaCy
# ─────────────────────────────────────────────────────────────────────────────

def test_normalizer_defaults():
    """El normalizador con defaults produce valores en [0, 1]."""
    norm = SAScoreNormalizer()
    for raw in [0.0, 0.1, 0.5, 1.0, 5.0]:
        result = norm.normalize(raw)
        assert 0.0 <= result <= 1.0, f"Score fuera de rango para raw={raw}: {result}"


def test_normalizer_fit():
    """fit() actualiza los parámetros y produce normalización correcta."""
    norm = SAScoreNormalizer()
    sample = [0.0, 0.05, 0.1, 0.2, 0.15, 0.08, 0.3, 0.01, 0.12, 0.07]
    norm.fit(sample)
    assert norm._fitted
    assert 0.0 <= norm.normalize(0.1) <= 1.0


def test_normalizer_zscore():
    """Estrategia zscore produce valores en [0, 1]."""
    norm = SAScoreNormalizer(method="zscore")
    norm.fit([0.0] * 5 + [0.1] * 5 + [0.5] * 5)
    for raw in [0.0, 0.1, 0.5]:
        assert 0.0 <= norm.normalize(raw) <= 1.0


def test_normalizer_serialize(tmp_path):
    """save() / load() produce el mismo normalizador."""
    norm = SAScoreNormalizer()
    norm.fit([0.0, 0.1, 0.2, 0.3, 0.5])
    path = tmp_path / "norm_sa.json"
    norm.save(path)
    norm2 = SAScoreNormalizer.load(path)
    assert abs(norm.normalize(0.15) - norm2.normalize(0.15)) < 1e-6


def test_normalizer_minmax():
    """Estrategia minmax produce valores en [0, 1]."""
    norm = SAScoreNormalizer(method="minmax")
    norm.fit([0.0, 0.1, 0.2, 0.3, 0.5])
    assert norm.normalize(0.0) == pytest.approx(0.0, abs=0.01)
    assert norm.normalize(0.5) == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Tests del extractor — requieren spaCy
# ─────────────────────────────────────────────────────────────────────────────

@requires_spacy
def test_texto_vacio(extractor):
    """Texto vacío produce score 0 y warning."""
    result = extractor.extract("", doc_id="test", section_id="HECHOS")
    assert result.score == 0.0
    assert result.n_instances == 0
    assert result.warning is not None


@requires_spacy
def test_texto_sin_sa(extractor):
    """Texto sin SA produce score bajo."""
    texto = (
        "El comandante Pérez ordenó que sus tropas abrieran fuego. "
        "El capitán Gómez disparó su arma. "
        "El teniente Rodríguez presentó el informe firmado."
    )
    result = extractor.extract(texto, doc_id="test", section_id="CARGOS")
    assert isinstance(result, SAExtractionResult)
    assert result.is_valid
    # Con agentes explícitos, el score SA debe ser bajo
    assert result.score < 0.5


@requires_spacy
def test_pasiva_sin_agente_detectada(extractor):
    """Pasivas sin complemento agente son detectadas como SA."""
    # Texto con pasivas prototípicas sin agente
    texto = (
        "El ciudadano fue dado de baja en la vereda El Toro. "
        "El resultado operacional fue reportado conforme al protocolo. "
        "Los cuerpos fueron encontrados con armamento que no portaban."
    )
    result = extractor.extract(texto, doc_id="test", section_id="HECHOS")
    # Debe detectar instancias SA
    assert result.n_instances > 0
    # El score SA debe ser mayor que en texto sin SA
    assert result.score_raw > 0.0


@requires_spacy
def test_patron_lexical_detectado(extractor):
    """Los patrones léxicos SA son detectados correctamente."""
    texto = (
        "Se presentó como resultado operacional la baja del individuo. "
        "Se procedió conforme al protocolo de la Directiva 029."
    )
    result = extractor.extract(texto, doc_id="test", section_id="HECHOS")
    assert result.n_patron_lexical > 0


@requires_spacy
def test_comparacion_corpus_a_vs_b(extractor):
    """Fragmento corpus A tiene mayor SA que fragmento corpus B (REP)."""
    # Fragmento A — alta SA
    texto_a = (
        "Fue dado de baja en combate en la vereda Los Mangos. "
        "El resultado operacional fue reportado positivo. "
        "Se procedió a la verificación del material incautado."
    )
    # Fragmento B (JEP) — baja SA: el agente es explícito
    texto_b = (
        "Reconozco que yo ordené la misión táctica. "
        "El general Coronado dispuso que el batallón procediera. "
        "Los comparecientes aceptaron su responsabilidad individual."
    )
    result_a = extractor.extract(texto_a, doc_id="a001", section_id="HECHOS", corpus_type="A")
    result_b = extractor.extract(texto_b, doc_id="b001", section_id="RECONOCIMIENTO", corpus_type="B")

    # El corpus A debe tener mayor SA que el corpus B en estos ejemplos
    assert result_a.score_raw >= result_b.score_raw, (
        f"Corpus A (raw={result_a.score_raw:.3f}) debe tener mayor SA "
        f"que corpus B (raw={result_b.score_raw:.3f})"
    )


@requires_spacy
def test_resultado_serializable(extractor):
    """to_dict() produce un diccionario serializable."""
    import json
    texto = "El individuo fue dado de baja y reportado como positivo operacional."
    result = extractor.extract(texto, doc_id="test", section_id="HECHOS")
    d = result.to_dict()
    json_str = json.dumps(d)  # no debe lanzar excepción
    assert "y2_sa_score" in d
    assert 0.0 <= d["y2_sa_score"] <= 1.0


@requires_spacy
def test_descomposicion_por_mecanismo(extractor):
    """La suma de mecanismos coincide con el total de instancias."""
    texto = (
        "Fue dado de baja en combate. "
        "Se procedió a la verificación. "
        "La presentación del resultado fue realizada conforme al protocolo. "
        "El batallón reportó la novedad."
    )
    result = extractor.extract(texto, doc_id="test", section_id="HECHOS")
    total_por_mecanismo = (
        result.n_pasiva_sin_agente +
        result.n_se_impersonal +
        result.n_nominalizacion +
        result.n_sujeto_institucional +
        result.n_patron_lexical
    )
    assert total_por_mecanismo == result.n_instances


@requires_spacy
def test_tiempo_procesamiento_razonable(extractor):
    """Un segmento típico de 500 palabras se procesa en menos de 10 segundos."""
    texto = (
        "El individuo fue dado de baja en combate por miembros del Ejército. "
        "El resultado operacional fue reportado conforme al protocolo. "
    ) * 25  # ~500 palabras
    result = extractor.extract(texto, doc_id="test", section_id="HECHOS")
    assert result.processing_time_s < 10.0, (
        f"Procesamiento demasiado lento: {result.processing_time_s:.1f}s"
    )
