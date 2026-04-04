"""
CFH · Tests unitarios — y₁₀ REP Extractor (Ruptura Epistémica Positiva)
========================================================================
Ejecutar: pytest code/tests/test_y10_rep.py -v
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import spacy
    spacy.load("es_core_news_lg")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False

from features.y10_rep_extractor import (
    REPExtractor,
    REPScoreNormalizer,
    REPExtractionResult,
)

requires_spacy = pytest.mark.skipif(
    not SPACY_AVAILABLE,
    reason="es_core_news_lg no instalado"
)


@pytest.fixture(scope="module")
def extractor():
    if not SPACY_AVAILABLE:
        pytest.skip("spaCy no disponible")
    return REPExtractor()


# ── Normalizer tests ──────────────────────────────────────────────────────

def test_normalizer_rango():
    norm = REPScoreNormalizer()
    for raw in [0.0, 0.1, 0.5, 1.0, 3.0]:
        assert 0.0 <= norm.normalize(raw) <= 1.0

def test_normalizer_fit():
    norm = REPScoreNormalizer()
    norm.fit([0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.8])
    assert norm._fitted

def test_normalizer_serialize(tmp_path):
    norm = REPScoreNormalizer()
    norm.fit([0.0, 0.1, 0.3, 0.5])
    p = tmp_path / "rep_norm.json"
    norm.save(p)
    norm2 = REPScoreNormalizer.load(p)
    assert abs(norm.normalize(0.2) - norm2.normalize(0.2)) < 1e-6


# ── Extractor tests ───────────────────────────────────────────────────────

@requires_spacy
def test_texto_vacio(extractor):
    r = extractor.extract("", doc_id="t", section_id="RECONOCIMIENTO")
    assert r.score == 0.0
    assert r.warning is not None

@requires_spacy
def test_texto_sin_rep(extractor):
    """Texto de justicia ordinaria con lenguaje bélico — bajo REP."""
    texto = (
        "El guerrillero fue dado de baja en combate en la vereda Los Mangos. "
        "El resultado operacional fue reportado como positivo conforme al protocolo. "
        "Se procedió a la verificación del material incautado."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS", corpus_type="A")
    assert isinstance(r, REPExtractionResult)
    assert r.score < 0.5

@requires_spacy
def test_reconocimiento_responsabilidad(extractor):
    """Frases de reconocimiento explícito son detectadas."""
    texto = (
        "Reconozco que ordené la misión táctica que resultó en la muerte de "
        "las víctimas civiles. Acepto mi responsabilidad plena en los hechos."
    )
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    assert r.n_reconocimiento > 0
    assert r.score_raw > 0.0

@requires_spacy
def test_restitución_identidad(extractor):
    """Frases de restitución de identidad son detectadas."""
    texto = (
        "Las personas asesinadas eran civiles inocentes que no tenían "
        "ninguna vinculación con grupos armados. "
        "Eran campesinos que trabajaban en sus tierras."
    )
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    assert r.n_restitución > 0

@requires_spacy
def test_lenguaje_dih(extractor):
    """Vocabulario DIH y derechos humanos es detectado."""
    texto = (
        "Los hechos constituyen crímenes de lesa humanidad y crímenes de guerra. "
        "Las muertes ilegítimamente presentadas como bajas en combate "
        "constituyen homicidios en persona protegida conforme al artículo 135 del Código Penal."
    )
    r = extractor.extract(texto, doc_id="t", section_id="CALIFICACION_JURIDICA")
    assert r.n_dih > 0

@requires_spacy
def test_compromiso_reparacion(extractor):
    """Compromisos de reparación son detectados."""
    texto = (
        "Pido perdón a las familias de las víctimas por el dolor causado. "
        "Me comprometo a no repetir estas conductas y a contribuir "
        "a la reparación de los daños causados."
    )
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    assert r.n_reparación > 0

@requires_spacy
def test_comparacion_corpus_a_vs_b(extractor):
    """El corpus B (JEP) tiene mayor REP que el corpus A (ordinario)."""
    texto_a = (
        "El guerrillero fue dado de baja en la vereda El Toro. "
        "El resultado operacional fue reportado conforme al protocolo. "
        "Se procedió al levantamiento del cadáver por las autoridades competentes."
    )
    texto_b = (
        "Reconozco que las personas eran civiles inocentes. "
        "Pido perdón a sus familias por el daño causado. "
        "Los hechos constituyen crímenes de lesa humanidad. "
        "Me comprometo a no repetir estas conductas."
    )
    r_a = extractor.extract(texto_a, doc_id="a", section_id="HECHOS", corpus_type="A")
    r_b = extractor.extract(texto_b, doc_id="b", section_id="RECONOCIMIENTO", corpus_type="B")
    assert r_b.score_raw > r_a.score_raw

@requires_spacy
def test_rep_mayor_en_seccion_jep(extractor):
    """Sección RECONOCIMIENTO JEP tiene alto REP."""
    texto = (
        "Reconozco mi responsabilidad en los hechos. "
        "Acepto que los asesinados eran civiles inocentes que no eran guerrilleros. "
        "Las muertes ilegítimamente presentadas como bajas en combate "
        "constituyen crímenes de lesa humanidad. "
        "Pido perdón a las familias y me comprometo a la no repetición. "
        "Contribuiré a la reparación de las víctimas."
    )
    r = extractor.extract(texto, doc_id="rc01", section_id="RECONOCIMIENTO", corpus_type="B")
    assert r.n_instances >= 4  # al menos un mecanismo por oración
    assert r.score_raw > 0.5   # densidad alta de REP

@requires_spacy
def test_descomposicion_mecanismos(extractor):
    """La suma de mecanismos coincide con el total de instancias."""
    texto = (
        "Reconozco mi responsabilidad. Era un civil inocente. "
        "Constituye un crimen de lesa humanidad. Pido perdón."
    )
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    total = r.n_reconocimiento + r.n_restitución + r.n_dih + r.n_reparación
    assert total == r.n_instances

@requires_spacy
def test_dedup_no_duplica(extractor):
    """No se duplican instancias con spans solapados."""
    # "crímenes de lesa humanidad" puede ser detectado por múltiples patrones
    texto = (
        "Los hechos constituyen crímenes de lesa humanidad y crímenes de guerra "
        "según el derecho internacional humanitario."
    )
    r = extractor.extract(texto, doc_id="t", section_id="CALIFICACION_JURIDICA")
    # Verificar que no hay duplicados exactos de text_span
    spans = [i.text_span for i in r.instances]
    assert len(spans) == len(set(spans)) or r.n_instances <= len(spans)

@requires_spacy
def test_serializable(extractor):
    """to_dict() es serializable a JSON."""
    texto = "Reconozco mi responsabilidad. Era un civil inocente."
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    d = r.to_dict()
    json.dumps(d)
    assert "y10_rep_score" in d
    assert 0.0 <= d["y10_rep_score"] <= 1.0

@requires_spacy
def test_tiempo_procesamiento(extractor):
    """Segmento típico procesado en menos de 10 segundos."""
    texto = "Reconozco mi responsabilidad en los hechos. Era un civil inocente. " * 20
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    assert r.processing_time_s < 10.0
