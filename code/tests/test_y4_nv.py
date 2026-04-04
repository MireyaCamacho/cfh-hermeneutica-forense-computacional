"""
CFH · Tests unitarios — y₄ NV Extractor (Negación de Victimización)
====================================================================
Ejecutar: pytest code/tests/test_y4_nv.py -v
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

from features.y4_nv_extractor import (
    NVExtractor,
    NVScoreNormalizer,
    NVExtractionResult,
)

requires_spacy = pytest.mark.skipif(
    not SPACY_AVAILABLE,
    reason="es_core_news_lg no instalado"
)


@pytest.fixture(scope="module")
def extractor():
    if not SPACY_AVAILABLE:
        pytest.skip("spaCy no disponible")
    return NVExtractor()


# ── Normalizer tests ──────────────────────────────────────────────────────

def test_normalizer_rango():
    norm = NVScoreNormalizer()
    for raw in [0.0, 0.1, 0.5, 1.0, 3.0]:
        assert 0.0 <= norm.normalize(raw) <= 1.0

def test_normalizer_fit():
    norm = NVScoreNormalizer()
    norm.fit([0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    assert norm._fitted
    assert 0.0 <= norm.normalize(0.3) <= 1.0

def test_normalizer_serialize(tmp_path):
    norm = NVScoreNormalizer()
    norm.fit([0.0, 0.2, 0.4, 0.6])
    p = tmp_path / "nv_norm.json"
    norm.save(p)
    norm2 = NVScoreNormalizer.load(p)
    assert abs(norm.normalize(0.3) - norm2.normalize(0.3)) < 1e-6


# ── Extractor tests ───────────────────────────────────────────────────────

@requires_spacy
def test_texto_vacio(extractor):
    r = extractor.extract("", doc_id="t", section_id="HECHOS")
    assert r.score == 0.0
    assert r.warning is not None

@requires_spacy
def test_texto_sin_nv(extractor):
    """Texto con agente y víctima claramente identificados — bajo NV."""
    texto = (
        "El señor Juan García era un campesino que trabajaba en su finca. "
        "Tenía 35 años y vivía con su familia en la vereda La Esperanza. "
        "Era una persona honesta y trabajadora."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    assert r.score < 0.5

@requires_spacy
def test_recategorizacion_combatiente(extractor):
    """Vocabulario de recategorización es detectado."""
    texto = (
        "El guerrillero fue dado de baja en combate en la vereda Los Mangos. "
        "El delincuente portaba un fusil de uso privativo de las fuerzas militares."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    assert r.n_recategorizacion > 0
    assert r.score_raw > 0.0

@requires_spacy
def test_atribucion_armamento(extractor):
    """Atribución de armamento post-mortem es detectada."""
    texto = (
        "El occiso portaba un fusil Galil AK número 12345. "
        "Vestía prendas de uso privativo de las fuerzas militares."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    assert r.n_atribucion_armamento > 0

@requires_spacy
def test_deshumanizacion(extractor):
    """Términos deshumanizantes son detectados."""
    texto = (
        "El individuo fue encontrado sin vida en el sector. "
        "El sujeto presentaba heridas de bala. "
        "El elemento fue identificado posteriormente."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    assert r.n_deshumanizacion > 0

@requires_spacy
def test_nv_cuestionado_tiene_peso_reducido(extractor):
    """NV en contexto de cuestionamiento tiene peso reducido."""
    # NV sin cuestionamiento
    texto_nv_puro = (
        "El guerrillero portaba armamento de uso privativo. "
        "El individuo fue dado de baja en combate."
    )
    # NV con cuestionamiento explícito
    texto_nv_cuestionado = (
        "Fue presentado fraudulentamente como guerrillero, "
        "cuando en realidad era un civil inocente que no portaba armas. "
        "El individuo fue presentado falsamente como delincuente."
    )
    r_puro = extractor.extract(texto_nv_puro, doc_id="p", section_id="HECHOS")
    r_cuestionado = extractor.extract(texto_nv_cuestionado, doc_id="c", section_id="HECHOS")

    # El NV cuestionado debe tener menor score que el NV puro
    assert r_cuestionado.n_questioned > 0
    assert r_cuestionado.score_raw <= r_puro.score_raw

@requires_spacy
def test_comparacion_corpus_a_vs_b(extractor):
    """Fragmento corpus A tiene mayor NV que fragmento REP del corpus B."""
    texto_a = (
        "El guerrillero fue dado de baja en combate. "
        "El individuo portaba armamento de uso privativo. "
        "El sujeto era integrante de grupo al margen de la ley."
    )
    texto_b = (
        "Reconozco que la persona era un civil inocente. "
        "No era guerrillero ni tenía vinculación con grupos armados. "
        "Era un campesino que fue engañado con falsas promesas de trabajo."
    )
    r_a = extractor.extract(texto_a, doc_id="a", section_id="HECHOS", corpus_type="A")
    r_b = extractor.extract(texto_b, doc_id="b", section_id="RECONOCIMIENTO", corpus_type="B")
    assert r_a.score_raw > r_b.score_raw

@requires_spacy
def test_descomposicion_mecanismos(extractor):
    """La suma de mecanismos coincide con el total."""
    texto = (
        "El guerrillero portaba un fusil. "
        "El individuo fue dado de baja. "
        "Era de baja trayectoria social."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    total = (r.n_recategorizacion + r.n_atribucion_armamento +
             r.n_deshumanizacion + r.n_descalificacion)
    assert total == r.n_instances

@requires_spacy
def test_serializable(extractor):
    """to_dict() es serializable a JSON."""
    texto = "El guerrillero fue dado de baja y el individuo portaba armamento."
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    d = r.to_dict()
    json.dumps(d)  # no debe lanzar excepción
    assert "y4_nv_score" in d
    assert 0.0 <= d["y4_nv_score"] <= 1.0

@requires_spacy
def test_tiempo_procesamiento(extractor):
    """Segmento típico procesado en menos de 10 segundos."""
    texto = "El guerrillero fue dado de baja en la vereda. " * 20
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    assert r.processing_time_s < 10.0
