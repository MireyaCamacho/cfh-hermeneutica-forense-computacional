"""
CFH · Tests unitarios — y₃ Civil Lexicon Extractor (Distancia Léxico Civil)
============================================================================
Ejecutar: pytest code/tests/test_y3_civil.py -v
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.y3_civil_extractor import (
    CivilLexiconExtractor,
    CivilDistanceNormalizer,
    CivilDistanceResult,
    LEXICO_CIVIL_PLANO,
    VOCABULARIO_ANTI_CIVIL,
)


@pytest.fixture(scope="module")
def extractor():
    return CivilLexiconExtractor()


# ── Normalizer tests ──────────────────────────────────────────────────────

def test_normalizer_passthrough():
    norm = CivilDistanceNormalizer(method="passthrough")
    for v in [0.0, 0.3, 0.7, 1.0]:
        assert 0.0 <= norm.normalize(v) <= 1.0

def test_normalizer_fit():
    norm = CivilDistanceNormalizer(method="percentile")
    norm.fit([0.3, 0.5, 0.7, 0.8, 0.9])
    assert norm._fitted
    assert 0.0 <= norm.normalize(0.5) <= 1.0

def test_normalizer_serialize(tmp_path):
    norm = CivilDistanceNormalizer()
    norm.fit([0.2, 0.4, 0.6, 0.8])
    p = tmp_path / "civil_norm.json"
    norm.save(p)
    norm2 = CivilDistanceNormalizer.load(p)
    assert abs(norm.normalize(0.5) - norm2.normalize(0.5)) < 1e-6


# ── Extractor tests ───────────────────────────────────────────────────────

def test_texto_vacio(extractor):
    r = extractor.extract("", doc_id="t", section_id="HECHOS")
    assert r.score == 1.0  # máxima distancia cuando no hay texto
    assert r.warning is not None

def test_texto_anti_civil_tiene_mayor_distancia(extractor):
    """Texto con vocabulario militar tiene mayor distancia que texto civil."""
    texto_militar = (
        "El guerrillero fue dado de baja en combate conforme a la misión táctica. "
        "El resultado operacional fue reportado positivo. "
        "El individuo neutralizado portaba armamento de uso privativo."
    )
    texto_civil = (
        "El joven campesino trabajaba en su finca cultivando maíz. "
        "Era hijo de familia humilde que buscaba trabajo para sostener a su madre. "
        "Era una persona honesta y trabajadora de la vereda."
    )
    r_mil = extractor.extract(texto_militar, doc_id="m", section_id="HECHOS")
    r_civ = extractor.extract(texto_civil, doc_id="c", section_id="RECONOCIMIENTO")

    # El texto civil debe tener MENOR distancia (mayor similitud) al léxico civil
    assert r_civ.score < r_mil.score, (
        f"Texto civil (y₃={r_civ.score:.3f}) debe tener menor distancia "
        f"que texto militar (y₃={r_mil.score:.3f})"
    )

def test_lexicol_civil_presente_en_texto_civil(extractor):
    """El extractor detecta tokens civiles en texto civil."""
    texto = (
        "El joven era un carpintero que trabajaba en su barrio. "
        "Su madre lo buscó durante semanas sin encontrarlo."
    )
    r = extractor.extract(texto, doc_id="t", section_id="RECONOCIMIENTO")
    assert len(r.top_civil_tokens) > 0

def test_comparacion_corpus_a_vs_b(extractor):
    """Corpus B (JEP) tiene menor distancia civil que corpus A (ordinario)."""
    texto_a = (
        "En cumplimiento de la misión táctica, miembros del batallón "
        "dieron de baja al individuo que fue reportado como resultado operacional. "
        "El sujeto portaba un fusil Galil de uso privativo."
    )
    texto_b = (
        "Fair Leonardo Porras Bernal era un joven carpintero de 16 años "
        "que buscaba trabajo para ayudar a su madre. "
        "Era un civil inocente ajeno al conflicto armado."
    )
    r_a = extractor.extract(texto_a, doc_id="a", section_id="HECHOS", corpus_type="A")
    r_b = extractor.extract(texto_b, doc_id="b", section_id="RECONOCIMIENTO", corpus_type="B")

    assert r_b.score < r_a.score, (
        f"Corpus B (y₃={r_b.score:.3f}) debe tener menor distancia civil "
        f"que corpus A (y₃={r_a.score:.3f})"
    )

def test_score_en_rango(extractor):
    """El score siempre está en [0, 1]."""
    textos = [
        "El guerrillero fue dado de baja en combate.",
        "El joven campesino buscaba trabajo en su vereda.",
        "La sala consideró que los hechos constituyen responsabilidad del estado.",
        "Reconozco que la persona era un civil inocente y pido perdón.",
    ]
    for texto in textos:
        r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
        assert 0.0 <= r.score <= 1.0, f"Score fuera de rango: {r.score}"

def test_dimensiones_presentes(extractor):
    """La descomposición por dimensión incluye las 5 categorías."""
    # Texto con suficientes tokens civiles después de filtrar stop words con spaCy
    texto = (
        "El joven campesino trabajaba cultivando maíz en su finca de la vereda. "
        "Era un trabajador honesto que buscaba sustento para su familia humilde."
    )
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    dimensiones_esperadas = {"identidad", "trabajo", "cotidianidad", "vulnerabilidad", "memoria"}
    assert set(r.similarity_by_dimension.keys()) == dimensiones_esperadas

def test_serializable(extractor):
    """to_dict() es serializable a JSON."""
    texto = "El campesino fue dado de baja. Su familia lo buscó por semanas."
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    d = r.to_dict()
    json.dumps(d)
    assert "y3_civil_distance" in d
    assert 0.0 <= d["y3_civil_distance"] <= 1.0

def test_lexico_civil_no_vacio():
    """El léxico civil tiene cobertura suficiente."""
    assert len(LEXICO_CIVIL_PLANO) >= 50
    assert len(VOCABULARIO_ANTI_CIVIL) >= 5

def test_tiempo_procesamiento(extractor):
    """Procesamiento en tiempo razonable."""
    import time
    texto = "El joven campesino fue dado de baja en la vereda El Toro. " * 20
    t0 = time.perf_counter()
    r = extractor.extract(texto, doc_id="t", section_id="HECHOS")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
