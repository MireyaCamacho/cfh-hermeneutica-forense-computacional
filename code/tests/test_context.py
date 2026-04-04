"""
CFH · Tests unitarios — Context Extractor (y₅ e y₆)
====================================================
Ejecutar: pytest code/tests/test_context.py -v

Estos tests usan JSONs reales del corpus procesado cuando están disponibles,
y JSONs sintéticos mínimos cuando no lo están — para probar la lógica de
extracción sin depender del corpus completo.
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.context_extractor import (
    ContextExtractor,
    ContextExtractionResult,
    CORPUS_TYPE_ENCODING,
    AÑO_MIN, AÑO_MAX,
)

import numpy as np


@pytest.fixture
def extractor():
    return ContextExtractor()


@pytest.fixture
def json_ce(tmp_path):
    """JSON mínimo de sentencia Consejo de Estado."""
    data = {
        "doc_id": "05001-23-31-000-2006-00039-01(38757)",
        "metadata": {
            "tribunal": "Consejo de Estado · Sección Tercera",
            "case_number": "05001-23-31-000-2006-00039-01(38757)",
            "date_issued": "2006-11-30",
            "extraction_confidence": 0.90,
        },
        "segmentation": {"corpus_type": "A", "total_sections": 4}
    }
    p = tmp_path / "ce_test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def json_csj(tmp_path):
    """JSON mínimo de sentencia Corte Suprema de Justicia."""
    data = {
        "doc_id": "SP036-2018(42374)",
        "metadata": {
            "tribunal": "Corte Suprema · Sala Penal",
            "case_number": "SP036-2018(42374)",
            "date_issued": "2018-01-12",
            "extraction_confidence": 0.90,
        },
        "segmentation": {"corpus_type": "A", "total_sections": 5}
    }
    p = tmp_path / "csj_test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def json_jep(tmp_path):
    """JSON mínimo de auto JEP."""
    data = {
        "doc_id": "Auto_125_2021_Norte_Santander",
        "metadata": {
            "tribunal": "JEP · Sala de Reconocimiento",
            "case_number": "Auto 125 de 2021",
            "date_issued": "2021-07-02",
            "extraction_confidence": 0.90,
        },
        "segmentation": {"corpus_type": "B", "total_sections": 9}
    }
    p = tmp_path / "jep_test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── Tests de y₅ (tipo de corpus) ─────────────────────────────────────────

def test_y5_consejo_estado(extractor, json_ce):
    r = extractor.extract_from_json(json_ce)
    assert r.y5_corpus_type == 0
    assert r.corpus_type_raw == "A-CE"

def test_y5_corte_suprema(extractor, json_csj):
    r = extractor.extract_from_json(json_csj)
    assert r.y5_corpus_type == 1
    assert r.corpus_type_raw == "A-CSJ"

def test_y5_jep_escrita(extractor, json_jep):
    r = extractor.extract_from_json(json_jep)
    assert r.y5_corpus_type == 2
    assert r.corpus_type_raw == "B"

def test_y5_inferido_del_docid(extractor, tmp_path):
    """y₅ se infiere del doc_id cuando el tribunal no está disponible."""
    data = {
        "doc_id": "SP036-2018(42374)",
        "metadata": {"extraction_confidence": 0.5},
        "segmentation": {"corpus_type": "A"}
    }
    p = tmp_path / "sin_tribunal.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.y5_corpus_type == 1  # SP → CSJ

def test_y5_ce_inferido_del_docid(extractor, tmp_path):
    """Radicado CE se infiere del doc_id."""
    data = {
        "doc_id": "05001-23-31-000-2006-00039-01(38757)",
        "metadata": {"extraction_confidence": 0.5},
        "segmentation": {"corpus_type": "A"}
    }
    p = tmp_path / "ce_sin_tribunal.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.y5_corpus_type == 0  # radicado CE

def test_y5_encoding_consistente():
    """Los valores de y₅ son consistentes con el esquema del SEM."""
    assert CORPUS_TYPE_ENCODING["A-CE"]  == 0
    assert CORPUS_TYPE_ENCODING["A-CSJ"] == 1
    assert CORPUS_TYPE_ENCODING["B"]     == 2
    assert CORPUS_TYPE_ENCODING["C"]     == 3


# ── Tests de y₆ (período normativo) ──────────────────────────────────────

def test_y6_rango_valido(extractor, json_ce):
    r = extractor.extract_from_json(json_ce)
    assert 0.0 <= r.y6_period <= 1.0

def test_y6_año_minimo(extractor, tmp_path):
    """Año mínimo del corpus produce y₆ = 0."""
    data = {
        "doc_id": "test_1994",
        "metadata": {"date_issued": "1994-01-01", "extraction_confidence": 0.9},
        "segmentation": {"corpus_type": "A"}
    }
    p = tmp_path / "año_min.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.y6_period == pytest.approx(0.0, abs=0.01)

def test_y6_año_maximo(extractor, tmp_path):
    """Año máximo del corpus produce y₆ = 1."""
    data = {
        "doc_id": "test_2023",
        "metadata": {"date_issued": "2023-12-31", "extraction_confidence": 0.9},
        "segmentation": {"corpus_type": "B"}
    }
    p = tmp_path / "año_max.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.y6_period == pytest.approx(1.0, abs=0.01)

def test_y6_año_jep_mayor_que_ce(extractor, json_ce, json_jep):
    """Un auto JEP (2021) tiene y₆ mayor que una sentencia CE (2006)."""
    r_ce  = extractor.extract_from_json(json_ce)
    r_jep = extractor.extract_from_json(json_jep)
    assert r_jep.y6_period > r_ce.y6_period

def test_y6_inferido_del_docid(extractor, tmp_path):
    """Año inferido del doc_id cuando date_issued no está disponible."""
    data = {
        "doc_id": "AP4064-2016(46318)",
        "metadata": {"extraction_confidence": 0.5},
        "segmentation": {"corpus_type": "A"}
    }
    p = tmp_path / "sin_fecha.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.year == 2016
    assert r.year_inferred is True

def test_y6_sin_fecha_usa_media(extractor, tmp_path):
    """Sin fecha ni inferencia, y₆ = 0.5 (media del corpus)."""
    data = {
        "doc_id": "documento_sin_fecha",
        "metadata": {"extraction_confidence": 0.3},
        "segmentation": {"corpus_type": "A"}
    }
    p = tmp_path / "sin_fecha_ni_id.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = extractor.extract_from_json(p)
    assert r.y6_period == pytest.approx(0.5, abs=0.1)


# ── Tests del batch extractor ─────────────────────────────────────────────

def test_batch_extractor(extractor, tmp_path):
    """El batch extractor procesa múltiples JSONs correctamente."""
    # Crear los JSONs directamente en tmp_path
    fixtures = [
        ("ce_batch.json", {
            "doc_id": "05001-23-31-000-2006-00039-01(38757)",
            "metadata": {"tribunal": "Consejo de Estado · Sección Tercera",
                         "date_issued": "2006-11-30", "extraction_confidence": 0.9},
            "segmentation": {"corpus_type": "A"}
        }),
        ("csj_batch.json", {
            "doc_id": "SP036-2018(42374)",
            "metadata": {"tribunal": "Corte Suprema · Sala Penal",
                         "date_issued": "2018-01-12", "extraction_confidence": 0.9},
            "segmentation": {"corpus_type": "A"}
        }),
        ("jep_batch.json", {
            "doc_id": "Auto_125_2021_Norte_Santander",
            "metadata": {"tribunal": "JEP · Sala de Reconocimiento",
                         "date_issued": "2021-07-02", "extraction_confidence": 0.9},
            "segmentation": {"corpus_type": "B"}
        }),
    ]
    for nombre, data in fixtures:
        (tmp_path / nombre).write_text(json.dumps(data), encoding="utf-8")

    results = extractor.extract_batch(tmp_path)
    assert len(results) == 3
    y5_values = {r.y5_corpus_type for r in results}
    assert 0 in y5_values  # CE
    assert 1 in y5_values  # CSJ
    assert 2 in y5_values  # JEP

def test_feature_matrix(extractor, json_ce, json_csj, json_jep):
    """to_feature_matrix() produce matriz [n, 2] correcta."""
    results = [
        extractor.extract_from_json(json_ce),
        extractor.extract_from_json(json_csj),
        extractor.extract_from_json(json_jep),
    ]
    matrix = extractor.to_feature_matrix(results)
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float64
    # Columna 0: y₅ en {0, 1, 2}
    assert set(matrix[:, 0].astype(int)) <= {0, 1, 2}
    # Columna 1: y₆ en [0, 1]
    assert all(0.0 <= v <= 1.0 for v in matrix[:, 1])


# ── Tests de serialización ────────────────────────────────────────────────

def test_to_dict_serializable(extractor, json_ce):
    """to_dict() produce un diccionario serializable a JSON."""
    import json as json_lib
    r = extractor.extract_from_json(json_ce)
    d = r.to_dict()
    json_lib.dumps(d)  # no debe lanzar excepción
    assert "y5_corpus_type" in d
    assert "y6_period" in d
    assert "period_label" in d

def test_period_label_coherente(extractor, json_jep):
    """El period_label refleja correctamente el año."""
    r = extractor.extract_from_json(json_jep)
    assert "2021" in r.period_label or "JEP" in r.period_label

def test_extract_from_dict(extractor):
    """extract_from_dict() produce el mismo resultado que extract_from_json()."""
    data = {
        "doc_id": "SP036-2018(42374)",
        "metadata": {
            "tribunal": "Corte Suprema · Sala Penal",
            "date_issued": "2018-01-12",
            "extraction_confidence": 0.90,
        },
        "segmentation": {"corpus_type": "A"}
    }
    r = extractor.extract_from_dict(data)
    assert r.y5_corpus_type == 1
    assert r.year == 2018
    assert 0.0 <= r.y6_period <= 1.0
