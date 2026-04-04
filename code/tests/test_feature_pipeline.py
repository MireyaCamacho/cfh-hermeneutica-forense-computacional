"""
CFH · Tests unitarios — Feature Pipeline
=========================================
Ejecutar: pytest code/tests/test_feature_pipeline.py -v

Estos tests verifican que el pipeline orquesta correctamente todos los
extractores y produce un DataFrame con el esquema esperado.
No requieren el corpus completo — usan JSONs mínimos en tmp_path.
"""

import sys
import json
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import spacy
    spacy.load("es_core_news_lg")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False

requires_spacy = pytest.mark.skipif(
    not SPACY_AVAILABLE,
    reason="es_core_news_lg no instalado"
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    if not SPACY_AVAILABLE:
        pytest.skip("spaCy no disponible")
    from features.feature_pipeline import FeaturePipeline
    return FeaturePipeline(use_ebi=False)


def _make_json(tmp_path, filename, doc_id, tribunal, date, corpus_type, sections):
    """Construye un JSON mínimo del pipeline de ingesta."""
    data = {
        "doc_id": doc_id,
        "metadata": {
            "tribunal": tribunal,
            "date_issued": date,
            "extraction_confidence": 0.9,
        },
        "segmentation": {
            "corpus_type": corpus_type,
            "sections": sections,
        }
    }
    p = tmp_path / filename
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def corpus_a_dir(tmp_path):
    """Directorio con 2 JSONs mínimos de Corpus A."""
    secs_ce = [
        {
            "section_id": "HECHOS",
            "is_target": True,
            "text": (
                "El joven fue dado de baja en la vereda Los Mangos conforme a la "
                "misión táctica número 34. El resultado operacional fue reportado "
                "positivo. El individuo portaba un fusil Galil de uso privativo."
            ),
            "word_count": 40,
        },
        {
            "section_id": "CONSIDERACIONES",
            "is_target": True,
            "text": (
                "La sala considera que se configuró la falla probada del servicio "
                "toda vez que el occiso fue presentado como guerrillero sin que "
                "existiera prueba de su pertenencia a grupo armado ilegal alguno. "
                "Se procedió a la reparación directa de los perjuicios causados."
            ),
            "word_count": 50,
        },
    ]
    _make_json(tmp_path, "ce_001.json",
               "05001-23-31-000-2006-00039-01(38757)",
               "Consejo de Estado · Sección Tercera",
               "2006-11-30", "A", secs_ce)

    secs_csj = [
        {
            "section_id": "HECHOS_JURIDICAMENTE_RELEVANTES",
            "is_target": True,
            "text": (
                "El procesado en su condición de comandante ordenó a los miembros "
                "del pelotón que procedieran a dar de baja al delincuente que fue "
                "presentado como guerrillero abatido en combate legítimo conforme "
                "al protocolo operacional vigente."
            ),
            "word_count": 45,
        },
        {
            "section_id": "CARGOS",
            "is_target": True,
            "text": (
                "Homicidio en persona protegida según el artículo 135 del Código "
                "Penal colombiano. La víctima era un civil inocente que no tenía "
                "ninguna vinculación con grupos armados al margen de la ley."
            ),
            "word_count": 35,
        },
    ]
    _make_json(tmp_path, "csj_001.json",
               "SP036-2018(42374)",
               "Corte Suprema · Sala Penal",
               "2018-01-12", "A", secs_csj)

    return tmp_path


@pytest.fixture
def corpus_b_dir(tmp_path):
    """Directorio con 1 JSON mínimo de Corpus B."""
    secs_jep = [
        {
            "section_id": "RECONOCIMIENTO",
            "is_target": True,
            "text": (
                "Reconozco que ordené la misión táctica que resultó en la muerte "
                "de civiles inocentes. Las personas asesinadas eran campesinos "
                "que no tenían vinculación con grupos armados. Pido perdón a sus "
                "familias y me comprometo a la no repetición de estas conductas. "
                "Los hechos constituyen crímenes de lesa humanidad."
            ),
            "word_count": 60,
        },
        {
            "section_id": "CALIFICACION_JURIDICA",
            "is_target": True,
            "text": (
                "Las muertes ilegítimamente presentadas como bajas en combate "
                "constituyen homicidios en persona protegida conforme al artículo "
                "135 del Código Penal y crímenes de guerra según el derecho "
                "internacional humanitario aplicable al conflicto armado colombiano."
            ),
            "word_count": 45,
        },
    ]
    _make_json(tmp_path, "jep_001.json",
               "Auto_125_2021_Norte_Santander",
               "JEP · Sala de Reconocimiento",
               "2021-07-02", "B", secs_jep)
    return tmp_path


# ── Tests del pipeline ────────────────────────────────────────────────────

@requires_spacy
def test_run_corpus_a_produce_dataframe(pipeline, corpus_a_dir):
    """run_corpus_a() produce un DataFrame no vacío."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

@requires_spacy
def test_run_corpus_a_columnas_correctas(pipeline, corpus_a_dir):
    """El DataFrame tiene todas las columnas esperadas."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    columnas_esperadas = {
        "doc_id", "section_id", "corpus_type", "year",
        "y2_sa", "y3_civil", "y4_nv", "y5_corpus_type",
        "y6_period", "y10_rep",
    }
    assert columnas_esperadas.issubset(set(df.columns))

@requires_spacy
def test_scores_en_rango(pipeline, corpus_a_dir):
    """Todos los scores están en [0, 1]."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    for col in ["y2_sa", "y3_civil", "y4_nv", "y10_rep", "y6_period"]:
        assert df[col].between(0.0, 1.0).all(), (
            f"Score fuera de rango en {col}: {df[col].tolist()}"
        )

@requires_spacy
def test_y5_valores_validos(pipeline, corpus_a_dir):
    """y₅ tiene valores en {0, 1, 2, 3}."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    assert df["y5_corpus_type"].isin({0, 1, 2, 3}).all()

@requires_spacy
def test_corpus_b_mayor_rep(pipeline, corpus_a_dir, corpus_b_dir):
    """El corpus B (JEP) tiene mayor REP promedio que el corpus A."""
    df_a = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    df_b = pipeline.run_corpus_b(corpus_dir=corpus_b_dir)

    rep_a = df_a["y10_rep"].mean()
    rep_b = df_b["y10_rep"].mean()

    assert rep_b >= rep_a, (
        f"Corpus B REP ({rep_b:.3f}) debe ser >= Corpus A REP ({rep_a:.3f})"
    )

@requires_spacy
def test_secciones_distintas(pipeline, corpus_a_dir):
    """El pipeline produce filas para secciones distintas."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    assert df["section_id"].nunique() > 1

@requires_spacy
def test_run_all_concatena_correctamente(pipeline, corpus_a_dir, corpus_b_dir):
    """run_all() concatena corpus A y B correctamente."""
    df = pipeline.run_all(
        corpus_a_dir=corpus_a_dir,
        corpus_b_dir=corpus_b_dir,
    )
    assert "A-CE" in df["corpus_type"].values or "A-CSJ" in df["corpus_type"].values
    assert "B" in df["corpus_type"].values

@requires_spacy
def test_output_csv(pipeline, corpus_a_dir, tmp_path):
    """El pipeline guarda el CSV correctamente."""
    output = tmp_path / "features" / "test_output.csv"
    df = pipeline.run_corpus_a(
        corpus_dir=corpus_a_dir,
        output_path=output,
    )
    assert output.exists()
    df_loaded = pd.read_csv(output)
    assert len(df_loaded) == len(df)

@requires_spacy
def test_max_docs(pipeline, corpus_a_dir):
    """max_docs limita el número de documentos procesados."""
    df_1 = pipeline.run_corpus_a(corpus_dir=corpus_a_dir, max_docs=1)
    df_all = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    assert len(df_1) <= len(df_all)

@requires_spacy
def test_metadatos_en_output(pipeline, corpus_a_dir):
    """Los metadatos de calidad están en el DataFrame."""
    df = pipeline.run_corpus_a(corpus_dir=corpus_a_dir)
    assert "text_length_chars" in df.columns
    assert "processing_time_s" in df.columns
    assert "has_warning" in df.columns
    assert (df["text_length_chars"] > 0).all()
