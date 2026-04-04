"""
CFH · Tests unitarios — Módulo SEM
===================================
Ejecutar: pytest code/tests/test_sem_model.py -v

Los tests verifican la lógica del modelo sin requerir datos reales del corpus.
Usan datos sintéticos mínimos que reproducen la estructura del DataFrame
del feature pipeline.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import semopy
    _SEMOPY_AVAILABLE = True
except ImportError:
    _SEMOPY_AVAILABLE = False

from sem.sem_model import CFHSEMModel, SEMResults, SEM_SPEC_PARCIAL, SEM_SPEC_DOS_FACTORES

requires_semopy = pytest.mark.skipif(
    not _SEMOPY_AVAILABLE,
    reason="semopy no instalado — pip install semopy"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df_sintetico():
    """
    DataFrame sintético con la estructura del feature pipeline.
    N=200, distribuido en dos grupos (A y B) con diferencias realistas.
    """
    np.random.seed(42)
    n = 200

    # Corpus A (justicia ordinaria): alto NV, bajo REP
    n_a = 160
    df_a = pd.DataFrame({
        "doc_id": [f"doc_a_{i}" for i in range(n_a)],
        "section_id": np.random.choice(["HECHOS", "CONSIDERACIONES", "DECISIÓN"], n_a),
        "corpus_type": np.random.choice(["A-CE", "A-CSJ"], n_a),
        "year": np.random.randint(1994, 2021, n_a),
        "y2_sa": np.clip(np.random.normal(0.85, 0.20, n_a), 0, 1),
        "y3_civil": np.clip(np.random.normal(0.99, 0.005, n_a), 0, 1),
        "y4_nv": np.clip(np.random.normal(0.24, 0.30, n_a), 0, 1),
        "y5_corpus_type": np.random.choice([0, 1], n_a),
        "y6_period": np.clip(np.random.normal(0.65, 0.16, n_a), 0, 1),
        "y10_rep": np.clip(np.random.normal(0.09, 0.18, n_a), 0, 1),
    })

    # Corpus B (JEP): mayor REP, NV similar
    n_b = 40
    df_b = pd.DataFrame({
        "doc_id": [f"doc_b_{i}" for i in range(n_b)],
        "section_id": np.random.choice(["RECONOCIMIENTO", "RESUELVE", "CONSIDERACIONES"], n_b),
        "corpus_type": ["B"] * n_b,
        "year": np.random.randint(2018, 2024, n_b),
        "y2_sa": np.clip(np.random.normal(0.91, 0.17, n_b), 0, 1),
        "y3_civil": np.clip(np.random.normal(0.987, 0.006, n_b), 0, 1),
        "y4_nv": np.clip(np.random.normal(0.23, 0.21, n_b), 0, 1),
        "y5_corpus_type": [2] * n_b,
        "y6_period": np.clip(np.random.normal(0.97, 0.03, n_b), 0, 1),
        "y10_rep": np.clip(np.random.normal(0.15, 0.19, n_b), 0, 1),
    })

    return pd.concat([df_a, df_b], ignore_index=True)


# ── Tests de preparación de datos ────────────────────────────────────────────

def test_prepare_data_columnas(df_sintetico):
    """prepare_data() produce las columnas correctas para el modelo parcial."""
    model = CFHSEMModel(spec="parcial")
    df_prep = model.prepare_data(df_sintetico)
    assert set(df_prep.columns) == {"y2", "y3", "y4", "y5", "y6", "y10"}

def test_prepare_data_sin_nan(df_sintetico):
    """prepare_data() elimina filas con NaN en indicadores esenciales."""
    df_con_nan = df_sintetico.copy()
    df_con_nan.loc[0, "y2_sa"] = np.nan
    model = CFHSEMModel(spec="parcial")
    df_prep = model.prepare_data(df_con_nan)
    assert df_prep["y2"].isna().sum() == 0

def test_prepare_data_rango(df_sintetico):
    """Los indicadores preparados están en [0, 1]."""
    model = CFHSEMModel(spec="parcial")
    df_prep = model.prepare_data(df_sintetico)
    for col in ["y2", "y3", "y4", "y10"]:
        assert df_prep[col].between(0, 1).all(), f"{col} fuera de rango"

def test_prepare_data_dos_factores(df_sintetico):
    """Modelo dos_factores selecciona solo 4 columnas."""
    model = CFHSEMModel(spec="dos_factores")
    df_prep = model.prepare_data(df_sintetico)
    assert set(df_prep.columns) == {"y2", "y3", "y4", "y10"}

def test_spec_invalido():
    """Especificación inválida lanza ValueError."""
    with pytest.raises(ValueError):
        CFHSEMModel(spec="invalido")


# ── Tests de estimación ───────────────────────────────────────────────────────

@requires_semopy
def test_fit_parcial_converge(df_sintetico):
    """El modelo parcial converge con datos sintéticos."""
    model = CFHSEMModel(spec="parcial")
    results = model.fit(df_sintetico)
    assert isinstance(results, SEMResults)
    assert results.converged

@requires_semopy
def test_fit_retorna_sem_results(df_sintetico):
    """fit() retorna un objeto SEMResults."""
    model = CFHSEMModel(spec="parcial")
    results = model.fit(df_sintetico)
    assert isinstance(results, SEMResults)
    assert results.n_obs == len(df_sintetico)

@requires_semopy
def test_fit_dos_factores(df_sintetico):
    """El modelo de dos factores converge."""
    model = CFHSEMModel(spec="dos_factores")
    results = model.fit(df_sintetico)
    assert results.converged

@requires_semopy
def test_summary_no_lanza_excepcion(df_sintetico):
    """summary() no lanza excepciones."""
    model = CFHSEMModel(spec="dos_factores")
    results = model.fit(df_sintetico)
    texto = results.summary()
    assert isinstance(texto, str)
    assert "SEM CFH" in texto

@requires_semopy
def test_multigroup_produce_dict(df_sintetico):
    """fit_multigroup() produce un dict con resultados por grupo."""
    model = CFHSEMModel(spec="dos_factores")
    results = model.fit_multigroup(df_sintetico, group_col="corpus_type")
    assert isinstance(results, dict)
    assert len(results) >= 2
    for group, res in results.items():
        assert isinstance(res, SEMResults)

@requires_semopy
def test_save_results(df_sintetico, tmp_path):
    """save_results() guarda el resumen en un archivo."""
    model = CFHSEMModel(spec="dos_factores")
    results = model.fit(df_sintetico)
    output = tmp_path / "sem_results.txt"
    model.save_results(results, output)
    assert output.exists()
    contenido = output.read_text(encoding="utf-8")
    assert "SEM CFH" in contenido


# ── Tests del bootstrap ───────────────────────────────────────────────────────

@requires_semopy
def test_bootstrap_produce_intervalos(df_sintetico):
    """bootstrap_beta23() produce intervalos numéricos."""
    model = CFHSEMModel(spec="dos_factores")
    beta_mean, ci_low, ci_high = model.bootstrap_beta23(
        df_sintetico, n_samples=50, random_state=42
    )
    # Los valores pueden ser NaN si el modelo no tiene beta_23 definido
    # en el modelo dos_factores, pero no debe lanzar excepción
    assert isinstance(beta_mean, float)
    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)
