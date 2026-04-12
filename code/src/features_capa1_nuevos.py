"""
Features nuevos Capa 1 — Hermenéutica Forense Computacional
============================================================
Implementa los tres features nuevos de la Capa 1 léxica:

1. Persona gramatical — ratio 1ª persona activa vs pasivas/impersonales
   Operacionaliza la accountability individual vs. institucional (Zhu et al., 2023)

2. Hedging — marcadores de evasión epistémica
   Indica evitación de compromisos afirmativos

3. Léxico emocional — carga emocional en 8 dimensiones NRC
   Captura la valencia emocional del discurso (NRC Emotion Lexicon)

Uso:
    python code/src/features_capa1_nuevos.py \
        --input data/processed/corpus_b/ \
        --output data/features/capa1_nuevos.csv

Requiere:
    pip install spacy pandas nrclex pysentimiento
    python -m spacy download es_core_news_lg
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Diccionarios ───────────────────────────────────────────────────────────────

# Marcadores de primera persona activa (accountability directa)
PRIMERA_PERSONA_ACTIVA = [
    "yo", "nosotros", "ordené", "ordenamos", "maté", "matamos",
    "ejecuté", "ejecutamos", "participé", "participamos",
    "reconozco", "reconocemos", "acepto", "aceptamos",
    "asumo", "asumimos", "fui", "fuimos", "di", "dimos",
    "tomé", "tomamos", "cometí", "cometimos"
]

# Construcciones pasivas e impersonales (supresión de agentividad)
CONSTRUCCIONES_SA = [
    "se produjo", "se presentó", "se dio de baja", "fue dado de baja",
    "fue presentado", "se encontró", "fue hallado", "se reportó",
    "fue reportado", "se procedió", "fue muerto en", "resultó muerto",
    "se identificó", "fue identificado como", "se neutralizó"
]

# Marcadores de hedging (evasión epistémica)
HEDGING_MARKERS = {
    "tal vez": 1.0, "quizás": 1.0, "quizá": 1.0,
    "posiblemente": 1.0, "probablemente": 1.0,
    "en cierta medida": 0.8, "en alguna medida": 0.8,
    "podría": 0.7, "podrían": 0.7, "pudiera": 0.7,
    "parecería": 0.8, "pareciera": 0.8,
    "aparentemente": 0.9, "presuntamente": 1.0,
    "supuestamente": 1.0, "al parecer": 0.9,
    "se estima": 0.6, "se considera": 0.5,
    "eventualmente": 0.6, "en principio": 0.5,
    "en teoría": 0.7, "según se indica": 0.8,
    "de alguna manera": 0.7, "de cierta forma": 0.7,
    "cabe la posibilidad": 0.9, "no se descarta": 0.8,
    "habría": 0.8, "habrían": 0.8, "hubiera": 0.7,
}

# Léxico emocional NRC simplificado en español
# Palabras clave por dimensión emocional relevantes para el corpus CFH
LEXICO_EMOCIONAL = {
    "culpa": [
        "responsabilidad", "responsable", "culpable", "culpa",
        "reconozco", "acepto", "perdón", "disculpa", "lamento",
        "arrepiento", "arrepentimiento", "vergüenza"
    ],
    "tristeza": [
        "dolor", "sufrimiento", "pérdida", "duelo", "luto",
        "llanto", "lágrimas", "angustia", "pena", "aflicción",
        "desconsuelo", "desolación", "muerte", "asesinato"
    ],
    "miedo": [
        "amenaza", "peligro", "terror", "miedo", "temor",
        "intimidación", "presión", "coerción", "riesgo",
        "vulnerabilidad", "indefensión"
    ],
    "ira": [
        "indignación", "rabia", "furia", "injusticia", "impunidad",
        "rechazo", "condena", "repudio", "denuncia", "reclamo"
    ],
    "confianza": [
        "verdad", "justicia", "reparación", "reconocimiento",
        "garantía", "compromiso", "esperanza", "fe", "dignidad",
        "derecho", "restauración"
    ],
    "anticipacion": [
        "futuro", "promesa", "compromiso", "garantía", "no repetición",
        "prevención", "transformación", "cambio", "proceso"
    ],
    "violencia_institucional": [
        "baja en combate", "resultado operacional", "dado de baja",
        "neutralizado", "abatido", "operación", "misión táctica",
        "guerrillero", "subversivo", "delincuente"
    ],
    "reconocimiento_victimas": [
        "víctima", "civil", "inocente", "persona protegida",
        "familiar", "madre", "hijo", "padre", "hermano",
        "comunidad", "pueblo", "campesino", "trabajador"
    ]
}


# ── Extractor de persona gramatical ───────────────────────────────────────────

def extraer_persona_gramatical(texto, nlp=None):
    """
    Calcula el ratio de accountability directa vs. supresión de agentividad.

    Returns:
        dict con:
        - primera_persona_ratio: proporción de tokens de 1ª persona activa
        - sa_ratio: proporción de construcciones pasivas/impersonales
        - accountability_score: primera_persona_ratio / (primera_persona_ratio + sa_ratio + 0.001)
    """
    texto_lower = texto.lower()
    tokens = texto_lower.split()
    n_tokens = max(len(tokens), 1)

    # Contar primera persona activa
    primera_persona_count = sum(
        texto_lower.count(term) for term in PRIMERA_PERSONA_ACTIVA
    )

    # Contar construcciones SA
    sa_count = sum(
        texto_lower.count(term) for term in CONSTRUCCIONES_SA
    )

    primera_persona_ratio = primera_persona_count / n_tokens
    sa_ratio = sa_count / n_tokens
    accountability_score = primera_persona_ratio / (
        primera_persona_ratio + sa_ratio + 0.001
    )

    # Si spaCy disponible, refinar con POS tagging
    if nlp:
        try:
            doc = nlp(texto[:50000])
            pron_primera = sum(
                1 for token in doc
                if token.pos_ == "PRON" and token.morph.get("Person") == ["1"]
            )
            verbos_activos = sum(
                1 for token in doc
                if token.pos_ == "VERB"
                and token.morph.get("Voice") != ["Pass"]
                and token.morph.get("Person") == ["1"]
            )
            primera_persona_ratio = (pron_primera + verbos_activos) / n_tokens
        except:
            pass

    return {
        "primera_persona_ratio": round(primera_persona_ratio, 4),
        "sa_ratio":              round(sa_ratio, 4),
        "accountability_score":  round(accountability_score, 4)
    }


# ── Extractor de hedging ──────────────────────────────────────────────────────

def extraer_hedging(texto):
    """
    Calcula la densidad y peso de marcadores de evasión epistémica.

    Returns:
        dict con:
        - hedging_count: número de marcadores encontrados
        - hedging_density: marcadores por 100 palabras
        - hedging_weight: suma ponderada de marcadores (algunos pesan más)
    """
    texto_lower = texto.lower()
    tokens = texto_lower.split()
    n_tokens = max(len(tokens), 1)

    hedging_count = 0
    hedging_weight = 0.0
    marcadores_encontrados = []

    for marcador, peso in HEDGING_MARKERS.items():
        ocurrencias = texto_lower.count(marcador)
        if ocurrencias > 0:
            hedging_count += ocurrencias
            hedging_weight += ocurrencias * peso
            marcadores_encontrados.append(f"{marcador}({ocurrencias})")

    hedging_density = (hedging_count / n_tokens) * 100

    return {
        "hedging_count":   hedging_count,
        "hedging_density": round(hedging_density, 4),
        "hedging_weight":  round(hedging_weight, 4),
        "marcadores":      ", ".join(marcadores_encontrados[:10])
    }


# ── Extractor de léxico emocional ─────────────────────────────────────────────

def extraer_lexico_emocional(texto):
    """
    Calcula la carga emocional en 8 dimensiones relevantes para el corpus CFH.

    Returns:
        dict con score por dimensión emocional (0-1)
    """
    texto_lower = texto.lower()
    tokens = texto_lower.split()
    n_tokens = max(len(tokens), 1)

    scores = {}
    for dimension, palabras in LEXICO_EMOCIONAL.items():
        count = sum(texto_lower.count(palabra) for palabra in palabras)
        scores[f"emo_{dimension}"] = round(count / n_tokens * 100, 4)

    # Score compuesto: reconocimiento_victimas vs. violencia_institucional
    # Cuanto mayor, más orientado al lenguaje de las víctimas
    scores["emo_balance_victimas"] = round(
        scores["emo_reconocimiento_victimas"] /
        (scores["emo_violencia_institucional"] + scores["emo_reconocimiento_victimas"] + 0.001),
        4
    )

    return scores


# ── Procesador de corpus ──────────────────────────────────────────────────────

def procesar_corpus(input_dir, output_path, corpus_type="B"):
    """Procesa todos los JSONs del corpus y calcula features Capa 1 nuevos."""

    input_dir = Path(input_dir)
    registros = []

    # Cargar spaCy si disponible
    nlp = None
    try:
        import spacy
        nlp = spacy.load("es_core_news_lg")
        print("✓ spaCy cargado")
    except:
        print("⚠ spaCy no disponible — usando solo diccionarios")

    archivos = list(input_dir.glob("*.json"))
    print(f"\nProcesando {len(archivos)} documentos de {input_dir}...")

    for path_json in archivos:
        with open(path_json, encoding='utf-8') as f:
            doc = json.load(f)

        doc_id = doc.get("doc_id", path_json.stem)
        subcaso = doc.get("subcaso", "")
        año = doc.get("año", 0)

        # ── Estructura Corpus B: doc["secciones"] es dict ──
        if "secciones" in doc and isinstance(doc["secciones"], dict):
            for nombre_sec, contenido_sec in doc["secciones"].items():
                texto = contenido_sec.get("texto", "") if isinstance(contenido_sec, dict) else str(contenido_sec)
                if len(texto) < 100:
                    continue
                registro = _calcular_features(texto, doc_id, nombre_sec, subcaso, año, corpus_type, nlp)
                registros.append(registro)

        # ── Estructura Corpus A: doc["segmentation"]["sections"] es lista ──
        elif "segmentation" in doc:
            seg = doc["segmentation"]
            sections = seg.get("sections", [])

            # Leer texto limpio desde el .txt correspondiente
            txt_path = path_json.with_suffix('.txt')
            if not txt_path.exists():
                continue
            with open(txt_path, encoding='utf-8') as ft:
                texto_completo = ft.read()

            meta = doc.get("metadata", {})
            doc_id = meta.get("radicado", path_json.stem)
            año_raw = str(meta.get("fecha", "0"))
            año = int(año_raw[:4]) if año_raw and año_raw[:4].isdigit() else 0
            subcaso = meta.get("tribunal", corpus_type)

            for sec in sections:
                nombre_sec = sec.get("section_id", "CUERPO")
                char_range = sec.get("char_range", [0, len(texto_completo)])
                texto = texto_completo[char_range[0]:char_range[1]]
                if len(texto) < 100:
                    continue
                registro = _calcular_features(texto, doc_id, nombre_sec, subcaso, año, corpus_type, nlp)
                registros.append(registro)

    df = pd.DataFrame(registros)
    if len(df) == 0:
        print("⚠ Sin registros — verifica la estructura de los JSONs")
        return df

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Features guardados: {output_path}")
    print(f"  Registros: {len(df)}")

    print("\nEstadísticas por sección (accountability_score):")
    if "seccion" in df.columns and "accountability_score" in df.columns:
        print(df.groupby("seccion")["accountability_score"].mean().round(3).to_string())

    return df


def _calcular_features(texto, doc_id, nombre_sec, subcaso, año, corpus_type, nlp):
    """Calcula los tres grupos de features para un segmento de texto."""
    persona  = extraer_persona_gramatical(texto, nlp)
    hedging  = extraer_hedging(texto)
    emocional = extraer_lexico_emocional(texto)
    return {
        "doc_id":      doc_id,
        "seccion":     nombre_sec,
        "subcaso":     subcaso,
        "año":         año,
        "corpus_type": corpus_type,
        "chars":       len(texto),
        **persona,
        **hedging,
        **emocional
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Features Capa 1 nuevos — CFH")
    parser.add_argument("--input",  default="data/processed/corpus_b",
                        help="Directorio con JSONs procesados")
    parser.add_argument("--output", default="data/features/capa1_nuevos_corpus_b.csv",
                        help="Ruta del CSV de salida")
    parser.add_argument("--corpus_type", default="B",
                        help="Tipo de corpus (A, B, C)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = procesar_corpus(args.input, args.output, args.corpus_type)

    print("\nTop 5 secciones con mayor accountability:")
    if "accountability_score" in df.columns:
        print(df.nlargest(5, "accountability_score")[
            ["doc_id", "seccion", "accountability_score", "primera_persona_ratio"]
        ].to_string())

    print("\nTop 5 secciones con mayor hedging:")
    if "hedging_density" in df.columns:
        print(df.nlargest(5, "hedging_density")[
            ["doc_id", "seccion", "hedging_density", "marcadores"]
        ].to_string())
