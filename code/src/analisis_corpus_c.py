"""
Análisis del Corpus C — Audiencias JEP con indicadores CFH
===========================================================
Aplica los 6 indicadores CFH a las transcripciones de Whisper
y compara con Corpus A y B (escrito vs oral).

Requiere en Colab:
- ConfliBERT-Spanish cargado (model_cs, tokenizer_cs)
- centroides cargados (centroide_mafapo_cs, centroide_cidh_cs)
- df con indicadores A y B
- Archivos corpus_c/*.txt

Uso:
    Correr celda por celda en Colab
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# ── Configuración ─────────────────────────────────────────────────────────────

CORPUS_C_DIR = "corpus_c"
BLOCK_SIZE   = 2000  # chars por bloque de análisis

AUDIOS = {
    "catatumbo":         "catatumbo_audiencia_reconocimiento.txt",
    "costa_caribe":      "costa_caribe.txt",
    "casanare_torres":   "casanare_torres.txt",
    "dabeiba_antioquia": "dabeiba_antioquia.txt",
    "huila":             "huila.txt",
}

SUBCASO_META = {
    "catatumbo":         {"subcaso": "Norte de Santander", "fecha": "2022-04-26", "tipo": "audiencia_reconocimiento"},
    "costa_caribe":      {"subcaso": "Costa Caribe",       "fecha": "2022-07-18", "tipo": "audiencia_reconocimiento"},
    "casanare_torres":   {"subcaso": "Casanare",           "fecha": "2020-02-06", "tipo": "version_voluntaria"},
    "dabeiba_antioquia": {"subcaso": "Antioquia",          "fecha": "2023-06-27", "tipo": "audiencia_reconocimiento"},
    "huila":             {"subcaso": "Huila",              "fecha": "2024-08-10", "tipo": "audiencia_reconocimiento"},
}

# ── Paso 1: Segmentar transcripciones en bloques ───────────────────────────────

def segmentar_transcripcion(texto, nombre, block_size=BLOCK_SIZE):
    """Divide el texto en bloques de ~block_size chars con solapamiento mínimo."""
    bloques = []
    palabras = texto.split()
    bloque_actual = []
    char_count = 0
    bloque_id = 0

    for palabra in palabras:
        bloque_actual.append(palabra)
        char_count += len(palabra) + 1
        if char_count >= block_size:
            texto_bloque = " ".join(bloque_actual)
            bloques.append({
                "audio":     nombre,
                "bloque_id": f"{nombre}_b{bloque_id:04d}",
                "texto":     texto_bloque,
                "chars":     len(texto_bloque),
                **SUBCASO_META.get(nombre, {}),
            })
            bloque_actual = []
            char_count = 0
            bloque_id += 1

    # Último bloque si queda contenido
    if bloque_actual and len(bloque_actual) > 20:
        texto_bloque = " ".join(bloque_actual)
        bloques.append({
            "audio":     nombre,
            "bloque_id": f"{nombre}_b{bloque_id:04d}",
            "texto":     texto_bloque,
            "chars":     len(texto_bloque),
            **SUBCASO_META.get(nombre, {}),
        })

    return bloques

print("== PASO 1: Segmentando transcripciones ==")
bloques_totales = []
for nombre, archivo in AUDIOS.items():
    path = os.path.join(CORPUS_C_DIR, archivo)
    if not os.path.exists(path):
        print(f"  ⚠ No encontrado: {path}")
        continue
    with open(path, encoding="utf-8") as f:
        texto = f.read()
    bloques = segmentar_transcripcion(texto, nombre)
    bloques_totales.extend(bloques)
    print(f"  ✓ {nombre}: {len(texto):,} chars → {len(bloques)} bloques")

df_c = pd.DataFrame(bloques_totales)
print(f"\nTotal bloques Corpus C: {len(df_c)}")
print(df_c["audio"].value_counts())


# ── Paso 2: Embeddings ConfliBERT-Spanish ────────────────────────────────────

def get_embedding_cs(text, tokenizer, model, device, max_length=512):
    if not text or len(text.strip()) < 10:
        return np.zeros(768)
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length,
        truncation=True, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def distancia_coseno(emb, centroide):
    if np.all(emb == 0):
        return np.nan
    sim = cosine_similarity(emb.reshape(1,-1), centroide.reshape(1,-1))[0][0]
    return float(1 - sim)

print("\n== PASO 2: Extrayendo embeddings ConfliBERT-Spanish ==")
print("Cargando modelo ConfliBERT-Spanish...")

from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
tokenizer_cs = AutoTokenizer.from_pretrained(model_name)
model_cs = AutoModel.from_pretrained(model_name).to(device)
model_cs.eval()
print(f"✓ ConfliBERT-Spanish cargado en {device}")

# Cargar centroides guardados
try:
    centroide_mafapo_cs = np.load("centroide_mafapo_cs.npy")
    centroide_cidh_cs   = np.load("centroide_cidh_cs.npy")
    print("✓ Centroides cargados desde archivo")
except:
    print("⚠ Centroides no encontrados — recalcular manualmente")

# Extraer embeddings
print(f"Extrayendo embeddings para {len(df_c)} bloques...")
embeddings_c = []
y8_scores = []
y9_scores = []

for _, row in tqdm(df_c.iterrows(), total=len(df_c)):
    emb = get_embedding_cs(row["texto"][:8000], tokenizer_cs, model_cs, device)
    embeddings_c.append(emb)
    y8_scores.append(distancia_coseno(emb, centroide_mafapo_cs))
    y9_scores.append(distancia_coseno(emb, centroide_cidh_cs))

df_c["y8_mafapo_cs"] = y8_scores
df_c["y9_cidh_cs"]   = y9_scores
print("✓ Embeddings y₈ y y₉ calculados")


# ── Paso 3: Indicadores léxico-sintácticos ────────────────────────────────────

print("\n== PASO 3: Indicadores léxico-sintácticos ==")
print("Importando módulos CFH...")

import sys
sys.path.insert(0, "cfh-hermeneutica-forense-computacional/code/src")

try:
    from features.y2_sa.sa_extractor import SAExtractor
    from features.y3_civil.civil_extractor import CivilExtractor
    from features.y4_nv.nv_extractor import NVExtractor
    from features.y10_rep.rep_extractor import REPExtractor

    sa_ext   = SAExtractor()
    civ_ext  = CivilExtractor()
    nv_ext   = NVExtractor()
    rep_ext  = REPExtractor()

    y2_scores, y3_scores, y4_scores, y10_scores = [], [], [], []

    for _, row in tqdm(df_c.iterrows(), total=len(df_c)):
        texto = row["texto"][:8000]
        try:
            y2_scores.append(sa_ext.extract(texto))
            y3_scores.append(civ_ext.extract(texto))
            y4_scores.append(nv_ext.extract(texto))
            y10_scores.append(rep_ext.extract(texto))
        except:
            y2_scores.append(np.nan)
            y3_scores.append(np.nan)
            y4_scores.append(np.nan)
            y10_scores.append(np.nan)

    df_c["y2_sa"]     = y2_scores
    df_c["y3_civil"]  = y3_scores
    df_c["y4_nv"]     = y4_scores
    df_c["y10_rep"]   = y10_scores
    print("✓ Indicadores léxico-sintácticos calculados")

except Exception as e:
    print(f"⚠ Error en extractores: {e}")
    print("  Continuar con y8 y y9 únicamente")


# ── Paso 4: Comparación A vs B vs C ──────────────────────────────────────────

print("\n== PASO 4: Comparación A vs B vs C ==")

# Cargar df A y B
try:
    df_ab = pd.read_csv("indicators_final_completo.csv")
    df_ab_a = df_ab[df_ab["corpus_type"] != "B"]
    df_ab_b = df_ab[df_ab["corpus_type"] == "B"]

    print(f"Corpus A: {len(df_ab_a)} secciones")
    print(f"Corpus B: {len(df_ab_b)} secciones")
    print(f"Corpus C: {len(df_c)} bloques")
    print()

    for col, nombre in [
        ("y8_mafapo_cs", "Dist. MAFAPO (y₈)"),
        ("y9_cidh_cs",   "Dist. CIDH (y₉)"),
    ]:
        col_ab = col
        ma = df_ab_a[col_ab].mean() if col_ab in df_ab_a.columns else np.nan
        mb = df_ab_b[col_ab].mean() if col_ab in df_ab_b.columns else np.nan
        mc = df_c[col].mean()

        print(f"{nombre}:")
        print(f"  Corpus A (escrito ordinario): {ma:.3f}")
        print(f"  Corpus B (JEP escrito):       {mb:.3f}")
        print(f"  Corpus C (JEP oral):          {mc:.3f}")

        # Test A vs C
        if col_ab in df_ab_a.columns:
            _, p_ac = stats.mannwhitneyu(
                df_ab_a[col_ab].dropna(),
                df_c[col].dropna(),
                alternative="two-sided"
            )
            sig = "***" if p_ac < 0.001 else "**" if p_ac < 0.01 else "*" if p_ac < 0.05 else "n.s."
            print(f"  A vs C: p={p_ac:.4f} {sig}")
        print()

except Exception as e:
    print(f"⚠ No se pudo cargar df A/B: {e}")
    print("Estadísticas Corpus C:")
    print(df_c[["y8_mafapo_cs", "y9_cidh_cs"]].describe().round(3))


# ── Paso 5: Análisis por subcaso ──────────────────────────────────────────────

print("\n== PASO 5: Análisis por subcaso ==")
print("\nMedias por audio:")
cols_disp = [c for c in ["y8_mafapo_cs", "y9_cidh_cs", "y2_sa", "y3_civil", "y4_nv", "y10_rep"] if c in df_c.columns]
print(df_c.groupby("audio")[cols_disp].mean().round(3).to_string())


# ── Paso 6: Guardar resultados ────────────────────────────────────────────────

print("\n== PASO 6: Guardando resultados ==")
df_c.drop(columns=["texto"], errors="ignore").to_csv(
    "indicators_corpus_c.csv", index=False, encoding="utf-8-sig"
)
print("✓ indicators_corpus_c.csv guardado")

from google.colab import files
files.download("indicators_corpus_c.csv")
print("✓ Descarga iniciada")
