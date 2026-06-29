"""
VERIFICACIÓN y8/y9 con centroide MAFAPO v5
===========================================
Recalcula y8 (dist. MAFAPO) e y9 (dist. CIDH) sobre el Corpus C usando
los centroides locales explícitos (v5 / v3) y compara con los valores
del CSV actual (data/features/indicators_corpus_c.csv).

Propósito: determinar si los y8/y9 que están en la tesis ya usan el
centroide v5 (293 textos) o uno anterior. NO modifica nada: solo lee y compara.

Uso:
    conda activate cfh
    python cfh_verificar_y8y9_centroide_v5.py
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ── Rutas (relativas a la raíz del repo) ──
CORPUS_C_DIR = "corpus_c"
BLOCK_SIZE   = 2000
CENTROIDE_MAFAPO_V5 = "data/referencias/centroide_mafapo_v5.npy"
CENTROIDE_CIDH_V3   = "data/referencias/centroide_cidh_v3.npy"
CSV_ACTUAL          = "data/features/indicators_corpus_c.csv"
MODEL_NAME          = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"

AUDIOS = {
    "catatumbo":         "catatumbo_audiencia_reconocimiento.txt",
    "costa_caribe":      "costa_caribe.txt",
    "casanare_torres":   "casanare_torres.txt",
    "dabeiba_antioquia": "dabeiba_antioquia.txt",
    "huila":             "huila.txt",
}

SUBCASO_META = {
    "catatumbo":         {"subcaso": "Norte de Santander"},
    "costa_caribe":      {"subcaso": "Costa Caribe"},
    "casanare_torres":   {"subcaso": "Casanare"},
    "dabeiba_antioquia": {"subcaso": "Antioquia"},
    "huila":             {"subcaso": "Huila"},
}


# ── Segmentación IDÉNTICA al script canónico ──
def segmentar_transcripcion(texto, nombre, block_size=BLOCK_SIZE):
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
                "audio": nombre,
                "bloque_id": f"{nombre}_b{bloque_id:04d}",
                "texto": texto_bloque,
                **SUBCASO_META.get(nombre, {}),
            })
            bloque_actual = []
            char_count = 0
            bloque_id += 1
    if bloque_actual and len(bloque_actual) > 20:
        texto_bloque = " ".join(bloque_actual)
        bloques.append({
            "audio": nombre,
            "bloque_id": f"{nombre}_b{bloque_id:04d}",
            "texto": texto_bloque,
            **SUBCASO_META.get(nombre, {}),
        })
    return bloques


def get_embedding_cs(text, tokenizer, model, device, max_length=512):
    if not text or len(text.strip()) < 10:
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def distancia_coseno(emb, centroide):
    if np.all(emb == 0):
        return np.nan
    sim = cosine_similarity(emb.reshape(1, -1), centroide.reshape(1, -1))[0][0]
    return float(1 - sim)


# ── PASO 1: Cargar centroides locales ──
print("== Verificación y8/y9 con centroide v5 ==\n")
print("Cargando centroides locales...")
assert os.path.exists(CENTROIDE_MAFAPO_V5), f"No existe {CENTROIDE_MAFAPO_V5}"
assert os.path.exists(CENTROIDE_CIDH_V3), f"No existe {CENTROIDE_CIDH_V3}"
cen_mafapo = np.load(CENTROIDE_MAFAPO_V5)
cen_cidh   = np.load(CENTROIDE_CIDH_V3)
print(f"  centroide MAFAPO v5: shape={cen_mafapo.shape}, norma={np.linalg.norm(cen_mafapo):.4f}")
print(f"  centroide CIDH v3:   shape={cen_cidh.shape}, norma={np.linalg.norm(cen_cidh):.4f}")
print(f"  (inventario v5 dice norma=14.2777 — debe coincidir)\n")

# ── PASO 2: Segmentar Corpus C ──
print("Segmentando Corpus C (2000 chars)...")
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
    print(f"  ✓ {nombre}: {len(bloques)} bloques")
df_c = pd.DataFrame(bloques_totales)
print(f"  Total: {len(df_c)} bloques\n")

# ── PASO 3: Cargar modelo y recalcular y8/y9 con v5 ──
print("Cargando ConfliBERT-Spanish...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"  ✓ cargado en {device}\n")

print("Recalculando y8/y9 con centroide v5...")
y8_v5, y9_v5 = [], []
for i, row in df_c.iterrows():
    emb = get_embedding_cs(row["texto"][:8000], tokenizer, model, device)
    y8_v5.append(distancia_coseno(emb, cen_mafapo))
    y9_v5.append(distancia_coseno(emb, cen_cidh))
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{len(df_c)}")
df_c["y8_v5"] = y8_v5
df_c["y9_v5"] = y9_v5
print("  ✓ recalculado\n")

# ── PASO 4: Comparar con el CSV actual ──
print("== COMPARACIÓN: CSV actual vs recálculo v5 ==\n")
df_actual = pd.read_csv(CSV_ACTUAL)

# Unir por bloque_id si existe, si no comparar medias globales
if "bloque_id" in df_actual.columns:
    merged = df_c.merge(df_actual[["bloque_id", "y8_mafapo_cs", "y9_cidh_cs"]],
                        on="bloque_id", how="inner")
    print(f"Bloques comparables: {len(merged)}")
    if len(merged) > 0:
        d8 = (merged["y8_v5"] - merged["y8_mafapo_cs"]).abs()
        d9 = (merged["y9_v5"] - merged["y9_cidh_cs"]).abs()
        print(f"\ny8 — diferencia |v5 − CSV actual|:")
        print(f"  media={d8.mean():.5f}  máx={d8.max():.5f}  mediana={d8.median():.5f}")
        print(f"y9 — diferencia |v5 − CSV actual|:")
        print(f"  media={d9.mean():.5f}  máx={d9.max():.5f}  mediana={d9.median():.5f}")

        print(f"\nMedias globales:")
        print(f"  y8: CSV actual={merged['y8_mafapo_cs'].mean():.4f}  |  v5={merged['y8_v5'].mean():.4f}")
        print(f"  y9: CSV actual={merged['y9_cidh_cs'].mean():.4f}  |  v5={merged['y9_v5'].mean():.4f}")

        print(f"\n== VEREDICTO ==")
        if d8.mean() < 0.001 and d9.mean() < 0.001:
            print("  ✓ Los y8/y9 del CSV YA usan el centroide v5 (diferencia despreciable).")
            print("    No hay que regenerar nada. Solo documentar.")
        else:
            print("  ⚠ Los y8/y9 del CSV NO coinciden con el centroide v5.")
            print("    Fueron calculados con un centroide anterior.")
            print("    Decisión: regenerar y8/y9 con v5 y propagar a Cap5/apéndice/IEI.")
else:
    print("CSV actual sin bloque_id. Comparando medias globales:")
    print(f"  y8: CSV={df_actual['y8_mafapo_cs'].mean():.4f}  v5={df_c['y8_v5'].mean():.4f}")
    print(f"  y9: CSV={df_actual['y9_cidh_cs'].mean():.4f}  v5={df_c['y9_v5'].mean():.4f}")

# ── PASO 5: Guardar el recálculo (sin sobrescribir el original) ──
df_c.drop(columns=["texto"], errors="ignore").to_csv(
    "data/indicators_corpus_c_y8y9_v5_VERIFICACION.csv",
    index=False, encoding="utf-8-sig")
print("\n✓ Recálculo guardado en data/indicators_corpus_c_y8y9_v5_VERIFICACION.csv")
print("  (archivo aparte — NO sobrescribe el original)")
