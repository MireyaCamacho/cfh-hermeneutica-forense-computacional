"""
CFH — Pipeline completo Colab Pro
==================================
1. Transcribir MP3 nuevos con Whisper large-v3
2. Diarizar con pyannote-audio
3. Filtrar segmentos de voz de víctimas
4. Calcular centroide MAFAPO v4
5. Calcular embeddings Corpus B (y8, y9)
6. Recalcular DIS / IEI completos

Ejecutar celda por celda en Colab Pro con GPU T4 o A100.
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 1 — Instalar dependencias
# ═══════════════════════════════════════════════════════════════
"""
!pip install -q openai-whisper pyannote.audio transformers torch
!pip install -q scipy pandas numpy
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 2 — Montar Drive
# ═══════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os
CORPUS_C = '/content/drive/MyDrive/CHF_Corpus/corpus_c'
print("Archivos en corpus_c:")
for f in sorted(os.listdir(CORPUS_C)):
    print(f"  {f}")
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 3 — Identificar MP3 nuevos a transcribir
# ═══════════════════════════════════════════════════════════════
"""
import os

CORPUS_C = '/content/drive/MyDrive/CHF_Corpus/corpus_c'

# MP3 ya transcritos (tienen .txt o .json correspondiente)
ya_procesados = {
    'casanare_torres',
    'catatumbo',
    'costa_caribe',
    'dabeiba_antioquia',
    'huila',
}

# MP3 nuevos a transcribir
mp3_nuevos = []
for f in sorted(os.listdir(CORPUS_C)):
    if not f.endswith('.mp3'):
        continue
    # Verificar si ya tiene transcripción
    nombre = f.replace('.mp3', '')
    txt_existe = os.path.exists(f"{CORPUS_C}/{nombre}.txt")
    json_existe = os.path.exists(f"{CORPUS_C}/{nombre}_segments.json")
    if not txt_existe and not json_existe:
        mp3_nuevos.append(f)
        print(f"  NUEVO: {f}")
    else:
        print(f"  ya procesado: {f}")

print(f"\nTotal MP3 nuevos a transcribir: {len(mp3_nuevos)}")
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 4 — Transcribir con Whisper large-v3
# ═══════════════════════════════════════════════════════════════
"""
import whisper
import json
import os

model = whisper.load_model("large-v3")
print("Modelo Whisper cargado")

CORPUS_C = '/content/drive/MyDrive/CHF_Corpus/corpus_c'

for mp3_file in mp3_nuevos:
    ruta = f"{CORPUS_C}/{mp3_file}"
    nombre = mp3_file.replace('.mp3', '')
    print(f"\nTranscribiendo: {mp3_file}")

    result = model.transcribe(
        ruta,
        language="es",
        word_timestamps=True,
        verbose=False
    )

    # Guardar TXT
    txt_path = f"{CORPUS_C}/{nombre}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    print(f"  ✓ TXT guardado: {txt_path}")

    # Guardar segmentos JSON con timestamps
    segs_path = f"{CORPUS_C}/{nombre}_whisper_segments.json"
    with open(segs_path, 'w', encoding='utf-8') as f:
        json.dump(result['segments'], f, ensure_ascii=False, indent=2)
    print(f"  ✓ Segmentos JSON guardados: {segs_path}")

print("\n✓ Transcripciones completadas")
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 5 — Filtrar segmentos de víctimas (sin diarización)
# ═══════════════════════════════════════════════════════════════
"""
import json
import os

CORPUS_C = '/content/drive/MyDrive/CHF_Corpus/corpus_c'

# Lexicón de voz directa de víctimas
LEXICON_VICTIMAS = [
    "mi hijo", "mi hija", "mi hermano", "mi hermana", "mi madre", "mi padre",
    "mi esposo", "mi esposa", "mi familiar", "nuestro hijo", "nuestros hijos",
    "nos mataron", "lo mataron", "la mataron", "los mataron", "lo asesinaron",
    "lo llevaron", "nunca volvió", "nunca regresó", "era inocente", "era civil",
    "no era guerrillero", "no era guerrillera", "buscando justicia",
    "buscando verdad", "busco a mi", "saber qué pasó", "quiero saber",
    "pedimos perdón", "exigimos verdad", "necesitamos saber",
    "soy la mamá", "soy la madre", "soy el padre", "soy hermana",
    "soy hermano", "soy la esposa", "vengo por mi",
    "dolor", "sufrimiento", "duelo", "llorar", "lloramos",
    "inocente", "civil", "campesino", "trabajador",
]

FRASES_INSTITUCIONALES = [
    "sala de reconocimiento", "jurisdicción especial",
    "ruta dialógica", "compareciente", "magistrada", "magistrado",
    "subcaso", "macrocaso", "resolución de conclusiones",
    "sección de reconocimiento", "apoderado judicial",
]

def score_victima(texto):
    t = texto.lower()
    score = sum(1 for f in LEXICON_VICTIMAS if f in t)
    penalidad = sum(1 for f in FRASES_INSTITUCIONALES if f in t)
    return max(0, score - penalidad)

todos_segmentos = []

for f in sorted(os.listdir(CORPUS_C)):
    if not f.endswith('_whisper_segments.json'):
        continue
    audiencia = f.replace('_whisper_segments.json', '')
    with open(f"{CORPUS_C}/{f}", encoding='utf-8') as fh:
        segs = json.load(fh)

    candidatos = []
    for s in segs:
        texto = s.get('text', '').strip()
        if len(texto) < 30:
            continue
        sc = score_victima(texto)
        if sc >= 2:
            candidatos.append({
                'texto': texto,
                'audiencia': audiencia,
                'score': sc,
                'start': s.get('start', 0),
                'end': s.get('end', 0),
            })

    # Tomar los mejores por audiencia
    candidatos_sorted = sorted(candidatos, key=lambda x: x['score'], reverse=True)
    mejores = candidatos_sorted[:30]  # máx 30 por audiencia
    todos_segmentos.extend(mejores)
    print(f"  {audiencia}: {len(candidatos)} candidatos → {len(mejores)} seleccionados")

print(f"\nTotal segmentos víctimas: {len(todos_segmentos)}")

# Guardar
output_path = f"{CORPUS_C}/corpus_c_victimas_observaciones.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total': len(todos_segmentos),
        'segmentos': todos_segmentos
    }, f, ensure_ascii=False, indent=2)
print(f"✓ Guardado: {output_path}")
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 6 — Calcular centroide MAFAPO v4
# ═══════════════════════════════════════════════════════════════
"""
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine as cosine_dist

DEVICE = torch.device('cuda')
MODEL_NAME = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"✓ ConfliBERT cargado en {DEVICE}")

def get_emb(texto):
    if not texto or len(texto.strip()) < 10:
        return np.zeros(768)
    inp = tokenizer(texto, return_tensors="pt", max_length=512,
                    truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

CORPUS_C = '/content/drive/MyDrive/CHF_Corpus/corpus_c'
REF_DIR = '/content/drive/MyDrive/CHF_Corpus/referencias'
os.makedirs(REF_DIR, exist_ok=True)

# Cargar textos v3b existentes
centroide_v3b = np.load(f"{REF_DIR}/centroide_mafapo_v3b.npy")

# Cargar nuevos segmentos de observaciones
with open(f"{CORPUS_C}/corpus_c_victimas_observaciones.json") as f:
    data = json.load(f)
textos_nuevos = [s['texto'] for s in data['segmentos'] if s['score'] >= 3]
print(f"Textos nuevos de alta calidad (score>=3): {len(textos_nuevos)}")

# Calcular embeddings nuevos
print("Calculando embeddings nuevos...")
embs_nuevos = []
for i, t in enumerate(textos_nuevos):
    embs_nuevos.append(get_emb(t))
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(textos_nuevos)}")

# Combinar con centroide v3b (promedio ponderado)
# v3b tiene 67 textos con peso 1.0
# nuevos tienen peso 1.8 (voz directa en audiencia)
n_v3b = 67
peso_v3b = 1.0
peso_nuevos = 1.8

# Reconstruir centroide ponderado
vec_v3b = centroide_v3b * n_v3b * peso_v3b
vec_nuevos = np.sum([e * peso_nuevos for e in embs_nuevos], axis=0) if embs_nuevos else np.zeros(768)
n_total_ponderado = n_v3b * peso_v3b + len(embs_nuevos) * peso_nuevos

centroide_v4 = (vec_v3b + vec_nuevos) / n_total_ponderado
centroide_v4 = centroide_v4 / np.linalg.norm(centroide_v4)  # normalizar

# Verificar saturación
dist = cosine_dist(centroide_v3b, centroide_v4)
n_efectivo = n_v3b + len(embs_nuevos)
margen = 100 / np.sqrt(n_efectivo)

print(f"\n{'='*50}")
print(f"CENTROIDE MAFAPO v4")
print(f"{'='*50}")
print(f"  Textos v3b:     {n_v3b}")
print(f"  Textos nuevos:  {len(textos_nuevos)}")
print(f"  Total:          {n_efectivo}")
print(f"  Margen error:   ±{margen:.1f}%")
print(f"  Dist v3b→v4:    {dist:.4f}")
print(f"  {'✓ SATURADO' if dist < 0.005 else '— variación presente'}")

# Guardar
np.save(f"{REF_DIR}/centroide_mafapo_v4.npy", centroide_v4)
print(f"\n✓ Centroide v4 guardado")
"""

# ═══════════════════════════════════════════════════════════════
# CELDA 7 — Embeddings Corpus B (y8, y9) — N=199 bloques
# ═══════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

CORPUS_B_JSON = '/content/drive/MyDrive/CHF_Corpus/corpus_b_json'
REF_DIR = '/content/drive/MyDrive/CHF_Corpus/referencias'

centroide_mafapo = np.load(f"{REF_DIR}/centroide_mafapo_v4.npy")
centroide_cidh   = np.load(f"{REF_DIR}/centroide_cidh_v3.npy")

print("Procesando bloques Corpus B...")
resultados = []

for jf in sorted(Path(CORPUS_B_JSON).glob('*.json')):
    try:
        data = json.loads(jf.read_text(encoding='utf-8'))
        secciones = data.get('segmentation', {}).get('sections', [])
        doc_id = data.get('metadata', {}).get('doc_id', jf.stem)

        for sec in secciones:
            if not sec.get('is_target', False):
                continue
            texto = sec.get('text', '').strip()
            if len(texto) < 30:
                continue
            texto = texto[:1000]

            emb = get_emb(texto)
            y8 = float(cosine_dist(emb, centroide_mafapo))
            y9 = float(cosine_dist(emb, centroide_cidh))

            resultados.append({
                'doc_id': doc_id,
                'section_id': sec.get('section_id', ''),
                'corpus_type': 'B',
                'y8_mafapo_v4': round(y8, 4),
                'y9_cidh_v3':   round(y9, 4),
            })
    except Exception as e:
        print(f"  Error {jf.name}: {e}")

df_emb = pd.DataFrame(resultados)
print(f"✓ {len(df_emb)} bloques con embeddings")
print(df_emb[['y8_mafapo_v4', 'y9_cidh_v3']].describe())

output = '/content/drive/MyDrive/CHF_Corpus/outputs/embeddings_corpus_b_v4.csv'
df_emb.to_csv(output, index=False)
print(f"✓ Guardado: {output}")
"""

print("Notebook CFH generado. Pega cada celda en Colab Pro.")
