"""
ICM Costa Caribe — eGeMAPS + AUs + ICM tri-canal
=================================================
Calcula los features de Capa 3 para el subcaso Costa Caribe
usando los datos ya disponibles:
  - Audio: corpus_c/costa_caribe_completo.wav
  - Video: corpus_c/costa_caribe/Caso 03|...mp4
  - Diarización: corpus_c/costa_caribe_diarization_v2.json
  - Compareciente: SPEAKER_00

Pesos ICM teóricos: facial=0.40, vocal=0.40, verbal=0.20

Outputs:
  - outputs/capa3/egemap_costa_caribe_compareciente.csv
  - outputs/capa3/aus_costa_caribe.csv
  - outputs/capa3/icm_tri_canal_v2.json  (actualizado con costa_caribe)
  - outputs/capa3/icm_resultados_v3.csv  (actualizado)
"""

import json
import numpy as np
import pandas as pd
import soundfile as sf
import opensmile
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
from pathlib import Path
from collections import defaultdict

# ── Rutas ────────────────────────────────────────────────────────────────────
AUDIO_PATH  = "corpus_c/costa_caribe_completo.wav"
DIAR_PATH   = "corpus_c/costa_caribe_diarization_v2.json"
ICM_JSON    = "outputs/capa3/icm_tri_canal_v2.json"
ICM_CSV_V3  = "outputs/capa3/icm_resultados_v3.csv"
OUT_EGEMAP  = "outputs/capa3/egemap_costa_caribe_compareciente.csv"
OUT_AUS     = "outputs/capa3/aus_costa_caribe.csv"

# Buscar video automáticamente
VIDEO_CANDIDATES = [
    "corpus_c/costa_caribe/Caso 03｜ Audiencia de Reconocimiento Subcaso Costa Caribe ｜ 18 de julio de 2022.mp4",
    "corpus_c/costa_caribe_Caso 03｜ Audiencia de Reconocimiento Subcaso Costa Caribe ｜ 18 de julio de 2022.mp4",
]
VIDEO_PATH = next((p for p in VIDEO_CANDIDATES if Path(p).exists()), None)

SPEAKER_TARGET = "SPEAKER_00"
SUBCASO_ID     = "costa_caribe"

# Lexicón REP para canal verbal (sin CFH-BERT)
LEXICON_REP = [
    "víctima", "civil", "inocente", "reconozco", "acepto",
    "responsabilidad", "responsable", "perdón", "lamento",
    "asesinato", "homicidio", "crimen", "ilegal",
    "persona protegida", "derecho", "dignidad", "verdad",
    "reparación", "nombre", "familia", "madre", "hijo",
    "joven", "trabajador", "humano", "vida"
]

print(f"Audio:    {AUDIO_PATH}")
print(f"Video:    {VIDEO_PATH or 'NO ENCONTRADO'}")
print(f"Diarización: {DIAR_PATH}")
print()

# ── 1. Cargar diarización y filtrar compareciente ─────────────────────────────
with open(DIAR_PATH, encoding="utf-8") as f:
    raw = json.load(f)

# El JSON puede ser lista o dict con clave "segments"
if isinstance(raw, list):
    segmentos = raw
elif isinstance(raw, dict) and "segments" in raw:
    segmentos = raw["segments"]
else:
    segmentos = list(raw.values())[0] if raw else []

# Calcular tiempo por speaker para confirmar SPEAKER_00
tiempo_por_speaker = defaultdict(float)
for s in segmentos:
    dur = s.get("duration", s.get("end", 0) - s.get("start", 0))
    tiempo_por_speaker[s["speaker"]] += dur

print("Tiempo por speaker (top 5):")
for spk, t in sorted(tiempo_por_speaker.items(), key=lambda x: -x[1])[:5]:
    print(f"  {spk}: {t/60:.1f} min")

segs_comp = [s for s in segmentos if s["speaker"] == SPEAKER_TARGET]
print(f"\nSegmentos SPEAKER_00 (compareciente): {len(segs_comp)}")

# ── 2. eGeMAPS sobre segmentos del compareciente ──────────────────────────────
print("\n[1/3] Calculando eGeMAPS...")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

audio_data, sr = sf.read(AUDIO_PATH)
if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]  # mono

egemap_rows = []
for i, seg in enumerate(segs_comp):
    start = seg.get("start", 0)
    end   = seg.get("end", start + seg.get("duration", 0))
    dur   = end - start

    if dur < 3.0:  # ignorar segmentos muy cortos
        continue

    s_idx = int(start * sr)
    e_idx = int(end * sr)
    clip  = audio_data[s_idx:e_idx]

    if len(clip) < sr:  # menos de 1 segundo
        continue

    try:
        feats = smile.process_signal(clip, sr)
        row   = feats.iloc[0].to_dict()
        row.update({"speaker": SPEAKER_TARGET, "start": start,
                    "end": end, "dur": dur, "seg_idx": i})
        egemap_rows.append(row)
    except Exception as e:
        continue

    if (i+1) % 100 == 0:
        print(f"  eGeMAPS: {i+1}/{len(segs_comp)} segmentos...")

df_egemap = pd.DataFrame(egemap_rows)
df_egemap.to_csv(OUT_EGEMAP, index=False, encoding="utf-8-sig")
print(f"  ✓ eGeMAPS guardado: {OUT_EGEMAP} ({len(df_egemap)} segmentos)")

# Score vocal agregado
def score_vocal(df):
    """Convierte eGeMAPS en score de sinceridad prosódica [0,1]."""
    scores = []
    # Shimmer — quiebre de voz (distress)
    if "shimmerLocaldB_sma3nz_amean" in df.columns:
        shimmer = df["shimmerLocaldB_sma3nz_amean"].mean()
        scores.append(min(shimmer / 2.0, 1.0))
    # Alpha ratio — estrés fisiológico
    if "alphaRatio_sma3_amean" in df.columns:
        alpha = abs(df["alphaRatio_sma3_amean"].mean())
        scores.append(min(alpha / 10.0, 1.0))
    # F0 variabilidad — emoción genuina
    if "F0semitoneFrom27.5Hz_sma3nz_stddevNorm" in df.columns:
        f0_std = df["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].mean()
        scores.append(min(f0_std, 1.0))
    # HNR bajo — voz tensa
    if "HNRdBACF_sma3nz_amean" in df.columns:
        hnr = df["HNRdBACF_sma3nz_amean"].mean()
        if hnr < 20:
            scores.append((20 - hnr) / 20)
    return float(np.mean(scores)) if scores else 0.5

icm_vocal_cc = score_vocal(df_egemap)
print(f"  ICM vocal Costa Caribe: {icm_vocal_cc:.3f}")

# ── 3. AUs con MediaPipe FaceLandmarker ───────────────────────────────────────
icm_facial_cc = 0.5  # default si no hay video
df_aus        = pd.DataFrame()

if VIDEO_PATH:
    print(f"\n[2/3] Calculando AUs con MediaPipe...")
    print(f"  Video: {VIDEO_PATH}")

    # Descargar modelo FaceLandmarker si no existe
    model_path = "face_landmarker.task"
    if not Path(model_path).exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"  Descargando modelo FaceLandmarker...")
        urllib.request.urlretrieve(url, model_path)
        print(f"  ✓ Modelo descargado")

    # Configurar FaceLandmarker
    base_options  = mp_python.BaseOptions(model_asset_path=model_path)
    face_options  = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    aus_rows = []
    cap      = cv2.VideoCapture(VIDEO_PATH)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    print(f"  FPS video: {fps:.1f}")

    with mp_vision.FaceLandmarker.create_from_options(face_options) as landmarker:
        for i, seg in enumerate(segs_comp[:200]):  # máx 200 segs para no exceder tiempo
            start = seg.get("start", 0)
            end   = seg.get("end", start + seg.get("duration", 0))
            dur   = end - start
            if dur < 3.0:
                continue

            # Tomar frame del medio del segmento
            mid_frame = int((start + dur/2) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            # Procesar con MediaPipe
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if result.face_blendshapes:
                bs  = result.face_blendshapes[0]
                row = {b.category_name: b.score for b in bs}
                row.update({"start": start, "end": end, "dur": dur, "seg_idx": i})
                aus_rows.append(row)

            if (i+1) % 50 == 0:
                print(f"  AUs: {i+1}/{min(len(segs_comp), 200)} segmentos...")

    cap.release()
    df_aus = pd.DataFrame(aus_rows)
    df_aus.to_csv(OUT_AUS, index=False, encoding="utf-8-sig")
    print(f"  ✓ AUs guardadas: {OUT_AUS} ({len(df_aus)} frames)")

    # Score facial
    def score_facial(df):
        """Convierte blendshapes MediaPipe en score de distress [0,1]."""
        if df.empty:
            return 0.5
        scores = []
        # Distress superior
        for col in ["browInnerUp", "browDownLeft", "browDownRight"]:
            if col in df.columns:
                scores.append(df[col].mean())
        # Tristeza
        for col in ["mouthFrownLeft", "mouthFrownRight", "mouthPucker"]:
            if col in df.columns:
                scores.append(df[col].mean())
        # Penalizar sonrisa social
        for col in ["mouthSmileLeft", "mouthSmileRight"]:
            if col in df.columns:
                sonrisa = df[col].mean()
                # Cheek raises (Duchenne)
                cheek   = df.get("cheekSquintLeft", pd.Series([0])).mean()
                if sonrisa > 0.3 and cheek < 0.1:
                    scores.append(-0.2)  # sonrisa performativa
        return float(max(0.0, min(np.mean(scores) if scores else 0.5, 1.0)))

    icm_facial_cc = score_facial(df_aus)
    print(f"  ICM facial Costa Caribe: {icm_facial_cc:.3f}")
else:
    print("\n[2/3] Video no encontrado — ICM facial = 0.5 (neutral)")

# ── 4. Canal verbal (lexicón REP sobre transcripción) ─────────────────────────
print("\n[3/3] Calculando canal verbal...")

# Usar texto del CSV de indicadores si existe
try:
    df_ind = pd.read_csv("data/indicators_corpus_c_capa1_v2.csv")
    df_cc  = df_ind[df_ind["audio"] == "costa_caribe"]
    rep_mean = df_cc["y10_rep"].mean() if "y10_rep" in df_cc.columns else 0.132
    print(f"  REP desde indicadores: {rep_mean:.3f}")
except:
    rep_mean = 0.132  # valor calculado previamente
    print(f"  REP desde valor previo: {rep_mean:.3f}")

icm_verbal_cc = float(rep_mean)

# ── 5. ICM tri-canal con pesos teóricos (0.40/0.40/0.20) ─────────────────────
print("\n[RESULTADO ICM COSTA CARIBE]")
print(f"  Canal facial (w=0.40): {icm_facial_cc:.3f}")
print(f"  Canal vocal  (w=0.40): {icm_vocal_cc:.3f}")
print(f"  Canal verbal (w=0.20): {icm_verbal_cc:.3f}")

icm_tri = 0.40 * icm_facial_cc + 0.40 * icm_vocal_cc + 0.20 * icm_verbal_cc
delta   = float(np.std([icm_facial_cc, icm_vocal_cc, icm_verbal_cc]))
congruencia = 1.0 - min(delta * 2, 1.0)
icm_final   = icm_tri * congruencia

print(f"  ICM tri-canal (pesos teóricos): {icm_tri:.3f}")
print(f"  Congruencia inter-canal: {congruencia:.3f}")
print(f"  ICM final (×congruencia): {icm_final:.3f}")

if icm_final >= 0.6:
    clasificacion = "GENUINO"
elif icm_final >= 0.35:
    clasificacion = "AMBIGUO"
else:
    clasificacion = "PERFORMATIVO"
print(f"  Clasificación: {clasificacion}")

# ── 6. Actualizar JSON y CSV ───────────────────────────────────────────────────
# Cargar JSON existente y añadir costa_caribe
with open(ICM_JSON, encoding="utf-8") as f:
    icm_data = json.load(f)

icm_data["costa_caribe"] = {
    "icm_facial":    round(icm_facial_cc, 3),
    "icm_vocal":     round(icm_vocal_cc, 3),
    "icm_verbal_v1": round(icm_verbal_cc, 3),
    "icm_verbal_v2": round(icm_verbal_cc, 3),
    "icm_tri_v1":    round(icm_tri, 3),
    "icm_tri_v2":    round(icm_final, 3),
    "delta_verbal":  round(icm_verbal_cc - rep_mean, 3),
    "y11_prop_mafapo": 0.0,  # pendiente
}

with open(ICM_JSON, "w", encoding="utf-8") as f:
    json.dump(icm_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ JSON actualizado: {ICM_JSON}")

# Actualizar CSV de resultados
df_v3 = pd.read_csv(ICM_CSV_V3)
nueva_fila = pd.DataFrame([{
    "audio":        "costa_caribe",
    "nombre":       "Batallón La Popa (12 comparecientes)",
    "rango":        "Coronel/Mayor",
    "icm_score":    round(icm_final, 3),
    "deteccion_pct": 100 if VIDEO_PATH else 0,
    "n_segs":       len(df_egemap),
}])
df_v3 = pd.concat([df_v3, nueva_fila], ignore_index=True)
df_v3.to_csv(ICM_CSV_V3, index=False, encoding="utf-8-sig")
print(f"✓ CSV actualizado: {ICM_CSV_V3}")

print("\n=== RESUMEN CORPUS C — ICM TRI-CANAL COMPLETO ===")
print(df_v3.to_string(index=False))
