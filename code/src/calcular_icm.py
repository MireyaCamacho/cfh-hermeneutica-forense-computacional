"""
Índice de Congruencia Multimodal (ICM) — Hermenéutica Forense Computacional
============================================================================
Calcula la congruencia entre los tres canales de comunicación en
las audiencias orales de la JEP (Corpus C):

  Canal verbal  → ConfliBERT REP score (texto transcrito)
  Canal vocal   → OpenSMILE eGeMAPS 88 features (prosodia)
  Canal facial  → OpenFace 3.0 AUs + head pose (expresión)

ICM alto (congruente)   → reconocimiento GENUINO
ICM bajo (incongruente) → reconocimiento PERFORMATIVO

Basado en:
  - Fraser (1995): reconocimiento genuino vs. performativo
  - Zehr (2002): autenticidad en justicia restaurativa
  - Baird & Coutinho (2019): firma prosódica de sinceridad
  - Scientific Reports (2024): AUs de culpa y distress

Uso:
    python code/src/calcular_icm.py \
        --diarization corpus_c/casanare_torres_diarization.json \
        --audio corpus_c/casanare_torres.wav \
        --video corpus_c/videos/casanare_torres.mp4 \
        --transcript corpus_c/casanare_torres_transcript.json \
        --output data/features/icm_casanare.csv

Requiere (instalar en Colab):
    pip install opensmile pandas numpy scipy
    # OpenFace 3.0: compilar desde https://github.com/TadasBaltrusaitis/OpenFace
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Configuración ICM ─────────────────────────────────────────────────────────

# AUs relevantes para distress/culpa (Scientific Reports, 2024)
AU_DISTRESS = ["AU01", "AU04", "AU15", "AU17"]   # inner brow raise, brow lowerer, lip corner depressor, chin raiser
AU_SOCIAL_SMILE = ["AU06", "AU12"]                # cheek raiser + lip corner puller (sonrisa Duchenne)
AU_SOCIAL_ONLY = ["AU12"]                          # sonrisa sin AU06 = social/performativa

# Features eGeMAPS más relevantes para sinceridad (Baird & Coutinho, 2019)
PROSODY_SINCERIDAD = [
    "shimmerLocaldB_sma3nz",     # variación de amplitud — quiebre de voz
    "F0semitoneFrom27.5Hz_sma3nz", # pitch
    "loudness_sma3",             # intensidad
    "alphaRatio_sma3",           # estrés involuntario
    "HNRdBACF_sma3nz",          # harmonics-to-noise ratio
    "F1frequency_sma3nz",        # primer formante
    "logRelF0-H1-H2_sma3nz",    # diferencia armónicas
]

# Hablante compareciente por audiencia (identificado en diarización)
COMPARECIENTE_SPEAKER = {
    "casanare_torres":  "SPEAKER_03",  # 58.7% del tiempo
    "catatumbo":        None,           # pendiente de identificar
    "costa_caribe":     None,
    "dabeiba":          None,
    "huila":            None,
}

# ── Módulo 1: Canal verbal (REP score del texto) ───────────────────────────────

def calcular_rep_score_segmento(texto, cfh_bert=None):
    """
    Calcula el REP score de un segmento de texto.
    Si CFH-BERT no está disponible, usa el lexicón de aproximación.
    """
    if cfh_bert:
        # CFH-BERT v2 — requiere GPU
        try:
            inputs = cfh_bert["tokenizer"](
                texto, return_tensors="pt",
                max_length=512, truncation=True
            )
            outputs = cfh_bert["model"](**inputs)
            probs = outputs.logits.softmax(-1)
            # Índice REP en el vocabulario de etiquetas
            rep_idx = cfh_bert["label2id"].get("B-REP", 1)
            rep_score = probs[:, :, rep_idx].mean().item()
            return rep_score
        except Exception as e:
            pass

    # Lexicón de aproximación sin GPU
    LEXICON_REP = [
        "víctima", "civil", "inocente", "reconozco", "acepto",
        "responsabilidad", "responsable", "perdón", "lamento",
        "asesinato", "homicidio", "crimen", "delito", "ilegal",
        "persona protegida", "derecho", "dignidad", "verdad",
        "reparación", "nombre", "familia", "madre", "hijo"
    ]
    texto_lower = texto.lower()
    n_tokens = max(len(texto_lower.split()), 1)
    rep_count = sum(texto_lower.count(term) for term in LEXICON_REP)
    return min(rep_count / n_tokens * 10, 1.0)


# ── Módulo 2: Canal vocal (prosodia eGeMAPS) ──────────────────────────────────

def extraer_features_prosodicas(audio_path, start_time, end_time, opensmile_path=None):
    """
    Extrae features eGeMAPS del segmento de audio.
    Requiere OpenSMILE instalado.
    """
    features = {}

    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Cargar segmento de audio
        import soundfile as sf
        import numpy as np

        audio_data, sr = sf.read(audio_path)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio_data[start_sample:end_sample]

        # Calcular features
        y = smile.process_signal(segment, sr)
        features = y.iloc[0].to_dict()

    except Exception as e:
        # Features de aproximación si OpenSMILE no disponible
        features = {feat: 0.0 for feat in PROSODY_SINCERIDAD}

    return features


def calcular_score_prosodico(features):
    """
    Convierte features eGeMAPS en un score de sinceridad prosódica.
    Score alto = marcadores de distress emocional (genuino)
    Score bajo = prosodia monótona (performativo)

    Basado en Baird & Coutinho (2019):
    - Shimmer alto → quiebre de voz → distress
    - Alpha ratio alto → estrés involuntario
    - F0 variable → emoción genuina vs. plana = lectura
    """
    score = 0.0
    n_features = 0

    # Shimmer — variación de amplitud (quiebre de voz)
    shimmer = features.get("shimmerLocaldB_sma3nz_amean", 0)
    if shimmer > 0:
        score += min(shimmer / 2.0, 1.0)
        n_features += 1

    # Alpha ratio — estrés fisiológico involuntario
    alpha = features.get("alphaRatio_sma3_amean", 0)
    if alpha != 0:
        score += min(abs(alpha) / 10.0, 1.0)
        n_features += 1

    # F0 variabilidad — emoción vs. monotonía
    f0_std = features.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0)
    if f0_std > 0:
        score += min(f0_std, 1.0)
        n_features += 1

    # HNR bajo — voz tensa/quebrada
    hnr = features.get("HNRdBACF_sma3nz_amean", 20)
    if hnr < 20:
        score += (20 - hnr) / 20
        n_features += 1

    return score / max(n_features, 1)


# ── Módulo 3: Canal facial (OpenFace AUs) ────────────────────────────────────

def extraer_features_faciales(video_path, start_time, end_time, openface_path=None):
    """
    Extrae Action Units del segmento de video con OpenFace 3.0.
    Requiere OpenFace compilado.
    """
    features = {}

    if openface_path and os.path.exists(openface_path):
        try:
            import subprocess
            import tempfile

            # Extraer clip de video
            clip_path = "/tmp/clip_icm.mp4"
            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-y", clip_path
            ], capture_output=True)

            # Correr OpenFace
            out_dir = "/tmp/openface_out"
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run([
                openface_path, "-f", clip_path,
                "-out_dir", out_dir,
                "-aus", "-pose", "-gaze"
            ], capture_output=True)

            # Leer CSV de OpenFace
            csv_files = list(Path(out_dir).glob("*.csv"))
            if csv_files:
                df_of = pd.read_csv(csv_files[0])
                # Promediar AUs sobre el segmento
                au_cols = [c for c in df_of.columns if c.startswith("AU") and "_r" in c]
                for col in au_cols:
                    features[col.replace("_r", "")] = df_of[col].mean()

                # Head pose
                features["pose_Rx"] = df_of["pose_Rx"].mean() if "pose_Rx" in df_of else 0
                features["pose_Ry"] = df_of["pose_Ry"].mean() if "pose_Ry" in df_of else 0
                features["gaze_angle_x"] = df_of["gaze_angle_x"].mean() if "gaze_angle_x" in df_of else 0

        except Exception as e:
            print(f"    ⚠ OpenFace error: {e}")

    return features


def calcular_score_facial(features):
    """
    Convierte AUs en un score de distress facial.
    Score alto = AUs de distress genuino
    Score bajo = sin distress o sonrisa social

    Basado en Scientific Reports (2024):
    - AU1+AU4 = inner brow raise + brow lowerer = distress
    - AU15+AU17 = tristeza, contención
    - AU6+AU12 = sonrisa Duchenne (genuina)
    - AU12 solo = sonrisa social (performativa)
    - Cabeza inclinada hacia abajo (pose_Rx < 0) = vergüenza
    """
    if not features:
        return 0.5  # neutral si no hay datos

    score = 0.0
    n_components = 0

    # Distress superior (AU1 + AU4)
    au1 = features.get("AU01", 0)
    au4 = features.get("AU04", 0)
    distress_superior = (au1 + au4) / 2
    score += min(distress_superior / 2.0, 1.0)
    n_components += 1

    # Tristeza (AU15 + AU17)
    au15 = features.get("AU15", 0)
    au17 = features.get("AU17", 0)
    tristeza = (au15 + au17) / 2
    score += min(tristeza / 2.0, 1.0)
    n_components += 1

    # Penalizar sonrisa social sin Duchenne (AU12 sin AU6)
    au6 = features.get("AU06", 0)
    au12 = features.get("AU12", 0)
    if au12 > 1.0 and au6 < 0.5:
        score -= 0.3  # sonrisa performativa

    # Head pose: cabeza inclinada = vergüenza/humildad
    pose_rx = features.get("pose_Rx", 0)
    if pose_rx < -0.1:  # inclinado hacia abajo
        score += 0.2
        n_components += 1

    return max(0.0, min(score / max(n_components, 1), 1.0))


# ── ICM: Integración de tres canales ─────────────────────────────────────────

def calcular_icm(rep_score, prosody_score, facial_score, pesos=(0.4, 0.35, 0.25)):
    """
    Calcula el Índice de Congruencia Multimodal.

    Args:
        rep_score:     score canal verbal (0-1, alto = REP alto)
        prosody_score: score canal vocal  (0-1, alto = distress prosódico)
        facial_score:  score canal facial (0-1, alto = distress facial)
        pesos:         (verbal, vocal, facial) — suma = 1.0

    Returns:
        dict con ICM score y clasificación
    """
    scores = np.array([rep_score, prosody_score, facial_score])
    pesos_arr = np.array(pesos)

    # Score ponderado
    icm_score = np.dot(scores, pesos_arr)

    # Congruencia: desviación estándar entre canales
    # Alta desviación = canales incongruentes = performativo
    congruencia = 1.0 - min(np.std(scores) * 2, 1.0)

    # ICM final: score ponderado × congruencia
    icm_final = icm_score * congruencia

    # Clasificación
    if icm_final >= 0.6:
        clasificacion = "GENUINO"
    elif icm_final >= 0.35:
        clasificacion = "AMBIGUO"
    else:
        clasificacion = "PERFORMATIVO"

    return {
        "icm_score":        round(icm_final, 4),
        "icm_verbal":       round(rep_score, 4),
        "icm_vocal":        round(prosody_score, 4),
        "icm_facial":       round(facial_score, 4),
        "icm_congruencia":  round(congruencia, 4),
        "icm_clasificacion": clasificacion
    }


# ── Procesador principal ──────────────────────────────────────────────────────

def procesar_audiencia(diarization_path, audio_path=None, video_path=None,
                       transcript_path=None, audiencia_id=None, output_path=None):
    """
    Calcula el ICM para todos los segmentos del compareciente
    en una audiencia diarizada.
    """
    print(f"\n== ICM: {audiencia_id} ==")

    # Cargar diarización
    with open(diarization_path, encoding='utf-8') as f:
        segmentos = json.load(f)
    print(f"✓ {len(segmentos)} segmentos diarizados")

    # Identificar hablante compareciente
    speaker_target = COMPARECIENTE_SPEAKER.get(audiencia_id)
    if not speaker_target:
        # Calcular automáticamente: el hablante con más tiempo
        tiempo_por_speaker = defaultdict(float)
        for s in segmentos:
            tiempo_por_speaker[s["speaker"]] += s["duracion"]
        speaker_target = max(tiempo_por_speaker, key=tiempo_por_speaker.get)
        print(f"✓ Compareciente identificado automáticamente: {speaker_target}")
    else:
        print(f"✓ Compareciente: {speaker_target}")

    # Filtrar segmentos del compareciente
    segs_compareciente = [s for s in segmentos if s["speaker"] == speaker_target]
    print(f"✓ Segmentos compareciente: {len(segs_compareciente)}")

    # Cargar transcripción si disponible
    transcripcion = {}
    if transcript_path and Path(transcript_path).exists():
        with open(transcript_path, encoding='utf-8') as f:
            transcripcion = json.load(f)

    # Cargar CFH-BERT si disponible
    cfh_bert = None
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        cfh_bert = {
            "tokenizer": AutoTokenizer.from_pretrained("eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"),
            "model": AutoModelForTokenClassification.from_pretrained("eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"),
            "label2id": {"O": 0, "B-REP": 1, "I-REP": 2}
        }
        print("✓ CFH-BERT cargado")
    except:
        print("⚠ CFH-BERT no disponible — usando lexicón")

    # Calcular ICM por segmento
    registros = []
    for idx, seg in enumerate(segs_compareciente):
        start = seg["start"]
        end = seg["end"]
        duracion = seg["duracion"]

        if duracion < 5:  # ignorar segmentos muy cortos
            continue

        # Canal verbal
        texto = transcripcion.get(f"{start:.1f}", "")
        rep_score = calcular_rep_score_segmento(texto, cfh_bert)

        # Canal vocal
        prosody_features = {}
        if audio_path and Path(audio_path).exists():
            prosody_features = extraer_features_prosodicas(audio_path, start, end)
        prosody_score = calcular_score_prosodico(prosody_features)

        # Canal facial
        facial_features = {}
        if video_path and Path(video_path).exists():
            facial_features = extraer_features_faciales(video_path, start, end)
        facial_score = calcular_score_facial(facial_features)

        # ICM
        icm = calcular_icm(rep_score, prosody_score, facial_score)

        registro = {
            "audiencia":  audiencia_id,
            "speaker":    speaker_target,
            "start":      start,
            "end":        end,
            "duracion":   duracion,
            "texto_preview": texto[:100] if texto else "",
            **icm
        }
        registros.append(registro)

        if idx % 50 == 0:
            print(f"  Procesados {idx}/{len(segs_compareciente)} segmentos...")

    df = pd.DataFrame(registros)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ ICM guardado: {output_path}")

    # Estadísticas
    print(f"\n== RESUMEN ICM {audiencia_id} ==")
    print(f"Segmentos procesados: {len(df)}")
    if len(df) > 0:
        print(f"ICM promedio: {df['icm_score'].mean():.3f}")
        print(f"Clasificación:")
        print(df['icm_clasificacion'].value_counts().to_string())
        print(f"\nTop 5 momentos más genuinos:")
        print(df.nlargest(5, 'icm_score')[
            ['start', 'duracion', 'icm_score', 'icm_clasificacion', 'texto_preview']
        ].to_string())

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICM — Índice de Congruencia Multimodal CFH")
    parser.add_argument("--diarization",  required=True, help="JSON de diarización")
    parser.add_argument("--audio",        default=None,  help="WAV del audio")
    parser.add_argument("--video",        default=None,  help="MP4 del video")
    parser.add_argument("--transcript",   default=None,  help="JSON de transcripción")
    parser.add_argument("--audiencia_id", default="audiencia", help="ID de la audiencia")
    parser.add_argument("--output",       default="data/features/icm.csv", help="CSV de salida")
    args = parser.parse_args()

    df = procesar_audiencia(
        diarization_path=args.diarization,
        audio_path=args.audio,
        video_path=args.video,
        transcript_path=args.transcript,
        audiencia_id=args.audiencia_id,
        output_path=args.output
    )
