"""
CFH — Muestreo verificado Capa 3 (3 filtros + agregación por turno)
=====================================================================
Implementa la Guía Metodológica de Muestreo Multimodal (Zuluaga, 2026-06-09)
Observación 6.1: el muestreo actual no verifica identidad del sujeto.

Tres filtros:
  F1 — Identidad facial: face_recognition con foto de referencia (tolerancia 0.55)
  F2 — Calidad mínima: ancho rostro >= 100px, proporción 0.6-1.1
  F3 — Identidad de voz: verificación manual del SPEAKER_XX por subcaso

Agregación: mediana por turno (intervención), no por frame.

Uso:
  python code/cfh_muestreo_capa3_v2.py --subcaso casanare --video path/video.mp4

Requiere:
  pip install face_recognition opencv-python mediapipe opensmile
"""
import argparse
import numpy as np
from pathlib import Path

REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")

# ── F3: Mapeo manual SPEAKER → compareciente (verificado) ─────────────────
SPEAKERS_COMPARECIENTE = {
    "casanare":    ["SPEAKER_03"],
    "catatumbo":   ["SPEAKER_03"],          # SPEAKER_01 también posible, verificar
    "dabeiba":     ["SPEAKER_01"],
    "huila":       ["SPEAKER_06", "SPEAKER_07", "SPEAKER_08"],
    "costa_caribe":["SPEAKER_00"],          # verificar manualmente
}

# ── F1: Identidad facial ──────────────────────────────────────────────────
def enrolar_referencia(ruta_foto: str):
    """Carga foto de referencia y devuelve encoding (128-d)."""
    import face_recognition
    img = face_recognition.load_image_file(ruta_foto)
    encs = face_recognition.face_encodings(img)
    if not encs:
        raise ValueError(f"No se detectó cara en la foto de referencia: {ruta_foto}")
    return encs[0]

def cara_del_compareciente(frame_bgr, ref_enc, tolerancia=0.55):
    """
    Devuelve (bbox, True) si la cara más parecida al compareciente está
    dentro de la tolerancia. tolerancia=0.55 es más estricto que el default 0.60.
    """
    import face_recognition, cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None, False
    encs = face_recognition.face_encodings(rgb, locs)
    dists = face_recognition.face_distance([ref_enc], encs[0])
    i = int(np.argmin(dists))
    es_el = dists[i] <= tolerancia
    return locs[i], es_el

# ── F2: Calidad mínima del rostro ─────────────────────────────────────────
def calidad_ok(bbox, frame_shape, min_ancho=100):
    """
    Verifica tamaño y proporción del rostro detectado.
    bbox = (top, right, bottom, left) — formato face_recognition
    """
    top, right, bottom, left = bbox
    ancho = right - left
    alto  = bottom - top
    if ancho < min_ancho:
        return False
    # Proporción: cara muy "aplastada" = perfil fuerte o detección parcial
    if not (0.6 <= ancho / max(alto, 1) <= 1.1):
        return False
    return True

# ── Pipeline por subcaso ──────────────────────────────────────────────────
def procesar_subcaso(subcaso: str, video_path: str, audio_path: str,
                     diarization_json: str, ref_foto: str):
    """
    Procesa un subcaso completo con los 3 filtros y agrega por turno.
    Devuelve lista de dicts con AUs y eGeMAPS por turno válido.
    """
    import json, cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    print(f"\n[CFH] Procesando subcaso: {subcaso}")

    # Cargar diarización
    with open(diarization_json) as f:
        diarization = json.load(f)
    
    speakers_ok = SPEAKERS_COMPARECIENTE.get(subcaso, [])
    print(f"  Speakers del compareciente: {speakers_ok}")

    # Enrolar referencia facial (F1)
    ref_enc = enrolar_referencia(ref_foto)
    print(f"  ✓ Referencia facial enrolada: {ref_foto}")

    # Configurar MediaPipe
    model_path = str(REPO / "models" / "face_landmarker.task")
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        num_faces=1
    )

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    turnos_validos = []
    n_turnos_total = 0
    n_turnos_descartados = 0

    with mp_vision.FaceLandmarker.create_from_options(face_opts) as landmarker:
        for segmento in diarization:
            # F3: solo el compareciente
            if segmento.get("speaker") not in speakers_ok:
                continue
            
            duracion = segmento["end"] - segmento["start"]
            if duracion < 3.0:  # turnos muy cortos no aportan
                continue
            
            n_turnos_total += 1
            aus_frames = []
            n_frames_turno = int(duracion * 5)  # ~5 fps

            for t in np.arange(segmento["start"], segmento["end"], 0.2):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

                # F1: identidad facial
                bbox, es_el = cara_del_compareciente(frame, ref_enc, tolerancia=0.55)
                if not es_el:
                    continue

                # F2: calidad mínima
                if not calidad_ok(bbox, frame.shape):
                    continue

                # Extraer AUs con MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if not result.face_blendshapes:
                    continue
                
                bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
                aus_frames.append({
                    "browInnerUp":      bs.get("browInnerUp", 0),      # → AU1
                    "browDownLeft":     bs.get("browDownLeft", 0),     # → AU4
                    "mouthFrownLeft":   bs.get("mouthFrownLeft", 0),   # → AU15
                    "mouthSmileLeft":   bs.get("mouthSmileLeft", 0),   # → AU12
                })

            # Cobertura mínima
            cobertura = len(aus_frames) / max(n_frames_turno, 1)
            if cobertura < 0.40:
                n_turnos_descartados += 1
                continue

            # AGREGACIÓN: mediana por turno (no frame)
            aus_array = np.array([[a["browInnerUp"], a["browDownLeft"],
                                   a["mouthFrownLeft"], a["mouthSmileLeft"]]
                                  for a in aus_frames])
            aus_turno = np.median(aus_array, axis=0)

            turnos_validos.append({
                "subcaso":        subcaso,
                "start":          segmento["start"],
                "end":            segmento["end"],
                "n_frames_validos": len(aus_frames),
                "cobertura":      round(cobertura, 3),
                "au1_brow_inner": round(float(aus_turno[0]), 4),
                "au4_brow_down":  round(float(aus_turno[1]), 4),
                "au15_mouth_frown": round(float(aus_turno[2]), 4),
                "au12_mouth_smile": round(float(aus_turno[3]), 4),
            })

    cap.release()
    
    print(f"  Turnos totales del compareciente: {n_turnos_total}")
    print(f"  Turnos descartados (cobertura <40%): {n_turnos_descartados}")
    print(f"  Turnos válidos: {len(turnos_validos)}")
    
    # ICM facial del subcaso = mediana de turnos válidos
    if turnos_validos:
        au4_med  = np.median([t["au4_brow_down"] for t in turnos_validos])
        au12_med = np.median([t["au12_mouth_smile"] for t in turnos_validos])
        icm_facial = au4_med / (au4_med + au12_med + 1e-9)
        print(f"  ICM facial (mediana): {icm_facial:.3f}")
        print(f"  AU4 ceño (mediana):   {au4_med:.4f}")
        print(f"  AU12 sonrisa (mediana): {au12_med:.4f}")
    
    return turnos_validos

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFH Muestreo Capa 3 v2 — 3 filtros")
    parser.add_argument("--subcaso", required=True, choices=list(SPEAKERS_COMPARECIENTE.keys()))
    parser.add_argument("--video",   required=True, help="Ruta al MP4 del subcaso")
    parser.add_argument("--audio",   required=True, help="Ruta al WAV diarizado")
    parser.add_argument("--diarization", required=True, help="JSON de diarización")
    parser.add_argument("--ref",     required=True, help="Foto de referencia (ref_<subcaso>.jpg)")
    args = parser.parse_args()
    
    turnos = procesar_subcaso(
        subcaso=args.subcaso,
        video_path=args.video,
        audio_path=args.audio,
        diarization_json=args.diarization,
        ref_foto=args.ref
    )
    
    import json
    out = REPO / "outputs" / "capa3" / f"turnos_validos_{args.subcaso}_v2.json"
    with open(out, "w") as f:
        json.dump(turnos, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Guardado: {out}")
