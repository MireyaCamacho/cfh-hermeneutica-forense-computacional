"""
AUs Costa Caribe v2 — muestreo múltiple por segmento
=====================================================
El problema con v1: se tomaba 1 frame por segmento (el del medio).
Si en ese momento la cámara apunta a otra persona, no hay detección.

Solución: tomar N frames por segmento en intervalos regulares
y quedarse con el primero que tenga detección facial.

Output:
  - outputs/capa3/aus_costa_caribe_v2.csv
  - outputs/capa3/icm_tri_canal_v2.json  (ICM facial actualizado)
  - outputs/capa3/icm_resultados_v3.csv  (actualizado)
"""

import cv2, json, numpy as np, pandas as pd, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

VIDEO = "corpus_c/costa_caribe/Caso 03\uff5c Audiencia de Reconocimiento Subcaso Costa Caribe \uff5c 18 de julio de 2022.mp4"
DIAR  = "corpus_c/costa_caribe_diarization_v2.json"
MODEL = "face_landmarker.task"
OUT_AUS  = "outputs/capa3/aus_costa_caribe_v2.csv"
ICM_JSON = "outputs/capa3/icm_tri_canal_v2.json"
ICM_CSV  = "outputs/capa3/icm_resultados_v3.csv"
SPEAKER  = "SPEAKER_00"
N_FRAMES = 5   # frames a probar por segmento
MAX_SEGS = 300 # máximo de segmentos a procesar

# ── Cargar diarización ────────────────────────────────────────────────────────
with open(DIAR, encoding="utf-8") as f:
    raw = json.load(f)
segs = raw if isinstance(raw, list) else raw.get("segments", list(raw.values())[0])
segs_comp = [s for s in segs if s["speaker"] == SPEAKER]
print(f"Segmentos SPEAKER_00: {len(segs_comp)} — procesando máx {MAX_SEGS}")

# ── Configurar MediaPipe ──────────────────────────────────────────────────────
base_opt  = mp_python.BaseOptions(model_asset_path=MODEL)
face_opt  = mp_vision.FaceLandmarkerOptions(
    base_options=base_opt,
    output_face_blendshapes=True,
    num_faces=1,
    running_mode=mp_vision.RunningMode.IMAGE,
)

# ── Procesar video ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {fps:.0f}fps, {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps/3600:.1f}h")

aus_rows      = []
n_detectados  = 0
n_procesados  = 0

with mp_vision.FaceLandmarker.create_from_options(face_opt) as lm:
    for seg in segs_comp[:MAX_SEGS]:
        start = seg.get("start", 0)
        end   = seg.get("end", start + seg.get("duration", 0))
        dur   = end - start
        if dur < 3.0:
            continue

        n_procesados += 1
        detectado     = False

        # Probar N_FRAMES posiciones dentro del segmento
        for k in range(N_FRAMES):
            t      = start + dur * (k + 1) / (N_FRAMES + 1)
            f_idx  = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lm.detect(mp_img)

            if result.face_blendshapes:
                bs  = result.face_blendshapes[0]
                row = {b.category_name: b.score for b in bs}
                row.update({"start": start, "end": end,
                            "dur": dur, "frame_t": t, "frame_k": k})
                aus_rows.append(row)
                n_detectados += 1
                detectado = True
                break  # con una detección por segmento es suficiente

        if n_procesados % 50 == 0:
            pct = 100 * n_detectados / n_procesados
            print(f"  {n_procesados}/{MAX_SEGS} segs — detección {pct:.0f}%")

cap.release()

df_aus = pd.DataFrame(aus_rows)
df_aus.to_csv(OUT_AUS, index=False, encoding="utf-8-sig")
pct_det = 100 * n_detectados / max(n_procesados, 1)
print(f"\n✓ AUs v2: {n_detectados}/{n_procesados} segmentos ({pct_det:.0f}% detección)")
print(f"  Guardado: {OUT_AUS}")

# ── Score facial ──────────────────────────────────────────────────────────────
def score_facial(df):
    if df.empty:
        return 0.5, 0.0
    scores = []
    for col in ["browInnerUp","browDownLeft","browDownRight"]:
        if col in df.columns: scores.append(df[col].mean())
    for col in ["mouthFrownLeft","mouthFrownRight","mouthPucker"]:
        if col in df.columns: scores.append(df[col].mean())
    sonrisa = 0
    for col in ["mouthSmileLeft","mouthSmileRight"]:
        if col in df.columns: sonrisa += df[col].mean() / 2
    cheek = df.get("cheekSquintLeft", pd.Series([0])).mean()
    if sonrisa > 0.3 and cheek < 0.1:
        scores.append(-0.2)
    val = float(max(0.0, min(np.mean(scores) if scores else 0.5, 1.0)))
    return val, pct_det

icm_facial, pct = score_facial(df_aus)
print(f"\nICM facial v2: {icm_facial:.3f} (detección {pct:.0f}%)")

# ── Recuperar vocal y verbal del JSON existente ────────────────────────────────
with open(ICM_JSON, encoding="utf-8") as f:
    icm_data = json.load(f)

icm_vocal  = icm_data["costa_caribe"]["icm_vocal"]
icm_verbal = icm_data["costa_caribe"]["icm_verbal_v2"]

# ── ICM tri-canal actualizado ─────────────────────────────────────────────────
icm_tri    = 0.40 * icm_facial + 0.40 * icm_vocal + 0.20 * icm_verbal
delta      = float(np.std([icm_facial, icm_vocal, icm_verbal]))
congruencia = 1.0 - min(delta * 2, 1.0)
icm_final  = icm_tri * congruencia

if icm_final >= 0.6:   clas = "GENUINO"
elif icm_final >= 0.35: clas = "AMBIGUO"
else:                   clas = "PERFORMATIVO"

print(f"\n[ICM COSTA CARIBE v2]")
print(f"  Facial (w=0.40): {icm_facial:.3f}")
print(f"  Vocal  (w=0.40): {icm_vocal:.3f}")
print(f"  Verbal (w=0.20): {icm_verbal:.3f}")
print(f"  ICM tri-canal:   {icm_tri:.3f}")
print(f"  Congruencia:     {congruencia:.3f}")
print(f"  ICM final:       {icm_final:.3f} — {clas}")

# ── Actualizar JSON ───────────────────────────────────────────────────────────
icm_data["costa_caribe"].update({
    "icm_facial":   round(icm_facial, 3),
    "icm_tri_v1":   round(icm_tri, 3),
    "icm_tri_v2":   round(icm_final, 3),
    "deteccion_pct": round(pct_det, 1),
})
with open(ICM_JSON, "w", encoding="utf-8") as f:
    json.dump(icm_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ JSON actualizado")

# ── Actualizar CSV ────────────────────────────────────────────────────────────
df_v3 = pd.read_csv(ICM_CSV)
df_v3.loc[df_v3["audio"] == "costa_caribe", "icm_score"]      = round(icm_final, 3)
df_v3.loc[df_v3["audio"] == "costa_caribe", "deteccion_pct"]  = round(pct_det, 1)
df_v3.to_csv(ICM_CSV, index=False, encoding="utf-8-sig")

print("\n=== CORPUS C — ICM COMPLETO FINAL ===")
print(df_v3.to_string(index=False))
