# -*- coding: utf-8 -*-
"""
calcular_aus_v3_multi.py
================================================================================
CFH — Extraccion facial v3 parametrizada por subcaso (los 5 del Corpus C)

OBJETO:
    Extraer Action Units faciales (via MediaPipe FaceLandmarker) de los
    comparecientes de cada audiencia, de forma HOMOGENEA para los 5 subcasos.
    Atribucion por TIEMPO+IDENTIDAD anclada en la marcacion manual del inventario
    (no por speaker de diarizacion, que mezcla comparecientes). Guarda en
    aus_<subcaso>_v3.csv con columna 'identidad' por frame.

METODO (identico en los 5 subcasos):
    - MediaPipe FaceLandmarker + mapeo blendshapes->AU (AU1,AU4,AU6,AU12,AU15,AU17)
    - recorre los tramos de comparecientes (uso=ANALISIS) del inventario, por tiempo
    - filtro presencial: excluye video documental/externo, bloques, eventos
      estructurales (no es rostro presencial atribuible al compareciente)
    - N_FRAMES=8 muestras por ventana de 3s; guarda todos los frames detectados
    - etiqueta cada frame con la identidad del compareciente

VIDEOS:
    - Casanare, Catatumbo, Dabeiba, Huila: copia local en D:\\ (<4GB, caben FAT32)
    - Costa Caribe: en C:\\...\\corpus_c\\costa_caribe\\ (5.22GB, no cabe en D:).
      Nombre con caracteres especiales; el script busca el MP4 automaticamente
      en la carpeta si la ruta exacta no coincide.
    - Google Drive (G:) NO es legible por Python (symlinks): usar copia local.

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\calcular_aus_v3_multi.py" --subcaso CostaCaribe

Entorno: Python 3.11, conda env cfh. Requiere face_landmarker.task en la raiz.
================================================================================
"""

import cv2, json, argparse
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
MODEL = BASE / "face_landmarker.task"

# Config por subcaso: video local, diarizacion, inventario, audio_id.
SUBCASOS = {
    "Casanare": {
        "video": Path(r"D:\casanare_torres.mp4"),
        "diar":  BASE / "corpus_c" / "casanare_torres_diarization.json",
        "inv":   BASE / "data" / "marcacion" / "inventario_Casanare.csv",
        "audio": "casanare",
    },
    "Catatumbo": {
        "video": Path(r"D:\catatumbo.mp4"),
        "diar":  BASE / "corpus_c" / "catatumbo_diarization.json",
        "inv":   BASE / "data" / "marcacion" / "inventario_Catatumbo.csv",
        "audio": "catatumbo",
    },
    "Dabeiba": {
        "video": Path(r"D:\dabeiba.mp4"),
        "diar":  BASE / "corpus_c" / "dabeiba_antioquia_diarization.json",
        "inv":   BASE / "data" / "marcacion" / "inventario_Dabeiba.csv",
        "audio": "dabeiba",
    },
    "Huila": {
        "video": Path(r"D:\huila.mp4"),
        "diar":  BASE / "corpus_c" / "huila_diarization.json",
        "inv":   BASE / "data" / "marcacion" / "inventario_Huila.csv",
        "audio": "huila",
    },
    "CostaCaribe": {
        # Video en C: (5.22GB no cabe en D: FAT32). Nombre con caracteres
        # especiales -> busqueda automatica del MP4 en la carpeta (ver main).
        "video": BASE / "corpus_c" / "costa_caribe" / "costa_caribe.mp4",
        "diar":  BASE / "corpus_c" / "costa_caribe_diarization_v2.json",
        "inv":   BASE / "data" / "marcacion" / "inventario_CostaCaribe.csv",
        "audio": "costa_caribe",
    },
}

N_FRAMES = 8
MIN_DUR_SEG = 2.0


def blendshapes_a_au(bs):
    """Mapeo MediaPipe blendshapes -> Action Units (rango 0-1)."""
    g = lambda k: float(bs.get(k, 0.0))
    return {
        "AU1":  g("browInnerUp"),
        "AU4":  np.mean([g("browDownLeft"), g("browDownRight")]),
        "AU6":  np.mean([g("cheekSquintLeft"), g("cheekSquintRight")]),
        "AU12": np.mean([g("mouthSmileLeft"), g("mouthSmileRight")]),
        "AU15": np.mean([g("mouthFrownLeft"), g("mouthFrownRight")]),
        "AU17": g("mouthPucker"),
    }


def hms(s):
    s = str(s).strip()
    if ":" not in s:
        return None
    p = [int(x) for x in s.split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


def resolver_video(cfg):
    """Devuelve la ruta del video; si la exacta no existe, busca un .mp4 en la carpeta."""
    v = cfg["video"]
    if v.exists():
        return v
    carpeta = v.parent
    if carpeta.exists():
        mp4s = sorted(carpeta.glob("*.mp4"))
        if mp4s:
            print(f"  [auto] usando MP4 hallado en {carpeta.name}: {mp4s[0].name}")
            return mp4s[0]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subcaso", required=True, choices=list(SUBCASOS.keys()))
    args = ap.parse_args()
    cfg = SUBCASOS[args.subcaso]

    print(f"CFH -- Extraccion facial v3: {args.subcaso}")
    print("="*72)

    if not MODEL.exists():
        print(f"[ERROR] Falta el modelo: {MODEL}"); return

    cfg["video"] = resolver_video(cfg)
    if not cfg["video"].exists():
        print(f"[ERROR] No encuentro el video local: {cfg['video']}")
        print(f"  Verifica la carpeta del subcaso.")
        return
    print(f"  Video: {cfg['video']}")

    inv = pd.read_csv(cfg["inv"])
    comp = inv[inv["uso"] == "ANALISIS"].copy()

    # -- Filtrar SOLO intervenciones presenciales individuales (para facial) --
    # Excluir: eventos estructurales/bloques, y apariciones en video documental
    # o video externo (no es el rostro presencial atribuible al compareciente).
    def es_presencial(row):
        rol = str(row.get("rol", "")).upper()
        obs = str(row.get("observaciones", "")).lower()
        ident = str(row.get("identidad", "")).upper()
        if "EVENTO_ESTRUCTURAL" in rol:
            return False
        if "BLOQUE" in ident or "VIDEO" in ident:
            return False
        if "en video documental" in obs or "video_externo" in obs or "video externo" in obs:
            return False
        if "en bloque" in obs:
            return False
        return True

    n_antes = len(comp)
    comp = comp[comp.apply(es_presencial, axis=1)].copy()
    n_excluidos = n_antes - len(comp)
    if n_excluidos:
        print(f"  [filtro presencial] {n_excluidos} tramos de video/bloque excluidos del facial")
    comp["ini_s"] = comp["inicio"].apply(hms)
    comp["fin_s"] = comp["fin"].apply(hms)
    comp = comp[comp["ini_s"].notna() & comp["fin_s"].notna()]
    print(f"  Tramos presenciales individuales: {len(comp)}")

    raw = json.load(open(cfg["diar"], encoding="utf-8"))
    diar = raw if isinstance(raw, list) else raw.get("segments", list(raw.values())[0])

    def speaker_en(t):
        for d in diar:
            if float(d["start"]) <= t <= float(d["end"]):
                return d["speaker"]
        return "UNICO"

    base_opt = mp_python.BaseOptions(model_asset_path=str(MODEL))
    face_opt = mp_vision.FaceLandmarkerOptions(
        base_options=base_opt, output_face_blendshapes=True,
        num_faces=1, running_mode=mp_vision.RunningMode.IMAGE)

    cap = cv2.VideoCapture(str(cfg["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {fps:.0f} fps\n")

    filas = []
    with mp_vision.FaceLandmarker.create_from_options(face_opt) as lm:
        for _, r in comp.iterrows():
            ident = r["identidad"]; a0, a1 = r["ini_s"], r["fin_s"]
            n_det = n_try = 0
            t = a0
            while t < a1:
                seg0, seg1 = t, min(t + 3.0, a1)
                dur = seg1 - seg0
                if dur < MIN_DUR_SEG:
                    t += 3.0; continue
                for k in range(N_FRAMES):
                    tt = seg0 + dur * (k + 1) / (N_FRAMES + 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * fps))
                    ok, frame = cap.read()
                    n_try += 1
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = lm.detect(mimg)
                    if res.face_blendshapes:
                        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
                        au = blendshapes_a_au(bs)
                        au.update({"audio": cfg["audio"], "identidad": ident,
                                   "speaker": speaker_en(tt),
                                   "start": round(tt, 3),
                                   "end": round(tt + dur/N_FRAMES, 3),
                                   "duracion": round(dur, 3)})
                        filas.append(au); n_det += 1
                t += 3.0
            pct = 100*n_det/max(n_try, 1)
            print(f"    {ident[:34]:34s} -> {n_det} frames ({pct:.0f}%)")

    cap.release()

    if not filas:
        print("\n[AVISO] Sin deteccion facial en los tramos.")
        return

    df = pd.DataFrame(filas)
    cols = ["AU1","AU4","AU6","AU12","AU15","AU17","audio","identidad","speaker","start","end","duracion"]
    df = df[[c for c in cols if c in df.columns]]
    out = BASE / "outputs" / "capa3" / f"aus_{cfg['audio']}_v3.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[GUARDADO] {out}  ({len(df)} frames)")
    print(f"  (atribuido por tiempo+identidad, homogeneo con los otros subcasos)")
    print(f"\n  Cobertura por compareciente:")
    for ident, g in df.groupby("identidad"):
        print(f"    {ident[:34]:34s} {len(g):5d} frames")
    print(f"\n  Rangos AU (deben ser 0-1):")
    for au in ["AU1","AU4","AU12","AU15","AU17"]:
        if au in df.columns:
            print(f"    {au}: {df[au].min():.3f} - {df[au].max():.3f} (media {df[au].mean():.3f})")


if __name__ == "__main__":
    main()
