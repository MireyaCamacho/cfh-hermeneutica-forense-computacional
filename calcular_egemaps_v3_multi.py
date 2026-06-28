# -*- coding: utf-8 -*-
"""
calcular_egemaps_v3_multi.py
================================================================================
CFH — Re-extracción VOCAL (eGeMAPS) v3 parametrizada por subcaso

OBJETO:
    Clonar EXACTAMENTE la arquitectura del facial v3 (calcular_aus_v3_multi.py)
    pero para el canal vocal. Arregla los defectos del notebook viejo de Colab:
      · NO atribuye por SPEAKER de pyannote (que mezcla comparecientes) — recorre
        los TRAMOS del inventario por TIEMPO, igual que el facial.
      · Etiqueta cada ventana con la IDENTIDAD del compareciente.
      · Filtra video/bloques (mismo filtro presencial del facial).
      · Procesa los 5 subcasos, INCLUIDO Costa Caribe (ya no hay DRM).
      · Guarda las 88 features eGeMAPS crudas por ventana + identidad, para que
        cfh_icm_tricanal_final.py aplique su ESCALA FIJA de literatura.

    NO calcula aquí el ICM vocal (eso lo hace el script final con escala absoluta).
    Solo extrae features homogéneas y atribuidas correctamente.

REQUISITOS:
    pip install opensmile soundfile  (en el env cfh)
    ffmpeg en el PATH (para extraer WAV del MP4 si no existe)

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\calcular_egemaps_v3_multi.py" --subcaso Casanare

    (correr uno a uno, igual que el facial; el más pesado es Huila)

Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import argparse, subprocess, os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
# WAV temporales en D: (C: casi lleno; los WAV son temporales para extraer eGeMAPS).
# D: es FAT32 pero los WAV (<2 GB) caben bajo el límite de 4 GB.
WAV_DIR = Path(r"D:\cfh_wavs")
OUT_DIR = BASE / "outputs" / "capa3"

# Config por subcaso: mismos videos locales (D:) que usó el facial.
# 'audio' = prefijo de salida (egemap_<audio>_compareciente.csv)
SUBCASOS = {
    "Casanare": {
        "video": Path(r"D:\casanare_torres.mp4"),
        "inv":   BASE / "data" / "marcacion" / "inventario_Casanare.csv",
        "audio": "casanare",
    },
    "Catatumbo": {
        "video": Path(r"D:\catatumbo.mp4"),
        "inv":   BASE / "data" / "marcacion" / "inventario_Catatumbo.csv",
        "audio": "catatumbo",
    },
    "Dabeiba": {
        "video": Path(r"D:\dabeiba.mp4"),
        "inv":   BASE / "data" / "marcacion" / "inventario_Dabeiba.csv",
        "audio": "dabeiba",
    },
    "Huila": {
        "video": Path(r"D:\huila.mp4"),
        "inv":   BASE / "data" / "marcacion" / "inventario_Huila.csv",
        "audio": "huila",
    },
    "CostaCaribe": {
        "video": BASE / "corpus_c" / "costa_caribe" / "Caso 03｜ Audiencia de Reconocimiento Subcaso Costa Caribe ｜ 18 de julio de 2022.mp4",
        "inv":   BASE / "data" / "marcacion" / "inventario_CostaCaribe.csv",
        "audio": "costa_caribe",
    },
}

# Ventana de análisis prosódico. eGeMAPS necesita >2-3s para shimmer/jitter
# confiables. Ventanas de 3s con paso 3s (no solapadas), igual densidad temporal
# que el facial (que muestrea cada 3s).
VENTANA_S = 3.0
PASO_S = 3.0
MIN_DUR_VENTANA = 2.0


def hms(s):
    s = str(s).strip()
    if ":" not in s:
        return None
    p = [int(x) for x in s.split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


def es_presencial(row):
    """Mismo filtro que el facial: excluye video/bloque/estructural."""
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


def asegurar_wav(video_path, audio_id):
    """Devuelve la ruta al WAV 16k mono; lo extrae del MP4 si no existe."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    wav = WAV_DIR / f"{audio_id}_16k.wav"
    if wav.exists() and wav.stat().st_size > 1024:
        print(f"  WAV ya existe: {wav.name}")
        return wav
    if not video_path.exists():
        print(f"[ERROR] No existe el video local: {video_path}")
        print(f"  Cópialo de G:\\Mi unidad\\CFH_videos\\ a esa ruta primero.")
        return None
    print(f"  Extrayendo WAV 16k de {video_path.name} (puede tardar)...")
    cmd = ["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1",
           "-vn", str(wav), "-y", "-loglevel", "error"]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not wav.exists():
        print(f"[ERROR] ffmpeg fallo extrayendo el WAV.")
        return None
    print(f"  WAV creado: {wav.name} ({wav.stat().st_size//(1024*1024)} MB)")
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subcaso", required=True, choices=list(SUBCASOS.keys()))
    args = ap.parse_args()
    cfg = SUBCASOS[args.subcaso]

    print(f"CFH — Re-extracción VOCAL eGeMAPS v3: {args.subcaso}")
    print("="*72)

    # ── imports pesados aquí para fallar limpio si falta algo ──
    try:
        import opensmile
        import soundfile as sf
    except ImportError as e:
        print(f"[ERROR] Falta una librería: {e}")
        print("  Instala:  pip install opensmile soundfile")
        return

    # ── 1. WAV ──
    # Si la ruta exacta del video no existe (p.ej. Costa Caribe con caracteres
    # raros en el nombre), buscar cualquier .mp4 en la carpeta del video.
    if not cfg["video"].exists():
        carpeta = cfg["video"].parent
        if carpeta.exists():
            mp4s = sorted(carpeta.glob("*.mp4"))
            if len(mp4s) == 1:
                print(f"  [auto] usando el único MP4 en {carpeta.name}: {mp4s[0].name}")
                cfg["video"] = mp4s[0]
            elif len(mp4s) > 1:
                print(f"  [aviso] varios MP4 en {carpeta.name}; usando el primero: {mp4s[0].name}")
                cfg["video"] = mp4s[0]
    wav = asegurar_wav(cfg["video"], cfg["audio"])
    if wav is None:
        return

    # ── 2. Inventario → tramos presenciales individuales (igual que facial) ──
    inv = pd.read_csv(cfg["inv"])
    comp = inv[inv["uso"] == "ANALISIS"].copy()
    n_antes = len(comp)
    comp = comp[comp.apply(es_presencial, axis=1)].copy()
    n_excl = n_antes - len(comp)
    if n_excl:
        print(f"  [filtro presencial] {n_excl} tramos de video/bloque excluidos del vocal")
    comp["ini_s"] = comp["inicio"].apply(hms)
    comp["fin_s"] = comp["fin"].apply(hms)
    comp = comp[comp["ini_s"].notna() & comp["fin_s"].notna()]
    print(f"  Tramos presenciales individuales: {len(comp)}")

    # ── 3. OpenSMILE eGeMAPS v02 Functionals ──
    print(f"  Cargando audio (16k) con soundfile...")
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:          # estéreo → mono
        audio = audio.mean(axis=1)
    if sr != 16000:
        print(f"  [AVISO] sr={sr}, esperaba 16000.")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print(f"  OpenSMILE eGeMAPS v02 — {len(smile.feature_names)} features\n")

    # ── 4. Recorrer tramos, extraer features por ventana, etiquetar identidad ──
    filas = []
    for _, r in comp.iterrows():
        ident = r["identidad"]; a0, a1 = r["ini_s"], r["fin_s"]
        n_vent = 0
        t = a0
        while t < a1:
            seg0, seg1 = t, min(t + VENTANA_S, a1)
            dur = seg1 - seg0
            if dur < MIN_DUR_VENTANA:
                t += PASO_S; continue
            s0, s1 = int(seg0 * sr), int(seg1 * sr)
            seg_audio = audio[s0:s1]
            # saltar silencio casi total (energía muy baja)
            if seg_audio.size == 0 or float(np.sqrt(np.mean(seg_audio**2))) < 1e-4:
                t += PASO_S; continue
            try:
                feats = smile.process_signal(seg_audio, sr)
                d = feats.iloc[0].to_dict()
                # NOTA METODOLÓGICA: se CONSERVAN las ventanas de silencio
                # (VoicedSegmentsPerSec=0). Verificado empíricamente que son
                # silencio genuino (loudness ~12x menor, sin segmentos sonoros),
                # pero el silencio/pausa ES parte del acto de reconocimiento
                # (Zehr 2002; Baird & Coutinho 2019: las pausas marcan el perdón
                # sincero). Excluirlas borraría señal prosódica teóricamente
                # relevante. Decisión informada: conservar.
                d.update({"audio": cfg["audio"], "identidad": ident,
                          "start": round(seg0, 3), "end": round(seg1, 3),
                          "duracion": round(dur, 3)})
                filas.append(d); n_vent += 1
            except Exception:
                pass
            t += PASO_S
        print(f"    {ident[:34]:34s} → {n_vent} ventanas")

    if not filas:
        print("\n[AVISO] Sin ventanas vocales extraídas.")
        return

    df = pd.DataFrame(filas)
    # poner identidad/start/end al frente para legibilidad
    front = ["audio", "identidad", "start", "end", "duracion"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]
    out = OUT_DIR / f"egemap_{cfg['audio']}_compareciente.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[GUARDADO] {out}  ({len(df)} ventanas)")
    print(f"  (atribuido por tiempo+identidad, igual que el facial v3)")

    # Diagnóstico rápido de los 3 marcadores que usa el ICM (escala cruda)
    print(f"\n  Marcadores clave (crudos, sin normalizar):")
    for col, lab in [("shimmerLocaldB_sma3nz_amean", "shimmer dB"),
                     ("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "F0 stddevNorm"),
                     ("HNRdBACF_sma3nz_amean", "HNR dB")]:
        if col in df.columns:
            print(f"    {lab:16s}: media={df[col].mean():.3f} "
                  f"[{df[col].min():.3f} – {df[col].max():.3f}]")
    print(f"\n  Cobertura por compareciente (ventanas):")
    vc = df.groupby("identidad").size().sort_values(ascending=False)
    for ident, n in vc.items():
        marca = "" if n >= 10 else "  ⚠ <10 (bajo piso vocal)"
        print(f"    {ident[:34]:34s} {n}{marca}")


if __name__ == "__main__":
    main()
