"""
CFH — Pipeline integrado: descarga + WAV + Whisper large-v3 (CPU, toda la noche)
=================================================================================
Hace TODO en una sola corrida para una audiencia de YouTube:
  1. Descarga el video (yt-dlp formato 18) si no existe el WAV
  2. Extrae audio a WAV 16kHz mono (ffmpeg)
  3. Borra el .mp4 para liberar espacio
  4. Transcribe con faster-whisper large-v3 en CPU (máxima calidad)
  5. Guarda <nombre>_segments.json en corpus_c/ (formato del pipeline CFH)

Robusto para dejar de noche: progreso parcial cada 100 segmentos, ETA, VAD.

REQUISITOS: yt-dlp, ffmpeg en PATH, faster-whisper instalado.

Uso:
  conda activate cfh
  python code/cfh_pipeline_audiencia.py "URL_YOUTUBE" obs_barranquilla_dia1

  (el 2º argumento es el nombre base; el output será obs_barranquilla_dia1_segments.json)

Si ya tienes el WAV descargado, igual funciona: detecta el WAV y salta a transcribir.
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────────
MODELO       = "large-v3"   # máxima calidad (no sacrificar calidad)
IDIOMA       = "es"
GUARDAR_CADA = 100          # segmentos entre guardados parciales

REPO     = Path(__file__).resolve().parent.parent
CORPUS_C = REPO / "corpus_c"
CORPUS_C.mkdir(exist_ok=True)


def run(cmd):
    """Ejecuta un comando y muestra salida en vivo."""
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    return r.returncode == 0


def paso_descarga(url, base):
    mp4 = CORPUS_C / f"{base}.mp4"
    wav = CORPUS_C / f"{base}.wav"

    if wav.exists():
        print(f"  = WAV ya existe ({wav.stat().st_size//(1024*1024)} MB) — salto descarga")
        return wav

    if not mp4.exists():
        print(f"\n[1/4] Descargando video de YouTube...")
        ok = run(["yt-dlp", "-f", "18", url, "-o", str(mp4)])
        if not ok or not mp4.exists():
            print("  x Error en la descarga"); sys.exit(1)
    else:
        print(f"  = MP4 ya existe — salto descarga")

    print(f"\n[2/4] Extrayendo audio a WAV 16kHz mono...")
    ok = run(["ffmpeg", "-y", "-i", str(mp4), "-ar", "16000", "-ac", "1", str(wav)])
    if not ok or not wav.exists():
        print("  x Error en ffmpeg"); sys.exit(1)

    print(f"\n[3/4] Borrando MP4 para liberar espacio...")
    try:
        mp4.unlink()
        print(f"  ✓ MP4 borrado")
    except Exception as e:
        print(f"  ! No se pudo borrar el MP4: {e}")

    return wav


def paso_whisper(wav):
    from faster_whisper import WhisperModel

    out_json = wav.with_name(wav.stem + "_segments.json")
    parcial  = wav.with_name(wav.stem + "_segments.parcial.json")

    print(f"\n[4/4] Transcripción Whisper CPU | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  WAV:    {wav.name}")
    print(f"  Modelo: {MODELO} (CPU, int8)")
    print(f"  Salida: {out_json.name}")
    print(f"  Cargando modelo (la primera vez descarga ~3GB)...")

    model = WhisperModel(MODELO, device="cpu", compute_type="int8")
    print(f"  ✓ Modelo cargado. Iniciando transcripción...\n")

    t0 = time.time()
    segmentos = []

    seg_iter, info = model.transcribe(
        str(wav),
        language=IDIOMA,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
    )
    print(f"  Duración detectada: {info.duration/3600:.2f} horas\n")

    for i, s in enumerate(seg_iter):
        segmentos.append({
            "start": round(s.start, 3),
            "end":   round(s.end, 3),
            "text":  s.text.strip(),
        })
        if (i + 1) % GUARDAR_CADA == 0:
            with open(parcial, "w", encoding="utf-8") as f:
                json.dump(segmentos, f, ensure_ascii=False)
            transcurrido = time.time() - t0
            pos = s.end
            vel = pos / transcurrido if transcurrido > 0 else 0
            rest = (info.duration - pos) / vel if vel > 0 else 0
            print(f"  [{datetime.now():%H:%M:%S}] seg {i+1} | "
                  f"audio {pos/3600:.2f}h/{info.duration/3600:.2f}h | "
                  f"ETA ~{rest/3600:.1f}h")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segmentos, f, ensure_ascii=False, indent=1)
    if parcial.exists():
        parcial.unlink()

    dur = time.time() - t0
    print(f"\n  ✓ Transcripción completada")
    print(f"    Segmentos: {len(segmentos)}")
    print(f"    Tiempo total: {dur/3600:.2f} horas")
    print(f"    → {out_json}")
    return out_json


def main():
    if len(sys.argv) < 3:
        print("Uso: python code/cfh_pipeline_audiencia.py \"URL_YOUTUBE\" <nombre_base>")
        print("Ej:  python code/cfh_pipeline_audiencia.py \"https://www.youtube.com/watch?v=fMKo4JE13UA\" obs_barranquilla_dia1")
        sys.exit(1)

    url  = sys.argv[1]
    base = sys.argv[2]
    if base.endswith("_segments"):
        base = base[:-9]

    print("=" * 60)
    print(f"CFH — Pipeline audiencia: {base}")
    print("=" * 60)

    wav = paso_descarga(url, base)
    out = paso_whisper(wav)

    print(f"\n[CFH] LISTO. Siguiente paso (mañana):")
    print(f"  python code/cfh_descargar_y_extraer_v5.py")
    print(f"  python code/cfh_centroide_mafapo_v5_final.py")


if __name__ == "__main__":
    main()
