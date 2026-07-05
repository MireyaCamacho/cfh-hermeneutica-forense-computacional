"""
CFH — Whisper CPU por CHUNKS (evita error de memoria RAM)
==========================================================
Transcribe un WAV largo partiéndolo en bloques de 30 min con ffmpeg,
transcribiendo cada bloque por separado (poca RAM por bloque) y uniendo
todos los segmentos al final con offset de tiempo correcto.

Soluciona: numpy._core._exceptions._ArrayMemoryError (RAM insuficiente
al cargar audio completo de 7h+ de una sola vez).

Robusto para dejar de noche:
  - Guarda cada chunk transcrito en disco → si se corta, retoma donde quedó
  - Borra el WAV temporal de cada chunk tras transcribirlo (ahorra espacio)
  - Imprime avance por chunk con marca de tiempo

Uso:
  conda activate cfh
  python code/cfh_whisper_chunks.py corpus_c/obs_barranquilla_dia1.wav
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

MODELO        = "large-v3"
IDIOMA        = "es"
CHUNK_MIN     = 30          # minutos por bloque (menos RAM = bajar a 20 o 15)
SR            = 16000


def duracion_wav(wav):
    """Duración en segundos vía ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
        capture_output=True, text=True)
    return float(r.stdout.strip())


def cortar_chunk(wav, inicio_s, dur_s, salida):
    """Extrae un trozo del WAV con ffmpeg (rápido, sin recodificar el pcm)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav),
         "-ss", str(inicio_s), "-t", str(dur_s),
         "-ar", str(SR), "-ac", "1", str(salida)],
        check=True)


def main():
    if len(sys.argv) < 2:
        print("Uso: python code/cfh_whisper_chunks.py <ruta_wav>")
        sys.exit(1)

    wav = Path(sys.argv[1])
    if not wav.exists():
        print(f"x No existe el WAV: {wav}")
        sys.exit(1)

    out_json   = wav.with_name(wav.stem + "_segments.json")
    chunks_dir = wav.with_name(wav.stem + "_chunks")
    chunks_dir.mkdir(exist_ok=True)

    from faster_whisper import WhisperModel

    print("=" * 60)
    print(f"CFH — Whisper CPU por chunks | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    dur_total = duracion_wav(wav)
    chunk_s   = CHUNK_MIN * 60
    n_chunks  = int(dur_total // chunk_s) + (1 if dur_total % chunk_s else 0)

    print(f"  WAV:      {wav.name}")
    print(f"  Duración: {dur_total/3600:.2f} horas")
    print(f"  Chunks:   {n_chunks} de {CHUNK_MIN} min c/u")
    print(f"  Modelo:   {MODELO} (CPU, int8)\n")

    print(f"  Cargando modelo...")
    model = WhisperModel(MODELO, device="cpu", compute_type="int8")
    print(f"  ✓ Modelo cargado\n")

    t0 = time.time()
    todos_segmentos = []

    for idx in range(n_chunks):
        inicio_s = idx * chunk_s
        dur_s    = min(chunk_s, dur_total - inicio_s)
        parcial_chunk = chunks_dir / f"chunk_{idx:03d}.json"

        # Si ya está transcrito (corrida previa interrumpida), retomar
        if parcial_chunk.exists():
            with open(parcial_chunk, encoding="utf-8") as f:
                segs_chunk = json.load(f)
            todos_segmentos.extend(segs_chunk)
            print(f"  [{idx+1}/{n_chunks}] ya existe — retomado ({len(segs_chunk)} segs)")
            continue

        # Cortar el trozo de audio
        wav_chunk = chunks_dir / f"chunk_{idx:03d}.wav"
        cortar_chunk(wav, inicio_s, dur_s, wav_chunk)

        # Transcribir el trozo
        seg_iter, info = model.transcribe(
            str(wav_chunk),
            language=IDIOMA,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
        )

    