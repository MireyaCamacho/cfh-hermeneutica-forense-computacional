# -*- coding: utf-8 -*-
"""
cfh_retranscribir_v2.py
================================================================================
CFH — Re-transcripción Corpus C (faster-whisper) — VERSIÓN RAM-SEGURA

CORRIGE el MemoryError de la v1:
    clip_timestamps cargaba el WAV completo (9.5h → 5 GB en RAM). Esta versión
    RECORTA físicamente el audio en trozos de N minutos con ffmpeg ANTES de
    transcribir, así Whisper nunca carga más de N minutos a la vez. Es la
    arquitectura orquestador/worker para RAM limitada.

    Cada trozo se escribe a un WAV temporal pequeño en disco de trabajo (puede
    ser D: que tiene espacio), se transcribe, y se borra. RAM usada: solo el
    trozo actual.

SIRVE PARA Catatumbo Y Costa Caribe (ambos dañados). Cambia --wav y --salida.

ENTRADAS:
    --wav    ruta al WAV 16k (ej. G:\\Mi unidad\\CFH_wavs\\catatumbo_16k.wav)
    --tmp    carpeta de trabajo para trozos (recomendado en D: con espacio)

SALIDA:
    corpus_c/<nombre>_retranscrito_segments.json

USO (Catatumbo):
    python cfh_retranscribir_v2.py ^
        --wav "G:\\Mi unidad\\CFH_wavs\\catatumbo_16k.wav" ^
        --nombre catatumbo ^
        --tmp "D:\\cfh_tmp_audio"

USO (Costa Caribe, cuando toque):
    python cfh_retranscribir_v2.py ^
        --wav "<ruta al wav de costa caribe>" ^
        --nombre costa_caribe ^
        --tmp "D:\\cfh_tmp_audio"

REQUISITOS:
    · faster-whisper instalado
    · ffmpeg en PATH  (comprueba con: ffmpeg -version)
      Si no está: descarga de https://www.gyan.dev/ffmpeg/builds/ y añade a PATH,
      o instala con: winget install ffmpeg

Entorno: Python 3.11, conda env cfh, Windows 11.
================================================================================
"""

import argparse
import json
import gc
import subprocess
import sys
from pathlib import Path

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"


def duracion_wav(ruta):
    try:
        import soundfile as sf
        info = sf.info(str(ruta))
        return info.frames / info.samplerate
    except Exception:
        import wave
        with wave.open(str(ruta), "rb") as w:
            return w.getnframes() / w.getframerate()


def recortar_ffmpeg(wav_in, ini_s, dur_s, wav_out):
    """Recorta [ini_s, ini_s+dur_s] del wav usando ffmpeg. Sin recodificar info extra."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(ini_s), "-t", str(dur_s),
        "-i", str(wav_in),
        "-ar", "16000", "-ac", "1",
        str(wav_out),
    ]
    subprocess.run(cmd, check=True)


def verificar_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--nombre", required=True, help="prefijo de salida, ej. catatumbo")
    ap.add_argument("--tmp", default=r"D:\cfh_tmp_audio", help="carpeta para trozos temporales")
    ap.add_argument("--chunk-min", type=int, default=10)
    ap.add_argument("--reanudar", action="store_true")
    args = ap.parse_args()

    print("CFH — Re-transcripción RAM-segura (recorte físico con ffmpeg)")
    print("Anti-bucle: condition_on_previous_text=False + VAD\n")

    wav = Path(args.wav)
    if not wav.exists():
        print(f"[ERROR] No existe el WAV: {wav}")
        sys.exit(1)
    if not verificar_ffmpeg():
        print("[ERROR] ffmpeg no está en el PATH.")
        print("  Instálalo: winget install ffmpeg")
        print("  o descárgalo de https://www.gyan.dev/ffmpeg/builds/ y añádelo al PATH.")
        sys.exit(1)

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    out_dir = base / "corpus_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_json_dir = out_dir / f"_{args.nombre}_chunks"
    chunk_json_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"{args.nombre}_retranscrito_segments.json"

    dur = duracion_wav(wav)
    chunk_s = args.chunk_min * 60
    n_chunks = int(dur // chunk_s) + 1
    print(f"WAV: {wav}  ({dur/3600:.2f} h)")
    print(f"Trozos: {args.chunk_min} min → {n_chunks} | temporales en: {tmp}")
    print("="*72)

    from faster_whisper import WhisperModel
    print("Cargando modelo large-v3 (1ª vez tarda)...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("Modelo cargado.\n")

    todos = []
    for i in range(n_chunks):
        ini = i * chunk_s
        if ini >= dur:
            break
        dur_chunk = min(chunk_s, dur - ini)
        chunk_json = chunk_json_dir / f"chunk_{i:03d}.json"

        if args.reanudar and chunk_json.exists():
            print(f"  chunk {i:03d} [{ini/60:.0f}–{(ini+dur_chunk)/60:.0f} min] ya hecho.")
            todos.extend(json.load(open(chunk_json, encoding="utf-8")))
            continue

        wav_trozo = tmp / f"{args.nombre}_chunk_{i:03d}.wav"
        print(f"  chunk {i:03d} [{ini/60:.0f}–{(ini+dur_chunk)/60:.0f} min] recortando...", flush=True)
        try:
            recortar_ffmpeg(wav, ini, dur_chunk, wav_trozo)
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR ffmpeg] {e}")
            continue

        print(f"             transcribiendo...", flush=True)
        segments, info = model.transcribe(
            str(wav_trozo),
            language="es",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            condition_on_previous_text=False,   # CLAVE anti-bucle "y y y y"
        )
        # offset: los tiempos del trozo empiezan en 0, hay que sumarle ini.
        chunk_segs = []
        for s in segments:
            chunk_segs.append({"start": round(s.start + ini, 3),
                               "end": round(s.end + ini, 3),
                               "text": s.text})
        json.dump(chunk_segs, open(chunk_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        todos.extend(chunk_segs)
        print(f"             → {len(chunk_segs)} seg. Total: {len(todos)}")

        # limpiar: borrar el trozo de audio y liberar RAM.
        try:
            wav_trozo.unlink()
        except OSError:
            pass
        gc.collect()

    # Consolidar.
    todos.sort(key=lambda x: x["start"])
    json.dump(todos, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n{'='*72}")
    print(f"[GUARDADO] {out_json}  ({len(todos)} segmentos)")

    # Control de calidad.
    vacios = sum(1 for s in todos if len(str(s["text"]).strip()) < 2)
    mono = sum(1 for s in todos
               if len(str(s["text"]).split()) >= 5 and
               sum(1 for p in str(s["text"]).split() if len(p) <= 1) / max(1,len(str(s["text"]).split())) > 0.6)
    pct = 100.0 * (vacios + mono) / max(1, len(todos))
    print(f"Control de calidad: {pct:.1f}% dañado "
          f"({'LIMPIA ✓' if pct < 5 else 'revisar'})")
    print(f"\nSi quedó limpia, esta transcripción reemplaza a la dañada para el verbal.")


if __name__ == "__main__":
    main()
