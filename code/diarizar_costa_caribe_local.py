"""
CFH — diarizar_costa_caribe_local.py v3
Procesa en chunks de 2h para evitar OOM de RAM
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import pandas as pd
from pyannote.audio import Pipeline

HF_TOKEN = "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"  # pasar como argumento o variable de entorno
WAV_PATH  = Path("corpus_c/costa_caribe_completo.wav")
OUT_PATH  = Path("corpus_c/costa_caribe_diarization_v2.json")
CKPT_PATH = Path("corpus_c/diarization_chunks_ckpt.json")

CHUNK_H   = 2.0   # horas por chunk
SR        = 16000

print("=" * 55)
print("CFH - Diarizacion Costa Caribe (chunks 2h)")
print("=" * 55)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", token=HF_TOKEN
).to(torch.device("cpu"))
print("Pipeline cargado")

# Cargar audio completo con soundfile
print("Cargando audio...")
data, sr_orig = sf.read(str(WAV_PATH), dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)

# Resamplear si es necesario
if sr_orig != SR:
    import torchaudio
    t = torch.tensor(data).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr_orig, SR)
    data = t.squeeze(0).numpy()

duracion_s = len(data) / SR
duracion_h = duracion_s / 3600
print(f"Audio: {duracion_h:.2f}h ({len(data)} samples)")

# Calcular chunks
chunk_samples = int(CHUNK_H * 3600 * SR)
n_chunks = int(np.ceil(len(data) / chunk_samples))
print(f"Chunks: {n_chunks} x {CHUNK_H}h")

# Cargar checkpoint si existe
inicio_chunk = 0
todos_segmentos = []
if CKPT_PATH.exists():
    with open(CKPT_PATH) as f:
        ckpt = json.load(f)
    todos_segmentos = ckpt["segmentos"]
    inicio_chunk    = ckpt["ultimo_chunk"] + 1
    print(f"Checkpoint: continuando desde chunk {inicio_chunk}")

# Procesar chunks
t_total = time.time()
for i in range(inicio_chunk, n_chunks):
    t0 = time.time()
    inicio_s  = i * chunk_samples
    fin_s     = min((i+1) * chunk_samples, len(data))
    offset_s  = inicio_s / SR

    chunk = data[inicio_s:fin_s]
    waveform = torch.tensor(chunk).unsqueeze(0)

    print(f"\nChunk {i+1}/{n_chunks}: {offset_s/3600:.2f}h - {fin_s/SR/3600:.2f}h "
          f"({len(chunk)/SR/60:.0f} min) — {time.strftime('%H:%M:%S')}")

    try:
        diarization = pipeline({"waveform": waveform, "sample_rate": SR})
        sd = diarization.speaker_diarization
        n_seg = 0
        for turn, _, speaker in sd.itertracks(yield_label=True):
            todos_segmentos.append({
                "start":    round(turn.start + offset_s, 3),
                "end":      round(turn.end   + offset_s, 3),
                "speaker":  speaker,
                "duracion": round(turn.end - turn.start, 3),
                "chunk":    i
            })
            n_seg += 1
        elapsed = time.time() - t0
        print(f"  {n_seg} segmentos en {elapsed/60:.1f} min")
    except Exception as e:
        print(f"  ERROR en chunk {i}: {e}")

    # Guardar checkpoint
    with open(CKPT_PATH, "w") as f:
        json.dump({"ultimo_chunk": i, "segmentos": todos_segmentos}, f)

# Resultado final
todos_segmentos.sort(key=lambda s: s["start"])
duracion_max = max(s["end"] for s in todos_segmentos)
speakers     = set(s["speaker"] for s in todos_segmentos)

print(f"\n{'='*55}")
print(f"COMPLETADO en {(time.time()-t_total)/3600:.1f}h")
print(f"Segmentos: {len(todos_segmentos)}")
print(f"Cobertura: {duracion_max/3600:.2f}h")
print(f"Speakers:  {len(speakers)}")

df = pd.DataFrame(todos_segmentos)
stats = df.groupby("speaker")["duracion"].sum().sort_values(ascending=False)
print(f"\nTop 10 speakers (minutos):")
print((stats.head(10)/60).round(1).to_string())

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(todos_segmentos, f, indent=2)
print(f"\nGuardado: {OUT_PATH}")

# Limpiar checkpoint
CKPT_PATH.unlink(missing_ok=True)
