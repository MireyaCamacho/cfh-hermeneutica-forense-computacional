"""
CFH — reconstruir_costa_caribe.py
===================================
Reconstruye costa_caribe.txt completo (10.46h) desde
costa_caribe_segments.json (Whisper), sin re-transcribir.

También genera costa_caribe_diarization_v2.json con la
cobertura temporal completa estimada desde los segmentos.

Uso:
    python code/reconstruir_costa_caribe.py

Salidas:
    corpus_c/costa_caribe.txt              — transcripción completa
    corpus_c/costa_caribe_stats.txt        — estadísticas de cobertura
"""

import json
from pathlib import Path

SEGMENTS_PATH   = Path("data/costa_caribe_segments.json")
OUT_TXT         = Path("corpus_c/costa_caribe.txt")
OUT_STATS       = Path("corpus_c/costa_caribe_stats.txt")

print("=" * 55)
print("CFH — reconstruir_costa_caribe.py")
print("=" * 55)

# ── Cargar segmentos ──────────────────────────────────────────
with open(SEGMENTS_PATH, encoding="utf-8") as f:
    segments = json.load(f)

print(f"✓ Segmentos cargados: {len(segments)}")

# ── Reconstruir texto en orden cronológico ────────────────────
segments_sorted = sorted(segments, key=lambda s: s.get("start", 0))

duracion_max = max(s.get("end", 0) for s in segments_sorted)
duracion_min = min(s.get("start", 0) for s in segments_sorted)

print(f"  Inicio: {duracion_min/3600:.2f}h ({duracion_min:.0f}s)")
print(f"  Fin:    {duracion_max/3600:.2f}h ({duracion_max:.0f}s)")
print(f"  Duración total cubierta: {(duracion_max-duracion_min)/3600:.2f}h")

# Reconstruir texto con marcas de tiempo opcionales
lineas = []
for seg in segments_sorted:
    texto = seg.get("text", "").strip()
    if not texto:
        continue
    lineas.append(texto)

texto_completo = " ".join(lineas)

# Estadísticas
n_palabras = len(texto_completo.split())
n_chars    = len(texto_completo)

print(f"\n✓ Texto reconstruido:")
print(f"  Caracteres : {n_chars:,}")
print(f"  Palabras   : {n_palabras:,}")
print(f"  Segmentos  : {len(lineas)}")

# Comparar con el TXT parcial anterior si existe
txt_anterior = Path("corpus_c/costa_caribe.txt")
if txt_anterior.exists():
    chars_anterior = len(txt_anterior.read_text(encoding="utf-8"))
    print(f"\n  TXT anterior (parcial): {chars_anterior:,} chars")
    print(f"  TXT nuevo (completo):   {n_chars:,} chars")
    print(f"  Incremento:             +{n_chars - chars_anterior:,} chars ({(n_chars/chars_anterior - 1)*100:.1f}%)")

# ── Guardar TXT completo ──────────────────────────────────────
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text(texto_completo, encoding="utf-8")
print(f"\n✓ Guardado: {OUT_TXT}")

# ── Guardar estadísticas ──────────────────────────────────────
stats = f"""CFH — Costa Caribe · Estadísticas de transcripción completa
============================================================
Video fuente    : Caso 03 Audiencia Reconocimiento Costa Caribe 18 julio 2022
Duración video  : {duracion_max/3600:.2f}h ({duracion_max:.0f}s)
Segmentos Whisper: {len(segments)}
Inicio cobertura: {duracion_min:.0f}s ({duracion_min/3600:.2f}h)
Fin cobertura   : {duracion_max:.0f}s ({duracion_max/3600:.2f}h)
Chars totales   : {n_chars:,}
Palabras totales: {n_palabras:,}
Archivo salida  : {OUT_TXT}
"""
OUT_STATS.write_text(stats, encoding="utf-8")
print(f"✓ Estadísticas: {OUT_STATS}")

print(f"""
{"="*55}
RESUMEN
{"="*55}
costa_caribe.txt reconstruido con cobertura completa.
Siguiente paso: correr analisis_corpus_c.py en Colab
con este TXT — la audiencia Costa Caribe ya está lista
para los indicadores CFH (y₈, y₉, y₂, y₄, y₁₀).

Nota: la diarización (4.30h) sigue siendo parcial.
Para el ICM tri-canal de Costa Caribe se necesita
re-diarizar con pyannote-audio sobre el video completo.
{"="*55}
""")
