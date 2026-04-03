import sys, re, logging
sys.path.insert(0, 'code/src')
logging.disable(logging.CRITICAL)

from ingestion.cleaner import JudicialTextCleaner
from pathlib import Path

cleaner = JudicialTextCleaner()
texto_raw = Path('data/processed/corpus_b/Auto_125_2021_Norte_Santander.txt').read_text(
    encoding='utf-8', errors='replace'
)
resultado = cleaner.clean(texto_raw)
texto_limpio = resultado.text


print(f"Chars antes: {len(texto_raw)} | después: {len(texto_limpio)}")
print()
print("=== PRIMEROS 1500 CHARS POST-CLEANER ===")
print(texto_limpio[:1500])
print()
print("=== LÍNEAS EN MAYÚSCULAS (posibles títulos) ===")
for i, linea in enumerate(texto_limpio.split('\n')):
    l = linea.strip()
    if len(l) > 3 and l == l.upper() and len(l) < 80:
        print(f"  línea {i:4d}: {repr(l)}")