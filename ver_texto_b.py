import sys, re
sys.path.insert(0, 'code/src')
from pathlib import Path

texto = Path('data/processed/corpus_b/Auto_125_2021_Norte_Santander.txt').read_text(
    encoding='utf-8', errors='replace'
)
print(f"Total chars: {len(texto)}")
print()
print("=== PRIMEROS 2000 CHARS ===")
print(texto[:2000])
print()
print("=== TITULOS DE SECCIÓN ENCONTRADOS ===")
patrones = [
    'DETERMINACI', 'HECHOS Y CONDUCTAS', 'PATRONES', 'CALIFICACI',
    'RECONOCIMIENTO', 'RESUELVE', 'CONSIDERACIONES', 'ANTECEDENTES',
    'SUBCASO', 'CASO 03', 'ASESINATOS'
]
for p in patrones:
    matches = [m.start() for m in re.finditer(p, texto, re.IGNORECASE)]
    if matches:
        pos = matches[0]
        print(f"\n'{p}' en pos {pos}:")
        print(repr(texto[max(0,pos-30):pos+100]))