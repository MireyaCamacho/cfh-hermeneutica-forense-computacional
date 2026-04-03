"""
CFH · OCR del Auto 005/2018 — PDF escaneado
"""
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF = Path("data/raw/corpus_b/Auto_005_2018_Apertura_Caso003.pdf")
DESTINO = Path("data/processed/corpus_b/Auto_005_2018_Apertura_Caso003.txt")

print(f"Procesando: {PDF.name}")
print("Convirtiendo páginas a imágenes...")

paginas = convert_from_path(str(PDF), dpi=300, fmt="jpeg")
print(f"Páginas encontradas: {len(paginas)}")

texto_total = []
for i, pagina in enumerate(paginas, 1):
    print(f"  OCR página {i}/{len(paginas)}...", end="\r")
    texto = pytesseract.image_to_string(pagina, lang="spa")
    texto_total.append(texto)

texto_final = "\n".join(texto_total)
DESTINO.write_text(texto_final, encoding="utf-8")

print(f"\n✓ Completado: {len(texto_final)/1024:.0f} KB extraídos")
print(f"  Guardado en: {DESTINO}")