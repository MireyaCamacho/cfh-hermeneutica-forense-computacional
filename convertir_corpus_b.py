"""
CFH · Conversión Corpus B — PDFs JEP a TXT
Usa pdfminer.six para extracción de texto de PDFs judiciales
"""
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

ORIGEN = Path("data/raw/corpus_b")
DESTINO = Path("data/processed/corpus_b")
DESTINO.mkdir(parents=True, exist_ok=True)

pdfs = list(ORIGEN.glob("*.pdf"))
print(f"PDFs encontrados: {len(pdfs)}")
print()

convertidos = 0
errores = []

laparams = LAParams(
    line_margin=0.5,
    word_margin=0.1,
    char_margin=2.0,
    boxes_flow=0.5,
)

for pdf in pdfs:
    destino_txt = DESTINO / (pdf.stem + ".txt")
    if destino_txt.exists():
        print(f"  ⏭ Ya existe: {pdf.name}")
        continue

    try:
        texto = extract_text(str(pdf), laparams=laparams)
        if texto and len(texto.strip()) > 100:
            destino_txt.write_text(texto, encoding="utf-8")
            kb = len(texto) / 1024
            print(f"  ✓ {pdf.name} ({kb:.0f} KB texto)")
            convertidos += 1
        else:
            errores.append(f"{pdf.name}: texto vacío o muy corto")
            print(f"  ✗ {pdf.name}: texto insuficiente — PDF escaneado sin OCR")
    except Exception as e:
        errores.append(f"{pdf.name}: {e}")
        print(f"  ✗ {pdf.name}: {e}")

print()
print(f"Convertidos: {convertidos}/{len(pdfs)}")
if errores:
    for e in errores:
        print(f"  ⚠ {e}")