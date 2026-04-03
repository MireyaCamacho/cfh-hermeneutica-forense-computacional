"""
Convierte doc/docx/html del corpus A a .txt para el pipeline CFH
"""
from pathlib import Path
from bs4 import BeautifulSoup
import subprocess


ORIGEN = Path(r"C:\Users\LENOVO\OneDrive - Universidad Externado de Colombia\CIENCIA DE DATOS\PROYECTO CIENCIA DE DATOS\FALSOS POSITIVOS\SENTENCIAS CSJ\SENTENCIAS CSJ")
DESTINO = Path(r"data\raw\corpus_a")
DESTINO.mkdir(parents=True, exist_ok=True)
LIBREOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"

def convertir_html(path):
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    return soup.get_text(separator="\n", strip=True)

def convertir_docx(path):
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

convertidos = 0
errores = []

for archivo in ORIGEN.iterdir():
    ext = archivo.suffix.lower()
    if ext not in (".doc", ".docx", ".html"):
        continue

    destino_txt = DESTINO / (archivo.stem + ".txt")
    if destino_txt.exists():
        continue

    try:
        if ext == ".html":
            texto = convertir_html(archivo)
            destino_txt.write_text(texto, encoding="utf-8")
            convertidos += 1
            print(f"  ✓ {archivo.name}")

        elif ext == ".docx":
            texto = convertir_docx(archivo)
            destino_txt.write_text(texto, encoding="utf-8")
            convertidos += 1
            print(f"  ✓ {archivo.name}")

        elif ext == ".doc":
            result = subprocess.run([
                LIBREOFFICE, "--headless", "--convert-to", "txt:Text",
                "--outdir", str(DESTINO), str(archivo)
            ], capture_output=True, timeout=60)
            if result.returncode == 0:
                convertidos += 1
                print(f"  ✓ {archivo.name}")
            else:
                errores.append(archivo.name)
                print(f"  ✗ {archivo.name}")

    except Exception as e:
        errores.append(f"{archivo.name}: {e}")
        print(f"  ✗ {archivo.name}: {e}")

print(f"\nConvertidos: {convertidos}")
print(f"Errores: {len(errores)}")