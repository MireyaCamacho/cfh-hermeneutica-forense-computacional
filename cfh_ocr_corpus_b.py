"""
CFH — OCR y limpieza metadata Corpus B
========================================
1. OCR de auto-RC-AI-016-2025-dabeiba.pdf con pytesseract
2. Corrección etiqueta Auto_128 en metadata
3. Deduplicación adhc-055 / Auto_055

Ejecutar:
  python cfh_ocr_corpus_b.py
"""
import os
import pandas as pd
from pathlib import Path

REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
RAW_B = REPO / "data" / "raw" / "corpus_b"
PROC_B = REPO / "data" / "processed" / "corpus_b"
META = REPO / "data" / "metadata_corpus_b.csv"

print("="*60)
print("CFH — OCR y limpieza Corpus B")
print("="*60)

# ── PASO 1: OCR del PDF de Dabeiba 2025 ──────────────────────────
PDF_OCR = RAW_B / "auto-RC-AI-016-2025-dabeiba.pdf"
TXT_OCR = PROC_B / "auto-RC-AI-016-2025-dabeiba.txt"

if PDF_OCR.exists() and not TXT_OCR.exists():
    print(f"\n[1] OCR: {PDF_OCR.name}")
    print(f"    Tamaño: {PDF_OCR.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        print("    Convirtiendo PDF a imágenes...")
        pages = convert_from_path(str(PDF_OCR), dpi=200)
        print(f"    {len(pages)} páginas detectadas")
        
        texto_total = []
        for i, page in enumerate(pages, 1):
            if i % 50 == 0:
                print(f"    Procesando página {i}/{len(pages)}...")
            texto = pytesseract.image_to_string(page, lang='spa')
            texto_total.append(texto)
        
        texto_final = "\n\n".join(texto_total)
        TXT_OCR.write_text(texto_final, encoding='utf-8')
        print(f"    ✓ OCR completado: {len(texto_final):,} chars → {TXT_OCR.name}")
        
    except ImportError as e:
        print(f"    ✗ Librería no disponible: {e}")
        print("    Instalar: pip install pdf2image pytesseract")
        print("    También necesitas: poppler-utils para pdf2image")
else:
    if TXT_OCR.exists():
        print(f"\n[1] OCR ya existe: {TXT_OCR.name} ({TXT_OCR.stat().st_size:,} bytes)")
    else:
        print(f"\n[1] PDF no encontrado: {PDF_OCR}")

# ── PASO 2: Auto_005 — declarar como procedimental ───────────────
PDF_005 = RAW_B / "Auto_005_2018_Apertura_Caso003.pdf"
TXT_005 = PROC_B / "Auto_005_2018_Apertura_Caso003.txt"

if PDF_005.exists() and not TXT_005.exists():
    print(f"\n[2] Auto_005 — documento procedimental (10 págs)")
    nota = """NOTA CFH: Este documento es el Auto de Apertura del Caso 03 (2018).
Es un documento procedimental sin contenido sustantivo de análisis discursivo.
Se excluye del análisis de indicadores CFH por no contener hechos, conductas ni lenguaje de víctimas.
Incluido en el inventario del corpus como referencia institucional."""
    TXT_005.write_text(nota, encoding='utf-8')
    print(f"    ✓ Marcado como procedimental: {TXT_005.name}")
else:
    if TXT_005.exists():
        print(f"\n[2] Auto_005 ya procesado")

# ── PASO 3: Corrección metadata Auto_128 ─────────────────────────
print(f"\n[3] Corrigiendo metadata Auto_128...")

df = pd.read_csv(META)

# Auto_128 tiene subcaso "Norte de Santander" en metadata pero nombre "Costa_Caribe"
mask_128 = df['archivo'].str.contains('Auto_128', na=False)
if mask_128.any():
    print(f"    Antes: {df.loc[mask_128, 'subcaso'].values}")
    # El subcaso ya dice Norte de Santander en metadata — solo agregar nota
    df.loc[mask_128, 'nota_corpus'] = 'Nombre de archivo incorrecto: corresponde a Norte de Santander, no Costa Caribe'
    print(f"    ✓ Nota agregada a Auto_128")
else:
    print(f"    ✗ Auto_128 no encontrado en metadata")

# ── PASO 4: Marcar duplicados ─────────────────────────────────────
print(f"\n[4] Marcando duplicados...")

# adhc-055 = Auto_055 (mismos chars: 60362)
mask_055a = df['archivo'].str.contains('adhc-055', na=False)
mask_055b = df['archivo'].str.contains('Auto_055', na=False)

if mask_055a.any() and mask_055b.any():
    df.loc[mask_055a, 'nota_corpus'] = 'DUPLICADO de Auto_055_2022_Subcaso_Casanare.pdf — mismo documento, nombre alternativo'
    df.loc[mask_055b, 'nota_corpus'] = 'Versión canónica — duplicado: adhc-055-2022-casanare.pdf'
    print(f"    ✓ Duplicado adhc-055 / Auto_055 marcado")

# Auto_125 = Auto_128 (mismos chars: 46287)
mask_125 = df['archivo'].str.contains('Auto_125', na=False)
mask_128 = df['archivo'].str.contains('Auto_128', na=False)

if mask_125.any() and mask_128.any():
    chars_125 = df.loc[mask_125, 'n_chars_extraidos'].values[0]
    chars_128 = df.loc[mask_128, 'n_chars_extraidos'].values[0]
    if chars_125 == chars_128:
        df.loc[mask_128, 'nota_corpus'] = (df.loc[mask_128, 'nota_corpus'].fillna('') + 
            ' | POSIBLE DUPLICADO de Auto_125 (mismos chars). Nombre archivo incorrecto.')
        print(f"    ✓ Auto_128 marcado como posible duplicado de Auto_125")

# Caso 01 FARC — fuera del scope
mask_farc = df['archivo'].str.contains('caso01-farc', na=False)
if mask_farc.any():
    df.loc[mask_farc, 'nota_corpus'] = 'FUERA DE SCOPE: corresponde al Caso 01 FARC, no al Macrocaso 003'
    print(f"    ✓ sentencia-caso01-farc marcada fuera de scope")

# Guardar metadata corregida
META_V2 = REPO / "data" / "metadata_corpus_b_v2.csv"
df.to_csv(META_V2, index=False, encoding='utf-8-sig')
print(f"\n✓ Metadata corregida guardada: {META_V2.name}")

# ── RESUMEN ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESUMEN")
print(f"{'='*60}")
print(f"  Total documentos en metadata: {len(df)}")
print(f"  Con nota de corpus: {df['nota_corpus'].notna().sum() if 'nota_corpus' in df.columns else 0}")
print(f"  PDFs sin texto (0 chars): {(df['n_chars_extraidos'] == 0).sum()}")
print(f"\n[CFH] Completado.")
