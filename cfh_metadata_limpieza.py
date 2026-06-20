"""
CFH — Limpieza metadata Corpus B (sin OCR)
"""
import pandas as pd
from pathlib import Path

REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
META = REPO / "data" / "metadata_corpus_b.csv"
META_V2 = REPO / "data" / "metadata_corpus_b_v2.csv"

df = pd.read_csv(META)
if 'nota_corpus' not in df.columns:
    df['nota_corpus'] = ''

print("="*60)
print("LIMPIEZA METADATA CORPUS B")
print("="*60)

# 1. Auto_128 — nombre incorrecto
mask = df['archivo'].str.contains('Auto_128', na=False)
df.loc[mask, 'nota_corpus'] = 'NOMBRE INCORRECTO: archivo dice Costa_Caribe pero corresponde a Norte de Santander (subcaso confirmado en metadata)'
print(f"\n[1] Auto_128 corregido: {mask.sum()} fila(s)")

# 2. Duplicados adhc-055 / Auto_055
mask_a = df['archivo'].str.contains('adhc-055', na=False)
mask_b = df['archivo'].str.contains('Auto_055', na=False)
df.loc[mask_a, 'nota_corpus'] = 'DUPLICADO de Auto_055_2022_Subcaso_Casanare.pdf'
df.loc[mask_b, 'nota_corpus'] = 'VERSION CANÓNICA — duplicado: adhc-055-2022-casanare.pdf'
print(f"[2] Duplicados adhc-055/Auto_055 marcados")

# 3. Auto_125 / Auto_128 posible duplicado
mask_125 = df['archivo'].str.contains('Auto_125', na=False)
mask_128 = df['archivo'].str.contains('Auto_128', na=False)
if mask_125.any() and mask_128.any():
    c125 = df.loc[mask_125, 'n_chars_extraidos'].values[0]
    c128 = df.loc[mask_128, 'n_chars_extraidos'].values[0]
    if c125 == c128:
        df.loc[mask_128, 'nota_corpus'] += ' | POSIBLE DUPLICADO de Auto_125 (n_chars idénticos: ' + str(int(c125)) + ')'
        print(f"[3] Auto_128 marcado como posible duplicado de Auto_125 (chars={int(c125)})")

# 4. Caso 01 FARC — fuera de scope
mask_farc = df['archivo'].str.contains('caso01-farc', na=False)
df.loc[mask_farc, 'nota_corpus'] = 'FUERA DE SCOPE: Caso 01 FARC, no Macrocaso 003'
print(f"[4] sentencia-caso01-farc marcada fuera de scope")

# 5. PDFs sin texto — pendiente OCR
mask_0 = df['n_chars_extraidos'] == 0
df.loc[mask_0 & df['archivo'].str.contains('RC-AI-016', na=False), 'nota_corpus'] = 'PENDIENTE OCR: PDF escaneado sin texto extraído (Colab Pro)'
df.loc[mask_0 & df['archivo'].str.contains('Auto_005', na=False), 'nota_corpus'] = 'PROCEDIMENTAL: Auto apertura Caso 003, sin contenido sustantivo para análisis CFH'
print(f"[5] PDFs sin texto marcados: {mask_0.sum()} documento(s)")

# Guardar
df.to_csv(META_V2, index=False, encoding='utf-8-sig')
print(f"\n✓ Metadata v2 guardada: {META_V2.name}")
print(f"  Total documentos: {len(df)}")
print(f"  Con notas: {(df['nota_corpus'] != '').sum()}")
print(f"\nResumen notas:")
for _, row in df[df['nota_corpus'] != ''].iterrows():
    print(f"  {row['archivo'][:50]}: {row['nota_corpus'][:60]}")
