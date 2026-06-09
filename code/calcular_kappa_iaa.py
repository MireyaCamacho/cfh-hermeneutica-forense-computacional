"""
Cálculo de Cohen κ — IAA CFH
==============================
Uso: una vez que el segundo anotador devuelva el Excel completado,
correr este script sobre el archivo.

Instrucciones:
1. El segundo anotador devuelve CFH_IAA_Validacion_A2.xlsx con
   columnas C-F (EBI, SA, NV, REP) completadas en la hoja ANOTACION.
2. Copiar el archivo a esta carpeta con nombre:
   CFH_IAA_Validacion_A2_COMPLETADO.xlsx
3. Correr: python code\calcular_kappa_iaa.py

Método:
  - Nivel de comparación: fragmento (100 unidades)
  - Para cada categoría: presencia/ausencia en el fragmento (binario)
  - Cohen κ por categoría + κ global (macro-promedio)
  - Nivel de span: Jaccard similarity para coincidencia parcial

Output:
  - outputs/iaa_kappa_resultados.csv
  - outputs/iaa_kappa_reporte.txt
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# ── Rutas ────────────────────────────────────────────────────────────────────
JSON_A1  = 'data/annotations_mireya_v1_json.json'
EXCEL_A2 = 'outputs/CFH_IAA_Validacion_A2_COMPLETADO.xlsx'
OUT_CSV  = 'outputs/iaa_kappa_resultados.csv'
OUT_TXT  = 'outputs/iaa_kappa_reporte.txt'

CATEGORIAS = ['EBI', 'SA', 'NV', 'REP']

# ── Cargar anotaciones A1 (Mireya) ───────────────────────────────────────────
with open(JSON_A1, encoding='utf-8') as f:
    data_a1 = json.load(f)

# Construir matriz binaria A1: [n_fragmentos x n_categorias]
a1_matrix = {}
for item in data_a1:
    frag_id = item['id']
    cats = set()
    for lbl in item.get('label', []):
        cat = lbl.get('labels', ['?'])[0] if isinstance(lbl.get('labels'), list) else '?'
        if cat in CATEGORIAS:
            cats.add(cat)
    a1_matrix[frag_id] = {c: int(c in cats) for c in CATEGORIAS}

# También guardar spans para Jaccard
a1_spans = {}
for item in data_a1:
    frag_id = item['id']
    spans = {c: set() for c in CATEGORIAS}
    for lbl in item.get('label', []):
        cat = lbl.get('labels', ['?'])[0] if isinstance(lbl.get('labels'), list) else '?'
        txt = lbl.get('text', '').strip().lower()
        if cat in CATEGORIAS and txt:
            spans[cat].add(txt)
    a1_spans[frag_id] = spans

# ── Cargar anotaciones A2 (segundo anotador) ─────────────────────────────────
if not Path(EXCEL_A2).exists():
    print(f"ERROR: No se encontró {EXCEL_A2}")
    print("Instrucciones:")
    print("  1. Renombrar el Excel completado a: CFH_IAA_Validacion_A2_COMPLETADO.xlsx")
    print("  2. Copiarlo a la carpeta outputs/")
    print("  3. Correr este script de nuevo")
    exit(1)

df_a2 = pd.read_excel(EXCEL_A2, sheet_name='ANOTACION', header=0)
print(f"A2: {len(df_a2)} fragmentos cargados")
print(f"Columnas: {df_a2.columns.tolist()}")

# Construir matriz binaria A2
a2_matrix = {}
a2_spans  = {}

for _, row in df_a2.iterrows():
    frag_id = int(row.iloc[0])  # columna #
    cats = set()
    spans = {c: set() for c in CATEGORIAS}
    
    for j, cat in enumerate(CATEGORIAS):
        cell_val = row.iloc[j+2]  # columnas C,D,E,F = índices 2,3,4,5
        if pd.notna(cell_val) and str(cell_val).strip():
            cats.add(cat)
            # Parsear spans separados por |
            sp_list = [s.strip().lower() for s in str(cell_val).split('|') if s.strip()]
            spans[cat] = set(sp_list)
    
    a2_matrix[frag_id] = {c: int(c in cats) for c in CATEGORIAS}
    a2_spans[frag_id]  = spans

# ── Alinear fragmentos comunes ────────────────────────────────────────────────
ids_comunes = sorted(set(a1_matrix.keys()) & set(a2_matrix.keys()))
print(f"\nFragmentos comunes: {len(ids_comunes)}")

# ── Cálculo de Cohen κ por categoría ─────────────────────────────────────────
resultados = []
reporte    = []

reporte.append("=" * 60)
reporte.append("REPORTE IAA — Hermenéutica Forense Computacional")
reporte.append(f"N fragmentos evaluados: {len(ids_comunes)}")
reporte.append("=" * 60)

kappas = []
for cat in CATEGORIAS:
    y_a1 = [a1_matrix[i][cat] for i in ids_comunes]
    y_a2 = [a2_matrix[i][cat] for i in ids_comunes]
    
    # Cohen κ
    try:
        kappa = cohen_kappa_score(y_a1, y_a2)
    except Exception as e:
        kappa = float('nan')
    
    # Acuerdo observado
    po = sum(a == b for a,b in zip(y_a1, y_a2)) / len(ids_comunes)
    
    # Conteos
    n_a1 = sum(y_a1)
    n_a2 = sum(y_a2)
    n_ambos = sum(a==1 and b==1 for a,b in zip(y_a1, y_a2))
    n_ninguno = sum(a==0 and b==0 for a,b in zip(y_a1, y_a2))
    n_discordia = len(ids_comunes) - n_ambos - n_ninguno
    
    # Interpretación
    if np.isnan(kappa):
        interpretacion = "N/A"
    elif kappa >= 0.80:
        interpretacion = "SUSTANCIAL-EXCELENTE ✓"
    elif kappa >= 0.60:
        interpretacion = "MODERADO"
    elif kappa >= 0.40:
        interpretacion = "REGULAR"
    else:
        interpretacion = "DÉBIL"
    
    # Jaccard sobre spans (coincidencia parcial)
    jaccards = []
    for i in ids_comunes:
        sp1 = a1_spans[i][cat]
        sp2 = a2_spans[i][cat]
        if sp1 or sp2:
            inter = len(sp1 & sp2)
            union = len(sp1 | sp2)
            jaccards.append(inter/union if union > 0 else 0)
    jaccard_mean = np.mean(jaccards) if jaccards else float('nan')
    
    resultados.append({
        'categoria': cat,
        'kappa': round(kappa, 4) if not np.isnan(kappa) else None,
        'acuerdo_obs': round(po, 4),
        'n_A1': n_a1,
        'n_A2': n_a2,
        'ambos_presentes': n_ambos,
        'ambos_ausentes': n_ninguno,
        'discordancias': n_discordia,
        'jaccard_spans': round(jaccard_mean, 4) if not np.isnan(jaccard_mean) else None,
        'interpretacion': interpretacion,
    })
    
    reporte.append(f"\n--- {cat} ---")
    reporte.append(f"  κ Cohen:         {kappa:.4f} → {interpretacion}")
    reporte.append(f"  Acuerdo obs.:    {po:.4f} ({po*100:.1f}%)")
    reporte.append(f"  A1 marcó:        {n_a1} fragmentos")
    reporte.append(f"  A2 marcó:        {n_a2} fragmentos")
    reporte.append(f"  Ambos presentes: {n_ambos}")
    reporte.append(f"  Ambos ausentes:  {n_ninguno}")
    reporte.append(f"  Discordancias:   {n_discordia}")
    reporte.append(f"  Jaccard spans:   {jaccard_mean:.4f}" if not np.isnan(jaccard_mean) else "  Jaccard spans: N/A")
    
    kappas.append(kappa if not np.isnan(kappa) else 0)

# κ global (macro-promedio)
kappa_global = np.mean(kappas)
reporte.append(f"\n{'='*60}")
reporte.append(f"κ GLOBAL (macro-promedio): {kappa_global:.4f}")
if kappa_global >= 0.80:
    reporte.append("→ UMBRAL ALCANZADO ✓ (κ > 0.80)")
else:
    reporte.append(f"→ UMBRAL NO ALCANZADO (requiere κ > 0.80, diferencia: {0.80-kappa_global:.4f})")
reporte.append(f"{'='*60}")

# Fragmentos con mayor discordancia
reporte.append("\nTop 10 fragmentos con mayor discordancia:")
discordancias_por_frag = []
for i in ids_comunes:
    disc = sum(a1_matrix[i][c] != a2_matrix[i][c] for c in CATEGORIAS)
    discordancias_por_frag.append((i, disc))
discordancias_por_frag.sort(key=lambda x: -x[1])
for frag_id, disc in discordancias_por_frag[:10]:
    reporte.append(f"  Fragmento #{frag_id}: {disc} categorías discordantes")

# ── Guardar resultados ────────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados)
df_res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

reporte_texto = "\n".join(reporte)
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write(reporte_texto)

print("\n" + reporte_texto)
print(f"\n✓ CSV: {OUT_CSV}")
print(f"✓ Reporte: {OUT_TXT}")
