"""
CFH — Descriptivos consolidados desde cfh.db
Corre con: python cfh_descriptivos.py
Genera: cfh_descriptivos.txt con todos los resultados
"""

import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DB  = r'C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\cfh.db'
OUT = r'C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\cfh_descriptivos.txt'

conn = sqlite3.connect(DB)
lines = []

def h(titulo):
    lines.append("\n" + "="*70)
    lines.append(f"  {titulo}")
    lines.append("="*70)

def p(texto=""):
    lines.append(str(texto))

# ── Carga base ───────────────────────────────────────────────────────────────
docs  = pd.read_sql("SELECT * FROM documentos",  conn)
bloq  = pd.read_sql("SELECT * FROM bloques",     conn)
ind   = pd.read_sql("SELECT * FROM indicadores", conn)
aud   = pd.read_sql("SELECT * FROM audiencias",  conn)
segs  = pd.read_sql("SELECT * FROM segmentos_orales", conn)
anot  = pd.read_sql("SELECT * FROM anotaciones", conn)
corp  = pd.read_sql("SELECT * FROM corpora",     conn)
comp  = pd.read_sql("SELECT * FROM comparecientes", conn)

# Join principal: indicadores → bloques → documentos
ind_b = ind.merge(bloq[['id','documento_id','seccion']], left_on='bloque_id', right_on='id', suffixes=('','_b'))
ind_d = ind_b.merge(docs[['id','corpus','año','tipo_documento','departamento']], left_on='documento_id', right_on='id', suffixes=('','_d'))

# ════════════════════════════════════════════════════════════════════════════
h("1. PANORAMA GENERAL DEL CORPUS")
# ════════════════════════════════════════════════════════════════════════════

p(f"\nDocumentos totales       : {len(docs):>6,}")
p(f"Bloques totales          : {len(bloq):>6,}")
p(f"Indicadores calculados   : {len(ind):>6,}")
p(f"Segmentos orales (Capa 3): {len(segs):>6,}")
p(f"Anotaciones (Label Studio): {len(anot):>6,}")
p()

# Por corpus
p("Documentos por corpus:")
dc = docs.groupby('corpus').size().reset_index(name='n_docs')
bc = bloq.merge(docs[['id','corpus']], left_on='documento_id', right_on='id')
bc2 = bc.groupby('corpus').size().reset_index(name='n_bloques')
resumen = dc.merge(bc2, on='corpus', how='left')
p(resumen.to_string(index=False))

p()
p("Indicadores por corpus (n únicos de bloques con indicadores):")
ind_corp = ind_d.groupby('corpus')['bloque_id'].nunique().reset_index(name='bloques_con_ind')
ind_corp2 = ind_d.groupby('corpus').size().reset_index(name='n_indicadores')
p(ind_corp.merge(ind_corp2, on='corpus').to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
h("2. DISTRIBUCIÓN POR AÑO — CORPUS A")
# ════════════════════════════════════════════════════════════════════════════

docs_a = docs[docs['corpus'].isin(['A-CE','A-CSJ'])].copy()
docs_a['año'] = pd.to_numeric(docs_a['año'], errors='coerce')
por_año = docs_a.groupby(['año','corpus']).size().reset_index(name='n_docs')
por_año = por_año.dropna(subset=['año'])
por_año['año'] = por_año['año'].astype(int)
p()
p(por_año.sort_values('año').to_string(index=False))

p(f"\nRango temporal Corpus A: {int(docs_a['año'].min())} – {int(docs_a['año'].max())}")
p(f"Docs con año registrado: {docs_a['año'].notna().sum()} / {len(docs_a)}")

# ════════════════════════════════════════════════════════════════════════════
h("3. DISTRIBUCIÓN DE BLOQUES POR SECCIÓN")
# ════════════════════════════════════════════════════════════════════════════

bloq_d = bloq.merge(docs[['id','corpus']], left_on='documento_id', right_on='id')
secc   = bloq_d.groupby(['corpus','seccion']).size().reset_index(name='n_bloques')
secc   = secc.sort_values(['corpus','n_bloques'], ascending=[True,False])
p()
p(secc.to_string(index=False))

# Secciones únicas totales
p(f"\nSecciones únicas en todo el corpus: {bloq['seccion'].nunique()}")
top_secc = bloq['seccion'].value_counts().head(15)
p("\nTop 15 secciones por frecuencia global:")
p(top_secc.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("4. ESTADÍSTICOS DESCRIPTIVOS POR INDICADOR Y CORPUS")
# ════════════════════════════════════════════════════════════════════════════

indicadores_orden = ['y1_ebi','y2_sa','y3_civil','y4_nv','y7_surprisal',
                     'y8_mafapo','y9_cidh','y10_rep','y11_quotes',
                     'y12_verbos','y13_evidenciales']

for ind_cod in indicadores_orden:
    sub = ind_d[ind_d['codigo'] == ind_cod]
    if sub.empty:
        continue
    p(f"\n── {ind_cod.upper()} (n={len(sub):,}) ──")
    tab = sub.groupby('corpus')['valor'].agg(['count','mean','median','std','min','max'])
    tab.columns = ['n','media','mediana','std','min','max']
    tab = tab.round(4)
    p(tab.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("5. COMPARACIÓN A vs B — INDICADORES CLAVE")
# ════════════════════════════════════════════════════════════════════════════

from scipy import stats as scipy_stats

claves = ['y2_sa','y3_civil','y4_nv','y8_mafapo','y9_cidh','y10_rep']
p()
p(f"{'Indicador':<18} {'Media A':>8} {'Media B':>8} {'Δ':>8} {'p-valor':>10} {'Sig':>5}")
p("-"*60)

for cod in claves:
    a_vals = ind_d[(ind_d['codigo']==cod) & (ind_d['corpus'].isin(['A-CE','A-CSJ']))]['valor'].dropna()
    b_vals = ind_d[(ind_d['codigo']==cod) & (ind_d['corpus']=='B-JEP')]['valor'].dropna()
    if len(a_vals) < 5 or len(b_vals) < 5:
        continue
    stat, pval = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
    delta = b_vals.mean() - a_vals.mean()
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))
    p(f"{cod:<18} {a_vals.mean():>8.4f} {b_vals.mean():>8.4f} {delta:>+8.4f} {pval:>10.4f} {sig:>5}")

# ════════════════════════════════════════════════════════════════════════════
h("6. ANÁLISIS TEMPORAL — CORPUS A (y8 y y10 por año)")
# ════════════════════════════════════════════════════════════════════════════

ind_año = ind_d[ind_d['corpus'].isin(['A-CE','A-CSJ'])].copy()
ind_año['año'] = pd.to_numeric(ind_año['año'], errors='coerce')
ind_año = ind_año.dropna(subset=['año'])
ind_año['año'] = ind_año['año'].astype(int)

for cod in ['y8_mafapo','y10_rep','y4_nv']:
    sub = ind_año[ind_año['codigo']==cod]
    if sub.empty:
        continue
    tab = sub.groupby('año')['valor'].agg(['count','mean']).round(4)
    tab.columns = ['n','media']
    p(f"\n{cod.upper()} por año (Corpus A):")
    p(tab.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("7. DISTRIBUCIÓN POR SECCIÓN — INDICADORES CLAVE B-JEP")
# ════════════════════════════════════════════════════════════════════════════

ind_b_jep = ind_d[ind_d['corpus']=='B-JEP']
for cod in ['y10_rep','y4_nv','y8_mafapo']:
    sub = ind_b_jep[ind_b_jep['codigo']==cod]
    if sub.empty:
        continue
    tab = sub.groupby('seccion')['valor'].agg(['count','mean']).round(4)
    tab.columns = ['n','media']
    tab = tab.sort_values('media', ascending=False).head(12)
    p(f"\n{cod.upper()} por sección (B-JEP):")
    p(tab.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("8. CORPUS C — ICM Y ACTION UNITS POR AUDIENCIA")
# ════════════════════════════════════════════════════════════════════════════

segs_aud = segs.merge(aud[['id','subcaso','duracion_horas']], left_on='audiencia_id', right_on='id', suffixes=('','_a'))

p("\nSegmentos orales por subcaso:")
p(segs_aud.groupby('subcaso').size().reset_index(name='n_segmentos').to_string(index=False))

p("\nAction Units medias por subcaso (comparecientes):")
au_cols = ['au1','au4','au6','au12','au15','au17']
aus = segs_aud.copy()
for c in au_cols:
    aus[c] = pd.to_numeric(aus[c], errors='coerce')

tab_au = aus.groupby('subcaso')[au_cols].mean().round(4)
p(tab_au.to_string())

p("\nICM scores por subcaso:")
icm_cols = ['icm_facial','icm_vocal','icm_verbal','icm_tri']
for c in icm_cols:
    segs_aud[c] = pd.to_numeric(segs_aud[c], errors='coerce')

tab_icm = segs_aud.groupby('subcaso')[icm_cols].mean().round(4)
p(tab_icm.to_string())

p("\nProsódicos (f0, shimmer, jitter) por subcaso:")
pros_cols = ['f0_mean','f0_stddev','shimmer','jitter','hnr']
for c in pros_cols:
    segs_aud[c] = pd.to_numeric(segs_aud[c], errors='coerce')
tab_pros = segs_aud.groupby('subcaso')[pros_cols].mean().round(4)
p(tab_pros.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("9. MATRIZ DE CORRELACIONES — INDICADORES PRINCIPALES")
# ════════════════════════════════════════════════════════════════════════════

pivot = ind_d.pivot_table(index='bloque_id', columns='codigo', values='valor', aggfunc='mean')
cols_corr = [c for c in ['y2_sa','y3_civil','y4_nv','y8_mafapo','y9_cidh','y10_rep','y1_ebi'] if c in pivot.columns]
if len(cols_corr) >= 3:
    corr = pivot[cols_corr].corr().round(3)
    p()
    p(corr.to_string())

# ════════════════════════════════════════════════════════════════════════════
h("10. ANOTACIONES — DISTRIBUCIÓN DE ETIQUETAS")
# ════════════════════════════════════════════════════════════════════════════

p(f"\nTotal anotaciones: {len(anot)}")
p(f"Anotadores únicos: {anot['anotador'].nunique()}")
p()
p("Distribución por etiqueta:")
p(anot['label'].value_counts().to_string())

p("\nCombinaciones de etiquetas más frecuentes:")
p(anot['etiquetas_combinadas'].value_counts().head(15).to_string())

p("\nAnotaciones por tipo (resumen vs span):")
p(anot['es_resumen'].value_counts().to_string())

# ════════════════════════════════════════════════════════════════════════════
h("11. TAMAÑO DE BLOQUES — DISTRIBUCIÓN n_chars")
# ════════════════════════════════════════════════════════════════════════════

bloq['n_chars'] = pd.to_numeric(bloq['n_chars'], errors='coerce')
bloq_d2 = bloq.merge(docs[['id','corpus']], left_on='documento_id', right_on='id')

p()
tab_chars = bloq_d2.groupby('corpus')['n_chars'].agg(['count','mean','median','std','min','max']).round(1)
tab_chars.columns = ['n','media_chars','mediana','std','min','max']
p(tab_chars.to_string())

p(f"\nTotal caracteres procesados: {bloq['n_chars'].sum():,.0f}")
p(f"Promedio chars/bloque global: {bloq['n_chars'].mean():.0f}")

# ════════════════════════════════════════════════════════════════════════════
h("12. CFHBERT v2 — MÉTRICAS DEL CLASIFICADOR")
# ════════════════════════════════════════════════════════════════════════════

modelos = pd.read_sql("SELECT * FROM modelos", conn)
p()
for _, row in modelos.iterrows():
    if pd.notna(row['metricas_json']) and row['metricas_json'] != 'None':
        p(f"Modelo: {row['nombre']} {row['version']}")
        p(f"  Métricas: {row['metricas_json']}")
        p()

# ════════════════════════════════════════════════════════════════════════════
# Guardar output
# ════════════════════════════════════════════════════════════════════════════
conn.close()

texto = "\n".join(lines)
print(texto)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(texto)

print(f"\n✓ Guardado en: {OUT}")
