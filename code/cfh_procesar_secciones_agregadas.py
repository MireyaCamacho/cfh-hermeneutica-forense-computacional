"""
CFH — Procesar secciones faltantes desde corpus_b_sentencias_agregadas
=======================================================================
Procesa los 69 TXT de ST001_Catatumbo, ST002_CostaCaribe, SENIT8_Reparacion
que no pudieron cargarse desde los JSONs por rutas faltantes.
Concatena con indicators_corpus_b_v2.csv para llegar a N_B~214.

Ejecutar desde la raíz del repo:
  python cfh_procesar_secciones_agregadas.py
"""
import sys
import time
import pandas as pd
from pathlib import Path

# Rutas
REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
SECS_DIR = REPO / "data" / "processed" / "corpus_b_sentencias_agregadas"
OUTPUT_V2 = REPO / "outputs" / "indicators_corpus_b_v2.csv"
OUTPUT_V3 = REPO / "outputs" / "indicators_corpus_b_v3.csv"

sys.path.insert(0, str(REPO / "code" / "src"))

# Cargar extractores
print("[CFH] Cargando extractores...")
from features.y2_sa_extractor import SAExtractor
from features.y3_civil_extractor import CivilLexiconExtractor
from features.y4_nv_extractor import NVExtractor
from features.y10_rep_extractor import REPExtractor

sa  = SAExtractor()
civ = CivilLexiconExtractor()
nv  = NVExtractor()
rep = REPExtractor()
print("[CFH] Extractores listos.")

# Secciones target válidas
SECCIONES_TARGET = {
    "RECONOCIMIENTO", "HECHOS_Y_CONDUCTAS", "PATRONES_MACROCRIMINALES",
    "CALIFICACION_JURIDICA", "RESUELVE", "CONSIDERACIONES",
    "CONTRIBUCION_VERDAD", "SANCION_PROPIA"
}

def es_target(nombre_seccion):
    """True si la sección es target para el análisis."""
    for t in SECCIONES_TARGET:
        if t in nombre_seccion.upper():
            return True
    return False

def extraer_doc_y_seccion(filename):
    """Extrae doc_id y section_id del nombre del archivo."""
    # Formato: ST001_Catatumbo_SECCION_00.txt
    stem = Path(filename).stem
    partes = stem.split("_")
    # Buscar dónde empieza la sección (mayúsculas después del doc_id)
    for i, p in enumerate(partes):
        if p.isupper() or p in {"HECHOS", "CONTRIBUCION", "SANCION", "PATRONES",
                                 "CALIFICACION", "RECONOCIMIENTO", "RESUELVE",
                                 "CONSIDERACIONES", "Y", "CONDUCTAS", "VERDAD",
                                 "PROPIA", "JURIDICA", "MACROCRIMINALES"}:
            doc_id = "_".join(partes[:i])
            section_id = "_".join(partes[i:])
            return doc_id, section_id
    return stem, "DESCONOCIDA"

# Procesar secciones
print(f"\n[CFH] Procesando secciones en {SECS_DIR}...")
txt_files = sorted(SECS_DIR.glob("*.txt"))
print(f"  Total archivos encontrados: {len(txt_files)}")

resultados = []
n_ok = 0
n_skip = 0

for i, f in enumerate(txt_files, 1):
    doc_id, section_id = extraer_doc_y_seccion(f.name)

    if not es_target(section_id):
        n_skip += 1
        continue

    try:
        texto = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(texto) < 30:
            n_skip += 1
            continue

        # Truncar a 8000 chars como el pipeline original
        texto = texto[:8000]

        t0 = time.perf_counter()
        r_sa  = sa.extract(texto,  doc_id=doc_id, section_id=section_id, corpus_type="B")
        r_civ = civ.extract(texto, doc_id=doc_id, section_id=section_id, corpus_type="B")
        r_nv  = nv.extract(texto,  doc_id=doc_id, section_id=section_id, corpus_type="B")
        r_rep = rep.extract(texto, doc_id=doc_id, section_id=section_id, corpus_type="B")
        elapsed = time.perf_counter() - t0

        resultados.append({
            "doc_id":            doc_id,
            "section_id":        section_id,
            "corpus_type":       "B",
            "year":              2025,
            "y1_ebi":            0.0,
            "y2_sa":             round(r_sa.score, 4),
            "y3_civil":          round(r_civ.score, 4),
            "y4_nv":             round(r_nv.score, 4),
            "y5_corpus_type":    1,
            "y6_period":         1.0,
            "y10_rep":           round(r_rep.score, 4),
            "y7_surprisal":      float("nan"),
            "y8_mafapo":         float("nan"),
            "y9_cidh":           float("nan"),
            "y11_conv_rest":     float("nan"),
            "text_length_chars": len(texto),
            "n_sa_instances":    r_sa.n_instances,
            "n_nv_instances":    r_nv.n_instances,
            "n_rep_instances":   r_rep.n_instances,
            "n_nv_questioned":   r_nv.n_questioned,
            "processing_time_s": round(elapsed, 3),
            "has_warning":       any([r_sa.warning, r_civ.warning,
                                      r_nv.warning, r_rep.warning]),
        })
        n_ok += 1

        if i % 10 == 0:
            print(f"  [{i}/{len(txt_files)}] procesados: {n_ok} OK, {n_skip} skip")

    except Exception as e:
        print(f"  ✗ Error en {f.name}: {e}")
        n_skip += 1

print(f"\n[CFH] Completado: {n_ok} secciones procesadas, {n_skip} omitidas")

if not resultados:
    print("✗ No se procesaron secciones. Revisar rutas.")
    sys.exit(1)

# Crear DataFrame con las nuevas secciones
df_nuevas = pd.DataFrame(resultados)
print(f"\nResumen nuevas secciones:")
print(f"  y2_sa:  {df_nuevas['y2_sa'].mean():.3f} ± {df_nuevas['y2_sa'].std():.3f}")
print(f"  y4_nv:  {df_nuevas['y4_nv'].mean():.3f} ± {df_nuevas['y4_nv'].std():.3f}")
print(f"  y10_rep: {df_nuevas['y10_rep'].mean():.3f} ± {df_nuevas['y10_rep'].std():.3f}")

# Concatenar con v2
df_v2 = pd.read_csv(OUTPUT_V2)
df_v3 = pd.concat([df_v2, df_nuevas], ignore_index=True)
df_v3.to_csv(OUTPUT_V3, index=False, encoding="utf-8-sig")

print(f"\n[CFH] Corpus B v3 guardado: {OUTPUT_V3}")
print(f"  v2: {len(df_v2)} secciones")
print(f"  nuevas: {len(df_nuevas)} secciones")
print(f"  v3 total: {len(df_v3)} secciones")

# Recalcular Cohen's d
import numpy as np
from scipy.stats import mannwhitneyu

df_a = pd.read_csv(REPO / "outputs" / "nivel1_dis_iei_AB.csv")
a = df_a[df_a['corpus_type'].isin(['A-CE', 'A-CSJ'])]
b = df_v3

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return abs(x.mean() - y.mean()) / pooled

print(f"\n{'='*55}")
print(f"COHEN'S d — N_A={len(a)}, N_B={len(b)}")
print(f"{'='*55}")

for ind in ['y2_sa', 'y4_nv', 'y10_rep']:
    xa, xb = a[ind].dropna(), b[ind].dropna()
    d = cohen_d(xa, xb)
    u, p = mannwhitneyu(xa, xb)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {ind}: A={xa.mean():.3f}, B={xb.mean():.3f}, d={d:.3f}, p={p:.4f} {sig}")

# DIS sintético
def calc_dis(df):
    sa  = df['y2_sa'].fillna(df['y2_sa'].mean())
    nv  = df['y4_nv'].fillna(df['y4_nv'].mean())
    rep = df['y10_rep'].fillna(df['y10_rep'].mean())
    sa_n  = (sa  - sa.min())  / (sa.max()  - sa.min()  + 1e-9)
    nv_n  = (nv  - nv.min())  / (nv.max()  - nv.min()  + 1e-9)
    rep_n = (rep - rep.min()) / (rep.max() - rep.min() + 1e-9)
    return 0.35*sa_n + 0.35*nv_n + 0.30*(1 - rep_n)

dis_a = calc_dis(a)
dis_b = calc_dis(b)
d = cohen_d(dis_a, dis_b)
u, p = mannwhitneyu(dis_a, dis_b)
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
print(f"  DIS: A={dis_a.mean():.3f}, B={dis_b.mean():.3f}, d={d:.3f}, p={p:.4f} {sig}")

print("\n[CFH] Completado.")
