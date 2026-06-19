"""
CFH — Verificación de saturación semántica del centroide MAFAPO
================================================================
Replica la metodología del Amicus Curiae TJP Bogotá:
  - Compara variación entre versiones del centroide (v2=25, v3=57, v3b=67 textos)
  - Calcula margen de error estadístico por versión
  - Determina si se alcanzó saturación semántica (variación < ±0.005)
  - Calcula cuántos textos se necesitan para ±5% de margen de error

Ejecutar:
  python cfh_saturacion_centroide.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_dist

REPO    = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
REF_DIR = REPO / "data" / "referencias"
sys.path.insert(0, str(REPO / "code" / "src"))

print("="*60)
print("VERIFICACIÓN DE SATURACIÓN SEMÁNTICA — CENTROIDE MAFAPO")
print("="*60)

# ── PASO 1: Recalcular centroide v2 (25 textos originales) ──────────────
print("\n[1] Recalculando centroide v2 (25 textos originales)...")

import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Dispositivo: {DEVICE}")

MODEL_NAME = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("    ✓ Modelo cargado")

def get_emb(texto):
    if not texto or len(texto.strip()) < 10:
        return np.zeros(768)
    inp = tokenizer(texto, return_tensors="pt", max_length=512,
                    truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Textos originales de centroides_expandidos.py
from centroides_expandidos import TEXTOS_MAFAPO as TEXTOS_V2

print(f"    Textos v2: {len(TEXTOS_V2)}")
embs_v2 = np.array([get_emb(t) for t in TEXTOS_V2])
centroide_v2 = np.mean(embs_v2, axis=0)
print(f"    ✓ Centroide v2 calculado (norma={np.linalg.norm(centroide_v2):.3f})")

# ── PASO 2: Cargar centroides v3 y v3b ───────────────────────────────────
print("\n[2] Cargando centroides v3 (57 textos) y v3b (67 textos)...")

centroide_v3  = np.load(str(REF_DIR / "centroide_mafapo_v3.npy"))
centroide_v3b = np.load(str(REF_DIR / "centroide_mafapo_v3b.npy"))
print(f"    ✓ v3  cargado (norma={np.linalg.norm(centroide_v3):.3f})")
print(f"    ✓ v3b cargado (norma={np.linalg.norm(centroide_v3b):.3f})")

# ── PASO 3: Calcular variación entre versiones ───────────────────────────
print("\n[3] Calculando variación entre versiones...")

dist_v2_v3   = cosine_dist(centroide_v2,  centroide_v3)
dist_v3_v3b  = cosine_dist(centroide_v3,  centroide_v3b)
dist_v2_v3b  = cosine_dist(centroide_v2,  centroide_v3b)

print(f"\n    Distancia coseno v2(25) → v3(57):  {dist_v2_v3:.4f}")
print(f"    Distancia coseno v3(57) → v3b(67): {dist_v3_v3b:.4f}")
print(f"    Distancia coseno v2(25) → v3b(67): {dist_v2_v3b:.4f}")

# Criterio de saturación del Amicus: variación < ±0.005
SAT_THRESHOLD = 0.005
print(f"\n    Umbral de saturación (Amicus): < {SAT_THRESHOLD}")
print(f"    v3 → v3b: {dist_v3_v3b:.4f} {'✓ SATURADO' if dist_v3_v3b < SAT_THRESHOLD else '⚠ NO saturado todavía'}")

# ── PASO 4: Calcular variación del IEI sobre muestra del corpus ──────────
print("\n[4] Calculando variación del IEI sobre muestra del corpus...")

PROCESSED_DIR = REPO / "data" / "processed"
textos_muestra = []
for txt in list(PROCESSED_DIR.rglob("*.txt"))[:30]:
    try:
        c = txt.read_text(encoding="utf-8", errors="ignore")
        pp = [p.strip() for p in c.split("\n\n") if 200 <= len(p.strip()) <= 500]
        textos_muestra.extend(pp[:3])
    except: pass
textos_muestra = textos_muestra[:50]

if textos_muestra:
    iei_v2, iei_v3, iei_v3b = [], [], []
    for t in textos_muestra:
        e = get_emb(t)
        iei_v2.append(cosine_dist(e, centroide_v2))
        iei_v3.append(cosine_dist(e, centroide_v3))
        iei_v3b.append(cosine_dist(e, centroide_v3b))

    iei_v2  = np.array(iei_v2)
    iei_v3  = np.array(iei_v3)
    iei_v3b = np.array(iei_v3b)

    var_v2_v3   = np.abs(iei_v2  - iei_v3).max()
    var_v3_v3b  = np.abs(iei_v3  - iei_v3b).max()
    var_v2_v3b  = np.abs(iei_v2  - iei_v3b).max()

    print(f"\n    Variación máxima IEI por bloque:")
    print(f"    v2(25) → v3(57):  Δ_max = {var_v2_v3:.4f}")
    print(f"    v3(57) → v3b(67): Δ_max = {var_v3_v3b:.4f}")
    print(f"    v2(25) → v3b(67): Δ_max = {var_v2_v3b:.4f}")
    print(f"\n    Umbral Amicus: < ±0.004")
    print(f"    v3 → v3b: {'✓ SATURADO' if var_v3_v3b < 0.004 else '⚠ Aún hay variación'}")

# ── PASO 5: Margen de error estadístico por versión ──────────────────────
print("\n[5] Margen de error estadístico (metodología Amicus)...")
print("""
    Fórmula: margen_error = 1 / sqrt(n)  (aproximación para corpus semánticos)
    Esta es la fórmula usada en el Amicus para estimar la representatividad
    del centroide como promedio de n vectores independientes.
""")

for n, nombre in [(25,"v2"), (57,"v3"), (67,"v3b"), (100,"meta_100"), (347,"Amicus_JP")]:
    margen = 1 / np.sqrt(n)
    pct = margen * 100
    print(f"    n={n:3d} ({nombre:<12}): margen = ±{margen:.4f} ({pct:.1f}%)")

print(f"""
    Para alcanzar ±5% (como en el Amicus): n = {int(np.ceil(1/0.05**2))} textos mínimo
    Para alcanzar ±3%:                     n = {int(np.ceil(1/0.03**2))} textos mínimo
    Nota: el Amicus logró ±5.3% con n=347, ya que usó 1/sqrt(347)=0.0537
""")

# ── PASO 6: Resumen y recomendación ──────────────────────────────────────
print("="*60)
print("RESUMEN PARA LA TESIS")
print("="*60)

saturado = var_v3_v3b < 0.004 if textos_muestra else dist_v3_v3b < SAT_THRESHOLD

print(f"""
Centroide MAFAPO — evolución:
  v2:  25 textos → margen ±{100/np.sqrt(25):.1f}%
  v3:  57 textos → margen ±{100/np.sqrt(57):.1f}%
  v3b: 67 textos → margen ±{100/np.sqrt(67):.1f}%

Variación entre versiones:
  v2→v3:   distancia coseno = {dist_v2_v3:.4f}
  v3→v3b:  distancia coseno = {dist_v3_v3b:.4f}
  {'✓ La variación v3→v3b es marginal — señal de convergencia' if dist_v3_v3b < 0.01 else '⚠ El centroide sigue moviéndose — agregar más textos mejora'}

Para alcanzar ±5% (estándar Amicus): necesitas ~{int(np.ceil(1/0.05**2))} textos
Textos actuales: 67 → margen ±{100/np.sqrt(67):.1f}%
Textos adicionales necesarios: ~{max(0, int(np.ceil(1/0.05**2)) - 67)}

{'→ Con los textos actuales ya puedes argumentar convergencia semántica.' if saturado else '→ Agregar más textos mejorará el margen de error.'}
Texto para §3.8:
  "El centroide MAFAPO v3b (67 textos, 5 fuentes) presenta una variación
   de Δ={dist_v3_v3b:.4f} respecto a la versión anterior (57 textos),
   margen de error ±{100/np.sqrt(67):.1f}%. Para alcanzar el estándar
   ±5% del Amicus CFH-TJP (347 fragmentos) se requieren ~{max(0, int(np.ceil(1/0.05**2)) - 67)}
   textos adicionales, planificados en la Fase 2 post-defensa."
""")

print("[CFH] Completado.")
