"""
CFH — Centroide MAFAPO v3b: +15 textos Dejusticia + limpieza Corpus C
======================================================================
Cambios respecto a v3:
  1. +15 textos de voz directa de Ana Páez y Gloria Martínez (Dejusticia 2024)
  2. Limpieza de segmentos contaminados del Corpus C:
     - Excluir segmentos con lenguaje institucional JEP
       ("Audiencia Pública de Reconocimiento", "ruta dialógica", etc.)
     - Conservar solo voz directa de familiares/víctimas

Ejecutar en conda env `cfh`:
  python -c "import sys; sys.path.insert(0,'code/src'); exec(open('cfh_centroide_mafapo_v3b.py',encoding='utf-8').read())"

Outputs:
  data/referencias/centroide_mafapo_v3b.npy
  data/referencias/corpus_mafapo_v3b.json
  data/referencias/reporte_correlacion_v3b.txt
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine as cosine_dist

# ── Cargar corpus v3 existente ────────────────────────────────────────────
REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
REF_DIR = REPO / "data" / "referencias"

with open(REF_DIR / "corpus_mafapo_v3.json", encoding="utf-8") as f:
    corpus_v3 = json.load(f)

corpus_actual = corpus_v3["corpus_final"]
print(f"Corpus v3 cargado: {len(corpus_actual)} textos")

# ── LIMPIEZA: filtrar segmentos contaminados del Corpus C ─────────────────
FRASES_INSTITUCIONALES = [
    "audiencia pública de reconocimiento",
    "ruta dialógica",
    "reconocimiento de verdad y responsabilidad",
    "determinación de los hechos",
    "aceptación de responsabilidad",
    "fue imposible hacer justicia",
    "el conflicto armado nos dividió",
]

def es_contaminado(texto):
    t = texto.lower()
    return any(f in t for f in FRASES_INSTITUCIONALES)

corpus_limpio = []
excluidos_limpieza = []
for t in corpus_actual:
    if t.get("tipo") == "intervención_oral_audiencia_auto" and es_contaminado(t["texto"]):
        excluidos_limpieza.append(t)
    else:
        corpus_limpio.append(t)

print(f"  Segmentos contaminados excluidos: {len(excluidos_limpieza)}")
for e in excluidos_limpieza:
    print(f"    [{e['id']}] {e['texto'][:70]}...")

# ── NUEVOS TEXTOS: Dejusticia 2024 ────────────────────────────────────────
TEXTOS_DEJUSTICIA = [
    {"id": "dej_01", "texto": "Una madre conoce a su hijo. Yo sabía que ese era mi hijo, pero que lo estaban haciendo pasar por algo que no era: un guerrillero.", "fuente": "Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_02", "texto": "Mami, todo lo que voy a hacer es por usted, para que no trabaje tanto y mis hermanas estén tranquilas.", "fuente": "Daniel Martínez (víctima), última frase — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_03", "texto": "Lo único que puede hacer es cuidar a mi mamá, y dígale que no le voy a poder cumplir la promesa que le hice.", "fuente": "Daniel Martínez (víctima), última llamada — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_04", "texto": "Solo ellas saben el dolor que es perder a un hijo de esta manera tan cruel. Hay peleas y disgustos, pero ellas son para mí, mi familia.", "fuente": "Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_05", "texto": "La vida nos puso ahí, y nos tocó asumir ese rol.", "fuente": "Ana Páez y Gloria Martínez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_06", "texto": "El dolor ya no solo era procesar la muerte de nuestros hijos, sino también el sufrimiento de ver cómo manchaban sus nombres, cómo se les despojaba de su honra incluso en la muerte.", "fuente": "Gloria Martínez y Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_07", "texto": "Sabíamos que era una mentira, que esa acusación no tenía sentido. Quienes debían cuidarlos fueron los que los mataron.", "fuente": "Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_08", "texto": "El cuerpo de Eduardo llevaba unas botas puestas al revés y una mochila colgada al hombro. Pero yo sabía que él nunca usaba botas, y mucho menos una mochila. Siempre se vestía de manera formal.", "fuente": "Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_09", "texto": "Para encontrar a su hermano, tuvo que mover uno a uno los cuerpos. Daniel era el último de la pila.", "fuente": "Angie, hija de Gloria Martínez — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_10", "texto": "Sentíamos un alivio por haberlos hallado. Pero ese alivio estaba teñido de una tristeza profunda, pues las personas que más amábamos estaban muertas.", "fuente": "Ana Páez y Gloria Martínez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_11", "texto": "Gloria, debilitada por la depresión y por ataques de asma, no se sintió capaz de hacer el viaje sola a identificar el cuerpo de su hijo.", "fuente": "Gloria Martínez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_12", "texto": "Necesitábamos la verdad. Queríamos mostrarle al país que nuestros hijos no eran guerrilleros, que eran inocentes.", "fuente": "Ana Páez y Gloria Martínez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_13", "texto": "Para mí era muy difícil pensar que quienes debían cuidar a nuestros hijos fueron los que los mataron.", "fuente": "Ana Páez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_14", "texto": "Cuando el país nos ve con nuestras botas pintadas de colores vibrantes, sabe que ahí estamos, recordando las injusticias y resistiendo con fuerza.", "fuente": "MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
    {"id": "dej_15", "texto": "Sabemos que nuestros hijos, donde sea que estén, están muy orgullosos de todo lo que hemos ayudado.", "fuente": "Ana Páez y Gloria Martínez, MAFAPO — Dejusticia oct 2024", "tipo": "testimonio_directo", "verificado": True},
]

# ── Corpus final v3b ──────────────────────────────────────────────────────
corpus_v3b = corpus_limpio + TEXTOS_DEJUSTICIA
TEXTOS_V3B = [t["texto"] for t in corpus_v3b]

print(f"\n{'='*50}")
print(f"CORPUS MAFAPO v3b")
print(f"{'='*50}")
print(f"  v3 original:              {len(corpus_actual)} textos")
print(f"  - Contaminados excluidos: {len(excluidos_limpieza)}")
print(f"  + Dejusticia 2024:        {len(TEXTOS_DEJUSTICIA)}")
print(f"  = v3b total:              {len(corpus_v3b)} textos")
print(f"  Meta 75:                  {'✓ ALCANZADA' if len(corpus_v3b) >= 75 else f'⚠ Faltan {75-len(corpus_v3b)}'}")

# ── Cargar modelo ─────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Dispositivo: {DEVICE}")

MODEL_NAME = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"  ✓ Modelo cargado")

def get_embedding(texto):
    if not texto or len(texto.strip()) < 10:
        return np.zeros(768)
    inputs = tokenizer(texto, return_tensors="pt", max_length=512,
                      truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def calcular_centroide(textos, nombre):
    print(f"\n  Calculando centroide {nombre} ({len(textos)} textos)...")
    embs = []
    for i, t in enumerate(textos):
        embs.append(get_embedding(t))
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(textos)}...")
    c = np.mean(embs, axis=0)
    print(f"  ✓ {nombre}: norma={np.linalg.norm(c):.3f}")
    return c

# ── Calcular centroides v3b ───────────────────────────────────────────────
centroide_mafapo_v3b = calcular_centroide(TEXTOS_V3B, "MAFAPO v3b")

# CIDH sin cambios — cargar desde disco o recalcular
cidh_path = REF_DIR / "centroide_cidh_v3.npy"
if cidh_path.exists():
    centroide_cidh = np.load(str(cidh_path))
    print(f"\n  ✓ Centroide CIDH cargado desde disco (sin cambios)")
else:
    # Recalcular desde centroides_expandidos.py
    import sys
    sys.path.insert(0, str(REPO / "code" / "src"))
    from centroides_expandidos import TEXTOS_CIDH
    centroide_cidh = calcular_centroide(TEXTOS_CIDH, "CIDH v3b")
    np.save(str(REF_DIR / "centroide_cidh_v3b.npy"), centroide_cidh)

# ── Guardar ───────────────────────────────────────────────────────────────
np.save(str(REF_DIR / "centroide_mafapo_v3b.npy"), centroide_mafapo_v3b)

inventario_v3b = {
    "version": "v3b",
    "timestamp": datetime.now().isoformat(),
    "resumen": {
        "total_textos": len(corpus_v3b),
        "de_v3_conservados": len(corpus_limpio),
        "contaminados_excluidos": len(excluidos_limpieza),
        "nuevos_dejusticia": len(TEXTOS_DEJUSTICIA),
        "meta_75": len(corpus_v3b) >= 75
    },
    "corpus_final": corpus_v3b,
    "excluidos_limpieza": excluidos_limpieza
}
with open(str(REF_DIR / "corpus_mafapo_v3b.json"), "w", encoding="utf-8") as f:
    json.dump(inventario_v3b, f, ensure_ascii=False, indent=2)

# ── Verificación distinción y₈ vs y₉ ─────────────────────────────────────
print(f"\n{'='*50}")
print("VERIFICACIÓN DISTINCIÓN y₈ vs y₉")
print(f"{'='*50}")

dist_polos = cosine_dist(centroide_mafapo_v3b, centroide_cidh)
print(f"\n  Distancia coseno MAFAPO v3b vs CIDH: {dist_polos:.4f}")
print(f"  v3  referencia:                       0.0919")
print(f"  {'✓ Mejoró' if dist_polos > 0.0919 else '⚠ Igual o peor'}")

# Correlación sobre muestra
PROCESSED_DIR = REPO / "data" / "processed"
textos_muestra = []
for txt in list(PROCESSED_DIR.rglob("*.txt"))[:50]:
    try:
        contenido = txt.read_text(encoding="utf-8", errors="ignore")
        parrafos = [p.strip() for p in contenido.split("\n\n") if 200 <= len(p.strip()) <= 500]
        textos_muestra.extend(parrafos[:3])
    except:
        pass
textos_muestra = textos_muestra[:100]

if textos_muestra:
    d_maf, d_cid = [], []
    for t in textos_muestra:
        e = get_embedding(t)
        d_maf.append(cosine_dist(e, centroide_mafapo_v3b))
        d_cid.append(cosine_dist(e, centroide_cidh))

    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(d_maf, d_cid)
    r_s, p_s = spearmanr(d_maf, d_cid)

    reporte = f"""
REPORTE CORRELACIÓN y₈ vs y₉ — CFH v3b
========================================
Fecha: {datetime.now().isoformat()}
N muestra: {len(textos_muestra)} bloques

MAFAPO v3b: {len(corpus_v3b)} textos (v3 limpio + 15 Dejusticia 2024)
CIDH:       25 textos (sin cambios)

Distancia entre polos:
  v3  (antes): 0.0919
  v3b (ahora): {dist_polos:.4f}  {'↑ mejoró' if dist_polos > 0.0919 else '↓ empeoró'}

Correlación y₈ vs y₉:
  v3  Pearson r: 0.7316
  v3b Pearson r: {r_p:.4f}  (p={p_p:.4f})  {'↓ mejoró' if r_p < 0.7316 else '↑ empeoró'}
  Spearman ρ:   {r_s:.4f}  (p={p_s:.4f})

Meta: r < 0.80  →  {'✓ META ALCANZADA' if r_p < 0.80 else '⚠ NO ALCANZADA'}

Comparación v2 → v3 → v3b:
  v2:  r ≈ 0.86  (25 textos con contaminación)
  v3:  r = 0.7316 (57 textos, meta alcanzada)
  v3b: r = {r_p:.4f} ({len(corpus_v3b)} textos, limpio + Dejusticia)
"""
    print(reporte)

    with open(str(REF_DIR / "reporte_correlacion_v3b.txt"), "w", encoding="utf-8") as f:
        f.write(reporte)

print("\n[CFH] Centroide MAFAPO v3b completado.")
print(f"  centroide_mafapo_v3b.npy  ({len(corpus_v3b)} textos)")
print(f"  corpus_mafapo_v3b.json    (inventario trazable)")
print(f"  reporte_correlacion_v3b.txt")
