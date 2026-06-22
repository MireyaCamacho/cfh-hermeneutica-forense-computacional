"""
CFH — Auditoría del Centroide MAFAPO y Construcción del Corpus Ampliado
========================================================================
Propósito:
  1. Auditar qué textos se usaron para construir el centroide MAFAPO actual
     (reconstruir trazabilidad desde el repo).
  2. Inventariar fuentes disponibles para ampliar de 25 → ~75 textos.
  3. Calcular el nuevo centroide y compararlo con el anterior.
  4. Verificar que y₈ (MAFAPO) e y₉ (CIDH) siguen siendo empíricamente distintos.

Ejecutar en conda env `cfh`:
  python cfh_auditoria_centroide_mafapo.py

Requiere:
  pip install transformers torch sentence-transformers requests beautifulsoup4
  (ConfliBERT-Spanish ya debe estar descargado en el entorno)
"""

import os
import json
import hashlib
import sqlite3
import glob
import re
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.spatial.distance import cosine

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
DB_PATH   = REPO_ROOT / "data" / "cfh.db"
DATA_DIR  = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs" / "auditoria_centroide"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "lcampillos/ConfliBERT-Spanish-BETO-Cased-v1"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"[CFH] Auditoría centroide MAFAPO — {TIMESTAMP}")
print(f"[CFH] Repo: {REPO_ROOT}")
print(f"[CFH] Dispositivo: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 1: AUDITORÍA DEL CENTROIDE ACTUAL
# Objetivo: reconstruir qué 25 textos usaste originalmente
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 1: Auditoría del centroide actual")
print("="*60)

# 1.1 Buscar en el repo todos los scripts que mencionan "centroide" o "MAFAPO"
print("\n[1.1] Buscando scripts relacionados con el centroide MAFAPO...")

scripts_candidatos = []
extensiones = ["*.py", "*.ipynb", "*.R", "*.sh"]
patrones = ["centroid", "MAFAPO", "mafapo", "polo_victimas", "corpus_referencia",
            "textos_mafapo", "ref_mafapo", "centroide_mafapo"]

for ext in extensiones:
    for f in REPO_ROOT.rglob(ext):
        try:
            contenido = f.read_text(encoding="utf-8", errors="ignore")
            if any(p.lower() in contenido.lower() for p in patrones):
                scripts_candidatos.append({
                    "archivo": str(f.relative_to(REPO_ROOT)),
                    "modificado": datetime.fromtimestamp(f.stat().st_mtime).isoformat()[:16],
                    "tamaño_kb": round(f.stat().st_size / 1024, 1),
                    "menciones_mafapo": contenido.lower().count("mafapo"),
                    "menciones_centroide": contenido.lower().count("centroid")
                })
        except Exception:
            pass

scripts_candidatos.sort(key=lambda x: x["menciones_mafapo"] + x["menciones_centroide"], reverse=True)

print(f"  → Encontrados {len(scripts_candidatos)} scripts relacionados:")
for s in scripts_candidatos[:10]:
    print(f"    [{s['modificado']}] {s['archivo']}")
    print(f"      menciones MAFAPO: {s['menciones_mafapo']}  |  centroide: {s['menciones_centroide']}")

# 1.2 Buscar archivos de texto/CSV que podrían ser el corpus MAFAPO original
print("\n[1.2] Buscando archivos de corpus MAFAPO...")

corpus_candidatos = []
patrones_archivo = ["mafapo", "polo_victima", "referencia", "corpus_ref",
                    "textos_ref", "centroide", "victimas_ref"]

for ext in ["*.txt", "*.csv", "*.json", "*.pkl", "*.npy"]:
    for f in REPO_ROOT.rglob(ext):
        nombre = f.name.lower()
        if any(p in nombre for p in patrones_archivo):
            corpus_candidatos.append({
                "archivo": str(f.relative_to(REPO_ROOT)),
                "extension": f.suffix,
                "modificado": datetime.fromtimestamp(f.stat().st_mtime).isoformat()[:16],
                "tamaño_kb": round(f.stat().st_size / 1024, 1)
            })

corpus_candidatos.sort(key=lambda x: x["tamaño_kb"], reverse=True)
print(f"  → Encontrados {len(corpus_candidatos)} archivos candidatos:")
for c in corpus_candidatos:
    print(f"    {c['archivo']}  ({c['tamaño_kb']} KB)  [{c['modificado']}]")

# 1.3 Buscar vectores de centroide persistidos
print("\n[1.3] Buscando vectores de centroide persistidos...")

vectores_candidatos = []
for ext in ["*.npy", "*.pkl", "*.pt", "*.npz"]:
    for f in REPO_ROOT.rglob(ext):
        nombre = f.name.lower()
        if any(p in nombre for p in ["centroid", "mafapo", "cidh", "polo", "embedding_ref"]):
            vectores_candidatos.append({
                "archivo": str(f.relative_to(REPO_ROOT)),
                "modificado": datetime.fromtimestamp(f.stat().st_mtime).isoformat()[:16],
                "tamaño_kb": round(f.stat().st_size / 1024, 1)
            })

print(f"  → Encontrados {len(vectores_candidatos)} archivos de vectores:")
for v in vectores_candidatos:
    print(f"    {v['archivo']}  ({v['tamaño_kb']} KB)")

# 1.4 Intentar cargar centroide anterior si existe
centroide_anterior = None
for v in vectores_candidatos:
    ruta = REPO_ROOT / v["archivo"]
    try:
        if ruta.suffix == ".npy":
            centroide_anterior = np.load(str(ruta))
            print(f"\n  ✓ Centroide anterior cargado desde: {v['archivo']}")
            print(f"    Dimensión: {centroide_anterior.shape}")
            break
        elif ruta.suffix == ".pkl":
            with open(ruta, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, np.ndarray):
                    centroide_anterior = obj
                    print(f"\n  ✓ Centroide anterior cargado desde: {v['archivo']}")
                    break
                elif isinstance(obj, dict):
                    if "mafapo" in obj:
                        centroide_anterior = np.array(obj["mafapo"])
                        print(f"\n  ✓ Centroide MAFAPO extraído del dict: {v['archivo']}")
                        break
    except Exception as e:
        print(f"    ✗ No se pudo cargar {v['archivo']}: {e}")

if centroide_anterior is None:
    print("\n  ⚠ No se encontró centroide anterior persistido.")
    print("    El centroide deberá reconstruirse desde los textos fuente.")

# 1.5 Auditar cfh.db para trazabilidad
print("\n[1.4] Auditando cfh.db para trazabilidad del centroide...")

if DB_PATH.exists():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Buscar tablas relacionadas
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablas = [r[0] for r in cur.fetchall()]
    print(f"  Tablas en cfh.db: {tablas}")

    tablas_ref = [t for t in tablas if any(p in t.lower()
                  for p in ["mafapo", "cidh", "referencia", "centroid", "polo"])]

    if tablas_ref:
        for t in tablas_ref:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            n = cur.fetchone()[0]
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cur.fetchall()]
            print(f"  Tabla '{t}': {n} filas | columnas: {cols}")

            # Mostrar muestra de textos fuente
            if "texto" in cols or "text" in cols or "contenido" in cols:
                col_text = next(c for c in cols if c in ["texto", "text", "contenido"])
                cur.execute(f"SELECT {col_text} FROM {t} LIMIT 3")
                for row in cur.fetchall():
                    print(f"    Muestra: {str(row[0])[:100]}...")
    else:
        print("  ⚠ No se encontraron tablas de referencia MAFAPO/CIDH en la base.")
        print("    El corpus de referencia probablemente está en archivos sueltos.")

    conn.close()
else:
    print(f"  ⚠ No se encontró cfh.db en {DB_PATH}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2: INVENTARIO DE FUENTES PARA AMPLIAR A 75 TEXTOS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 2: Inventario de fuentes para ampliar a ~75 textos")
print("="*60)

FUENTES_MAFAPO = {
    "A_informe_principal": {
        "descripcion": "Informe 'Unidas por la Memoria y la Verdad' (2021)",
        "url": "https://mafapocolombia.org/wp-content/uploads/2021/03/Unidas-por-la-Memoria-y-la-Verdad.pdf",
        "tipo": "PDF",
        "textos_estimados": 15,
        "ya_incluido": True,  # presumiblemente ya está en los 25
        "acceso": "público",
        "nota": "Fuente primaria. Verificar cuántos fragmentos del PDF se usaron."
    },
    "B_sitio_web_mafapo": {
        "descripcion": "Sitio web mafapocolombia.org — secciones: Sobre nosotros, Manifiesto, Noticias, Comunicados",
        "url": "https://mafapocolombia.org/",
        "tipo": "web_scraping",
        "textos_estimados": 20,
        "ya_incluido": False,
        "acceso": "público",
        "nota": "Scraping ético — solo texto publicado por MAFAPO. Incluir: /sobre-nosotros/, /manifiesto/, /comunicados/, /noticias/"
    },
    "C_informe_jep_2018": {
        "descripcion": "Informe MAFAPO ante la JEP (septiembre 2018)",
        "url": "https://www.jep.gov.co/Sala-de-Prensa/Paginas/Madres-de-Soacha-entregan-segundo-informe-a-la-JEP.aspx",
        "tipo": "PDF",
        "textos_estimados": 10,
        "ya_incluido": False,
        "acceso": "público — descargable desde JEP",
        "nota": "Segundo informe ante la SRVR. Buscar enlace de descarga directo."
    },
    "D_declaraciones_cev": {
        "descripcion": "Declaraciones de integrantes de MAFAPO ante la Comisión de la Verdad",
        "url": "https://web.comisiondelaverdad.co/actualidad/noticias/queremos-una-verdad-completa-y-profunda-madres-de-victimas-de-falsos-positivos",
        "tipo": "web",
        "textos_estimados": 8,
        "ya_incluido": False,
        "acceso": "público",
        "nota": "Entrevistas y declaraciones publicadas por la CEV. Buscar en el repositorio de testimonios."
    },
    "E_corpus_c_segmentos_victimas": {
        "descripcion": "Intervenciones de familiares/víctimas en audiencias JEP Macrocaso 003 (Corpus C diarizado)",
        "url": "local — Corpus C diarizado",
        "tipo": "interno",
        "textos_estimados": 15,
        "ya_incluido": False,
        "acceso": "local — ya tienes el audio diarizado",
        "nota": "FUENTE MÁS VALIOSA: extraer segmentos de SPEAKER identificados como familiares/víctimas en las 5 audiencias. Pyannote ya los tiene separados por speaker. Esto amplía el polo MAFAPO con el lenguaje ORAL de las propias víctimas en el mismo corpus judicial."
    },
    "F_cnmh_mafapo": {
        "descripcion": "Archivo CNMH — textos sobre MAFAPO (Centro Nacional de Memoria Histórica)",
        "url": "https://centrodememoriahistorica.gov.co/tag/mafapo/",
        "tipo": "web",
        "textos_estimados": 5,
        "ya_incluido": False,
        "acceso": "público",
        "nota": "Crónicas y testimonios publicados por el CNMH con voz de integrantes de MAFAPO."
    },
    "G_movice_comunicados": {
        "descripcion": "Comunicados MOVICE sobre Macrocaso 003",
        "url": "https://movimientodevictimas.org/",
        "tipo": "web",
        "textos_estimados": 5,
        "ya_incluido": False,
        "acceso": "público",
        "nota": "Mismo universo semántico. Solo incluir comunicados específicos sobre falsos positivos."
    }
}

total_textos_nuevos = sum(
    f["textos_estimados"] for f in FUENTES_MAFAPO.values() if not f["ya_incluido"]
)
total_con_existentes = 25 + total_textos_nuevos

print(f"\n  Textos actuales (aproximado):  25")
print(f"  Textos nuevos disponibles:    ~{total_textos_nuevos}")
print(f"  Total estimado post-ampliación: ~{total_con_existentes}")
print(f"\n  Fuentes por prioridad:")

for clave, fuente in FUENTES_MAFAPO.items():
    estado = "✓ YA INCLUIDO" if fuente["ya_incluido"] else f"+ {fuente['textos_estimados']} textos"
    print(f"\n  [{clave}] {fuente['descripcion']}")
    print(f"    Estado: {estado}")
    print(f"    Tipo: {fuente['tipo']}  |  Acceso: {fuente['acceso']}")
    print(f"    ⚡ {fuente['nota']}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 3: SCRIPT DE RECOLECCIÓN AUTOMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 3: Recolección automática de textos nuevos")
print("="*60)

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("  ⚠ requests/beautifulsoup4 no instalados. Ejecutar:")
    print("    pip install requests beautifulsoup4")

textos_recolectados = []

if REQUESTS_OK:
    headers = {"User-Agent": "Mozilla/5.0 (academic research CFH-tesis)"}

    # ── B: Sitio web MAFAPO ───────────────────────────────────────────────
    print("\n  [B] Scraping mafapocolombia.org...")
    urls_mafapo = [
        "https://mafapocolombia.org/sobre-nosotros/",
        "https://mafapocolombia.org/manifiesto/",
        "https://mafapocolombia.org/",
    ]

    for url in urls_mafapo:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                # Eliminar nav, footer, scripts
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                # Extraer párrafos sustantivos (>50 chars)
                parrafos = [p.get_text(strip=True) for p in soup.find_all("p")
                           if len(p.get_text(strip=True)) > 50]
                for i, parrafo in enumerate(parrafos):
                    textos_recolectados.append({
                        "fuente": "B_sitio_web_mafapo",
                        "url": url,
                        "id_texto": f"B_{url.split('/')[-2]}_{i:03d}",
                        "texto": parrafo,
                        "longitud": len(parrafo)
                    })
                print(f"    ✓ {url} → {len(parrafos)} párrafos")
            else:
                print(f"    ✗ {url} → HTTP {r.status_code}")
        except Exception as e:
            print(f"    ✗ {url} → {e}")

    # ── D: Comisión de la Verdad ──────────────────────────────────────────
    print("\n  [D] Extrayendo declaraciones CEV...")
    urls_cev = [
        "https://web.comisiondelaverdad.co/actualidad/noticias/queremos-una-verdad-completa-y-profunda-madres-de-victimas-de-falsos-positivos",
    ]
    for url in urls_cev:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                parrafos = [p.get_text(strip=True) for p in soup.find_all("p")
                           if len(p.get_text(strip=True)) > 80]
                for i, parrafo in enumerate(parrafos):
                    textos_recolectados.append({
                        "fuente": "D_declaraciones_cev",
                        "url": url,
                        "id_texto": f"D_cev_{i:03d}",
                        "texto": parrafo,
                        "longitud": len(parrafo)
                    })
                print(f"    ✓ {url} → {len(parrafos)} párrafos")
        except Exception as e:
            print(f"    ✗ {url} → {e}")

    # ── F: CNMH ──────────────────────────────────────────────────────────
    print("\n  [F] Extrayendo textos CNMH...")
    urls_cnmh = [
        "http://experiencias.centromemoria.gov.co/mafapo/",
        "https://centrodememoriahistorica.gov.co/mujeres-con-las-botas-bien-puestas-las-madres-de-soacha-quieren-contarle-al-mundo-su-lucha-contra-la-impunidad/",
    ]
    for url in urls_cnmh:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                parrafos = [p.get_text(strip=True) for p in soup.find_all("p")
                           if len(p.get_text(strip=True)) > 80]
                for i, parrafo in enumerate(parrafos[:15]):  # max 15 por página
                    textos_recolectados.append({
                        "fuente": "F_cnmh",
                        "url": url,
                        "id_texto": f"F_cnmh_{i:03d}",
                        "texto": parrafo,
                        "longitud": len(parrafo)
                    })
                print(f"    ✓ {url} → {len(parrafos)} párrafos (tomados: {min(15, len(parrafos))})")
        except Exception as e:
            print(f"    ✗ {url} → {e}")

    print(f"\n  Total textos recolectados automáticamente: {len(textos_recolectados)}")

    # Guardar textos recolectados
    ruta_textos = OUTPUT_DIR / f"textos_nuevos_mafapo_{TIMESTAMP}.json"
    with open(ruta_textos, "w", encoding="utf-8") as f:
        json.dump(textos_recolectados, f, ensure_ascii=False, indent=2)
    print(f"  → Guardado: {ruta_textos}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 4: EXTRACCIÓN SEGMENTOS VÍCTIMAS DEL CORPUS C (FUENTE MÁS VALIOSA)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 4: Extracción segmentos víctimas del Corpus C diarizado")
print("="*60)
print("  → Esta es la fuente más valiosa: voz oral de familiares/víctimas")
print("    en el mismo corpus judicial.\n")

# Buscar transcripciones del Corpus C
transcripciones_c = []
patrones_trans = ["transcript", "diariz", "corpus_c", "audiencia", "catatumbo",
                  "casanare", "dabeiba", "huila", "costa_caribe"]

for ext in ["*.json", "*.csv", "*.txt"]:
    for f in REPO_ROOT.rglob(ext):
        nombre = f.name.lower()
        if any(p in nombre for p in patrones_trans):
            transcripciones_c.append({
                "archivo": str(f.relative_to(REPO_ROOT)),
                "tamaño_kb": round(f.stat().st_size / 1024, 1),
                "modificado": datetime.fromtimestamp(f.stat().st_mtime).isoformat()[:16]
            })

transcripciones_c.sort(key=lambda x: x["tamaño_kb"], reverse=True)
print(f"  Transcripciones Corpus C encontradas: {len(transcripciones_c)}")
for t in transcripciones_c[:8]:
    print(f"    {t['archivo']}  ({t['tamaño_kb']} KB)")

# Instrucción para extracción manual si no hay auto-detección
print("""
  INSTRUCCIÓN PARA EXTRACCIÓN MANUAL:
  Si tienes las diarizaciones en JSON (formato pyannote), ejecutar:

      python cfh_extraer_segmentos_victimas.py \\
          --diarizacion data/diarizaciones/ \\
          --transcripcion data/transcripciones_corpus_c/ \\
          --speaker_victimas SPEAKER_00,SPEAKER_02  \\  # identificar manualmente
          --output outputs/auditoria_centroide/segmentos_victimas.json

  Los speakers de víctimas/familiares se identifican por:
    - Turnos en secciones de intervención de víctimas (marcados en la agenda JEP)
    - Contraste léxico: mayor uso de "hijo", "madre", "familia" vs. jerga militar
""")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 5: CÁLCULO DEL NUEVO CENTROIDE
# (se ejecuta cuando ya tienes los textos completos)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 5: Cálculo del nuevo centroide (requiere textos completos)")
print("="*60)

def calcular_centroide(textos, modelo, tokenizer, device, batch_size=16):
    """Calcula el centroide semántico de una lista de textos."""
    import torch
    embeddings = []
    modelo.eval()
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = modelo(**enc)
            # CLS token embedding
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(emb)
        if (i // batch_size) % 5 == 0:
            print(f"    Procesados {min(i+batch_size, len(textos))}/{len(textos)} textos...")
    return np.mean(np.vstack(embeddings), axis=0)

def comparar_centroides(c_anterior, c_nuevo, nombre="MAFAPO"):
    """Compara dos centroides y reporta la distancia."""
    if c_anterior is None:
        print(f"  ⚠ No hay centroide anterior para comparar.")
        return
    dist = cosine(c_anterior, c_nuevo)
    similitud = 1 - dist
    print(f"\n  Comparación centroide {nombre}:")
    print(f"    Distancia coseno anterior vs nuevo: {dist:.4f}")
    print(f"    Similitud coseno:                   {similitud:.4f}")
    if dist < 0.05:
        print("    ✓ Centroides muy similares — ampliación consistente con el original.")
    elif dist < 0.15:
        print("    ⚠ Diferencia moderada — verificar que los textos nuevos son coherentes.")
    else:
        print("    ✗ Diferencia grande — revisar si los textos nuevos son apropiados.")

# Placeholder: ejecutar cuando textos_completos esté listo
CALCULAR_CENTROIDE = False  # cambiar a True cuando tengas los textos

if CALCULAR_CENTROIDE:
    print("\n  Cargando modelo ConfliBERT-Spanish...")
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        modelo = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        print(f"  ✓ Modelo cargado en {DEVICE}")

        # Cargar textos completos (combinar originales + nuevos)
        # textos_originales = [...]  # cargar desde el archivo auditado
        # textos_nuevos     = [t["texto"] for t in textos_recolectados]
        # textos_completos  = textos_originales + textos_nuevos

        # centroide_nuevo = calcular_centroide(textos_completos, modelo, tokenizer, DEVICE)
        # comparar_centroides(centroide_anterior, centroide_nuevo, "MAFAPO")

        # Guardar nuevo centroide
        # np.save(OUTPUT_DIR / f"centroide_mafapo_v2_{TIMESTAMP}.npy", centroide_nuevo)
        # print(f"  ✓ Centroide guardado: centroide_mafapo_v2_{TIMESTAMP}.npy")

    except Exception as e:
        print(f"  ✗ Error al cargar modelo: {e}")
else:
    print("\n  [MODO AUDITORÍA] Cálculo del centroide omitido.")
    print("  Para activar: cambiar CALCULAR_CENTROIDE = True")
    print("  y asegurarse de tener los textos completos cargados.")

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 6: VERIFICACIÓN DE DISTINCIÓN y₈ vs y₉
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PARTE 6: Verificación distinción y₈ (MAFAPO) vs y₉ (CIDH)")
print("="*60)
print("""
  Objetivo: confirmar que los dos polos de referencia miden dimensiones distintas.
  El director señaló correlación actual de 0.86 — demasiado alta.
  Meta: < 0.80 después de ampliar el corpus MAFAPO.

  Diagnóstico del problema:
    - El polo CIDH usa lenguaje técnico-jurídico de DDHH internacional.
    - El polo MAFAPO actual (25 textos) puede haber incorporado fragmentos
      con lenguaje jurídico de sus informes ante la JEP (que imitan el
      lenguaje legal), reduciendo la diferencia semántica entre polos.

  Solución con la ampliación:
    - Incluir MÁS testimonios orales y comunicados informales de MAFAPO
      (fuentes B, D, E, F) que usan lenguaje cotidiano, no jurídico.
    - EXCLUIR del corpus MAFAPO los fragmentos que copian lenguaje de
      resoluciones JEP o sentencias CIDH.

  Test de distinción a ejecutar post-ampliación:
    from scipy.stats import pearsonr
    # Calcular y₈ e y₉ sobre muestra de 100 bloques del corpus
    # r, p = pearsonr(distancias_mafapo, distancias_cidh)
    # Meta: r < 0.80 con p < 0.05
""")

# ═══════════════════════════════════════════════════════════════════════════
# REPORTE FINAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("REPORTE FINAL — PRÓXIMOS PASOS")
print("="*60)

reporte = {
    "timestamp": TIMESTAMP,
    "scripts_relacionados_encontrados": len(scripts_candidatos),
    "archivos_corpus_candidatos": len(corpus_candidatos),
    "vectores_centroide_encontrados": len(vectores_candidatos),
    "centroide_anterior_recuperado": centroide_anterior is not None,
    "textos_recolectados_automaticamente": len(textos_recolectados) if REQUESTS_OK else 0,
    "transcripciones_corpus_c_encontradas": len(transcripciones_c),
    "proximos_pasos": [
        "1. Revisar manualmente los scripts_candidatos para identificar EL script que generó el centroide original",
        "2. Reconstruir la lista exacta de 25 textos originales (para trazabilidad)",
        "3. Verificar textos recolectados automáticamente (outputs/auditoria_centroide/textos_nuevos_mafapo_*.json)",
        "4. Añadir segmentos de víctimas del Corpus C diarizado (fuente E — más valiosa)",
        "5. Descargar manualmente: Informe JEP 2018 (fuente C)",
        "6. Activar CALCULAR_CENTROIDE = True y recalcular con textos completos",
        "7. Verificar correlación y₈ vs y₉ < 0.80 post-ampliación",
        "8. Actualizar Tablas 5.3, 5.5, 5.9, 5.14, 5.16 con nuevos valores"
    ]
}

ruta_reporte = OUTPUT_DIR / f"reporte_auditoria_centroide_{TIMESTAMP}.json"
with open(ruta_reporte, "w", encoding="utf-8") as f:
    json.dump(reporte, f, ensure_ascii=False, indent=2)

print(f"\n  Reporte guardado: {ruta_reporte}")
print(f"\n  Scripts relacionados encontrados: {len(scripts_candidatos)}")
print(f"  Centroide anterior recuperado:    {'Sí' if centroide_anterior is not None else 'No — reconstruir desde textos'}")
print(f"  Textos nuevos recolectados:       {len(textos_recolectados) if REQUESTS_OK else '(instalar requests primero)'}")
print(f"  Transcripciones C encontradas:    {len(transcripciones_c)}")

print("""
  ─────────────────────────────────────────────────────
  PASO CRÍTICO INMEDIATO:
  Abrir outputs/auditoria_centroide/reporte_*.json
  y revisar qué script generó el centroide original.
  Ese script tiene la lista de los 25 textos — esa es
  la trazabilidad que necesitas para el director.
  ─────────────────────────────────────────────────────
""")

print("[CFH] Auditoría completada.")
