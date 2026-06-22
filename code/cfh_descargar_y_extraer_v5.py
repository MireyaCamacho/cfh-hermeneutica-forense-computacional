"""
CFH — Descarga segments.json y extrae candidatos para centroide v5
====================================================================
Descarga los 7 segments.json de observaciones de víctimas desde Drive,
los combina con los segments.json del corpus C ya existentes en local,
y extrae candidatos de voz de víctimas con los 3 niveles de filtro.

NO sobrescribe archivos existentes. NO usa rutas hardcodeadas de datos.

Ejecutar:
  conda activate cfh
  python code/cfh_descargar_y_extraer_v5.py
"""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPO       = Path(__file__).resolve().parent.parent
CORPUS_C   = REPO / "corpus_c"
OUTPUT_DIR = REPO / "data" / "referencias"
CORPUS_C.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paso 1: instalar gdown si falta ───────────────────────────────
try:
    import gdown
except ImportError:
    print("Instalando gdown...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
    import gdown

# ── Paso 2: descargar segments.json de observaciones de víctimas ──
# Solo audiencias donde habla la víctima (NO reconocimiento de comparecientes)
ARCHIVOS = {
    "obs_costa_caribe_atanquez_segments.json":   "1KkbGkuLLmZ4KC8hopdcaq6kXcD7Oj_1u",
    "obs_huila_2022_segments.json":              "10-pCJNEtnFlnkPc0D2v7PZPA4YPbTipq",
    "obs_dabeiba_antioquia_dia1_segments.json":  "1qybotbQWdDoZWUA8OdpOMcSK1LKLHMmZ",
    "obs_barranquilla_dia2_segments.json":       "13M7wN9xLIhi5sym36WhgYIFg7BUGRu3F",
    "obs_san_juan_cesar_segments.json":          "1W6go3ZES3onOtTkavz3aaD084wJKUWM8",
    "obs_fase_nacional_20260409_segments.json":  "1xuHids74ScdtRyiEOdg3Sx_d2HI5e-LY",
    "obs_rc05_20260424_segments.json":           "1kwmQ6TqH_LJgxprALHKYwyRLvBT8HnQc",
}

print("=" * 60)
print("CFH — Descarga segments.json para centroide v5")
print("=" * 60)

for nombre, fid in ARCHIVOS.items():
    ruta = CORPUS_C / nombre
    if ruta.exists():
        print(f"  = Ya existe: {nombre} ({ruta.stat().st_size // 1024} KB)")
    else:
        print(f"  Descargando {nombre}...")
        try:
            gdown.download(id=fid, output=str(ruta), quiet=True)
            print(f"  + OK: {nombre} ({ruta.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"  x Error: {nombre} -> {e}")

# ── Paso 3: lexicones de filtro (idénticos a cfh_extraer_victimas_ampliado) ──
FRASES_IDENTIDAD = [
    "soy la mamá", "soy la madre", "soy el padre", "soy hermana",
    "soy hermano", "soy la esposa", "soy el hijo", "soy la hija",
    "vengo por mi", "vengo en nombre", "yo soy la mamá", "yo soy madre",
    "mi nombre es", "represento a", "soy familiar",
]

LEXICON_VICTIMAS = [
    "mi hijo", "mi hija", "mi hermano", "mi hermana", "mi madre", "mi padre",
    "mi esposo", "mi esposa", "mi familiar", "nuestro hijo", "nuestros hijos",
    "nos mataron", "lo mataron", "la mataron", "los mataron", "lo asesinaron",
    "lo llevaron", "nunca volvió", "nunca regresó", "era inocente", "era civil",
    "no era guerrillero", "no era guerrillera", "buscando justicia",
    "buscando verdad", "busco a mi", "encontrar a mi", "saber qué pasó",
    "quiero saber", "pedimos perdón", "exigimos verdad", "necesitamos saber",
    "dolor", "sufrimiento", "duelo", "llorar", "lloramos",
    "familia", "comunidad", "pueblo", "víctima", "víctimas",
    "inocente", "civil", "campesino", "trabajador",
]

FRASES_INSTITUCIONALES = [
    "audiencia pública de reconocimiento", "sala de reconocimiento",
    "jurisdicción especial", "ruta dialógica",
    "reconocimiento de verdad y responsabilidad", "aceptación de responsabilidad",
    "determinación de los hechos", "subcaso", "macrocaso", "compareciente",
    "magistrada", "magistrado", "sala de primera instancia",
    "sección de reconocimiento",
]

def tiene_frases_identidad(t):
    t = t.lower()
    return sum(1 for f in FRASES_IDENTIDAD if f in t)

def score_victima(t):
    t = t.lower()
    return sum(1 for f in LEXICON_VICTIMAS if f in t)

def es_institucional(t):
    t = t.lower()
    return sum(1 for f in FRASES_INSTITUCIONALES if f in t) >= 2

def nivel_filtro(texto):
    if len(texto.strip()) < 30:
        return 0
    if es_institucional(texto):
        return 0
    identidad = tiene_frases_identidad(texto)
    score = score_victima(texto)
    if identidad >= 1 and score >= 2:
        return 1
    elif score >= 4:
        return 2
    elif score >= 2:
        return 3
    return 0

# ── Paso 4: procesar TODOS los segments.json del corpus C ─────────
print("\n" + "=" * 60)
print("EXTRACCIÓN — candidatos de voz de víctimas")
print("=" * 60)

resultados_por_nivel = defaultdict(list)
n_total_segs = 0

for jf in CORPUS_C.glob("*segments*.json"):
    audiencia = jf.stem.replace("_segments", "")
    try:
        with open(jf, encoding="utf-8") as f:
            datos = json.load(f)
        segs = datos if isinstance(datos, list) else next(
            (v for v in datos.values() if isinstance(v, list)), [])
        n_total_segs += len(segs)
        por_nivel = defaultdict(list)
        for s in segs:
            txt = s.get("text", s.get("texto", "")) if isinstance(s, dict) else str(s)
            niv = nivel_filtro(txt)
            if niv > 0:
                por_nivel[niv].append({
                    "texto": txt.strip(), "audiencia": audiencia, "nivel": niv,
                    "score": score_victima(txt),
                    "identidad": tiene_frases_identidad(txt),
                    "longitud": len(txt),
                })
        for niv in [1, 2, 3]:
            resultados_por_nivel[niv].extend(por_nivel[niv])
        print(f"  {audiencia[:50]}: {len(segs)} segs -> "
              f"N1={len(por_nivel[1])} N2={len(por_nivel[2])} N3={len(por_nivel[3])}")
    except Exception as e:
        print(f"  x Error en {jf.name}: {e}")

print(f"\n  Total segmentos procesados: {n_total_segs}")
for niv in [1, 2, 3]:
    print(f"  Nivel {niv}: {len(resultados_por_nivel[niv])} candidatos")

# ── Paso 5: selección estratificada por audiencia ────────────────
MAX_POR_AUDIENCIA = {1: 25, 2: 18, 3: 12}
seleccionados = []
audiencias = set()
for niv in [1, 2, 3]:
    for s in resultados_por_nivel[niv]:
        audiencias.add(s["audiencia"])

for audiencia in sorted(audiencias):
    for niv in [1, 2, 3]:
        cands = [s for s in resultados_por_nivel[niv] if s["audiencia"] == audiencia]
        cands_sorted = sorted(cands, key=lambda x: x["score"] + x["identidad"] * 3, reverse=True)
        seleccionados.extend(cands_sorted[:MAX_POR_AUDIENCIA[niv]])

# Deduplicar por texto
vistos, unicos = set(), []
for s in seleccionados:
    clave = s["texto"][:100]
    if clave not in vistos:
        vistos.add(clave)
        unicos.append(s)

print(f"\n  Seleccionados únicos: {len(unicos)}")

# ── Paso 6: guardar candidatos v5 ────────────────────────────────
output = {
    "version": "corpus_c_victimas_v5",
    "timestamp": datetime.now().isoformat(),
    "total": len(unicos),
    "por_nivel": {str(n): len([s for s in unicos if s["nivel"] == n]) for n in [1, 2, 3]},
    "por_audiencia": {a: len([s for s in unicos if s["audiencia"] == a]) for a in sorted(audiencias)},
    "segmentos": unicos,
}
ruta_out = OUTPUT_DIR / "corpus_c_victimas_v5.json"
with open(ruta_out, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n  -> Guardado: {ruta_out}")
print(f"\n  Total v5: {len(unicos)} segmentos nuevos")
print(f"  + 169 textos del centroide v4")
total_estimado = len(unicos) + 169
print(f"  -> Centroide v5 estimado: ~{total_estimado} textos")
print(f"  -> Margen de error estimado: ±{100 / (total_estimado ** 0.5):.1f}%")
print("\n[CFH] Extracción completada. Siguiente: calcular embeddings v5.")
