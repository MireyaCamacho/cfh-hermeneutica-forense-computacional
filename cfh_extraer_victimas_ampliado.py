"""
CFH — Extracción ampliada de segmentos de víctimas del Corpus C
================================================================
Objetivo: extraer el máximo de segmentos de voz directa de familiares
para ampliar el centroide MAFAPO de 67 → ~200 textos.

Filtros por nivel:
  - Nivel 1 (estricto):  ≥2 frases de identidad directa ("soy la mamá", "mi hijo")
  - Nivel 2 (moderado):  ≥3 términos del léxico de víctimas sin lenguaje institucional
  - Nivel 3 (amplio):    ≥2 términos léxico víctimas, descartando solo los claramente institucionales

Ejecutar:
  python cfh_extraer_victimas_ampliado.py
"""
import json
import re
from pathlib import Path
from collections import defaultdict

REPO       = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
CORPUS_C   = Path(r"G:\Mi unidad\CHF_Corpus\corpus_c")
OUTPUT_DIR = Path(r"G:\Mi unidad\CHF_Corpus\referencias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lexicones ──────────────────────────────────────────────────────────────
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
    "audiencia pública de reconocimiento",
    "sala de reconocimiento",
    "jurisdicción especial",
    "ruta dialógica",
    "reconocimiento de verdad y responsabilidad",
    "aceptación de responsabilidad",
    "determinación de los hechos",
    "subcaso", "macrocaso", "compareciente",
    "magistrada", "magistrado", "sala de primera instancia",
    "sección de reconocimiento",
]

def tiene_frases_identidad(texto):
    t = texto.lower()
    return sum(1 for f in FRASES_IDENTIDAD if f in t)

def score_victima(texto):
    t = texto.lower()
    return sum(1 for f in LEXICON_VICTIMAS if f in t)

def es_institucional(texto):
    t = texto.lower()
    return sum(1 for f in FRASES_INSTITUCIONALES if f in t) >= 2

def nivel_filtro(texto):
    """Retorna 1, 2, 3 según nivel de filtro que pasa, o 0 si no pasa ninguno."""
    if len(texto.strip()) < 30:
        return 0
    if es_institucional(texto):
        return 0

    identidad = tiene_frases_identidad(texto)
    score = score_victima(texto)

    if identidad >= 1 and score >= 2:
        return 1  # estricto — voz directa identificada
    elif score >= 4:
        return 2  # moderado — alta densidad léxico víctimas
    elif score >= 2:
        return 3  # amplio — mínima señal de voz víctima
    return 0

# ── Procesar todos los JSONs del Corpus C ──────────────────────────────────
print("="*60)
print("EXTRACCIÓN AMPLIADA — SEGMENTOS VÍCTIMAS CORPUS C")
print("="*60)

resultados_por_nivel = defaultdict(list)
n_total_segs = 0

for jf in CORPUS_C.glob("*segments*.json"):
    audiencia = jf.stem.replace("_segments","").replace("_audiencia_reconocimiento","")
    print(f"\n  Procesando: {jf.name}")

    try:
        with open(jf, encoding="utf-8") as f:
            datos = json.load(f)

        segs = datos if isinstance(datos, list) else next(
            (v for v in datos.values() if isinstance(v, list)), [])

        n_total_segs += len(segs)
        por_nivel = defaultdict(list)

        for s in segs:
            txt = s.get("text", s.get("texto", s.get("transcript",""))) if isinstance(s,dict) else str(s)
            nivel = nivel_filtro(txt)
            if nivel > 0:
                por_nivel[nivel].append({
                    "texto": txt,
                    "audiencia": audiencia,
                    "nivel": nivel,
                    "score": score_victima(txt),
                    "identidad": tiene_frases_identidad(txt),
                    "longitud": len(txt)
                })

        for niv in [1,2,3]:
            cands = sorted(por_nivel[niv], key=lambda x: x["score"], reverse=True)
            print(f"    Nivel {niv}: {len(cands)} candidatos")
            resultados_por_nivel[niv].extend(cands)

    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n  Total segmentos procesados: {n_total_segs}")
print(f"\n  Candidatos por nivel:")
for niv in [1,2,3]:
    print(f"    Nivel {niv} (estricto→amplio): {len(resultados_por_nivel[niv])}")

# ── Selección estratificada por audiencia ──────────────────────────────────
print("\n  Selección estratificada (máx por audiencia para representatividad):")

MAX_POR_AUDIENCIA = {1: 20, 2: 15, 3: 10}  # máx por nivel por audiencia
seleccionados = []

audiencias = set()
for niv in [1,2,3]:
    for s in resultados_por_nivel[niv]:
        audiencias.add(s["audiencia"])

for audiencia in sorted(audiencias):
    print(f"\n  [{audiencia}]")
    for niv in [1,2,3]:
        cands = [s for s in resultados_por_nivel[niv] if s["audiencia"]==audiencia]
        cands_sorted = sorted(cands, key=lambda x: x["score"]+x["identidad"]*3, reverse=True)
        tomar = min(MAX_POR_AUDIENCIA[niv], len(cands_sorted))
        seleccionados.extend(cands_sorted[:tomar])
        print(f"    Nivel {niv}: {len(cands)} candidatos → tomados {tomar}")

# Deduplicar por texto
textos_vistos = set()
seleccionados_unicos = []
for s in seleccionados:
    clave = s["texto"][:100]
    if clave not in textos_vistos:
        textos_vistos.add(clave)
        seleccionados_unicos.append(s)

print(f"\n  Total seleccionados únicos: {len(seleccionados_unicos)}")

# ── Guardar ────────────────────────────────────────────────────────────────
import json as json_mod
from datetime import datetime

output = {
    "version": "corpus_c_victimas_ampliado",
    "timestamp": datetime.now().isoformat(),
    "total": len(seleccionados_unicos),
    "por_nivel": {
        str(niv): len([s for s in seleccionados_unicos if s["nivel"]==niv])
        for niv in [1,2,3]
    },
    "por_audiencia": {
        aud: len([s for s in seleccionados_unicos if s["audiencia"]==aud])
        for aud in sorted(audiencias)
    },
    "segmentos": seleccionados_unicos
}

ruta = OUTPUT_DIR / "corpus_c_victimas_ampliado.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(ruta, "w", encoding="utf-8") as f:
    json_mod.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n  → Guardado: {ruta}")
print(f"\n  Muestra de nivel 1 (voz directa identificada):")
for s in [x for x in seleccionados_unicos if x["nivel"]==1][:5]:
    print(f"    [{s['audiencia']}] score={s['score']}: {s['texto'][:80]}...")

print(f"\n  Con {len(seleccionados_unicos)} segmentos nuevos + 67 existentes")
print(f"  → Total estimado centroide v4: ~{len(seleccionados_unicos) + 67} textos")
print(f"  → Margen de error: ±{100/((len(seleccionados_unicos)+67)**0.5):.1f}%")

print("\n[CFH] Completado.")
