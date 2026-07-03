# -*- coding: utf-8 -*-
"""
cfh_ampliar_patrones_rep.py
============================
Corrige el bug del y10_rep=0.000 para comparecientes que SI reconocen.

CAUSA (confirmada por diagnostico):
  Los patrones regex del REPExtractor fueron disenados para el lenguaje
  FORMAL de autos escritos JEP ("reconozco mi responsabilidad en calidad de").
  Los comparecientes en audiencia ORAL hablan distinto: "yo asesine",
  "le pido perdon," (con coma, no preposicion), "causamos dano", "yo fui".
  Esas formas no matchean -> score_raw=0 -> REP=0 (falso negativo).

QUE HACE ESTE SCRIPT (quirurgico, NO reescribe la logica):
  1. Hace backup del extractor.
  2. Inserta nuevos patrones ORALES en las listas existentes
     REP_RECONOCIMIENTO_FRASES y REP_REPARACION_FRASES, justo antes del
     cierre "]" de cada lista.
  3. Valida que el archivo siga compilando y que los patrones nuevos
     matcheen las frases reales de los comparecientes.

Los patrones nuevos respetan tildes (el texto real las tiene) y usan
alternancias tolerantes a puntuacion oral.

Uso:
    python cfh_ampliar_patrones_rep.py            # aplica
    python cfh_ampliar_patrones_rep.py --dry-run  # solo muestra que haria
"""

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVO = Path("code/src/features/y10_rep_extractor.py")

# ---------------------------------------------------------------------------
# NUEVOS PATRONES - habla oral de comparecientes (con tildes, como el texto real)
# ---------------------------------------------------------------------------

# Reconocimiento oral directo de la autoria del crimen
NUEVOS_RECONOCIMIENTO = [
    # confesion directa del homicidio en primera persona
    r'\byo\s+(?:asesiné|maté|ejecuté|disparé|le\s+disparé|los\s+maté|lo\s+maté)',
    r'\b(?:asesiné|maté|ejecuté)\s+(?:a\s+)?(?:ese|este|el|la|los|las|un|una|don|doña)',
    # "yo fui" (autoconfesion) seguido de puntuacion o pronombre, no solo quien/el que
    r'\byo\s+fui\b(?![\w\s]{0,10}(?:testigo|víctima))',
    r'\b(?:le\s+)?dije\s+yo\s+fui',
    # confesion en plural (accion colectiva reconocida)
    r'\bnosotros\s+(?:montamos|hicimos|cometimos|causamos|matamos|ejecutamos)',
    r'\b(?:causamos|cometimos|ocasionamos|provocamos)\s+(?:mucho\s+)?(?:daño|dolor|sufrimiento|muerte)',
    r'\bmontamos\s+un\s+retén',
    # reconocimiento del deshonor / falta
    r'\bdeshonré\s+(?:el|la|mi)',
    r'\b(?:le\s+)?fallé\s+(?:a\s+)?(?:mi|la|las|los|el)',
    # aceptacion coloquial
    r'\bno\s+me\s+lo\s+merezco',
    r'\b(?:acepto|reconozco|asumo)\s+(?:que\s+)?(?:lo\s+)?(?:hice|participé|estuve|cometí)',
    r'\breconozco\s+mi\s+responsabilidad\b',
    r'\baceptar\s+mi\s+responsabilidad\b',
]

# Pedido de perdon oral - sin exigir preposicion obligatoria
NUEVOS_REPARACION = [
    # "pido perdon" seguido de coma, punto, "y", "de", "a", nombre, o fin
    r'\b(?:le|les|te)?\s*pido\s+perdón\b',
    r'\bpedir(?:le|les)?\s+perdón\b',
    r'\bpedirles\s+perdón\b',
    r'\bquiero\s+pedir(?:le|les)?\s+perdón',
    # esperanza de ser perdonado (reconocimiento del dano)
    r'\bespero\s+que\s+(?:algún\s+día\s+)?(?:nos|me)\s+perdonen',
    r'\b(?:nos|me)\s+puedan\s+perdonar',
    r'\bsi\s+algún\s+día\s+me\s+pueden\s+perdonar',
    # disculpas orales
    r'\b(?:le|les)\s+(?:pido|ofrezco)\s+(?:mis\s+)?disculpas',
    r'\bpido\s+disculpas\b',
    # perdon a persona nombrada (nombre propio tras perdon, sin preposicion rigida)
    r'\bperdón\s+(?:a|de)?\s*[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ]',
]


def insertar_en_lista(contenido, nombre_lista, nuevos_patrones):
    """Inserta patrones antes del ']' de cierre de una lista de frases."""
    # localizar "NOMBRE_LISTA = [" y su "]" de cierre
    m = re.search(rf'{nombre_lista}\s*=\s*\[', contenido)
    if not m:
        raise ValueError(f"No se encontro la lista {nombre_lista}")
    inicio = m.end()
    # encontrar el ']' que cierra (contando corchetes)
    depth = 1
    i = inicio
    while i < len(contenido) and depth > 0:
        if contenido[i] == '[':
            depth += 1
        elif contenido[i] == ']':
            depth -= 1
        i += 1
    cierre = i - 1  # posicion del ']'

    # construir bloque de insercion
    bloque = "\n    # --- patrones ORALES anadidos (habla de comparecientes) ---\n"
    for p in nuevos_patrones:
        # escapar comillas simples internas usando raw string con comillas dobles
        bloque += f'    r"{p}",\n'

    nuevo = contenido[:cierre] + bloque + contenido[cierre:]
    return nuevo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ARCHIVO.exists():
        print(f"[ERROR] no existe {ARCHIVO}")
        sys.exit(1)

    contenido = ARCHIVO.read_text(encoding="utf-8")
    original = contenido

    print("=" * 70)
    print("AMPLIACION DE PATRONES REP (habla oral)")
    print("=" * 70)

    # insertar en las dos listas
    try:
        contenido = insertar_en_lista(contenido, "REP_RECONOCIMIENTO_FRASES", NUEVOS_RECONOCIMIENTO)
        print(f"  + {len(NUEVOS_RECONOCIMIENTO)} patrones en REP_RECONOCIMIENTO_FRASES")
        contenido = insertar_en_lista(contenido, "REP_REPARACION_FRASES", NUEVOS_REPARACION)
        print(f"  + {len(NUEVOS_REPARACION)} patrones en REP_REPARACION_FRASES")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # validar que compila
    try:
        compile(contenido, str(ARCHIVO), "exec")
        print("  [OK] el archivo modificado compila sin errores de sintaxis")
    except SyntaxError as e:
        print(f"  [ERROR] el archivo modificado NO compila: {e}")
        print("  No se escribe nada.")
        sys.exit(1)

    # validar que los patrones nuevos matchean frases reales
    print("\n  Validando patrones nuevos contra frases reales:")
    frases_test = [
        "Yo asesiné a ese señor, le pido perdón, disculpas",
        "le puse la cara y le dije yo fui",
        "Le pido perdón a todas las familias",
        "causamos mucho daño a todos ustedes",
        "espero que algún día nos perdonen",
        "nosotros montamos un retén",
        "deshonré el buen nombre de mi",
        "no me lo merezco pero le puse la cara",
        "quiero pedirles perdón a ustedes",
        "pido perdón, adiós a ustedes",
    ]
    todos_nuevos = NUEVOS_RECONOCIMIENTO + NUEVOS_REPARACION
    compilados = [re.compile(p, re.IGNORECASE) for p in todos_nuevos]
    for fr in frases_test:
        n = sum(1 for p in compilados if p.search(fr))
        estado = "OK" if n > 0 else "-- SIN MATCH"
        print(f"    [{n}] {estado:12s} '{fr[:50]}'")

    if args.dry_run:
        print("\n  [dry-run] NO se escribe el archivo.")
        return

    # backup y escritura
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVO.with_name(f"y10_rep_extractor_BACKUP_{ts}.py")
    shutil.copy2(ARCHIVO, backup)
    print(f"\n  backup -> {backup.name}")

    ARCHIVO.write_text(contenido, encoding="utf-8")
    print(f"  ESCRITO: {ARCHIVO}")
    print("\n  Siguiente paso: re-correr el REP sobre los 47 comparecientes")
    print("  y verificar que Restrepo y los otros falsos negativos ya puntuan.")


if __name__ == "__main__":
    main()
