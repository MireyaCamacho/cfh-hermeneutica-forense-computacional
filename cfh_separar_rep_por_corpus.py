# -*- coding: utf-8 -*-
"""
cfh_separar_rep_por_corpus.py
==============================
Corrige la CONTAMINACION CRUZADA del REPExtractor: hoy aplica los 4 detectores
a todos los corpus por igual, mezclando dos operacionalizaciones distintas del
REP que son teoricamente diferentes:

  CORPUS A / B (texto escrito judicial):
     REP = RUPTURA EPISTEMICA institucional = el tribunal deja de eufemizar
     ("baja en combate") y nombra el crimen en terminos de DIH, victimas y
     reparacion ("ejecucion extrajudicial", "persona protegida", "victima").
     -> detectores: restitucion + DIH + reparacion
     -> NO reconocimiento en 1a persona (los jueces no confiesan)

  CORPUS C (habla oral del compareciente):
     REP = RECONOCIMIENTO del compareciente en 1a persona (sing/plural):
     aceptacion de autoria + perdon + restitucion de la victima.
     -> detectores: reconocimiento + reparacion + restitucion
     -> NO el DIH tecnico del tribunal (el compareciente no cita articulos)

CAMBIO QUIRURGICO: reemplaza SOLO el bloque de 4 llamadas a los detectores
dentro de extract() por un condicional segun corpus_type. No toca nada mas.

Uso:
    python cfh_separar_rep_por_corpus.py --dry-run
    python cfh_separar_rep_por_corpus.py
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVO = Path("code/src/features/y10_rep_extractor.py")

# El bloque actual (las 4 llamadas seguidas). Se busca de forma tolerante a
# los nombres con tilde detectando las lineas por su parte estable.
# Usamos las lineas literales tal como estan en el archivo.
BLOQUE_VIEJO = """        all_instances.extend(self._detect_reconocimiento(text, sentences))
        all_instances.extend(self._detect_restitución(text, sentences))
        all_instances.extend(self._detect_dih(text, sentences))
        all_instances.extend(self._detect_reparación(text, sentences))"""

BLOQUE_NUEVO = """        # SEPARACION POR CORPUS (diseno teorico CFH):
        # A/B (texto judicial escrito): ruptura epistemica institucional
        #   = dejar de eufemizar y nombrar en terminos de DIH/victimas/reparacion.
        # C (habla oral del compareciente): reconocimiento en 1a persona
        #   = aceptacion de autoria + perdon + restitucion de la victima.
        if corpus_type in ("A", "B"):
            all_instances.extend(self._detect_restitución(text, sentences))
            all_instances.extend(self._detect_dih(text, sentences))
            all_instances.extend(self._detect_reparación(text, sentences))
        else:  # Corpus C (oral)
            all_instances.extend(self._detect_reconocimiento(text, sentences))
            all_instances.extend(self._detect_reparación(text, sentences))
            all_instances.extend(self._detect_restitución(text, sentences))"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ARCHIVO.exists():
        print(f"[ERROR] no existe {ARCHIVO}")
        sys.exit(1)

    contenido = ARCHIVO.read_text(encoding="utf-8")

    # verificar si ya esta aplicado
    if 'if corpus_type in ("A", "B"):' in contenido:
        print("  Ya parece estar aplicada la separacion por corpus.")
        try:
            compile(contenido, str(ARCHIVO), "exec")
            print("  [OK] el archivo compila.")
        except SyntaxError as e:
            print(f"  [ERROR] no compila: {e}")
        return

    if BLOQUE_VIEJO not in contenido:
        print("[ERROR] no se encontro el bloque de 4 detectores esperado.")
        print("  Puede que las tildes de los nombres difieran. Lineas con _detect_:")
        for i, l in enumerate(contenido.split("\n"), 1):
            if "_detect_" in l and "extend" in l:
                print(f"    linea {i}: {repr(l)}")
        print("\n  Copia esas 4 lineas EXACTAS y avisame para ajustar el script.")
        sys.exit(1)

    nuevo = contenido.replace(BLOQUE_VIEJO, BLOQUE_NUEVO, 1)

    print("=" * 70)
    print("SEPARACION DE DETECTORES REP POR CORPUS")
    print("=" * 70)
    print("  A/B -> restitucion + DIH + reparacion (ruptura epistemica institucional)")
    print("  C   -> reconocimiento + reparacion + restitucion (compareciente 1a persona)")

    # validar que compila
    try:
        compile(nuevo, str(ARCHIVO), "exec")
        print("\n  [OK] el archivo modificado compila sin errores")
    except SyntaxError as e:
        print(f"\n  [ERROR] no compila: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n  [dry-run] NO se escribe el archivo.")
        print("\n  Bloque que se insertaria:")
        for l in BLOQUE_NUEVO.split("\n"):
            print("    " + l)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVO.with_name(f"y10_rep_extractor_BACKUP_corpus_{ts}.py")
    shutil.copy2(ARCHIVO, backup)
    print(f"\n  backup -> {backup.name}")
    ARCHIVO.write_text(nuevo, encoding="utf-8")
    print(f"  ESCRITO: {ARCHIVO}")
    print("\n  Siguiente: python cfh_recorrer_rep_todo.py --dry-run")
    print("  Esperado: Corpus A cae casi a 0 en 'yo fui/yo dispare' (ya no cuenta),")
    print("  mantiene el vocabulario DIH; Corpus C mide reconocimiento oral puro.")


if __name__ == "__main__":
    main()
