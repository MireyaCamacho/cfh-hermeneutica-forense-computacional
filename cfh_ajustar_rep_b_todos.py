# -*- coding: utf-8 -*-
"""
cfh_ajustar_rep_b_todos.py
===========================
Refina la separacion de detectores REP por corpus (decision teorica de Mireya):

  A (justicia ordinaria):  SOLO ruptura epistemica institucional
                           -> restitucion + DIH + reparacion
  B (autos JEP):           TODOS los detectores
                           -> los autos RECOGEN los reconocimientos de los
                              comparecientes (justicia transicional), asi que
                              el reconocimiento en 1a persona SI cuenta,
                              ademas del lenguaje institucional.
  C (habla oral):          reconocimiento + perdon + restitucion
                           (sin el DIH tecnico del tribunal)

Reemplaza el bloque condicional insertado por cfh_separar_rep_por_corpus.py
(texto exacto conocido) por la version de tres ramas. No toca nada mas.

Uso:
    python cfh_ajustar_rep_b_todos.py --dry-run
    python cfh_ajustar_rep_b_todos.py
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVO = Path("code/src/features/y10_rep_extractor.py")

BLOQUE_VIEJO = """        # SEPARACION POR CORPUS (diseno teorico CFH):
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

BLOQUE_NUEVO = """        # SEPARACION POR CORPUS (diseno teorico CFH):
        # A (justicia ordinaria): ruptura epistemica institucional
        #   = dejar de eufemizar y nombrar en terminos de DIH/victimas/reparacion.
        # B (autos JEP): TODOS los detectores — el auto recoge los
        #   reconocimientos de los comparecientes (justicia transicional),
        #   ademas del lenguaje institucional.
        # C (habla oral del compareciente): reconocimiento en 1a persona
        #   = aceptacion de autoria + perdon + restitucion de la victima
        #   (sin el DIH tecnico del tribunal).
        if corpus_type == "A":
            all_instances.extend(self._detect_restitución(text, sentences))
            all_instances.extend(self._detect_dih(text, sentences))
            all_instances.extend(self._detect_reparación(text, sentences))
        elif corpus_type == "B":
            all_instances.extend(self._detect_reconocimiento(text, sentences))
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

    if 'elif corpus_type == "B":' in contenido:
        print("  Ya parece aplicada la version de tres ramas (A / B / C).")
        try:
            compile(contenido, str(ARCHIVO), "exec")
            print("  [OK] compila.")
        except SyntaxError as e:
            print(f"  [ERROR] no compila: {e}")
        return

    if BLOQUE_VIEJO not in contenido:
        print("[ERROR] no se encontro el bloque condicional esperado (verbatim).")
        print("  Lineas actuales con _detect_ / corpus_type en extract():")
        for i, l in enumerate(contenido.split("\n"), 1):
            if ("_detect_" in l and "extend" in l) or "corpus_type in" in l or "corpus_type ==" in l:
                print(f"    linea {i}: {repr(l)}")
        print("\n  Pega esas lineas exactas para ajustar el script. NO se toca nada.")
        sys.exit(1)

    nuevo = contenido.replace(BLOQUE_VIEJO, BLOQUE_NUEVO, 1)

    print("=" * 70)
    print("SEPARACION REP EN TRES RAMAS")
    print("=" * 70)
    print("  A -> restitucion + DIH + reparacion (institucional)")
    print("  B -> reconocimiento + restitucion + DIH + reparacion (TODOS)")
    print("  C -> reconocimiento + reparacion + restitucion (oral)")

    try:
        compile(nuevo, str(ARCHIVO), "exec")
        print("\n  [OK] compila sin errores")
    except SyntaxError as e:
        print(f"\n  [ERROR] no compila: {e}. NO se escribe.")
        sys.exit(1)

    if args.dry_run:
        print("\n  [dry-run] NO se escribe el archivo.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVO.with_name(f"y10_rep_extractor_BACKUP_3ramas_{ts}.py")
    shutil.copy2(ARCHIVO, backup)
    print(f"\n  backup -> {backup.name}")
    ARCHIVO.write_text(nuevo, encoding="utf-8")
    print(f"  ESCRITO: {ARCHIVO}")
    print("\n  Siguiente (en orden):")
    print("  1. python cfh_recorrer_rep_ab.py --dry-run   (B ahora con reconocimiento)")
    print("  2. aplicar sin --dry-run")
    print("  3. python cfh_ajustar_unificador_corpus_c.py --dry-run  (si aun no)")
    print("  4. python reproducible\\cfh_unificar_corpus_c.py")


if __name__ == "__main__":
    main()
