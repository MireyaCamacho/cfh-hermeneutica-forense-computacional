# -*- coding: utf-8 -*-
"""
cfh_fix_maxlength_rep_v2.py
============================
Version corregida: inserta self._nlp.max_length DESPUES del bloque try/except
completo (el fix anterior lo metia dentro del try y rompia la sintaxis).

Busca el patron:
    try:
        self._nlp = spacy.load(model_name)
    except OSError:
        raise OSError(
            ...
        )
y agrega, con la misma indentacion del 'try', la linea:
    self._nlp.max_length = 3_000_000

Uso:
    python cfh_fix_maxlength_rep_v2.py --dry-run
    python cfh_fix_maxlength_rep_v2.py
"""

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVO = Path("code/src/features/y10_rep_extractor.py")
MAXLEN = 3_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    contenido = ARCHIVO.read_text(encoding="utf-8")

    if f"self._nlp.max_length" in contenido:
        print("  Ya tiene self._nlp.max_length. Nada que hacer.")
        # verificar que compila igual
        try:
            compile(contenido, str(ARCHIVO), "exec")
            print("  [OK] el archivo compila.")
        except SyntaxError as e:
            print(f"  [ERROR] no compila: {e}")
        return

    # localizar el bloque try/except que carga spacy, capturando hasta el
    # cierre del except (la linea con solo ')' indentada, fin del raise OSError(...))
    # Estrategia: encontrar 'self._nlp = spacy.load' y luego el proximo
    # 'except OSError:' y su bloque, hasta la linea que cierra el raise.
    patron = re.compile(
        r"(?P<indent>[ \t]*)try:\s*\n"
        r"[ \t]*self\._nlp\s*=\s*spacy\.load\([^\n]*\)\s*\n"
        r"[ \t]*except\s+OSError:\s*\n"
        r"(?:[ \t]*raise\s+OSError\(\s*\n"
        r"(?:[^\n]*\n)*?"          # cuerpo del raise (varias lineas)
        r"[ \t]*\)\s*\n)"
    )

    m = patron.search(contenido)
    if not m:
        print("[ERROR] no se encontro el bloque try/except de carga de spaCy.")
        print("  Insertalo manualmente: despues del bloque try/except que carga")
        print(f"  spaCy, agrega (misma indentacion que 'try'):")
        print(f"      self._nlp.max_length = {MAXLEN}")
        sys.exit(1)

    indent = m.group("indent")
    bloque = m.group(0)
    insercion = f"{indent}self._nlp.max_length = {MAXLEN}  # autos JEP muy largos\n"
    nuevo = contenido.replace(bloque, bloque + insercion, 1)

    print("=" * 70)
    print("FIX v2 max_length (despues del try/except)")
    print("=" * 70)
    print("  bloque encontrado:")
    for l in bloque.rstrip().split("\n"):
        print(f"    | {l}")
    print(f"\n  se insertara despues: {insercion.strip()}")

    try:
        compile(nuevo, str(ARCHIVO), "exec")
        print("\n  [OK] compila sin errores")
    except SyntaxError as e:
        print(f"\n  [ERROR] no compila: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n  [dry-run] NO se escribe.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVO.with_name(f"y10_rep_extractor_BACKUP_maxlen2_{ts}.py")
    shutil.copy2(ARCHIVO, backup)
    print(f"\n  backup -> {backup.name}")
    ARCHIVO.write_text(nuevo, encoding="utf-8")
    print(f"  ESCRITO: {ARCHIVO}")
    print("\n  Ahora re-corre: python cfh_recorrer_rep_todo.py --dry-run")


if __name__ == "__main__":
    main()
