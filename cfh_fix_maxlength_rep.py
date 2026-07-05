# -*- coding: utf-8 -*-
"""
cfh_fix_maxlength_rep.py
=========================
Corrige el error [E088] de spaCy: secciones del Corpus B (autos JEP) superan
el limite de 1.000.000 chars de spaCy. El REPExtractor cae al llamar self._nlp.

SOLUCION: subir nlp.max_length a 3.000.000 justo despues de cargar el modelo
spaCy en el __init__ del REPExtractor. Es seguro porque el REP es analisis
lexico (regex + segmentacion), no depende del parser/NER pesado.

Inserta la linea:  self._nlp.max_length = 3_000_000
justo despues de la asignacion de self._nlp = spacy.load(...) (o similar).

Uso:
    python cfh_fix_maxlength_rep.py --dry-run
    python cfh_fix_maxlength_rep.py
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

    if not ARCHIVO.exists():
        print(f"[ERROR] no existe {ARCHIVO}")
        sys.exit(1)

    contenido = ARCHIVO.read_text(encoding="utf-8")

    if "max_length" in contenido and "3_000_000" in contenido or "3000000" in contenido:
        print("  Ya parece tener max_length ajustado. Revisar manualmente.")
        # continuar de todos modos para mostrar

    # localizar donde se asigna self._nlp = spacy.load(...)
    # patrones posibles
    patrones = [
        r"(self\._nlp\s*=\s*spacy\.load\([^\n]*\)\n)",
        r"(self\._nlp\s*=\s*_load[^\n]*\n)",
        r"(self\._nlp\s*=\s*nlp\n)",
        r"(self\._nlp\s*=\s*[^\n]+\n)",
    ]
    m = None
    for pat in patrones:
        m = re.search(pat, contenido)
        if m:
            break

    if not m:
        print("[ERROR] no se encontro la asignacion self._nlp = ...")
        print("  Busca manualmente en __init__ donde se carga spaCy y agrega:")
        print(f"      self._nlp.max_length = {MAXLEN}")
        # mostrar lineas con _nlp para ayudar
        for i, l in enumerate(contenido.split("\n"), 1):
            if "_nlp" in l and ("spacy" in l or "load" in l or "=" in l):
                print(f"    linea {i}: {l.strip()[:80]}")
        sys.exit(1)

    linea_asignacion = m.group(1)
    # detectar indentacion
    indent = re.match(r"(\s*)", linea_asignacion).group(1)
    insercion = f"{indent}self._nlp.max_length = {MAXLEN}  # autos JEP muy largos\n"

    if f"self._nlp.max_length = {MAXLEN}" in contenido:
        print("  Ya esta insertada la linea de max_length. Nada que hacer.")
        return

    nuevo = contenido.replace(linea_asignacion, linea_asignacion + insercion, 1)

    print("=" * 70)
    print("FIX max_length en REPExtractor")
    print("=" * 70)
    print(f"  asignacion encontrada: {linea_asignacion.strip()[:70]}")
    print(f"  se insertara:          {insercion.strip()}")

    # validar que compila
    try:
        compile(nuevo, str(ARCHIVO), "exec")
        print("  [OK] compila sin errores")
    except SyntaxError as e:
        print(f"  [ERROR] no compila: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n  [dry-run] NO se escribe.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVO.with_name(f"y10_rep_extractor_BACKUP_maxlen_{ts}.py")
    shutil.copy2(ARCHIVO, backup)
    print(f"\n  backup -> {backup.name}")
    ARCHIVO.write_text(nuevo, encoding="utf-8")
    print(f"  ESCRITO: {ARCHIVO}")
    print(f"\n  Ahora re-corre: python cfh_recorrer_rep_todo.py --dry-run")


if __name__ == "__main__":
    main()
