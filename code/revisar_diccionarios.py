# -*- coding: utf-8 -*-
"""Revisa los diccionarios/lexicones de los extractores SA, NV, REP
para auditar si cada uno mide su concepto correcto."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "code" / "src" / "features"

archivos = [
    ("y2_sa_extractor.py", "SA / y2 (Supresion de Agentividad)"),
    ("y4_nv_extractor.py", "NV / y4 (Negacion de Victimizacion - debe recaracterizar victima como combatiente)"),
    ("y10_rep_extractor.py", "REP / y10 (Ruptura Epistemica Positiva - valida perspectiva victimas)"),
]

for fname, desc in archivos:
    path = FEAT / fname
    if not path.exists():
        print(f"[no existe] {fname}")
        continue
    t = path.read_text(encoding="utf-8")
    print("=" * 70)
    print(desc)
    print(f"  ({fname}, {len(t)} chars)")
    print("=" * 70)
    # Buscar definiciones de listas/constantes (lexicones)
    lineas = t.split("\n")
    en_lista = False
    for i, l in enumerate(lineas):
        s = l.strip()
        # inicio de constante tipo NOMBRE = [ o NOMBRE = {
        if re.match(r'^[A-Z][A-Z0-9_]{2,}\s*[:=]', s) and ('[' in s or '{' in s):
            print(f"  L{i}: {s[:90]}")
            en_lista = True
        elif en_lista:
            # mostrar contenido de la lista (palabras entre comillas)
            if '"' in s or "'" in s:
                print(f"       {s[:90]}")
            if ']' in s or '}' in s:
                en_lista = False
    print()
