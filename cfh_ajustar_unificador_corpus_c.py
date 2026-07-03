# -*- coding: utf-8 -*-
"""
cfh_ajustar_unificador_corpus_c.py
===================================
AJUSTE QUIRURGICO VERIFICADO al unificador del Corpus C.

PROBLEMA (verificado en reproducible/cfh_unificar_corpus_c.py):
    def get_score(extractor, texto):
        r = extractor.extract(texto)      <- sin corpus_type
    ...
    y10.append(get_score(rep, t))

  extract() tiene default corpus_type="B". Con la separacion de detectores por
  corpus (A/B institucional vs C oral), los bloques del Corpus C se estarian
  procesando como texto institucional: NO se activarian los detectores de
  reconocimiento oral del compareciente.

CAMBIO (minimo, solo para el REP):
  1. Inserta una funcion get_score_rep() que llama extract(texto, corpus_type="C").
  2. Cambia la linea  y10.append(get_score(rep, t))
     por              y10.append(get_score_rep(rep, t))
  y2 (SA) e y4 (NV) quedan intactos: sus extractores no fueron separados por corpus.

Ajusta el archivo en reproducible\\ y, si existe, tambien la copia en la raiz
(ambas contienen el mismo codigo; asi no queda una version desactualizada).

Uso:
    python cfh_ajustar_unificador_corpus_c.py --dry-run
    python cfh_ajustar_unificador_corpus_c.py
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVOS = [
    Path("reproducible/cfh_unificar_corpus_c.py"),
    Path("cfh_unificar_corpus_c.py"),
]

# Bloque verificado en el archivo (definicion de get_score)
DEF_VIEJA = '''def get_score(extractor, texto):
    """Maneja el objeto resultado: devuelve .score (float)."""
    try:
        r = extractor.extract(texto)
        # el resultado es un *ExtractionResult con atributo .score
        return float(getattr(r, "score", r))
    except Exception as e:
        return np.nan'''

DEF_NUEVA = '''def get_score(extractor, texto):
    """Maneja el objeto resultado: devuelve .score (float)."""
    try:
        r = extractor.extract(texto)
        # el resultado es un *ExtractionResult con atributo .score
        return float(getattr(r, "score", r))
    except Exception as e:
        return np.nan
def get_score_rep(extractor, texto):
    """REP sobre Corpus C: corpus_type="C" activa los detectores orales
    (reconocimiento 1a persona + reparacion/perdon + restitucion)."""
    try:
        r = extractor.extract(texto, corpus_type="C")
        return float(getattr(r, "score", r))
    except Exception as e:
        return np.nan'''

LINEA_VIEJA = "    y10.append(get_score(rep, t))"
LINEA_NUEVA = "    y10.append(get_score_rep(rep, t))"


def ajustar(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        print(f"  [no existe] {path} - se omite")
        return False
    contenido = path.read_text(encoding="utf-8")

    if "get_score_rep" in contenido:
        print(f"  [ya ajustado] {path}")
        return False

    ok_def = DEF_VIEJA in contenido
    ok_lin = LINEA_VIEJA in contenido
    print(f"  {path}")
    print(f"    def get_score encontrada:      {ok_def}")
    print(f"    linea y10.append encontrada:   {ok_lin}")
    if not (ok_def and ok_lin):
        print("    [ERROR] el contenido no coincide con lo verificado. NO se toca.")
        print("    Pega las lineas reales de get_score y y10.append para ajustar el script.")
        return False

    nuevo = contenido.replace(DEF_VIEJA, DEF_NUEVA, 1)
    nuevo = nuevo.replace(LINEA_VIEJA, LINEA_NUEVA, 1)

    try:
        compile(nuevo, str(path), "exec")
        print("    [OK] compila tras el ajuste")
    except SyntaxError as e:
        print(f"    [ERROR] no compila: {e}. NO se escribe.")
        return False

    if dry_run:
        print("    [dry-run] NO se escribe.")
        return True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f"_BACKUP_{ts}.py")
    shutil.copy2(path, backup)
    path.write_text(nuevo, encoding="utf-8")
    print(f"    backup -> {backup.name}")
    print(f"    ESCRITO: {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("AJUSTE: REP del unificador Corpus C con corpus_type='C'")
    print("=" * 70)
    alguno = False
    for p in ARCHIVOS:
        alguno = ajustar(p, args.dry_run) or alguno

    if alguno and not args.dry_run:
        print("\nSiguiente paso (tu propio pipeline, sin cambios adicionales):")
        print("  python reproducible\\cfh_unificar_corpus_c.py")
        print("  (regenera y2/y4/y10 sobre los 588 bloques con el extractor corregido,")
        print("   une y8/y9 por bloque_id y recalcula DIS/IEI z-score+sigmoid)")


if __name__ == "__main__":
    main()
