#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cfh_explorar_csvs.py
====================
Examina los CSVs del proyecto CFH y produce un reporte Markdown con:
  - dimensiones (n filas, n columnas)
  - nombres y dtypes de cada columna
  - primeras 3 filas
  - detección de duplicados por hash MD5 del contenido binario
  - agrupación por carpeta
  - clasificación heurística por nombre (capa 1, capa 2, capa 3, etc.)

Uso típico:
    python cfh_explorar_csvs.py
    python cfh_explorar_csvs.py --raiz . --salida ./inventario_cfh
    python cfh_explorar_csvs.py --max-filas 5

Salida:
    inventario_cfh/csvs_<YYYYMMDD_HHMM>.md

El propósito de este script es generar el insumo necesario para escribir
`cfh_ingesta.py`: saber qué columnas tiene cada CSV permite mapearlas
a las tablas del esquema sin adivinar.

Autor: Mireya Camacho Celis (CFH)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

DIRS_EXCLUIDAS = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    "node_modules", ".venv", "venv", "env", ".conda",
    "dist", "build", "mlruns", "inventario_cfh",
}

# Heurística para clasificar el CSV en una "familia" por su nombre
def clasificar_csv(ruta: str) -> str:
    s = ruta.lower().replace("\\", "/")
    if "egemap" in s:                      return "Capa 3 — eGeMAPS (vocal)"
    if "aus_" in s or "/aus_" in s:        return "Capa 3 — Action Units (facial)"
    if "icm_resultados" in s:              return "Capa 3 — ICM síntesis"
    if "dis_iei" in s:                     return "Capa 2 — DIS / IEI"
    if "indicators" in s and "y11" in s:   return "Capa 1 — Beach (y11/y12/y13)"
    if "capa1_nuevos" in s:                return "Capa 1 — extracción nueva"
    if "indicators_corpus_a" in s:         return "Indicadores Corpus A"
    if "indicators_corpus_b" in s:         return "Indicadores Corpus B"
    if "indicators_corpus_c" in s:         return "Indicadores Corpus C"
    if "indicators_completo" in s or "indicators_final" in s:
                                           return "Indicadores agregados (todos los corpora)"
    if "metadata" in s:                    return "Metadata documental"
    if "inventario" in s:                  return "Inventario / no procesar"
    if "iaa" in s or "anotaciones" in s:   return "Anotaciones / IAA"
    return "Sin clasificar"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def md5_archivo(ruta: Path, bloque: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with ruta.open("rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def humanizar_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def leer_csv_robusto(ruta: Path, max_filas: int):
    """
    Intenta leer el CSV con varias combinaciones de encoding y separador.
    Devuelve (df, info_lectura) donde info_lectura es un dict con encoding/sep usados.
    """
    intentos = [
        {"encoding": "utf-8",      "sep": ","},
        {"encoding": "utf-8-sig",  "sep": ","},
        {"encoding": "latin-1",    "sep": ","},
        {"encoding": "utf-8",      "sep": ";"},
        {"encoding": "latin-1",    "sep": ";"},
        {"encoding": "utf-8",      "sep": "\t"},
    ]
    ultimo_error = None
    for opts in intentos:
        try:
            df_full = pd.read_csv(ruta, encoding=opts["encoding"], sep=opts["sep"],
                                  low_memory=False, on_bad_lines="warn")
            df_head = df_full.head(max_filas)
            return df_full, df_head, {**opts, "n_filas": len(df_full),
                                      "n_cols": df_full.shape[1]}
        except Exception as e:
            ultimo_error = e
            continue
    raise RuntimeError(f"No se pudo leer {ruta}: {ultimo_error}")


# --------------------------------------------------------------------------- #
# Recorrido
# --------------------------------------------------------------------------- #

def encontrar_csvs(raiz: Path) -> list[Path]:
    csvs = []
    for ruta in raiz.rglob("*.csv"):
        # Excluir si algún ancestro está en la lista negra
        if any(part in DIRS_EXCLUIDAS for part in ruta.relative_to(raiz).parts):
            continue
        csvs.append(ruta)
    return sorted(csvs)


def analizar_csvs(csvs: list[Path], raiz: Path, max_filas: int):
    resultados = []
    hashes = defaultdict(list)  # md5 -> [ruta_relativa, ...]

    for ruta in csvs:
        rel = str(ruta.relative_to(raiz)).replace("\\", "/")
        try:
            stat = ruta.stat()
            md5 = md5_archivo(ruta)
            hashes[md5].append(rel)
            try:
                df_full, df_head, info = leer_csv_robusto(ruta, max_filas)
                cols_info = [(c, str(df_full[c].dtype)) for c in df_full.columns]
                lectura_ok = True
                error = None
            except Exception as e:
                df_full = df_head = None
                cols_info = []
                info = {}
                lectura_ok = False
                error = str(e)

            resultados.append({
                "ruta": rel,
                "tamano_bytes": stat.st_size,
                "tamano_humano": humanizar_bytes(stat.st_size),
                "md5": md5,
                "lectura_ok": lectura_ok,
                "error": error,
                "cols_info": cols_info,
                "info": info,
                "head": df_head,
                "categoria": clasificar_csv(rel),
            })
        except Exception as e:
            print(f"  WARN: error inesperado con {rel}: {e}", file=sys.stderr)

    return resultados, hashes


# --------------------------------------------------------------------------- #
# Reporte Markdown
# --------------------------------------------------------------------------- #

def render_md(resultados, hashes, raiz: Path) -> str:
    n = len(resultados)
    out = []
    out.append(f"# Exploración de CSVs — proyecto CFH\n")
    out.append(f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    out.append(f"**Raíz:** `{raiz}`  ")
    out.append(f"**CSVs encontrados:** {n}\n")

    # ---- Sección 1: duplicados ----
    duplicados = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    out.append("## 1. Duplicados detectados (por hash MD5)\n")
    if not duplicados:
        out.append("_Ninguno._\n")
    else:
        out.append("Estos archivos tienen contenido **idéntico** byte a byte. "
                   "Se puede eliminar todos menos uno (el canónico).\n")
        for i, (md5, paths) in enumerate(sorted(duplicados.items(),
                                                 key=lambda kv: -len(kv[1])), 1):
            out.append(f"**Grupo {i}** (md5: `{md5[:10]}…`, {len(paths)} archivos):")
            for p in sorted(paths):
                out.append(f"  - `{p}`")
            out.append("")

    # ---- Sección 2: clasificación por categoría ----
    out.append("## 2. Clasificación heurística por nombre\n")
    por_cat = defaultdict(list)
    for r in resultados:
        por_cat[r["categoria"]].append(r)
    for cat in sorted(por_cat.keys()):
        items = por_cat[cat]
        out.append(f"### {cat}  *(n={len(items)})*\n")
        for r in items:
            out.append(f"- `{r['ruta']}` — {r['tamano_humano']}")
        out.append("")

    # ---- Sección 3: detalle por archivo ----
    out.append("## 3. Detalle por archivo\n")
    out.append("Una sub-sección por CSV con columnas, dtypes y muestra de filas.\n")

    for r in resultados:
        out.append(f"### `{r['ruta']}`")
        out.append(f"- **Tamaño:** {r['tamano_humano']}  ")
        out.append(f"- **MD5:** `{r['md5']}`  ")
        out.append(f"- **Categoría:** {r['categoria']}  ")
        if not r["lectura_ok"]:
            out.append(f"- **⚠ Error de lectura:** `{r['error']}`")
            out.append("")
            continue

        info = r["info"]
        out.append(f"- **Dimensiones:** {info['n_filas']:,} filas × "
                   f"{info['n_cols']} columnas  ")
        out.append(f"- **Encoding usado:** `{info['encoding']}`, "
                   f"separador: `{repr(info['sep'])}`\n")

        out.append("**Columnas:**\n")
        out.append("| # | nombre | dtype |")
        out.append("|---|--------|-------|")
        for i, (c, t) in enumerate(r["cols_info"], 1):
            out.append(f"| {i} | `{c}` | `{t}` |")
        out.append("")

        if r["head"] is not None and len(r["head"]) > 0:
            out.append("**Primeras filas:**\n")
            try:
                out.append("```")
                out.append(r["head"].to_string(index=False, max_cols=10,
                                               max_colwidth=40))
                out.append("```\n")
            except Exception as e:
                out.append(f"_(error mostrando filas: {e})_\n")
        out.append("---\n")

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explorador de CSVs CFH para diseño de la ingesta a SQLite."
    )
    parser.add_argument("--raiz", default=".",
                        help="Raíz del proyecto (default: directorio actual).")
    parser.add_argument("--salida", default="./inventario_cfh",
                        help="Carpeta donde escribir el reporte Markdown.")
    parser.add_argument("--max-filas", type=int, default=3,
                        help="Cuántas filas mostrar como muestra de cada CSV (default 3).")
    args = parser.parse_args()

    raiz = Path(args.raiz).expanduser().resolve()
    if not raiz.exists():
        print(f"ERROR: raíz no existe: {raiz}", file=sys.stderr)
        return 2

    salida = Path(args.salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    print(f"[CFH] Buscando CSVs en: {raiz}")
    csvs = encontrar_csvs(raiz)
    print(f"[CFH] {len(csvs)} CSVs encontrados.")
    if not csvs:
        return 0

    print(f"[CFH] Analizando (puede tardar varios minutos en CSVs grandes)…")
    resultados, hashes = analizar_csvs(csvs, raiz, args.max_filas)

    md = render_md(resultados, hashes, raiz)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    destino = salida / f"csvs_{timestamp}.md"
    destino.write_text(md, encoding="utf-8")

    # Resumen en consola
    n = len(resultados)
    n_ok = sum(1 for r in resultados if r["lectura_ok"])
    n_dup = sum(len(v) - 1 for v in hashes.values() if len(v) > 1)
    print()
    print(f"[CFH] CSVs analizados      : {n}")
    print(f"[CFH] Lecturas exitosas    : {n_ok}/{n}")
    print(f"[CFH] Archivos duplicados  : {n_dup} (se pueden eliminar sin pérdida)")
    print(f"[CFH] Reporte Markdown     : {destino}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
