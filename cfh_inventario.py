#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cfh_inventario.py
=================
Inventario del estado actual del proyecto CFH en disco.

Recorre la carpeta raíz del proyecto, clasifica cada archivo por tipo,
calcula hash SHA-256, e infiere a qué corpus pertenece según convenciones
de nombre/carpeta. Produce un CSV con el detalle completo y un reporte
de texto agregado.

Uso típico:
    python cfh_inventario.py
    python cfh_inventario.py --raiz "C:\\PROYECTOS 2026\\TESIS 2026\\CFH_Hermeneutica_Forense_Computacional"
    python cfh_inventario.py --raiz . --salida ./inventario --skip-hash

Salidas (en --salida, por defecto ./inventario_cfh):
    inventario_cfh_<YYYYMMDD_HHMM>.csv      detalle archivo por archivo
    inventario_cfh_<YYYYMMDD_HHMM>.txt      reporte agregado legible
    inventario_cfh_<YYYYMMDD_HHMM>.json     mismo detalle en JSON

Autor: Mireya Camacho Celis (CFH)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

# Carpetas que NO se recorren (ahorran tiempo y evitan ruido)
DIRS_EXCLUIDAS = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
    "node_modules",
    ".venv", "venv", "env", ".conda", "conda-meta",
    "dist", "build", ".egg-info",
    ".dvc", ".dvcignore_cache",
    "mlruns",  # MLflow runs: pueden tener miles de archivos
}

# Clasificación de archivos por extensión
TIPOS_ARCHIVO = {
    # Texto y documentos
    ".txt": "texto",
    ".md": "texto",
    ".rst": "texto",
    ".docx": "documento",
    ".doc": "documento",
    ".pdf": "documento",
    ".rtf": "documento",
    # Datos estructurados
    ".csv": "datos_tabular",
    ".tsv": "datos_tabular",
    ".xlsx": "datos_tabular",
    ".xls": "datos_tabular",
    ".parquet": "datos_tabular",
    ".feather": "datos_tabular",
    ".json": "datos_json",
    ".jsonl": "datos_json",
    ".xml": "datos_estructurado",
    ".yaml": "datos_estructurado",
    ".yml": "datos_estructurado",
    # Modelos y embeddings
    ".pkl": "modelo_o_embedding",
    ".pickle": "modelo_o_embedding",
    ".npy": "modelo_o_embedding",
    ".npz": "modelo_o_embedding",
    ".pt": "modelo_o_embedding",
    ".pth": "modelo_o_embedding",
    ".bin": "modelo_o_embedding",
    ".safetensors": "modelo_o_embedding",
    ".h5": "modelo_o_embedding",
    # Multimodal
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".mp4": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".mov": "video",
    ".srt": "subtitulo",
    ".vtt": "subtitulo",
    # Código
    ".py": "codigo",
    ".ipynb": "notebook",
    ".sh": "codigo",
    ".sql": "codigo",
    ".r": "codigo",
    # Otros
    ".log": "log",
    ".cfg": "config",
    ".ini": "config",
    ".toml": "config",
    ".lock": "config",
    ".png": "imagen",
    ".jpg": "imagen",
    ".jpeg": "imagen",
    ".svg": "imagen",
    ".pdf_thumb": "imagen",
}

# Patrones para inferir corpus
PATRONES_CORPUS = [
    # (regex, corpus)
    (re.compile(r"\bA[-_]?CE\b|consejo[-_]?(de[-_]?)?estado", re.I), "A-CE"),
    (re.compile(r"\bA[-_]?CSJ\b|corte[-_]?suprema|sala[-_]?penal|casacion", re.I), "A-CSJ"),
    (re.compile(r"\bB[-_]?JEP\b|srvr|macrocaso[-_]?003", re.I), "B-JEP"),
    (re.compile(r"\bC[-_]?JEP\b|audiencia|catatumbo|costa[-_]?caribe|casanare|dabeiba|huila|ollo|chaparro|torres[-_]?escalante|popa", re.I), "C-JEP-oral"),
    (re.compile(r"mafapo|madres[-_]?(de[-_]?)?(falsos|soacha)", re.I), "REF-MAFAPO"),
    (re.compile(r"cidh|villamizar|corteIDH|corte[-_]?interamericana", re.I), "REF-CIDH"),
]

# Tamaños "grandes" sobre los que se podría omitir el hash (en MB)
TAM_GRANDE_MB = 200


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def humanizar_bytes(n: int) -> str:
    """Devuelve una cadena tipo '1.3 GB' / '450 MB' / '12 KB'."""
    for unidad in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unidad == "TB":
            return f"{n:.1f} {unidad}" if unidad != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def hash_sha256(ruta: Path, bloque: int = 1024 * 1024) -> Optional[str]:
    """SHA-256 leyendo en bloques de 1 MB. Devuelve None si falla la lectura."""
    try:
        h = hashlib.sha256()
        with ruta.open("rb") as f:
            while chunk := f.read(bloque):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return None


def inferir_corpus(ruta_relativa: str, nombre: str) -> str:
    """Decide a qué corpus pertenece un archivo según su ruta + nombre."""
    blob = f"{ruta_relativa} {nombre}"
    for patron, corpus in PATRONES_CORPUS:
        if patron.search(blob):
            return corpus
    return "sin_clasificar"


def clasificar_tipo(extension: str) -> str:
    return TIPOS_ARCHIVO.get(extension.lower(), "otro")


# --------------------------------------------------------------------------- #
# Recorrido
# --------------------------------------------------------------------------- #

def recorrer(
    raiz: Path,
    skip_hash: bool = False,
    limite_hash_mb: float = TAM_GRANDE_MB,
    verbose: bool = True,
) -> list[dict]:
    """Recorre `raiz` y devuelve una lista de dicts con metadata por archivo."""
    registros: list[dict] = []
    n_archivos = 0
    bytes_totales = 0
    inicio = datetime.now()

    for dirpath, dirnames, filenames in os.walk(raiz):
        # Poda in-place de directorios excluidos
        dirnames[:] = [d for d in dirnames if d not in DIRS_EXCLUIDAS]

        for nombre in filenames:
            ruta_abs = Path(dirpath) / nombre
            try:
                stat = ruta_abs.stat()
            except (OSError, PermissionError):
                continue

            tam = stat.st_size
            ext = ruta_abs.suffix.lower()
            ruta_rel = str(ruta_abs.relative_to(raiz)).replace("\\", "/")

            # Hash: opcional, y se omite en archivos enormes salvo --force-hash
            calc_hash = (not skip_hash) and (tam <= limite_hash_mb * 1024 * 1024)
            sha = hash_sha256(ruta_abs) if calc_hash else None

            registros.append({
                "ruta_relativa": ruta_rel,
                "nombre": nombre,
                "extension": ext,
                "tipo": clasificar_tipo(ext),
                "corpus_inferido": inferir_corpus(ruta_rel, nombre),
                "tamano_bytes": tam,
                "tamano_humano": humanizar_bytes(tam),
                "modificado": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha,
                "hash_omitido": not calc_hash,
            })

            n_archivos += 1
            bytes_totales += tam
            if verbose and n_archivos % 200 == 0:
                print(f"  ... {n_archivos:,} archivos procesados "
                      f"({humanizar_bytes(bytes_totales)})", flush=True)

    if verbose:
        elapsed = (datetime.now() - inicio).total_seconds()
        print(f"  Total: {n_archivos:,} archivos, {humanizar_bytes(bytes_totales)}, "
              f"{elapsed:.1f}s", flush=True)
    return registros


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #

def construir_reporte(registros: list[dict], raiz: Path) -> str:
    """Reporte agregado en texto plano legible."""
    n = len(registros)
    total_bytes = sum(r["tamano_bytes"] for r in registros)

    por_tipo = Counter(r["tipo"] for r in registros)
    por_extension = Counter(r["extension"] for r in registros)
    por_corpus = Counter(r["corpus_inferido"] for r in registros)

    bytes_por_tipo: dict[str, int] = defaultdict(int)
    bytes_por_corpus: dict[str, int] = defaultdict(int)
    for r in registros:
        bytes_por_tipo[r["tipo"]] += r["tamano_bytes"]
        bytes_por_corpus[r["corpus_inferido"]] += r["tamano_bytes"]

    # Top 10 archivos más grandes
    top_grandes = sorted(registros, key=lambda r: r["tamano_bytes"], reverse=True)[:10]

    # Carpetas top por número de archivos
    por_carpeta: Counter = Counter()
    for r in registros:
        carpeta = r["ruta_relativa"].rsplit("/", 1)[0] if "/" in r["ruta_relativa"] else "."
        por_carpeta[carpeta] += 1
    top_carpetas = por_carpeta.most_common(15)

    lineas = []
    lineas.append("=" * 78)
    lineas.append(f"INVENTARIO CFH — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lineas.append(f"Raíz: {raiz}")
    lineas.append("=" * 78)
    lineas.append(f"Total de archivos     : {n:,}")
    lineas.append(f"Tamaño total          : {humanizar_bytes(total_bytes)}")
    lineas.append("")

    lineas.append("─── Por TIPO ──────────────────────────────────────────────────────────")
    for tipo, cnt in sorted(por_tipo.items(), key=lambda x: -x[1]):
        lineas.append(f"  {tipo:25s}  {cnt:6,}  {humanizar_bytes(bytes_por_tipo[tipo]):>12}")
    lineas.append("")

    lineas.append("─── Por CORPUS inferido ──────────────────────────────────────────────")
    for corpus, cnt in sorted(por_corpus.items(), key=lambda x: -x[1]):
        lineas.append(f"  {corpus:20s}  {cnt:6,}  {humanizar_bytes(bytes_por_corpus[corpus]):>12}")
    lineas.append("")

    lineas.append("─── Por EXTENSIÓN (top 20) ───────────────────────────────────────────")
    for ext, cnt in por_extension.most_common(20):
        ext_show = ext if ext else "(sin ext.)"
        lineas.append(f"  {ext_show:15s}  {cnt:6,}")
    lineas.append("")

    lineas.append("─── Top 10 archivos más grandes ──────────────────────────────────────")
    for r in top_grandes:
        lineas.append(f"  {r['tamano_humano']:>10}  {r['ruta_relativa']}")
    lineas.append("")

    lineas.append("─── Top 15 carpetas por nº de archivos ───────────────────────────────")
    for carpeta, cnt in top_carpetas:
        lineas.append(f"  {cnt:6,}  {carpeta}")
    lineas.append("")

    # Sugerencias automáticas
    lineas.append("─── Notas para diseño de ingesta ────────────────────────────────────")
    sin_clas = por_corpus.get("sin_clasificar", 0)
    if sin_clas:
        lineas.append(f"  • {sin_clas} archivos sin clasificar — revisar el CSV "
                      f"y ajustar PATRONES_CORPUS.")
    if por_tipo.get("audio", 0):
        lineas.append(f"  • {por_tipo['audio']} archivos de audio — candidatos para Capa 3 "
                      f"(diarización + eGeMAPS).")
    if por_tipo.get("video", 0):
        lineas.append(f"  • {por_tipo['video']} archivos de video — candidatos para Capa 3 "
                      f"facial (MediaPipe). Verificar el de Costa Caribe (DRM).")
    if por_tipo.get("modelo_o_embedding", 0):
        lineas.append(f"  • {por_tipo['modelo_o_embedding']} archivos de modelo/embedding — "
                      f"NO migrar a SQLite, mantener en disco y referenciar por ruta.")
    if por_tipo.get("notebook", 0):
        lineas.append(f"  • {por_tipo['notebook']} notebooks — revisar cuáles producen "
                      f"datasets canónicos (esos definen las tablas).")
    lineas.append("")
    lineas.append("=" * 78)
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #

def escribir_csv(registros: list[dict], destino: Path) -> None:
    if not registros:
        return
    campos = list(registros[0].keys())
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(registros)


def escribir_json(registros: list[dict], destino: Path) -> None:
    with destino.open("w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)


def escribir_reporte(reporte: str, destino: Path) -> None:
    with destino.open("w", encoding="utf-8") as f:
        f.write(reporte)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventario del proyecto CFH para diseño de la BD SQLite."
    )
    parser.add_argument(
        "--raiz", type=str, default=".",
        help="Carpeta raíz del proyecto a inventariar (default: directorio actual)."
    )
    parser.add_argument(
        "--salida", type=str, default="./inventario_cfh",
        help="Carpeta donde se escriben los archivos de inventario."
    )
    parser.add_argument(
        "--skip-hash", action="store_true",
        help="No calcular SHA-256 (mucho más rápido en discos grandes)."
    )
    parser.add_argument(
        "--limite-hash-mb", type=float, default=TAM_GRANDE_MB,
        help=f"No hashear archivos por encima de este tamaño en MB "
             f"(default {TAM_GRANDE_MB})."
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="No imprimir progreso parcial."
    )
    args = parser.parse_args()

    raiz = Path(args.raiz).expanduser().resolve()
    if not raiz.exists() or not raiz.is_dir():
        print(f"ERROR: la raíz '{raiz}' no existe o no es directorio.", file=sys.stderr)
        return 2

    salida = Path(args.salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    print(f"[CFH] Inventariando: {raiz}")
    print(f"[CFH] Salida en    : {salida}")
    if args.skip_hash:
        print("[CFH] Hash SHA-256 desactivado (--skip-hash).")

    registros = recorrer(
        raiz,
        skip_hash=args.skip_hash,
        limite_hash_mb=args.limite_hash_mb,
        verbose=not args.silent,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = salida / f"inventario_cfh_{timestamp}"
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")

    escribir_csv(registros, csv_path)
    escribir_json(registros, json_path)
    reporte = construir_reporte(registros, raiz)
    escribir_reporte(reporte, txt_path)

    print()
    print(reporte)
    print()
    print(f"[CFH] CSV     : {csv_path}")
    print(f"[CFH] JSON    : {json_path}")
    print(f"[CFH] Reporte : {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
