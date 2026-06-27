# -*- coding: utf-8 -*-
"""
cfh_buscar_inventarios.py
================================================================================
CFH — Localizar inventarios de marcación de comparecientes (Corpus C)

MOTIVO:
    El ICM tri-canal saltó Casanare, Huila y CostaCaribe por no encontrar
    data/marcacion/inventario_<Subcaso>.csv. Pero pueden existir con otro
    nombre o en otra carpeta (como pasó con Dabeiba). Esto los busca antes de
    asumir que hay que crearlos.

QUÉ HACE:
    Busca en local/G: cualquier CSV cuyo nombre o contenido sugiera marcación
    de comparecientes (inventario_, marcacion, con columnas speaker_diar/
    identidad/inicio/fin/uso). Para cada uno reporta subcaso, nº de filas,
    columnas clave y cuántos comparecientes ICM tiene.

USO:
    python cfh_buscar_inventarios.py

Entorno: Python 3.11, conda env cfh. Dependencias: pandas.
================================================================================
"""

import argparse
import os
from pathlib import Path
import pandas as pd

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"
RAICES_EXTRA = [r"G:\Mi unidad"]
SUBCASOS = ["casanare", "catatumbo", "dabeiba", "huila", "costa", "caribe"]
IGNORAR = {"node_modules", ".git", "__pycache__", ".venv", "$RECYCLE.BIN",
           "System Volume Information", ".Trashes"}

# Columnas que delatan una marcación de comparecientes.
COLS_MARCA = {"identidad", "inicio", "fin", "uso", "rol", "rango_militar", "speaker_diar"}


def listar_csv(raiz, max_prof=7):
    raiz = Path(raiz)
    if not raiz.exists():
        return []
    out = []
    rd = len(raiz.parts)
    for dp, dn, fn in os.walk(raiz):
        dn[:] = [d for d in dn if d not in IGNORAR]
        if len(Path(dp).parts) - rd > max_prof:
            dn[:] = []; continue
        for f in fn:
            low = f.lower()
            if low.endswith(".csv") and ("inventario" in low or "marcacion" in low or
                                         "marca" in low or "comparecien" in low):
                out.append(Path(dp) / f)
    return out


def subcaso_de(nombre):
    low = nombre.lower()
    for sc in ["casanare", "catatumbo", "dabeiba", "huila"]:
        if sc in low:
            return sc
    if "costa" in low or "caribe" in low:
        return "costa_caribe"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()
    base = Path(args.base)
    raices = [base] + [Path(r) for r in RAICES_EXTRA]

    print("CFH — Búsqueda de inventarios de marcación (Corpus C)")
    print("="*72)

    archivos, vistos = [], set()
    for r in raices:
        for a in listar_csv(r):
            if str(a).lower() not in vistos:
                vistos.add(str(a).lower())
                archivos.append(a)

    por_subcaso = {}
    for a in archivos:
        try:
            d = pd.read_csv(a, nrows=5)
            cols = set(c.lower() for c in d.columns)
        except Exception:
            continue
        # ¿es una marcación de comparecientes?
        if len(COLS_MARCA & cols) < 3:
            continue
        sc = subcaso_de(a.name)
        try:
            full = pd.read_csv(a)
            n = len(full)
            # comparecientes ICM
            if "uso" in full.columns or "rol" in full.columns:
                cond = pd.Series(False, index=full.index)
                if "uso" in full.columns:
                    cond = cond | full["uso"].astype(str).str.upper().eq("ANALISIS")
                if "rol" in full.columns:
                    cond = cond | full["rol"].astype(str).str.upper().eq("COMPARECIENTE")
                n_comp = int(cond.sum())
            else:
                n_comp = "?"
            tiene_sd = "speaker_diar" in full.columns
            sd_lleno = (full["speaker_diar"].notna().sum() if tiene_sd else 0)
        except Exception:
            n, n_comp, tiene_sd, sd_lleno = "?", "?", False, 0

        info = {"ruta": str(a), "subcaso": sc, "filas": n,
                "comparecientes_ICM": n_comp, "speaker_diar_lleno": sd_lleno,
                "cols": ", ".join(list(d.columns)[:10])}
        por_subcaso.setdefault(sc, []).append(info)

    # Reporte por subcaso objetivo.
    print()
    objetivo = ["casanare", "huila", "costa_caribe", "catatumbo", "dabeiba"]
    for sc in objetivo:
        print(f"\n{'='*72}\nSUBCASO: {sc}\n{'='*72}")
        items = por_subcaso.get(sc, [])
        if not items:
            print("  ✗ NO se encontró inventario de marcación.")
            print("    → Hay que crearlo (marcar comparecientes con inicio/fin/identidad).")
            continue
        for it in sorted(items, key=lambda x: -(x["comparecientes_ICM"] if isinstance(x["comparecientes_ICM"], int) else 0)):
            print(f"\n  • {it['ruta']}")
            print(f"      filas={it['filas']} | comparecientes ICM={it['comparecientes_ICM']} | "
                  f"speaker_diar lleno={it['speaker_diar_lleno']}")
            print(f"      cols: {it['cols']}")

    # Resumen de qué falta.
    print(f"\n{'='*72}\nRESUMEN\n{'='*72}")
    for sc in ["casanare", "huila", "costa_caribe"]:
        items = por_subcaso.get(sc, [])
        if items:
            mejor = max(items, key=lambda x: x["comparecientes_ICM"] if isinstance(x["comparecientes_ICM"], int) else 0)
            estado = "✓ existe" if (isinstance(mejor["comparecientes_ICM"], int) and mejor["comparecientes_ICM"] > 0) else "⚠ existe pero sin comparecientes ICM"
            print(f"  {sc:14} {estado}  → {Path(mejor['ruta']).name}")
        else:
            print(f"  {sc:14} ✗ FALTA — hay que crear la marcación")
    print("\nNota: el nombre debe ser data/marcacion/inventario_<Subcaso>.csv")
    print("con <Subcaso> en: Casanare, Huila, CostaCaribe (como los espera el pipeline).")


if __name__ == "__main__":
    main()
