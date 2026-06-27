# -*- coding: utf-8 -*-
"""
cfh_clasificar_audiencias.py
================================================================================
CFH — Clasificar audiencias del Corpus C: RECONOCIMIENTO vs OBSERVACIONES

MOTIVO:
    El ICM se calcula SOLO sobre audiencias de RECONOCIMIENTO (amarillo →
    comparecientes). Las de OBSERVACIONES de víctimas (verde) van al centroide,
    no al ICM. Los nombres de archivo son confusos y hay versiones mezcladas.
    Esto clasifica cada archivo para elegir la fuente correcta sin equivocarse.

QUÉ HACE:
    Lista todos los .json de corpus_c, los agrupa por subcaso, y clasifica cada
    uno como RECONOCIMIENTO / OBSERVACIONES / INDETERMINADO según señales en el
    nombre y una muestra del contenido (rol de quien más habla).

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_clasificar_audiencias.py"

Entorno: Python 3.11, conda env cfh. Dependencias: (ninguna especial).
================================================================================
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"

SUBCASOS = ["casanare", "catatumbo", "dabeiba", "huila", "costa", "caribe"]

# Señales en el nombre del archivo.
SIG_RECON = ["reconocimiento", "aceptaci", "responsabilidad", "reconocim"]
SIG_OBS = ["observacion", "observaciones", "obs_", "victima", "víctima", "atanquez"]


def subcaso_de(nombre):
    low = nombre.lower()
    for sc in ["casanare", "catatumbo", "dabeiba", "huila"]:
        if sc in low:
            return sc
    if "costa" in low or "caribe" in low:
        return "costa_caribe"
    return "?"


def clasificar_por_nombre(nombre):
    low = nombre.lower()
    rec = any(s in low for s in SIG_RECON)
    obs = any(s in low for s in SIG_OBS)
    if rec and not obs:
        return "RECONOCIMIENTO"
    if obs and not rec:
        return "OBSERVACIONES"
    if rec and obs:
        return "MIXTO?"
    return "INDETERMINADO"


def muestra_contenido(ruta, n=30):
    """Devuelve una muestra de texto del archivo (si es transcripción)."""
    try:
        d = json.load(open(ruta, encoding="utf-8"))
    except Exception:
        return ""
    segs = d if isinstance(d, list) else None
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list):
                segs = v; break
    if not segs:
        return ""
    textos = []
    for s in segs[:n]:
        if isinstance(s, dict):
            t = str(s.get("text", s.get("texto", ""))).strip()
            if t:
                textos.append(t)
    return " ".join(textos)[:300]


def pista_contenido(texto):
    """Heurística: ¿suena a reconocimiento o a observaciones de víctimas?"""
    low = texto.lower()
    rec_kw = ["reconozco", "reconocemos", "asumo", "mi responsabilidad", "pido perd",
              "comparezco", "como compareciente", "ordené", "acepto"]
    obs_kw = ["mi hijo", "mi hermano", "como víctima", "las víctimas exigimos",
              "queremos saber", "exigimos verdad", "representante de las víctimas"]
    nr = sum(1 for k in rec_kw if k in low)
    no = sum(1 for k in obs_kw if k in low)
    if nr > no:
        return "→ contenido sugiere RECONOCIMIENTO"
    if no > nr:
        return "→ contenido sugiere OBSERVACIONES"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()
    base = Path(args.base)

    print("CFH — Clasificación de audiencias: RECONOCIMIENTO vs OBSERVACIONES")
    print("ICM usa SOLO reconocimiento (amarillo). Observaciones (verde) → centroide.")
    print("="*72)

    archivos = glob.glob(str(base/"corpus_c"/"*.json"))
    por_sub = defaultdict(list)
    for f in archivos:
        sc = subcaso_de(Path(f).name)
        por_sub[sc].append(Path(f))

    for sc in ["casanare", "huila", "costa_caribe", "catatumbo", "dabeiba"]:
        print(f"\n{'='*72}\nSUBCASO: {sc}\n{'='*72}")
        items = por_sub.get(sc, [])
        if not items:
            print("  (sin archivos)")
            continue
        # separar diarización de transcripción
        for ruta in sorted(items):
            nombre = ruta.name
            tipo_arch = "DIARIZACIÓN" if "diariz" in nombre.lower() else \
                        ("TRANSCRIPCIÓN" if "segment" in nombre.lower() or "transcri" in nombre.lower() else "OTRO")
            clase = clasificar_por_nombre(nombre)
            pista = ""
            if tipo_arch == "TRANSCRIPCIÓN":
                pista = pista_contenido(muestra_contenido(ruta))
            marca = "★" if clase == "RECONOCIMIENTO" else " "
            print(f"  {marca} [{clase:14s}] [{tipo_arch:13s}] {nombre}")
            if pista:
                print(f'        {pista}')

        # Recomendación.
        recon = [p for p in items if clasificar_por_nombre(p.name) == "RECONOCIMIENTO"]
        print(f"\n  → Para el ICM usar (RECONOCIMIENTO ★):")
        if recon:
            diar = [p for p in recon if "diariz" in p.name.lower()]
            trans = [p for p in recon if "segment" in p.name.lower()]
            # si no hay diarización marcada como reconocimiento, usar la genérica del subcaso
            if not diar:
                diar = [p for p in items if "diariz" in p.name.lower()]
            print(f"      diarización: {diar[0].name if diar else '??? revisar'}")
            print(f"      transcripción: {trans[0].name if trans else '??? revisar'}")
        else:
            print("      ⚠ Ninguna marcada claramente como reconocimiento — revisar manual.")

    print(f"\n{'='*72}")
    print("Confirma las fuentes ★ y con esas genero las plantillas de marcación.")


if __name__ == "__main__":
    main()
