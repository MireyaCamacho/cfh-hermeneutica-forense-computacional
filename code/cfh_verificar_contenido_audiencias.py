# -*- coding: utf-8 -*-
"""
cfh_verificar_contenido_audiencias.py
================================================================================
CFH — Verificar por CONTENIDO si el archivo base es RECONOCIMIENTO u OBSERVACIONES

MOTIVO:
    El nombre no basta. Abrimos cada transcripción base (sin 'obs_') de
    Casanare, Huila y Costa Caribe y miramos QUÉ se dice en distintos momentos
    de la audiencia, para confirmar si son audiencias de RECONOCIMIENTO
    (comparecientes asumen responsabilidad → ICM) u OBSERVACIONES (víctimas).

QUÉ HACE:
    Para cada archivo base, toma muestras de texto al 10%, 30%, 50%, 70%, 90%
    de la audiencia y cuenta marcadores de cada tipo. Muestra extractos para que
    leas con tus ojos y confirmes.

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_verificar_contenido_audiencias.py"

Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import argparse
import json
from pathlib import Path

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"

ARCHIVOS = {
    "Casanare": "casanare_torres_segments.json",
    "Huila": "huila_segments.json",
    "CostaCaribe": "costa_caribe_segments.json",
}

# Marcadores de RECONOCIMIENTO (comparecientes / militares que asumen).
KW_RECON = [
    "reconozco", "reconocemos", "asumo", "asumimos", "mi responsabilidad",
    "nuestra responsabilidad", "pido perdón", "pedimos perdón", "ordené",
    "como compareciente", "comparezco", "acepto", "mayor", "coronel", "sargento",
    "soldado", "general", "batallón", "brigada", "yo participé", "di la orden",
]
# Marcadores de OBSERVACIONES (víctimas / representantes).
KW_OBS = [
    "mi hijo", "mi hermano", "mi esposo", "mi padre", "como víctima",
    "las víctimas", "exigimos", "queremos saber", "representante de las víctimas",
    "señora magistrada queremos", "nos arrebataron", "se lo llevaron",
    "lo desaparecieron", "madre de", "madres de",
]


def cargar(ruta):
    d = json.load(open(ruta, encoding="utf-8"))
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list):
                return v
    return d if isinstance(d, list) else []


def contar(texto, kws):
    low = texto.lower()
    return sum(low.count(k) for k in kws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()
    base = Path(args.base)

    print("CFH — Verificación por CONTENIDO (reconocimiento vs observaciones)")
    print("="*72)

    for subcaso, nombre in ARCHIVOS.items():
        ruta = base / "corpus_c" / nombre
        print(f"\n{'='*72}\n{subcaso}: {nombre}\n{'='*72}")
        if not ruta.exists():
            print("  [NO EXISTE] revisar ruta.")
            continue
        segs = cargar(ruta)
        # filtrar segmentos con texto real
        segs = [s for s in segs if len(str(s.get("text", s.get("texto",""))).strip()) > 3]
        n = len(segs)
        if n == 0:
            print("  [VACÍO]")
            continue

        # texto completo (para conteo global)
        full = " ".join(str(s.get("text", s.get("texto",""))) for s in segs)
        nr, no = contar(full, KW_RECON), contar(full, KW_OBS)

        print(f"  Segmentos con texto: {n}")
        print(f"  Marcadores RECONOCIMIENTO: {nr}  |  OBSERVACIONES: {no}")
        veredicto = "RECONOCIMIENTO ✓ (→ ICM)" if nr > no*1.3 else \
                    ("OBSERVACIONES ⚠ (→ centroide, NO ICM)" if no > nr*1.3 else
                     "MIXTO / revisar manual")
        print(f"  → VEREDICTO POR CONTENIDO: {veredicto}")

        # muestras a lo largo de la audiencia
        print(f"\n  --- Extractos (lee para confirmar) ---")
        for pct in [0.10, 0.30, 0.50, 0.70, 0.90]:
            idx = int(n * pct)
            txt = str(segs[idx].get("text", segs[idx].get("texto",""))).strip()
            t = segs[idx].get("start", 0)
            print(f"    [{pct*100:.0f}% · {int(t)//60}min] «{txt[:120]}»")

    print(f"\n{'='*72}")
    print("Si el veredicto y los extractos confirman RECONOCIMIENTO, uso ese")
    print("archivo base para la plantilla de marcación. Si alguno es OBSERVACIONES,")
    print("buscamos la versión de reconocimiento correcta para ese subcaso.")


if __name__ == "__main__":
    main()
