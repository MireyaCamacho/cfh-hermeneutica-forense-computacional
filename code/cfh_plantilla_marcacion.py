# -*- coding: utf-8 -*-
"""
cfh_plantilla_marcacion.py
================================================================================
CFH — Generador de plantillas de marcación pre-rellenadas (Corpus C)

MOTIVO:
    Casanare, Huila y Costa Caribe no tienen inventario de marcación. Marcarlos
    desde cero es lento. Este script PRE-RELLENA la plantilla: toma la
    diarización (quién habla y cuándo) + la transcripción limpia (qué dice), y
    arma por SPEAKER una fila con tramos, tiempo total y muestra de texto.

    Tú solo completas 3 columnas mirando el video:
        identidad  ·  rol  ·  uso

ESTRUCTURA DE SALIDA (compatible con el pipeline ICM):
    audiencia, video_url, speaker_diar, inicio, fin, n_segmentos,
    tiempo_total_s, identidad[VACÍO], rol[VACÍO], rango_militar[VACÍO],
    evidencia_identificacion, confianza, uso[VACÍO], muestra_texto

    · Una fila por SPEAKER (consolidando todas sus intervenciones).
    · inicio/fin = primer/último tramo del speaker (formato HH:MM:SS).
    · muestra_texto = primeras frases que dijo (para reconocerlo).
    · uso sugerido automáticamente: speakers que hablan poco→EXCLUIR (probable
      magistrado/secretaría); el resto en blanco para que tú decidas ICM/CENTROIDE.

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_plantilla_marcacion.py"

Entorno: Python 3.11, conda env cfh. Dependencias: pandas, numpy.
================================================================================
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"

# Fuentes de RECONOCIMIENTO confirmadas por contenido (2026-06-26):
#   Casanare:  686 vs 48 marcadores -> reconocimiento OK
#   Huila:     376 vs 238           -> reconocimiento OK
#   CostaCaribe: 417 vs 183         -> reconocimiento OK
# Son los archivos BASE (sin 'obs_'). Las versiones obs_* son OBSERVACIONES (->centroide, NO ICM).
SUBCASOS = {
    "Casanare": {
        "diar_file": "casanare_torres_diarization.json",
        "trans_file": "casanare_torres_segments.json",
    },
    "Huila": {
        "diar_file": "huila_diarization.json",
        "trans_file": "huila_segments.json",     # NO obs_huila_2022 (observaciones)
    },
    "CostaCaribe": {
        "diar_file": "costa_caribe_diarization.json",
        "trans_file": "costa_caribe_segments.json",  # NO obs_costa_caribe_atanquez (observaciones)
    },
}

RUIDO_RE = re.compile(r"(suscr[ií]bete|subscribe|\[m[uú]sica\]|gracias por ver)", re.I)


def seg_a_hms(s):
    s = int(round(s))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def cargar_json_lista(ruta):
    d = json.load(open(ruta, encoding="utf-8"))
    if isinstance(d, dict):
        for k in ["segments","segmentos","results","chunks"]:
            if k in d and isinstance(d[k], list):
                return d[k]
        for v in d.values():
            if isinstance(v, list):
                return v
    return d if isinstance(d, list) else []


def hallar(base, hints, must_have="diarization"):
    """Busca en corpus_c el archivo que contenga alguno de los hints."""
    todos = glob.glob(str(base/"corpus_c"/"*.json"))
    # priorizar por orden de hints
    for h in hints:
        for f in todos:
            if h.lower() in Path(f).name.lower() and must_have in Path(f).name.lower():
                return Path(f)
    # si must_have no aplica (transcripción), buscar sin él
    for h in hints:
        for f in todos:
            if h.lower() in Path(f).name.lower():
                return Path(f)
    return None


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def texto_de_speaker(speaker_tramos, segs):
    """Junta una muestra de texto de los tramos de un speaker."""
    partes = []
    for s in segs:
        s0, s1 = float(s.get("start",0)), float(s.get("end",0))
        if s1 <= s0:
            continue
        txt = str(s.get("text", s.get("texto",""))).strip()
        if not txt or RUIDO_RE.search(txt):
            continue
        for (t0, t1) in speaker_tramos:
            if overlap(t0, t1, s0, s1) > 0.5:
                partes.append(txt)
                break
        if len(" ".join(partes)) > 400:
            break
    return " ".join(partes)[:400]


def procesar(subcaso, cfg, base):
    base = Path(base)
    print(f"\n{'='*72}\nSUBCASO: {subcaso}\n{'='*72}")

    r_diar = base / "corpus_c" / cfg["diar_file"]
    r_trans = base / "corpus_c" / cfg["trans_file"]
    if not r_diar.exists():
        print(f"  [SALTADO] sin diarización: {r_diar.name}"); return None
    print(f"  Diarización: {r_diar.name}")
    print(f"  Transcripción (RECONOCIMIENTO): "
          f"{r_trans.name if r_trans.exists() else 'NO encontrada'}")

    diar = cargar_json_lista(r_diar)
    segs = cargar_json_lista(r_trans) if r_trans.exists() else []

    # Agrupar tramos por speaker.
    por_spk = defaultdict(list)
    for d in diar:
        s0, s1 = float(d["start"]), float(d["end"])
        por_spk[d["speaker"]].append((s0, s1))

    filas = []
    for spk, tramos in sorted(por_spk.items()):
        tramos.sort()
        t_total = sum(t1 - t0 for t0, t1 in tramos)
        ini = min(t0 for t0, _ in tramos)
        fin = max(t1 for _, t1 in tramos)
        muestra = texto_de_speaker(tramos, segs) if segs else ""
        # Sugerencia de uso: poco tiempo → probablemente no compareciente.
        uso_sug = "EXCLUIR?" if t_total < 60 else ""
        filas.append({
            "audiencia": subcaso,
            "video_url": "",
            "speaker_diar": spk,
            "inicio": seg_a_hms(ini),
            "fin": seg_a_hms(fin),
            "n_segmentos": len(tramos),
            "tiempo_total_s": round(t_total, 1),
            "identidad": "",          # ← COMPLETAR
            "rol": "",                # ← COMPLETAR (COMPARECIENTE/VICTIMA/MAGISTRADO/...)
            "rango_militar": "",      # ← COMPLETAR si aplica
            "evidencia_identificacion": "",
            "confianza": "",
            "uso": uso_sug,           # ← COMPLETAR (ANALISIS/CENTROIDE/EXCLUIR)
            "muestra_texto": muestra,
        })

    df = pd.DataFrame(filas).sort_values("tiempo_total_s", ascending=False)
    out_dir = base / "data" / "marcacion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"inventario_{subcaso}_PLANTILLA.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"  Speakers detectados: {len(df)}")
    print(f"  --- Top speakers por tiempo (para que reconozcas comparecientes) ---")
    for _, r in df.head(8).iterrows():
        m = (r["muestra_texto"][:70] + "…") if r["muestra_texto"] else "(sin texto)"
        print(f"    {r['speaker_diar']:12s} {r['tiempo_total_s']:7.0f}s "
              f"[{r['inicio']}–{r['fin']}]  «{m}»")
    print(f"\n  [GUARDADO] {out}")
    print(f"    → Completa identidad/rol/uso y renómbralo a inventario_{subcaso}.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()

    print("CFH — Generador de plantillas de marcación pre-rellenadas")
    print("Completa SOLO: identidad · rol · uso  (mirando el video)")
    print("uso: ANALISIS=compareciente→ICM | CENTROIDE=víctima | EXCLUIR=magistrado/otro")

    generados = []
    for sc, cfg in SUBCASOS.items():
        try:
            out = procesar(sc, cfg, args.base)
            if out:
                generados.append(out)
        except Exception as e:
            print(f"  [ERROR] {sc}: {type(e).__name__}: {e}")

    print(f"\n{'='*72}\nPLANTILLAS GENERADAS: {len(generados)}")
    for g in generados:
        print(f"  · {g}")
    print("\nFLUJO:")
    print("  1. Abre cada *_PLANTILLA.csv en Excel.")
    print("  2. Para cada SPEAKER, mira el video en [inicio] y escribe identidad/rol/uso.")
    print("     (la muestra_texto te ayuda a reconocer de qué habla cada uno).")
    print("  3. Borra filas EXCLUIR si quieres, o déjalas marcadas.")
    print("  4. Guarda como inventario_<Subcaso>.csv (sin _PLANTILLA).")
    print("  5. Re-corre el ICM tri-canal: ya tomará los 3 subcasos nuevos.")


if __name__ == "__main__":
    main()
