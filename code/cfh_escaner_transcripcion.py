# -*- coding: utf-8 -*-
"""
cfh_escaner_transcripcion.py
================================================================================
CFH — Escáner de calidad de transcripción (Whisper) del Corpus C

DOS TAREAS EN UNA:
  (1) MEDIR EL DAÑO: detecta tramos degenerados de Whisper ("y y y y",
      "gracias gracias", token repetido N veces, segmentos vacíos) en los 5
      segments del Corpus C. Reporta % de daño por subcaso y en qué franjas
      temporales está concentrado.
  (2) BUSCAR ALTERNATIVAS: localiza TODAS las transcripciones de cada subcaso
      en local/G: (puede haber varias versiones) y compara su nivel de daño,
      para elegir la mejor.

CRITERIOS DE DEGENERACIÓN:
  · 'token_unico_repetido': un mismo token ocupa >70% de las palabras del seg.
  · 'mono_caracter': el texto es básicamente una letra repetida ("y y y").
  · 'frase_loop': la misma frase corta repetida >5 veces.
  · 'vacio': segmento sin texto útil.

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_escaner_transcripcion.py"

Entorno: Python 3.11, conda env cfh. Dependencias: pandas, numpy.
================================================================================
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"
RAICES_EXTRA = [r"G:\Mi unidad"]
SUBCASOS = ["casanare", "catatumbo", "dabeiba", "huila", "costa_caribe"]

IGNORAR = {"node_modules", ".git", "__pycache__", ".venv", "$RECYCLE.BIN",
           "System Volume Information", ".Trashes"}


def cargar_json_lista(ruta):
    try:
        d = json.load(open(ruta, encoding="utf-8"))
    except Exception:
        return []
    if isinstance(d, dict):
        for k in ["segments","segmentos","results","chunks"]:
            if k in d and isinstance(d[k], list):
                return d[k]
        for v in d.values():
            if isinstance(v, list):
                return v
    return d if isinstance(d, list) else []


def clasificar_segmento(txt):
    """Devuelve None si el segmento es válido, o el tipo de degeneración."""
    t = str(txt).strip()
    if len(t) < 2:
        return "vacio"
    palabras = t.split()
    if not palabras:
        return "vacio"
    # mono-caracter: casi todo son tokens de 1 letra
    cortos = sum(1 for p in palabras if len(p) <= 1)
    if cortos / len(palabras) > 0.6:
        return "mono_caracter"
    # token único dominante
    c = Counter(palabras)
    top, n = c.most_common(1)[0]
    if len(palabras) >= 5 and n / len(palabras) > 0.7:
        return "token_unico_repetido"
    # frase loop: pocas palabras distintas en texto largo
    if len(palabras) >= 15 and len(set(palabras)) <= 3:
        return "frase_loop"
    return None


def escanear_archivo(ruta):
    segs = cargar_json_lista(ruta)
    if not segs:
        return None
    total = len(segs)
    danados, tipos, t_danado = 0, Counter(), []
    dur_total, dur_danada = 0.0, 0.0
    for s in segs:
        txt = s.get("text", s.get("texto", ""))
        s0, s1 = float(s.get("start", 0)), float(s.get("end", 0))
        dur = max(0.0, s1 - s0)
        dur_total += dur
        tipo = clasificar_segmento(txt)
        if tipo:
            danados += 1
            tipos[tipo] += 1
            dur_danada += dur
            t_danado.append((s0, s1))
    pct_seg = 100.0 * danados / max(1, total)
    pct_dur = 100.0 * dur_danada / max(1e-9, dur_total)
    return {
        "total_seg": total, "danados": danados,
        "pct_segmentos": round(pct_seg, 1), "pct_duracion": round(pct_dur, 1),
        "tipos": dict(tipos),
        "franjas_danadas_s": t_danado,
    }


def listar_transcripciones(raices, pref):
    """Todas las transcripciones (segments/transcrip/whisper) de un subcaso."""
    out = []
    pats = [f"{pref}*segments*.json", f"{pref}*transcrip*.json",
            f"{pref}*whisper*.json", f"{pref}*.json"]
    for raiz in raices:
        raiz = Path(raiz)
        if not raiz.exists():
            continue
        rd = len(raiz.parts)
        for dp, dn, fn in os.walk(raiz):
            dn[:] = [d for d in dn if d not in IGNORAR]
            if len(Path(dp).parts) - rd > 7:
                dn[:] = []; continue
            for f in fn:
                low = f.lower()
                if low.endswith(".json") and pref in low and \
                   ("segment" in low or "transcri" in low or "whisper" in low):
                    out.append(Path(dp) / f)
    # dedup
    return sorted(set(out), key=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()
    base = Path(args.base)
    raices = [base, base/"corpus_c"] + [Path(r) for r in RAICES_EXTRA]

    print("CFH — Escáner de calidad de transcripción (Corpus C)")
    print("="*72)

    resumen = []
    for pref in SUBCASOS:
        print(f"\n{'='*72}\nSUBCASO: {pref}\n{'='*72}")
        candidatos = listar_transcripciones(raices, pref)
        if not candidatos:
            print("  (sin transcripciones encontradas)")
            continue
        mejores = []
        for ruta in candidatos:
            r = escanear_archivo(ruta)
            if r is None:
                continue
            estado = "LIMPIA ✓" if r["pct_duracion"] < 5 else \
                     ("DAÑO PARCIAL" if r["pct_duracion"] < 30 else "MUY DAÑADA ✗")
            print(f"\n  {ruta.name}")
            print(f"    [{estado}] segmentos dañados: {r['danados']}/{r['total_seg']} "
                  f"({r['pct_segmentos']}%) | duración dañada: {r['pct_duracion']}%")
            if r["tipos"]:
                print(f"    tipos: {r['tipos']}")
            mejores.append((r["pct_duracion"], ruta, r))
            resumen.append({"subcaso": pref, "archivo": str(ruta),
                            "pct_seg_danado": r["pct_segmentos"],
                            "pct_dur_danada": r["pct_duracion"],
                            "estado": estado, **{f"tipo_{k}": v for k,v in r["tipos"].items()}})
        # Recomendación por subcaso.
        if mejores:
            mejores.sort()
            mejor = mejores[0]
            print(f"\n  → MEJOR versión: {mejor[1].name} (daño {mejor[0]}%)")
            if mejor[0] >= 30:
                print(f"    ⚠ Todas las versiones muy dañadas → re-transcribir este subcaso.")
            elif mejor[0] >= 5:
                print(f"    ⚠ Daño parcial → usable con filtrado de tramos degenerados.")
            else:
                print(f"    ✓ Versión limpia disponible.")

    if resumen:
        df = pd.DataFrame(resumen)
        out = base / "outputs" / "capa3" / "escaner_transcripcion.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n{'='*72}\n[GUARDADO] {out}")
        print("\nRESUMEN — mejor versión por subcaso:")
        for pref in SUBCASOS:
            sub = df[df["subcaso"]==pref].sort_values("pct_dur_danada")
            if not sub.empty:
                b = sub.iloc[0]
                print(f"  {pref:14} mejor daño={b['pct_dur_danada']}%  ({b['estado']})")


if __name__ == "__main__":
    main()
