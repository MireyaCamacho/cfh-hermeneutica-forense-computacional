# -*- coding: utf-8 -*-
"""
cfh_diag_verbal.py
================================================================================
CFH — Diagnóstico del canal verbal por compareciente

OBJETO:
    (1) Ver QUÉ TEXTO REAL se le atribuye a cada compareciente de Catatumbo
        (para entender por qué Chaparro sale con y10_rep = 0.000).
    (2) Probar el efecto de un PISO DE TOKENS (>=100) que descarte textos
        demasiado cortos cuyos scores (0.0 / 1.0) son artefactos.

QUÉ HACE:
    Para el subcaso indicado, reconstruye el texto por compareciente (segments ×
    diarization × marcación) y muestra:
      · nº de tokens y de segmentos atribuidos
      · primeras y últimas 200 letras del texto (para ver si es la persona o ruido)
      · cuántos segmentos eran ruido YouTube y se filtraron
      · el y10_rep que produce el extractor real, con y sin piso de tokens

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_diag_verbal.py" --subcaso Catatumbo

Entorno: Python 3.11, conda env cfh. Requiere spaCy + es_core_news_lg.
================================================================================
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"
PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
USOS_ICM = {"ANALISIS"}
ROLES_ICM = {"COMPARECIENTE"}
NO_PERSONAS = {"BLOQUE_COMPARECIENTES", "BLOQUE_COMPARECIENTES_NO_MR"}
MIN_OVERLAP_S = 0.5
PISO_TOKENS = 100   # piso propuesto

RUIDO_RE = re.compile(
    r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\]|\[music\]|"
    r"activa la campanita|dale like|no olvides suscribirte)", re.I)


def t_a_seg(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return p[0]*3600+p[1]*60+p[2] if len(p) == 3 else p[0]*60+p[1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


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


def buscar(base, pref, suf):
    hits = glob.glob(str(base/"corpus_c"/f"{pref}*{suf}.json"))
    return Path(sorted(hits, key=len)[0]) if hits else None


def cargar_extractor(base):
    for src in [Path(base)/"code"/"src", Path(base)/"src"]:
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    from features.y10_rep_extractor import REPExtractor
    return REPExtractor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--subcaso", default="Catatumbo")
    args = ap.parse_args()
    base = Path(args.base)
    sc = args.subcaso
    pref = PREFIJO.get(sc, sc.lower())

    print(f"CFH — Diagnóstico verbal — {sc}")
    print("="*72)

    ext = cargar_extractor(base)
    print("REPExtractor cargado OK\n")

    diar = cargar_json_lista(buscar(base, pref, "diarization"))
    segs = cargar_json_lista(buscar(base, pref, "segments"))
    print(f"Diarización: {len(diar)} seg | Transcripción: {len(segs)} seg\n")

    m = pd.read_csv(base/"data"/"marcacion"/f"inventario_{sc}.csv")
    m["ini_s"] = m["inicio"].apply(t_a_seg)
    m["fin_s"] = m["fin"].apply(t_a_seg)
    cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
           (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
    m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"]>m["ini_s"])]
    m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]

    # Agrupar intervenciones por identidad.
    for ident, grupo in m.groupby("identidad"):
        partes, n_ruido, n_seg_ok = [], 0, 0
        intervalos = [(r["ini_s"], r["fin_s"]) for _, r in grupo.iterrows()]
        for s in segs:
            s0, s1 = float(s.get("start",0)), float(s.get("end",0))
            if s1 <= s0:
                continue
            if any(overlap(a0,a1,s0,s1) >= MIN_OVERLAP_S for a0,a1 in intervalos):
                txt = str(s.get("text", s.get("texto",""))).strip()
                if not txt:
                    continue
                if RUIDO_RE.search(txt):
                    n_ruido += 1
                    continue
                partes.append(txt)
                n_seg_ok += 1
        texto = " ".join(partes).strip()
        n_tok = len(texto.split())

        print(f"\n{'─'*72}")
        print(f"● {ident}")
        print(f"  intervalos: {[(round(a,0),round(b,0)) for a,b in intervalos]}")
        print(f"  segmentos atribuidos: {n_seg_ok} | ruido YouTube filtrado: {n_ruido}")
        print(f"  tokens: {n_tok}")
        if texto:
            print(f"  INICIO: «{texto[:200]}»")
            print(f"  FINAL : «{texto[-200:]}»")
        else:
            print("  (sin texto atribuido)")

        # Score con el extractor.
        if n_tok >= 20:
            r = ext.extract(texto, doc_id=str(ident), section_id="RECONOCIMIENTO", corpus_type="C")
            piso = "OK" if n_tok >= PISO_TOKENS else f"< piso {PISO_TOKENS} → NA"
            print(f"  y10_rep (extractor): {r.score:.3f}  | score_raw={getattr(r,'score_raw','?')}")
            print(f"  n_reconocimiento={getattr(r,'n_reconocimiento','?')} "
                  f"n_restitución={getattr(r,'n_restitución','?')} "
                  f"n_dih={getattr(r,'n_dih','?')} "
                  f"n_reparación={getattr(r,'n_reparación','?')}")
            print(f"  con PISO de tokens ({PISO_TOKENS}): {piso}")
        else:
            print(f"  texto < 20 tokens → extractor devuelve vacío")

    print(f"\n{'='*72}")
    print("Lectura: si el INICIO/FINAL del texto no corresponde a la persona,")
    print("el cruce de tiempo está mal (revisar offset diarización vs marcación).")
    print("Si el texto SÍ es suyo pero y10_rep=0, entonces no hubo marcadores REP")
    print("en su intervención (resultado legítimo, no error).")


if __name__ == "__main__":
    main()
