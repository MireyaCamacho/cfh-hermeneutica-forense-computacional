# -*- coding: utf-8 -*-
"""
cfh_diag_rep_extractor.py
==========================
El enganche ya se valido: a cada compareciente le llega su texto correcto.
Entonces el bug del y10_rep=0.000 esta en el REPEXTRACTOR mismo.

Este script, SIN hardcodear texto, recorre los comparecientes de un subcaso,
les arma el texto con el MISMO enganche del pipeline final, y corre el
REPExtractor PASO A PASO mostrando donde se pierde el score:

  [A] matching crudo de patrones regex (cuantos reconocimientos detecta)
  [B] score_raw y score normalizado del extractor real
  [C] diagnostico de sent_index (si las instancias caen en -1 y se descartan)
  [D] estado del normalizador (si usa defaults p_high=0.4 sin fit)

Uso:
    python cfh_diag_rep_extractor.py --subcaso Huila
    python cfh_diag_rep_extractor.py --subcaso Huila --persona "Restrepo"
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
sys.path.insert(0, str(BASE / "code" / "src"))

PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
USOS_ICM = {"ANALISIS"}
ROLES_ICM = {"COMPARECIENTE"}
NO_PERSONAS = {"BLOQUE_COMPARECIENTES", "BLOQUE_COMPARECIENTES_NO_MR"}
MIN_OVERLAP_S = 0.5
RUIDO_RE = re.compile(
    r"(suscr[i\u00ed]bete|subscribe|gracias por ver|\[m[u\u00fa]sica\]|\[music\])", re.I)


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
        for k in ["segments", "segmentos", "results", "chunks"]:
            if k in d and isinstance(d[k], list):
                return d[k]
        for v in d.values():
            if isinstance(v, list):
                return v
    return d if isinstance(d, list) else []


def buscar_archivo(pref, sufijo):
    hits = glob.glob(str(BASE / "corpus_c" / f"{pref}*{sufijo}.json"))
    return Path(sorted(hits, key=len)[0]) if hits else None


def cargar_intervalos(ruta_marc):
    m = pd.read_csv(ruta_marc)
    m["ini_s"] = m["inicio"].apply(t_a_seg)
    m["fin_s"] = m["fin"].apply(t_a_seg)
    cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
           (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
    m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"] > m["ini_s"])]
    m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]
    return m.reset_index(drop=True)


def texto_por_compareciente(segments, intervalos):
    out = {}
    for _, r in intervalos.iterrows():
        a0, a1 = r["ini_s"], r["fin_s"]
        partes = []
        for s in segments:
            s0, s1 = float(s.get("start", 0)), float(s.get("end", 0))
            if s1 <= s0:
                continue
            if overlap(a0, a1, s0, s1) >= MIN_OVERLAP_S:
                txt = str(s.get("text", "")).strip()
                if txt and not RUIDO_RE.search(txt):
                    partes.append(txt)
        ident = r["identidad"]
        out.setdefault(ident, []).append(" ".join(partes))
    return {k: " ".join(v).strip() for k, v in out.items()}


def diagnosticar_texto(ident, texto, ext):
    from features.y10_rep_extractor import (
        _REP_RECONOCIMIENTO_COMPILED, _REP_RESTITUCION_COMPILED,
        _REP_DIH_COMPILED, _REP_REPARACION_COMPILED,
    )
    print(f"\n{'='*72}")
    print(f"COMPARECIENTE: {ident}   ({len(texto)} chars, {len(texto.split())} tokens)")
    print(f"{'='*72}")

    # buscar frases de reconocimiento en el texto crudo (informativo)
    low = texto.lower()
    for frase in ["reconozco", "responsabilidad", "pido perd", "perd\u00f3n", "acepto",
                  "asumo", "v\u00edctima", "disculp"]:
        if frase in low:
            idx = low.find(frase)
            print(f"  contiene '{frase}': ...{texto[max(0,idx-20):idx+40]}...")

    # [A] matching crudo
    print("\n  [A] MATCHING REGEX CRUDO:")
    grupos = [("reconocimiento", _REP_RECONOCIMIENTO_COMPILED),
              ("restitucion", _REP_RESTITUCION_COMPILED),
              ("dih", _REP_DIH_COMPILED),
              ("reparacion", _REP_REPARACION_COMPILED)]
    total = 0
    matches_todos = []
    for nombre, patrones in grupos:
        ms = []
        for pat in patrones:
            for m in pat.finditer(texto):
                ms.append(m)
                matches_todos.append(m)
        total += len(ms)
        print(f"    {nombre:16s}: {len(ms)} matches", end="")
        if ms:
            print(f"  ej: '{ms[0].group()[:45]}'")
        else:
            print()
    print(f"    TOTAL: {total} matches")

    # [B] extractor real
    print("\n  [B] EXTRACTOR REAL:")
    res = ext.extract(texto, doc_id=str(ident), section_id="RECONOCIMIENTO", corpus_type="C")
    print(f"    n_sentences: {res.n_sentences}")
    print(f"    n_instances: {res.n_instances}")
    print(f"    score_raw:   {res.score_raw:.5f}")
    print(f"    score(norm): {res.score:.5f}")

    # [C] sent_index de las instancias
    print("\n  [C] sent_index de instancias (se descartan las de -1):")
    if res.instances:
        neg = sum(1 for i in res.instances if i.sent_index < 0)
        print(f"    instancias con sent_index=-1: {neg}/{len(res.instances)}")
        for inst in res.instances[:6]:
            print(f"      sent={inst.sent_index:3d} w={inst.weight:.2f} '{inst.text_span[:40]}'")
        if neg == len(res.instances):
            print("    [!! BUG] TODAS en -1 -> score_raw=0 aunque hay matches")
    else:
        print("    (sin instancias)")

    # veredicto local
    if total > 0 and res.score_raw == 0:
        print("\n  >> BUG: hay matches pero score_raw=0 (revisar sent_index / calculo)")
    elif total > 0 and res.score_raw > 0 and res.score == 0:
        print("\n  >> BUG: score_raw>0 pero normalizado=0 (revisar normalizador)")
    elif total == 0:
        print("\n  >> texto sin matches REP (posible: reconoce con otras palabras)")
    else:
        print(f"\n  >> OK: score={res.score:.3f}")
    return res.score_raw, res.score, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subcaso", default="Huila")
    ap.add_argument("--persona", default=None, help="filtra por substring del nombre")
    args = ap.parse_args()

    from features.y10_rep_extractor import REPExtractor, REPScoreNormalizer
    ext = REPExtractor()

    # [D] estado del normalizador
    print("[D] ESTADO DEL NORMALIZADOR:")
    nz = ext.normalizer
    print(f"    method={nz.method}  p_low={nz._p_low}  p_high={nz._p_high}  "
          f"fitted={getattr(nz,'_fitted',False)}")
    if not getattr(nz, "_fitted", False):
        print("    [aviso] normalizador SIN fit -> usa defaults (p_high=0.4).")
        print("            Si score_raw es pequeno, el normalizado puede quedar diminuto.")

    pref = PREFIJO.get(args.subcaso, args.subcaso.lower())
    r_seg = buscar_archivo(pref, "segments")
    segs = cargar_json_lista(r_seg)
    intervalos = cargar_intervalos(BASE / "data" / "marcacion" / f"inventario_{args.subcaso}.csv")
    textos = texto_por_compareciente(segs, intervalos)

    resumen = []
    for ident, txt in sorted(textos.items()):
        if args.persona and args.persona.lower() not in ident.lower():
            continue
        if len(txt.strip()) < 20:
            continue
        raw, norm, total = diagnosticar_texto(ident, txt, ext)
        resumen.append((ident, raw, norm, total))

    print(f"\n{'='*72}\nRESUMEN {args.subcaso}\n{'='*72}")
    print(f"  {'compareciente':40s} {'raw':>8s} {'norm':>8s} {'matches':>8s}")
    ceros = 0
    for ident, raw, norm, total in resumen:
        flag = "  <-- REP=0 con matches!" if (norm == 0 and total > 0) else ""
        if norm == 0:
            ceros += 1
        print(f"  {str(ident)[:40]:40s} {raw:>8.4f} {norm:>8.4f} {total:>8d}{flag}")
    print(f"\n  comparecientes con REP=0: {ceros}/{len(resumen)}")


if __name__ == "__main__":
    main()
