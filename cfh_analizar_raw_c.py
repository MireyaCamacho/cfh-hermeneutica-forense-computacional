# -*- coding: utf-8 -*-
"""
cfh_analizar_raw_c.py
======================
Investiga POR QUE el score_raw de REP es alto en Corpus C (comparecientes),
antes de decidir como normalizar.

Hipotesis (por el caso Oswaldo): el raw = suma_pesos_oraciones / n_oraciones.
En textos CORTOS con pocas oraciones, si varias son de reconocimiento, la
proporcion se dispara (1 reconocimiento / 1 oracion = 1.0). El raw alto seria
un artefacto de la longitud del texto capturado, no de "mas reconocimiento".

Este script, para cada compareciente de C, muestra:
  - n_chars, n_oraciones (sent_count), n_instancias REP
  - score_raw
  - densidad = n_instancias / n_oraciones
Y correlaciona raw con longitud para ver si los raw altos son los textos cortos.

Reproducible, sin hardcodear.

Uso:
    python cfh_analizar_raw_c.py > analisis_raw_c.txt 2>&1
    type analisis_raw_c.txt
"""

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
RUIDO_RE = re.compile(r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\]|\[music\])", re.I)


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


def main():
    from features.y10_rep_extractor import REPExtractor
    ext = REPExtractor()

    filas = []
    for sc, pref in PREFIJO.items():
        hits = glob.glob(str(BASE / "corpus_c" / f"{pref}*segments*.json"))
        ruta_marc = BASE / "data" / "marcacion" / f"inventario_{sc}.csv"
        if not hits or not ruta_marc.exists():
            continue
        segs = cargar_json_lista(sorted(hits, key=len)[0])
        m = pd.read_csv(ruta_marc)
        m["ini_s"] = m["inicio"].apply(t_a_seg)
        m["fin_s"] = m["fin"].apply(t_a_seg)
        cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
               (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
        m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"] > m["ini_s"])]
        m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]

        tx = {}
        for _, r in m.iterrows():
            partes = []
            for s in segs:
                s0, s1 = float(s.get("start", 0)), float(s.get("end", 0))
                if s1 > s0 and overlap(r["ini_s"], r["fin_s"], s0, s1) >= MIN_OVERLAP_S:
                    t = str(s.get("text", "")).strip()
                    if t and not RUIDO_RE.search(t):
                        partes.append(t)
            tx.setdefault(r["identidad"], []).append(" ".join(partes))
        tx = {k: " ".join(v).strip() for k, v in tx.items()}

        for ident, texto in tx.items():
            if len(texto) < 20:
                continue
            res = ext.extract(texto, doc_id=str(ident), section_id="RECONOCIMIENTO", corpus_type="C")
            dens = res.n_instances / res.n_sentences if res.n_sentences else 0
            filas.append({
                "subcaso": sc,
                "compareciente": ident,
                "chars": len(texto),
                "tokens": len(texto.split()),
                "n_sent": res.n_sentences,
                "n_inst": res.n_instances,
                "densidad": round(dens, 3),
                "raw": round(res.score_raw, 4),
            })

    df = pd.DataFrame(filas).sort_values("raw", ascending=False)

    print("=" * 95)
    print("ANALISIS DEL raw DE REP EN CORPUS C  (ordenado por raw desc)")
    print("=" * 95)
    print(f"{'subcaso':11s} {'compareciente':34s} {'chars':>6s} {'n_sent':>7s} "
          f"{'n_inst':>7s} {'dens':>6s} {'raw':>7s}")
    print("-" * 95)
    for _, r in df.iterrows():
        alerta = "  <-- corto?" if r["n_sent"] <= 3 else ""
        print(f"{r['subcaso']:11s} {str(r['compareciente'])[:34]:34s} "
              f"{r['chars']:>6d} {r['n_sent']:>7d} {r['n_inst']:>7d} "
              f"{r['densidad']:>6.2f} {r['raw']:>7.4f}{alerta}")

    print("\n" + "=" * 95)
    print("ESTADISTICAS")
    print("=" * 95)
    print(f"  n comparecientes: {len(df)}")
    print(f"  raw: min={df['raw'].min():.4f}  mediana={df['raw'].median():.4f}  "
          f"max={df['raw'].max():.4f}")
    print(f"  n_sent: min={df['n_sent'].min()}  mediana={df['n_sent'].median():.0f}  "
          f"max={df['n_sent'].max()}")

    # correlacion raw vs longitud
    if len(df) > 3:
        corr_chars = df["raw"].corr(df["chars"])
        corr_sent = df["raw"].corr(df["n_sent"])
        print(f"\n  correlacion raw vs chars:   {corr_chars:+.3f}")
        print(f"  correlacion raw vs n_sent:  {corr_sent:+.3f}")
        print("\n  INTERPRETACION:")
        if corr_sent < -0.3:
            print("  -> Correlacion NEGATIVa fuerte: los raw altos SON los textos cortos.")
            print("     El raw alto es artefacto de longitud, no de mas reconocimiento.")
            print("     Solucion: normalizar por densidad o ponderar por longitud, no")
            print("     solo proporcion de oraciones.")
        else:
            print("  -> No hay correlacion fuerte con longitud. El raw alto refleja")
            print("     reconocimiento real, no artefacto de texto corto.")

    # cuantos textos muy cortos
    cortos = (df["n_sent"] <= 3).sum()
    print(f"\n  comparecientes con <=3 oraciones: {cortos} / {len(df)}")
    print("  (esos son los candidatos a raw inflado por longitud)")

    out = BASE / "data" / "analisis_raw_c.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n  guardado: {out}")


if __name__ == "__main__":
    main()
