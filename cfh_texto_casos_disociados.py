# -*- coding: utf-8 -*-
"""
cfh_texto_casos_disociados.py
==============================
Extrae el TEXTO COMPLETO de los comparecientes disociados (y de contraste, los
congruentes) desde el enganche segments x marcacion, junto a su perfil de
canales (facial, vocal, verbal), disociacion y duracion, para el analisis
cualitativo de la conclusion.

Clasifica cada caso disociado de forma DESCRIPTIVA (solo direccion de la
incongruencia entre canales, SIN inferir estados internos ni efecto):
  VERBAL_ALTO_NOVERB_BAJO: verbal alto + facial/vocal bajo
           -> el canal verbal se despega hacia arriba de los no-verbales.
  VERBAL_BAJO_NOVERB_ALTO: verbal bajo + facial/vocal alto
           -> los canales no-verbales se despegan hacia arriba del verbal.
NOTA EPISTEMOLOGICA: el ICM mide CONGRUENCIA entre senales comunicativas, no
sinceridad, autenticidad ni efecto psicologico (Barrett et al. 2019). Por eso
las etiquetas describen la direccion del desajuste, no lo interpretan.

Salida:
  data/casos_disociados_texto.csv   (perfil + texto completo por caso)
  outputs/casos_disociados_analisis.txt  (informe legible por caso)

Uso:
    python cfh_texto_casos_disociados.py
"""

import glob
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
PERFIL = BASE / "data" / "perfil_disociacion_47.csv"
MARC_DIR = BASE / "data" / "marcacion"
AUD_DIR = BASE / "corpus_c"

PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
MIN_OVERLAP_S = 0.5
RUIDO_RE = re.compile(r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\])", re.I)

# umbral de disociacion para "disociado" (tercio superior aprox)
UMBRAL_DISOC = 1.0
# umbrales para clasificar patron (sobre valores 0-1 de cada canal)
VERBAL_ALTO = 0.35
NOVERB_BAJO = 0.35


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def t_a_seg(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def cargar_segments(pref):
    hits = glob.glob(str(AUD_DIR / f"{pref}*segments*.json"))
    if not hits:
        return []
    d = json.load(open(sorted(hits, key=len)[0], encoding="utf-8"))
    if isinstance(d, dict):
        for k in ["segments", "segmentos"]:
            if k in d:
                return d[k]
    return d if isinstance(d, list) else []


def texto_por_persona():
    out = {}
    for sub, pref in PREFIJO.items():
        mp = MARC_DIR / f"inventario_{sub}.csv"
        if not mp.exists():
            continue
        m = pd.read_csv(mp)
        m["ini_s"] = m["inicio"].apply(t_a_seg)
        m["fin_s"] = m["fin"].apply(t_a_seg)
        segs = cargar_segments(pref)
        for ident, g in m.groupby("identidad"):
            partes = []
            for _, r in g.iterrows():
                if pd.isna(r["ini_s"]) or pd.isna(r["fin_s"]):
                    continue
                for s in segs:
                    s0, s1 = float(s.get("start", 0)), float(s.get("end", 0))
                    if s1 > s0 and overlap(r["ini_s"], r["fin_s"], s0, s1) >= MIN_OVERLAP_S:
                        t = str(s.get("text", "")).strip()
                        if t and not RUIDO_RE.search(t):
                            partes.append(t)
            txt = " ".join(partes).strip()
            if txt:
                out[norm(ident)] = txt
    return out


def clasificar(fac, voc, verb):
    """Etiqueta DESCRIPTIVA de la direccion de la incongruencia (sin inferir efecto)."""
    noverb = np.nanmean([fac, voc])
    if verb >= VERBAL_ALTO and noverb < NOVERB_BAJO + 0.05 and verb - noverb > 0.15:
        return "verbal_alto_noverb_bajo"
    if verb < NOVERB_BAJO and noverb - verb > 0.15:
        return "verbal_bajo_noverb_alto"
    return "mixto"


def main():
    perf = pd.read_csv(PERFIL)
    textos = texto_por_persona()
    perf["_k"] = perf["identidad"].apply(norm)
    perf["texto_completo"] = perf["_k"].map(textos).fillna("")

    # solo con disociacion calculada
    d = perf.dropna(subset=["disociacion"]).copy()
    d["patron"] = d.apply(
        lambda r: clasificar(r["facial"], r["vocal"], r["verbal"]), axis=1)

    disociados = d[d["disociacion"] >= UMBRAL_DISOC].sort_values(
        "disociacion", ascending=False)
    congruentes = d.sort_values("disociacion").head(4)

    # guardar CSV
    cols = ["subcaso", "identidad", "MR", "facial", "vocal", "verbal",
            "disociacion", "duracion_s", "patron", "texto_completo"]
    cols = [c for c in cols if c in d.columns]
    d.sort_values("disociacion", ascending=False)[cols].to_csv(
        BASE / "data" / "casos_disociados_texto.csv",
        index=False, encoding="utf-8-sig")

    # informe legible
    out_txt = BASE / "outputs" / "casos_disociados_analisis.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("ANALISIS CUALITATIVO — CASOS DISOCIADOS (texto vs canales)")
    lines.append("=" * 78)
    lines.append(f"\nDisociados (disociacion >= {UMBRAL_DISOC}): {len(disociados)}\n")

    for _, r in disociados.iterrows():
        lines.append("-" * 78)
        lines.append(f"{r['identidad']}  [{r['subcaso']} | {r['MR']}]  "
                     f"PATRON: {r['patron']}")
        lines.append(f"  canales: facial={r['facial']:.2f}  vocal={r['vocal']:.2f}  "
                     f"verbal={r['verbal']:.2f}   disociacion={r['disociacion']:.3f}")
        if pd.notna(r.get("duracion_s")):
            lines.append(f"  duracion: {int(r['duracion_s'])}s")
        lines.append(f"  TEXTO:")
        txt = r["texto_completo"] or "(sin texto enganchado)"
        for i in range(0, len(txt), 76):
            lines.append("    " + txt[i:i+76])
        lines.append("")

    lines.append("=" * 78)
    lines.append("CONTRASTE — CASOS MAS CONGRUENTES (canales alineados)")
    lines.append("=" * 78)
    for _, r in congruentes.iterrows():
        lines.append("-" * 78)
        lines.append(f"{r['identidad']}  [{r['subcaso']} | {r['MR']}]  "
                     f"disociacion={r['disociacion']:.3f}")
        lines.append(f"  canales: facial={r['facial']:.2f}  vocal={r['vocal']:.2f}  "
                     f"verbal={r['verbal']:.2f}")
        txt = (r["texto_completo"] or "(sin texto)")[:400]
        lines.append("  TEXTO (extracto): " + txt)
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # resumen consola
    print("=" * 60)
    print("CASOS DISOCIADOS POR PATRON")
    print("=" * 60)
    print(disociados.groupby("patron").size().to_string())
    print(f"\n  Disociados totales: {len(disociados)}")
    print("\n  Detalle:")
    for _, r in disociados.iterrows():
        print(f"    {str(r['identidad'])[:28]:28s} {r['patron']:24s} "
              f"f={r['facial']:.2f} v={r['vocal']:.2f} verb={r['verbal']:.2f}")
    print(f"\n  Guardado: data/casos_disociados_texto.csv")
    print(f"           {out_txt}")
    print("\n  Abri el .txt para leer el texto completo de cada caso junto a su perfil.")


if __name__ == "__main__":
    main()
