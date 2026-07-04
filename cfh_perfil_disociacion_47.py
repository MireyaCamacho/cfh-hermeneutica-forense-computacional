# -*- coding: utf-8 -*-
"""
cfh_perfil_disociacion_47.py
=============================
Perfil individual COMPLETO de los 47 comparecientes para entender la
disociacion multimodal caso por caso, y un analisis de QUE distingue a los
mas disociados de los mas congruentes.

Para cada compareciente:
  - subcaso, etiqueta MR/no-MR
  - los 3 canales (facial, vocal, verbal/REP)
  - duracion real de la intervencion (tiempo_total_s de la marcacion)
  - n_tokens
  - disociacion (sd de los 3 canales estandarizados) o "EXCLUIDO" si falta canal
  - fragmento de su texto (primeros ~300 chars de su intervencion)

Los 3 sin los 3 canales (Alfonso Romero, Jaime Coral, Luis Fidel Arenas) se
incluyen marcados como EXCLUIDO_DISOCIACION (canal faltante) para no perder la
relacion de los 47.

Analisis final: compara perfil de canales entre el tercio mas disociado y el
tercio mas congruente, para explicar el fenomeno.

Salida:
  data/perfil_disociacion_47.csv
  consola: tabla + analisis disociados vs congruentes.

Uso:
    python cfh_perfil_disociacion_47.py
"""

import glob
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
ICM = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
ETIQ = BASE / "data" / "mr_asignacion_final.csv"
MARC_DIR = BASE / "data" / "marcacion"
AUD_DIR = BASE / "corpus_c"

CANALES = ["icm_facial", "icm_vocal", "y10_rep"]
PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
MIN_OVERLAP_S = 0.5
RUIDO_RE = re.compile(r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\])", re.I)


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
        return p[0]*3600+p[1]*60+p[2] if len(p) == 3 else p[0]*60+p[1]
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


def duracion_y_texto():
    """Devuelve dict key_norm -> (tiempo_total_s, texto) por compareciente."""
    out = {}
    for sub, pref in PREFIJO.items():
        marc_path = MARC_DIR / f"inventario_{sub}.csv"
        if not marc_path.exists():
            continue
        m = pd.read_csv(marc_path)
        m["ini_s"] = m["inicio"].apply(t_a_seg)
        m["fin_s"] = m["fin"].apply(t_a_seg)
        segs = cargar_segments(pref)
        for ident, g in m.groupby("identidad"):
            tt = g["tiempo_total_s"].sum() if "tiempo_total_s" in g.columns else np.nan
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
            out[norm(ident)] = (tt, " ".join(partes).strip())
    return out


def main():
    df = pd.read_csv(ICM)
    et = pd.read_csv(ETIQ)
    df["_k"] = df["identidad"].apply(norm)
    et["_k"] = et["compareciente"].apply(norm)
    df = df.merge(et[["_k", "etiqueta_MR"]], on="_k", how="left")

    # disociacion sobre los que tienen los 3 canales
    comp = df.dropna(subset=CANALES).copy()
    for c in CANALES:
        mu, sd = comp[c].mean(), comp[c].std()
        comp[c + "_z"] = (comp[c] - mu) / (sd if sd > 1e-9 else 1)
    zc = [c + "_z" for c in CANALES]
    comp["disociacion"] = comp[zc].std(axis=1)
    disoc_map = dict(zip(comp["_k"], comp["disociacion"]))

    # duracion y texto
    dt = duracion_y_texto()

    filas = []
    for _, r in df.iterrows():
        k = r["_k"]
        tiene3 = all(pd.notna(r[c]) for c in CANALES)
        disoc = disoc_map.get(k, np.nan)
        tt, texto = dt.get(k, (np.nan, ""))
        falta = "" if tiene3 else "falta:" + ",".join(
            {"icm_facial": "facial", "icm_vocal": "vocal", "y10_rep": "verbal"}[c]
            for c in CANALES if pd.isna(r[c]))
        filas.append({
            "subcaso": r["subcaso"],
            "identidad": r["identidad"],
            "MR": r.get("etiqueta_MR", ""),
            "facial": round(r["icm_facial"], 3) if pd.notna(r["icm_facial"]) else np.nan,
            "vocal": round(r["icm_vocal"], 3) if pd.notna(r["icm_vocal"]) else np.nan,
            "verbal": round(r["y10_rep"], 3) if pd.notna(r["y10_rep"]) else np.nan,
            "duracion_s": round(tt) if pd.notna(tt) else np.nan,
            "n_tokens": r.get("n_tokens", np.nan),
            "disociacion": round(disoc, 3) if pd.notna(disoc) else np.nan,
            "estado": "EXCLUIDO(" + falta + ")" if not tiene3 else "ok",
            "texto_frag": texto[:300].replace("\n", " "),
        })
    out = pd.DataFrame(filas).sort_values(
        "disociacion", ascending=False, na_position="last")
    out.to_csv(BASE / "data" / "perfil_disociacion_47.csv",
               index=False, encoding="utf-8-sig")

    print("=" * 92)
    print(f"PERFIL DE LOS {len(out)} COMPARECIENTES (ordenado por disociacion)")
    print("=" * 92)
    print(f"  {'compareciente':30s} {'sub':10s} {'MR':6s} "
          f"{'fac':>5s} {'voc':>5s} {'verb':>5s} {'dur_s':>6s} {'disoc':>6s} {'estado':>16s}")
    print("-" * 92)
    for _, r in out.iterrows():
        d = f"{r['disociacion']:.3f}" if pd.notna(r["disociacion"]) else "  ---"
        du = f"{int(r['duracion_s'])}" if pd.notna(r["duracion_s"]) else "?"
        fa = f"{r['facial']:.2f}" if pd.notna(r["facial"]) else " -- "
        vo = f"{r['vocal']:.2f}" if pd.notna(r["vocal"]) else " -- "
        ve = f"{r['verbal']:.2f}" if pd.notna(r["verbal"]) else " -- "
        print(f"  {str(r['identidad'])[:30]:30s} {str(r['subcaso'])[:10]:10s} "
              f"{str(r['MR'])[:6]:6s} {fa:>5s} {vo:>5s} {ve:>5s} {du:>6s} {d:>6s} "
              f"{r['estado']:>16s}")

    # analisis disociados vs congruentes (tercios)
    val = out.dropna(subset=["disociacion"]).sort_values("disociacion", ascending=False)
    n3 = max(1, len(val) // 3)
    alto = val.head(n3)
    bajo = val.tail(n3)
    print("\n" + "=" * 92)
    print("  QUE DISTINGUE A LOS DISOCIADOS DE LOS CONGRUENTES (tercios)")
    print("=" * 92)
    print(f"  {'':20s} {'facial':>10s} {'vocal':>10s} {'verbal':>10s} "
          f"{'duracion_s':>11s} {'%MR':>6s}")
    for nombre, grp in [("MAS DISOCIADOS", alto), ("MAS CONGRUENTES", bajo)]:
        pmr = 100 * (grp["MR"] == "MR").mean()
        print(f"  {nombre:20s} {grp['facial'].mean():>10.3f} {grp['vocal'].mean():>10.3f} "
              f"{grp['verbal'].mean():>10.3f} {grp['duracion_s'].mean():>11.0f} {pmr:>5.0f}%")

    # rango de cada canal en cada grupo (para ver cual canal se dispara)
    print("\n  Dispersion por canal (sd dentro de cada grupo):")
    for nombre, grp in [("disociados", alto), ("congruentes", bajo)]:
        print(f"    {nombre:12s}: facial sd={grp['facial'].std():.3f}  "
              f"vocal sd={grp['vocal'].std():.3f}  verbal sd={grp['verbal'].std():.3f}")

    print("\n  Guardado: data/perfil_disociacion_47.csv (con texto_frag por persona)")
    print("\n  LECTURA: comparar las medias por canal entre disociados y congruentes")
    print("  muestra QUE canal se despega. Si los disociados tienen verbal alto pero")
    print("  facial bajo -> reconocimiento performativo (dicen, cuerpo no acompana).")


if __name__ == "__main__":
    main()
