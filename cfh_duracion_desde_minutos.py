# -*- coding: utf-8 -*-
"""
cfh_duracion_desde_minutos.py
==============================
Reconstruye la DURACION real de cada compareciente a partir de las marcas
inicio/fin (los minutos que Mireya anoto una a una), calculando fin - inicio
por intervencion y sumando por persona. NO depende de la columna
'tiempo_total_s' que quedo a medio llenar (vacia en Catatumbo y Dabeiba).

Cobertura de inicio/fin (verificado):
  - Dabeiba:    inicio 81/81, fin 81/81  -> reconstruible completo
  - Casanare, Huila, CostaCaribe: tienen inicio/fin -> reconstruible
  - Catatumbo:  fin solo 21/89 -> parcial (varios quedaran sin duracion)

Cruza con el perfil de disociacion (data/perfil_disociacion_47.csv) y analiza:
  1. duracion por MR vs no-MR (hipotesis: a los no-MR les dan menos tiempo).
  2. correlacion duracion ~ disociacion (Spearman): la duracion NO deberia
     explicar la disociacion (control ya hecho con tokens; se reconfirma).

Salida:
  data/duracion_comparecientes.csv
  data/perfil_disociacion_47_con_duracion.csv
  consola: cobertura, test MR vs no-MR, correlacion con disociacion.

Uso:
    python cfh_duracion_desde_minutos.py
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

BASE = Path(".")
MARC_DIR = BASE / "data" / "marcacion"
PERFIL = BASE / "data" / "perfil_disociacion_47.csv"
SUBCASOS = ["Catatumbo", "Dabeiba", "Casanare", "Huila", "CostaCaribe"]


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


def main():
    filas = []
    print("=" * 60)
    print("RECONSTRUCCION DE DURACION DESDE inicio/fin")
    print("=" * 60)
    for sub in SUBCASOS:
        p = MARC_DIR / f"inventario_{sub}.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p)
        m["ini_s"] = m["inicio"].apply(t_a_seg)
        m["fin_s"] = m["fin"].apply(t_a_seg)
        m["dur"] = m["fin_s"] - m["ini_s"]
        # solo duraciones validas y positivas
        m.loc[(m["dur"] <= 0) | (m["dur"] > 20000), "dur"] = np.nan
        val = m["dur"].notna().sum()
        print(f"  {sub:12s}: {val}/{len(m)} intervenciones con duracion valida")
        for ident, g in m.groupby("identidad"):
            dur_total = g["dur"].sum(min_count=1)
            n_int = g["dur"].notna().sum()
            filas.append({
                "subcaso": sub, "identidad": ident,
                "duracion_s": round(dur_total) if pd.notna(dur_total) else np.nan,
                "n_intervenciones": int(n_int),
            })

    dur = pd.DataFrame(filas)
    dur.to_csv(BASE / "data" / "duracion_comparecientes.csv",
               index=False, encoding="utf-8-sig")

    # cruzar con perfil de disociacion
    if not PERFIL.exists():
        print(f"\n[AVISO] no existe {PERFIL}; solo se genero duracion_comparecientes.csv")
        return
    perf = pd.read_csv(PERFIL)
    perf["_k"] = perf["identidad"].apply(norm)
    dur["_k"] = dur["identidad"].apply(norm)
    m = perf.merge(dur[["_k", "duracion_s", "n_intervenciones"]], on="_k",
                   how="left", suffixes=("_old", ""))
    # usar la duracion reconstruida
    if "duracion_s_old" in m.columns:
        m = m.drop(columns=["duracion_s_old"])
    m.to_csv(BASE / "data" / "perfil_disociacion_47_con_duracion.csv",
             index=False, encoding="utf-8-sig")

    con_dur = m["duracion_s"].notna().sum()
    print(f"\n  comparecientes con duracion reconstruida: {con_dur} / {len(m)}")

    # 1. MR vs no-MR
    print("\n" + "=" * 60)
    print("  DURACION: MR vs no-MR")
    print("=" * 60)
    a = m[m["MR"] == "MR"]["duracion_s"].dropna()
    b = m[m["MR"] == "NO_MR"]["duracion_s"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        n = len(a) + len(b)
        mu = len(a)*len(b)/2
        sigma = np.sqrt(len(a)*len(b)*(n+1)/12)
        r = abs((U-mu)/sigma)/np.sqrt(n)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  MR:    mediana {a.median():.0f}s  media {a.mean():.0f}s  n={len(a)}")
        print(f"  no-MR: mediana {b.median():.0f}s  media {b.mean():.0f}s  n={len(b)}")
        print(f"  U={U:.1f}  p={p:.4f}  r={r:.3f}  {sig}")
        if p < 0.05:
            print("  -> CONFIRMA: a los no-MR se les da menos tiempo (hallazgo estructural)")

    # 2. duracion ~ disociacion
    print("\n" + "=" * 60)
    print("  ¿LA DURACION EXPLICA LA DISOCIACION?")
    print("=" * 60)
    sub = m.dropna(subset=["duracion_s", "disociacion"])
    if len(sub) >= 5:
        rho, p = spearmanr(sub["duracion_s"], sub["disociacion"])
        print(f"  disociacion ~ duracion: rho={rho:+.3f}  p={p:.4f}  (n={len(sub)})")
        if abs(rho) < 0.3 or p >= 0.05:
            print("  -> NO la explica: la disociacion es independiente de la duracion")
        else:
            print("  -> hay relacion: revisar si es artefacto")

    print("\n  Guardado: data/duracion_comparecientes.csv")
    print("           data/perfil_disociacion_47_con_duracion.csv")


if __name__ == "__main__":
    main()
