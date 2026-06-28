# -*- coding: utf-8 -*-
"""
cfh_diag_facial_distress.py
================================================================================
CFH — Diagnóstico del canal facial: distress vs sonrisa por compareciente

OBJETO (paso 3 antes de decidir):
    El ICM facial = distress/(distress+sonrisa) se satura hacia 1 cuando la
    sonrisa (AU12) ≈ 0, lo cual ocurre en casi todos los comparecientes de las
    audiencias de reconocimiento (nadie sonríe). Antes de reformular o normalizar,
    necesitamos VER los componentes crudos:
      · distress medio (AU1+AU4+AU15+AU17)
      · sonrisa media (AU12)
      · el ratio actual
    para los 4 subcasos con facial, y decidir con datos si:
      (a) el facial alto refleja distress real, o
      (b) es saturación por ausencia de sonrisa.

QUÉ MUESTRA:
    Tabla por compareciente (todos los subcasos) ordenada por subcaso:
      identidad | n_frames | distress | sonrisa | ratio_actual | AU4 | AU12
    + estadísticas por subcaso (media distress, media sonrisa, % con AU12<0.05)

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_diag_facial_distress.py"

Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
SUBCASOS = {
    "Catatumbo":  "catatumbo",
    "Dabeiba":    "dabeiba",
    "Casanare":   "casanare",
    "Huila":      "huila",
}
MIN_FRAMES = 25
DISTRESS_AUS = ["AU1", "AU4", "AU15", "AU17"]


def hms(s):
    s = str(s).strip()
    if s in ("", "nan", "NaN", "None") or ":" not in s:
        return np.nan
    try:
        p = [int(x) for x in s.split(":")]
    except ValueError:
        return np.nan
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


def cargar_aus(pref):
    files = glob.glob(str(BASE/"outputs"/"capa3"/f"aus_{pref}*.csv")) + \
            glob.glob(str(BASE/f"aus_{pref}*.csv"))
    dfs = []
    for f in sorted(set(files)):
        d = pd.read_csv(f)
        if {"start","end"} <= set(d.columns) and any(c.startswith("AU") for c in d.columns):
            if "speaker" not in d.columns:
                d["speaker"] = "UNICO"
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates(["speaker","start"])


def main():
    print("CFH — Diagnóstico facial: distress vs sonrisa (paso 3)")
    print("="*84)

    todas = []
    for subcaso, pref in SUBCASOS.items():
        marc_p = BASE / "data" / "marcacion" / f"inventario_{subcaso}.csv"
        if not marc_p.exists():
            continue
        marc = pd.read_csv(marc_p)
        marc = marc[marc["uso"] == "ANALISIS"].copy()
        marc["ini_s"] = marc["inicio"].apply(hms)
        marc["fin_s"] = marc["fin"].apply(hms)
        marc = marc[marc["ini_s"].notna() & marc["fin_s"].notna() &
                    (marc["fin_s"] > marc["ini_s"])]
        aus = cargar_aus(pref)
        if aus.empty:
            continue

        print(f"\n{'─'*84}\nSUBCASO: {subcaso}")
        print(f"  {'identidad':32s} {'frm':>4s} {'distress':>8s} {'sonrisa':>8s} "
              f"{'ratio':>6s} {'AU4':>6s} {'AU12':>6s}")
        filas_sub = []
        for ident, g in marc.groupby("identidad"):
            frames = []
            for _, r in g.iterrows():
                f = aus[(aus["end"] > r["ini_s"]) & (aus["start"] < r["fin_s"])]
                frames.append(f)
            fr = pd.concat(frames) if frames else pd.DataFrame()
            if len(fr) < MIN_FRAMES:
                continue
            d_aus = [fr[c].fillna(0).mean() for c in DISTRESS_AUS if c in fr.columns]
            distress = float(np.mean(d_aus)) if d_aus else 0.0
            sonrisa = float(fr["AU12"].fillna(0).mean()) if "AU12" in fr.columns else 0.0
            ratio = distress/(distress+sonrisa+1e-9)
            au4 = float(fr["AU4"].fillna(0).mean()) if "AU4" in fr.columns else 0.0
            print(f"  {ident[:32]:32s} {len(fr):4d} {distress:8.3f} {sonrisa:8.3f} "
                  f"{ratio:6.3f} {au4:6.3f} {sonrisa:6.3f}")
            filas_sub.append({"subcaso":subcaso, "identidad":ident, "n":len(fr),
                              "distress":distress, "sonrisa":sonrisa, "ratio":ratio})
        todas.extend(filas_sub)

        if filas_sub:
            df = pd.DataFrame(filas_sub)
            print(f"  ── Resumen {subcaso}: distress medio={df['distress'].mean():.3f} | "
                  f"sonrisa media={df['sonrisa'].mean():.3f} | "
                  f"% con sonrisa<0.05: {100*(df['sonrisa']<0.05).mean():.0f}%")

    # Comparación global.
    if todas:
        dfg = pd.DataFrame(todas)
        print(f"\n{'='*84}\nCOMPARACIÓN GLOBAL")
        print(f"  Correlación distress↔ratio: {dfg['distress'].corr(dfg['ratio']):.3f}")
        print(f"  Correlación sonrisa↔ratio:  {dfg['sonrisa'].corr(dfg['ratio']):.3f}")
        print(f"\n  Si |corr(sonrisa,ratio)| >> |corr(distress,ratio)|:")
        print(f"     → el ratio lo MANDA la (falta de) sonrisa, no el distress.")
        print(f"     → conviene normalizar por subcaso (paso 2) o reformular (paso 1).")
        print(f"\n  Rango de distress: {dfg['distress'].min():.3f}–{dfg['distress'].max():.3f}")
        print(f"  Rango de sonrisa:  {dfg['sonrisa'].min():.3f}–{dfg['sonrisa'].max():.3f}")
        # ¿La sonrisa discrimina entre subcasos?
        print(f"\n  Sonrisa media por subcaso (clave para interpretar saturación):")
        for sc, g in dfg.groupby("subcaso"):
            print(f"    {sc:12s} sonrisa={g['sonrisa'].mean():.3f}  distress={g['distress'].mean():.3f}")
        out = BASE/"outputs"/"capa3"/"diag_facial_distress.csv"
        dfg.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  [GUARDADO] {out}")


if __name__ == "__main__":
    main()
