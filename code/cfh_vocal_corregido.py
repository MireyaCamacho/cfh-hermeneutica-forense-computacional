# -*- coding: utf-8 -*-
"""
cfh_vocal_corregido.py
================================================================================
CFH — Canal vocal corregido (asignación única ventana→persona)

CORRIGE 4 PROBLEMAS detectados en el diagnóstico (2026-06-26):
  P1. Normalización por persona aplanaba todo a ~0.49.
      → Ahora se normaliza UNA sola vez a nivel subcaso (z-score sobre todas las
        ventanas del subcaso), antes de promediar por persona.
  P2. Personas distintas recibían las MISMAS ventanas (tramos solapados).
      → ASIGNACIÓN ÚNICA: cada ventana se asigna a una sola persona (mayor
        solapamiento). Nunca se duplica.
  P3. Ventanas en zona ambigua (2+ personas con solapamiento parejo).
      → Se EXCLUYEN: si el 2º candidato tiene >40% del solapamiento del 1º.
  P4. Etiquetas de panel (BLOQUE_COMPARECIENTES*) tratadas como personas.
      → Se EXCLUYEN antes de asignar.

RESULTADO:
    Un score vocal por compareciente, calculado sobre ventanas que le pertenecen
    de forma inequívoca. Resuelve el problema de unidad de análisis a nivel de
    ventana prosódica.

SALIDA:
    outputs/capa3/vocal_corregido_<subcaso>.csv  (score por persona + cobertura)

USO:
    python cfh_vocal_corregido.py
    python cfh_vocal_corregido.py --subcaso Dabeiba

Entorno: Python 3.11, conda env cfh. Dependencias: pandas, numpy.
================================================================================
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DEFAULT = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"
PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
USOS_ICM = {"ANALISIS"}
ROLES_ICM = {"COMPARECIENTE"}

# Etiquetas que NO son personas (paneles colectivos).
NO_PERSONAS = {"BLOQUE_COMPARECIENTES", "BLOQUE_COMPARECIENTES_NO_MR"}

# Umbral de ambigüedad: si 2º candidato tiene > este % del solapamiento del 1º,
# la ventana se considera ambigua y se excluye.
UMBRAL_AMBIGUO = 0.40

MIN_VENTANAS = 5   # mínimo de ventanas asignadas para reportar a una persona

FEATS = ["shimmerLocaldB_sma3nz_amean",
         "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
         "HNRdBACF_sma3nz_amean"]


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


def cargar_intervalos(ruta_marc):
    """Intervalos de comparecientes ICM, excluyendo paneles."""
    m = pd.read_csv(ruta_marc)
    m["ini_s"] = m["inicio"].apply(t_a_seg)
    m["fin_s"] = m["fin"].apply(t_a_seg)
    cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
           (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
    m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"] > m["ini_s"])]
    # Excluir paneles (P4).
    m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]
    return m[["identidad", "ini_s", "fin_s"]].reset_index(drop=True)


def asignar_ventanas_unicas(eg, intervalos):
    """
    ASIGNACIÓN ÚNICA (P2+P3): a cada ventana de egemap le asigna a lo sumo UNA
    persona. Para cada ventana, calcula solapamiento con cada intervalo de
    persona; gana el mayor; si el 2º es >40% del 1º, se marca ambigua y se
    descarta. Devuelve eg con columnas 'persona_asignada' y 'estado'.
    """
    centros = (eg["start_s"] + eg["end_s"]) / 2.0
    asign, estado = [], []
    for i in range(len(eg)):
        w0, w1 = eg.iloc[i]["start_s"], eg.iloc[i]["end_s"]
        sol = []
        for _, r in intervalos.iterrows():
            ov = overlap(w0, w1, r["ini_s"], r["fin_s"])
            if ov > 0:
                sol.append((ov, r["identidad"]))
        if not sol:
            asign.append(None); estado.append("sin_persona"); continue
        sol.sort(reverse=True)
        if len(sol) >= 2 and sol[1][0] > UMBRAL_AMBIGUO * sol[0][0]:
            asign.append(None); estado.append("ambigua_excluida"); continue
        asign.append(sol[0][1]); estado.append("ok")
    eg = eg.copy()
    eg["persona_asignada"] = asign
    eg["estado_asignacion"] = estado
    return eg


def vocal_score_subcaso(eg_asignado, eg_full):
    """
    Calcula score vocal por persona. NORMALIZACIÓN ÚNICA (P1): z-score sobre
    TODO el subcaso (eg_full), no por persona.
    """
    feats_ok = [f for f in FEATS if f in eg_full.columns]
    # Estadística global del subcaso para z-score.
    mu = {f: eg_full[f].mean() for f in feats_ok}
    sd = {f: eg_full[f].std() + 1e-9 for f in feats_ok}

    def score_row_mean(sub):
        vals = []
        for f in feats_ok:
            z = (sub[f] - mu[f]) / sd[f]
            if "HNR" in f:
                z = -z   # menor HNR = más irregular
            vals.append(np.tanh(z * 0.5))
        sc = np.mean(np.column_stack(vals), axis=1)
        return float(1/(1+np.exp(-np.mean(sc)*2)))

    filas = []
    ok = eg_asignado[eg_asignado["estado_asignacion"] == "ok"]
    for persona, sub in ok.groupby("persona_asignada"):
        if len(sub) < MIN_VENTANAS:
            filas.append({"identidad": persona, "n_ventanas": len(sub),
                          "icm_vocal": None, "estado": "pocas_ventanas"})
            continue
        filas.append({"identidad": persona, "n_ventanas": len(sub),
                      "icm_vocal": round(score_row_mean(sub), 3),
                      "shimmer_medio": round(sub[FEATS[0]].mean(), 3) if FEATS[0] in sub else None,
                      "estado": "ok"})
    return pd.DataFrame(filas)


def procesar(subcaso, base):
    base = Path(base)
    pref = PREFIJO.get(subcaso, subcaso.lower())
    ruta_marc = base / "data" / "marcacion" / f"inventario_{subcaso}.csv"
    p_eg = base / "outputs" / "capa3" / f"egemap_{pref}.csv"

    print(f"\n{'='*72}\nVOCAL CORREGIDO — {subcaso}\n{'='*72}")
    if not ruta_marc.exists() or not p_eg.exists():
        print("  [SALTADO] falta marcación o egemap.")
        return None

    eg = pd.read_csv(p_eg)
    intervalos = cargar_intervalos(ruta_marc)
    print(f"  Egemap: {len(eg)} ventanas | comparecientes (sin paneles): "
          f"{intervalos['identidad'].nunique()}")

    eg_asig = asignar_ventanas_unicas(eg, intervalos)
    n_ok = (eg_asig["estado_asignacion"] == "ok").sum()
    n_amb = (eg_asig["estado_asignacion"] == "ambigua_excluida").sum()
    n_sin = (eg_asig["estado_asignacion"] == "sin_persona").sum()
    print(f"  Ventanas asignadas únicas: {n_ok} | ambiguas excluidas: {n_amb} | "
          f"sin persona: {n_sin}")

    res = vocal_score_subcaso(eg_asig, eg)
    res = res.sort_values("icm_vocal", na_position="last")
    print(f"\n  --- Score vocal por compareciente (normalización a nivel subcaso) ---")
    for _, r in res.iterrows():
        v = f"{r['icm_vocal']:.3f}" if pd.notna(r['icm_vocal']) else "  NA "
        print(f"    {str(r['identidad'])[:34]:34s} vocal={v}  "
              f"(n={int(r['n_ventanas'])}, {r['estado']})")

    # Veredicto de discriminación.
    vv = res[res["icm_vocal"].notna()]["icm_vocal"]
    if len(vv) >= 2:
        rng = vv.max() - vv.min()
        print(f"\n  Rango del vocal entre personas: {rng:.3f} "
              f"({'DISCRIMINA ✓' if rng > 0.05 else 'aún plano ✗'})")

    res["subcaso"] = subcaso
    out = base / "outputs" / "capa3" / f"vocal_corregido_{pref}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [GUARDADO] {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--subcaso", default=None)
    args = ap.parse_args()
    subcasos = [args.subcaso] if args.subcaso else ["Catatumbo", "Dabeiba"]

    print("CFH — Canal vocal corregido (asignación única + normalización subcaso)")
    print("Correcciones: P1 normalización | P2 sin duplicar | P3 ambiguas fuera | P4 paneles fuera")

    todos = []
    for sc in subcasos:
        r = procesar(sc, args.base)
        if r is not None and not r.empty:
            todos.append(r)
    if todos:
        cons = pd.concat(todos, ignore_index=True)
        out = Path(args.base) / "outputs" / "capa3" / "vocal_corregido_resumen.csv"
        cons.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n{'='*72}\n[GUARDADO] {out}")
        print("Si el vocal ya discrimina, lo integramos al tri-canal y seguimos")
        print("con el Problema 2 (extractor real de y10_rep para el verbal).")


if __name__ == "__main__":
    main()
