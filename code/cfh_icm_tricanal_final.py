# -*- coding: utf-8 -*-
"""
cfh_icm_tricanal_final.py
================================================================================
CFH — ICM TRI-CANAL por compareciente — VERSIÓN FINAL
Verbal con el REPExtractor REAL (no proxy) + vocal corregido + facial.

RESUELVE:
  · P1 Vocal: normalización a nivel subcaso + asignación única de ventanas.
  · P2 Verbal: usa src/features/y10_rep_extractor.REPExtractor sobre el texto
       REAL de cada compareciente (aislado por segments × diarization × marcación).
       Ya NO es el lexicón proxy.
  · Tramos colectivos y paneles excluidos. Solo vista individual.

CADENA DE ENGANCHE:
    segments (texto+tiempo) × diarization (speaker+tiempo) × marcación (identidad)
    → texto por compareciente → REPExtractor → y10_rep real por persona.

PESOS ICM: 0.40 facial + 0.40 vocal + 0.20 verbal (renormalizados si falta canal).

ENTRADAS:
    data/marcacion/inventario_<Subcaso>.csv
    corpus_c/<subcaso>_*segments.json
    corpus_c/<subcaso>_*diarization.json
    outputs/capa3/egemap_<subcaso>.csv
    outputs/capa3/aus_<subcaso>*.csv
    src/features/y10_rep_extractor.py   (tu extractor real)

SALIDA:
    outputs/capa3/icm_tricanal_final.csv
    outputs/capa3/icm_tricanal_final_excluidos.csv

USO:
    python cfh_icm_tricanal_final.py
    python cfh_icm_tricanal_final.py --subcaso Catatumbo

REQUISITO: spaCy + es_core_news_lg instalado (lo usa el REPExtractor).
    python -m spacy download es_core_news_lg   # si no está

Entorno: Python 3.11, conda env cfh. Dependencias: pandas, numpy, spacy.
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

W_FACIAL, W_VOCAL, W_VERBAL = 0.40, 0.40, 0.20
UMBRAL_AMBIGUO = 0.40
MIN_VENTANAS = 5
MIN_OVERLAP_S = 0.5

RUIDO_RE = re.compile(
    r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\]|\[music\]|"
    r"activa la campanita|dale like|no olvides suscribirte)", re.I)

VOCAL_FEATS = ["shimmerLocaldB_sma3nz_amean",
               "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
               "HNRdBACF_sma3nz_amean"]


# ------------------------------------------------------------------------------
# Cargar el REPExtractor real
# ------------------------------------------------------------------------------

def cargar_rep_extractor(base):
    """Importa REPExtractor desde code/src/features/. Devuelve instancia o None."""
    # El extractor vive en code/src/features/ (confirmado). Probamos varias rutas.
    candidatos = [
        Path(base) / "code" / "src",
        Path(base) / "src",
    ]
    for src in candidatos:
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        from features.y10_rep_extractor import REPExtractor
        ext = REPExtractor()
        print("  REPExtractor real cargado OK (code/src/features)")
        return ext
    except Exception as e:
        print(f"  [aviso] no se pudo cargar REPExtractor real: {type(e).__name__}: {e}")
        print("          -> verbal quedara como NA (revisar spaCy/es_core_news_lg).")
        return None


# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------

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


def buscar_archivo(base, prefijo, sufijo):
    hits = glob.glob(str(base / "corpus_c" / f"{prefijo}*{sufijo}.json"))
    return Path(sorted(hits, key=len)[0]) if hits else None


def cargar_intervalos(ruta_marc):
    m = pd.read_csv(ruta_marc)
    m["ini_s"] = m["inicio"].apply(t_a_seg)
    m["fin_s"] = m["fin"].apply(t_a_seg)
    cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
           (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
    m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"] > m["ini_s"])]
    m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]
    # marcar colectivos
    m = m.reset_index(drop=True)
    m["colectivo"] = False
    for i in range(len(m)):
        a0, a1, ida = m.loc[i,"ini_s"], m.loc[i,"fin_s"], m.loc[i,"identidad"]
        dur = max(1e-9, a1-a0)
        for j in range(len(m)):
            if i == j or m.loc[j,"identidad"] == ida:
                continue
            if overlap(a0,a1,m.loc[j,"ini_s"],m.loc[j,"fin_s"])/dur > 0.5:
                m.loc[i,"colectivo"] = True; break
    return m


def texto_por_compareciente(segments, intervalos):
    """Une el texto de los segments que caen en cada intervalo (no colectivo)."""
    out = {}
    for _, r in intervalos.iterrows():
        if r["colectivo"]:
            continue
        a0, a1 = r["ini_s"], r["fin_s"]
        partes = []
        for s in segments:
            s0, s1 = float(s.get("start", 0)), float(s.get("end", 0))
            if s1 <= s0:
                continue
            if overlap(a0, a1, s0, s1) >= MIN_OVERLAP_S:
                txt = str(s.get("text", s.get("texto", ""))).strip()
                if txt and not RUIDO_RE.search(txt):
                    partes.append(txt)
        ident = r["identidad"]
        out.setdefault(ident, []).append(" ".join(partes))
    # unir por persona
    return {k: " ".join(v).strip() for k, v in out.items()}


def asignar_ventanas_unicas(eg, intervalos):
    """Asignación única ventana→persona (P2+P3+P4 ya filtrado en intervalos)."""
    interv = intervalos[~intervalos["colectivo"]]
    asign = []
    for i in range(len(eg)):
        w0, w1 = eg.iloc[i]["start_s"], eg.iloc[i]["end_s"]
        sol = []
        for _, r in interv.iterrows():
            ov = overlap(w0, w1, r["ini_s"], r["fin_s"])
            if ov > 0:
                sol.append((ov, r["identidad"]))
        if not sol:
            asign.append(None); continue
        sol.sort(reverse=True)
        if len(sol) >= 2 and sol[1][0] > UMBRAL_AMBIGUO * sol[0][0]:
            asign.append(None); continue
        asign.append(sol[0][1])
    eg = eg.copy()
    eg["persona"] = asign
    return eg


def vocal_por_persona(eg_asig, eg_full):
    feats = [f for f in VOCAL_FEATS if f in eg_full.columns]
    mu = {f: eg_full[f].mean() for f in feats}
    sd = {f: eg_full[f].std() + 1e-9 for f in feats}
    res = {}
    for persona, sub in eg_asig[eg_asig["persona"].notna()].groupby("persona"):
        if len(sub) < MIN_VENTANAS:
            continue
        vals = []
        for f in feats:
            z = (sub[f] - mu[f]) / sd[f]
            if "HNR" in f:
                z = -z
            vals.append(np.tanh(z * 0.5))
        sc = np.mean(np.column_stack(vals), axis=1)
        res[persona] = (float(1/(1+np.exp(-np.mean(sc)*2))), len(sub))
    return res


def facial_por_persona(base, prefijo, intervalos, diar):
    """AUs por speaker → identidad. Devuelve {identidad: (score, n)}."""
    files = glob.glob(str(base / "outputs" / "capa3" / f"aus_{prefijo}*.csv")) + \
            glob.glob(str(base / f"aus_{prefijo}*.csv"))
    dfs = []
    for a in sorted(set(files)):
        d = pd.read_csv(a)
        if {"start","end"} <= set(d.columns) and any(c.startswith("AU") for c in d.columns):
            if "speaker" not in d.columns:
                d["speaker"] = "UNICO"
            keep = ["speaker","start","end"]+[c for c in ["AU1","AU4","AU12","AU15","AU17"] if c in d.columns]
            dfs.append(d[keep])
    if not dfs:
        return {}
    aus = pd.concat(dfs, ignore_index=True).drop_duplicates(["speaker","start"])

    res = {}
    for _, r in intervalos[~intervalos["colectivo"]].iterrows():
        a0, a1 = r["ini_s"], r["fin_s"]
        # speaker dominante del intervalo
        acum = {}
        for d in diar:
            ov = overlap(a0, a1, float(d["start"]), float(d["end"]))
            if ov > 0:
                acum[d["speaker"]] = acum.get(d["speaker"], 0.0) + ov
        spk = max(acum, key=acum.get) if acum else None
        sub = aus[(aus["end"] > a0) & (aus["start"] < a1)]
        if spk and spk in set(aus["speaker"]):
            sub = sub[sub["speaker"] == spk]
        if len(sub) < MIN_VENTANAS:
            continue
        distress = np.zeros(len(sub)); n = 0
        for au in ["AU1","AU4","AU15","AU17"]:
            if au in sub.columns:
                distress += sub[au].fillna(0).values; n += 1
        distress /= max(1, n)
        son = sub["AU12"].fillna(0).values if "AU12" in sub.columns else np.zeros(len(sub))
        icm = np.clip(distress/(distress+son+1e-9), 0, 1)
        ident = r["identidad"]
        # acumular si varias intervenciones
        prev = res.get(ident, (0.0, 0))
        res[ident] = ((prev[0]*prev[1] + float(np.mean(icm))*len(sub))/(prev[1]+len(sub)),
                      prev[1]+len(sub))
    return res


# ------------------------------------------------------------------------------
# PROCESAR
# ------------------------------------------------------------------------------

def procesar(subcaso, base, rep_ext):
    base = Path(base)
    pref = PREFIJO.get(subcaso, subcaso.lower())
    print(f"\n{'='*72}\nSUBCASO: {subcaso}\n{'='*72}")

    ruta_marc = base / "data" / "marcacion" / f"inventario_{subcaso}.csv"
    if not ruta_marc.exists():
        print(f"  [SALTADO] sin marcación."); return None, None

    r_seg = buscar_archivo(base, pref, "segments")
    r_diar = buscar_archivo(base, pref, "diarization")
    if r_diar is None:
        print("  [SALTADO] sin diarización."); return None, None

    diar = cargar_json_lista(r_diar)
    segs = cargar_json_lista(r_seg) if r_seg else []
    intervalos = cargar_intervalos(ruta_marc)
    n_indiv = (~intervalos["colectivo"]).sum()
    print(f"  Comparecientes: {intervalos['identidad'].nunique()} | "
          f"intervenciones individuales: {n_indiv} | colectivas: {intervalos['colectivo'].sum()}")

    # --- VERBAL real ---
    verbal = {}
    if rep_ext is not None and segs:
        textos = texto_por_compareciente(segs, intervalos)
        for ident, txt in textos.items():
            if len(txt.strip()) < 20:
                continue
            try:
                r = rep_ext.extract(txt, doc_id=str(ident), section_id="RECONOCIMIENTO",
                                    corpus_type="C")
                verbal[ident] = (float(r.score), len(txt.split()))
            except Exception as e:
                print(f"    [aviso verbal] {ident}: {e}")

    # --- VOCAL corregido ---
    vocal = {}
    p_eg = base / "outputs" / "capa3" / f"egemap_{pref}.csv"
    if p_eg.exists():
        eg = pd.read_csv(p_eg)
        if "start_s" in eg.columns:
            eg_asig = asignar_ventanas_unicas(eg, intervalos)
            vocal = vocal_por_persona(eg_asig, eg)

    # --- FACIAL ---
    facial = facial_por_persona(base, pref, intervalos, diar)

    print(f"  Cobertura: verbal={len(verbal)} | vocal={len(vocal)} | facial={len(facial)} personas")

    # --- Combinar por identidad ---
    identidades = set(verbal) | set(vocal) | set(facial)
    filas, excl = [], []
    for ident in sorted(identidades):
        v = verbal.get(ident, (None, 0))
        vo = vocal.get(ident, (None, 0))
        f = facial.get(ident, (None, 0))
        comp = [(W_FACIAL, f[0]), (W_VOCAL, vo[0]), (W_VERBAL, v[0])]
        wsum = sum(w for w, x in comp if x is not None)
        if wsum == 0:
            excl.append({"subcaso": subcaso, "identidad": ident, "razon": "sin_canales"})
            continue
        icm = sum(w*x for w, x in comp if x is not None) / wsum
        canales = "+".join([c for c, ok in
                            [("F", f[0] is not None), ("V", vo[0] is not None),
                             ("Verb", v[0] is not None)] if ok])
        filas.append({
            "subcaso": subcaso, "identidad": ident,
            "icm_facial": round(f[0], 3) if f[0] is not None else None, "n_facial": f[1],
            "icm_vocal": round(vo[0], 3) if vo[0] is not None else None, "n_vocal": vo[1],
            "y10_rep": round(v[0], 3) if v[0] is not None else None, "n_tokens": v[1],
            "icm_tricanal": round(icm, 3), "canales": canales,
        })

    df = pd.DataFrame(filas)
    if not df.empty:
        print(f"\n  --- ICM tri-canal por compareciente ({subcaso}) ---")
        for _, x in df.sort_values("icm_tricanal").iterrows():
            print(f"    {str(x['identidad'])[:32]:32s} ICM={x['icm_tricanal']:.3f}  "
                  f"[{x['canales']:12s}] "
                  f"f={x['icm_facial']} v={x['icm_vocal']} rep={x['y10_rep']}")
    return df, pd.DataFrame(excl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--subcaso", default=None)
    args = ap.parse_args()
    subcasos = [args.subcaso] if args.subcaso else ["Catatumbo", "Dabeiba", "Casanare", "Huila", "CostaCaribe"]

    print("CFH — ICM tri-canal FINAL (verbal real + vocal corregido + facial)")
    print(f"Pesos: {W_FACIAL} facial / {W_VOCAL} vocal / {W_VERBAL} verbal")
    rep_ext = cargar_rep_extractor(args.base)

    todos, excls = [], []
    for sc in subcasos:
        try:
            df, excl = procesar(sc, args.base, rep_ext)
            if df is not None and not df.empty:
                todos.append(df)
            if excl is not None and not excl.empty:
                excls.append(excl)
        except Exception as e:
            print(f"\n  [ERROR] {sc}: {type(e).__name__}: {e}")

    out = Path(args.base) / "outputs" / "capa3"
    out.mkdir(parents=True, exist_ok=True)
    if todos:
        cons = pd.concat(todos, ignore_index=True)
        cons.to_csv(out / "icm_tricanal_final.csv", index=False, encoding="utf-8-sig")
        print(f"\n{'='*72}\nCONSOLIDADO FINAL\n{'='*72}")
        print(cons.sort_values(["subcaso","icm_tricanal"]).to_string(index=False))
        print(f"\n  [GUARDADO] {out/'icm_tricanal_final.csv'}")
    if excls:
        pd.concat(excls, ignore_index=True).to_csv(
            out / "icm_tricanal_final_excluidos.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
