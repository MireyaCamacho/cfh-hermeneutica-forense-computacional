# -*- coding: utf-8 -*-
"""
cfh_recorrer_rep_todo.py
=========================
Re-corre y10_rep para TODO el corpus (A, B y C) con el REPExtractor corregido
(patrones orales ampliados), porque el cambio de metodo afecta a todos, no solo
a los 47 comparecientes.

PRESERVA y7_surprisal, y8_mafapo, y9_cidh y todas las demas columnas: actualiza
UNICAMENTE y10_rep en los CSV del SEM. NO usa run_features.py (que regeneraria
los CSV con y7/y8/y9 en NaN, borrando ese trabajo).

FLUJO:
  1. Backup de indicators_corpus_a.csv, _b.csv (que tienen y7/y8/y9).
  2. CORPUS A y B (escrito): re-extrae y10_rep por seccion desde el texto limpio
     (data/processed/{corpus}/{hash}.txt + char_range del JSON), con el extractor
     corregido. Actualiza solo la columna y10_rep por (doc_id, section_id).
  3. CORPUS C (oral): re-extrae y10_rep por compareciente (enganche segments x
     marcacion ya validado). Guarda un CSV aparte.
  4. Re-fitea el normalizador sobre la distribucion CONJUNTA de score_raw (A+B+C)
     para que los scores normalizados sean comparables entre corpus.
  5. Compara antes/despues por corpus y reporta impacto global.

Uso:
    python cfh_recorrer_rep_todo.py
    python cfh_recorrer_rep_todo.py --dry-run   # no escribe, solo reporta
"""

import argparse
import glob
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
sys.path.insert(0, str(BASE / "code" / "src"))

CSV_A = BASE / "data" / "features" / "indicators_corpus_a.csv"
CSV_B = BASE / "data" / "features" / "indicators_corpus_b.csv"
PROC_A = BASE / "data" / "processed" / "corpus_a"
PROC_B = [BASE / "data" / "processed" / "corpus_b_json"]

# --- Corpus C (oral) ---
PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
USOS_ICM = {"ANALISIS"}
ROLES_ICM = {"COMPARECIENTE"}
NO_PERSONAS = {"BLOQUE_COMPARECIENTES", "BLOQUE_COMPARECIENTES_NO_MR"}
MIN_OVERLAP_S = 0.5
RUIDO_RE = re.compile(r"(suscr[ií]bete|subscribe|gracias por ver|\[m[uú]sica\]|\[music\])", re.I)


# ============================================================ CORPUS A/B ======
def iter_secciones(processed_dirs):
    """(doc_id, section_id, texto) por seccion is_target de cada JSON+txt."""
    for pdir in processed_dirs:
        if not pdir.exists():
            continue
        for jp in sorted(pdir.glob("*.json")):
            if jp.name.startswith("batch_summary"):
                continue
            tp = jp.with_suffix(".txt")
            if not tp.exists():
                alt = list(pdir.glob(jp.stem + "*.txt"))
                if not alt:
                    continue
                tp = alt[0]
            try:
                d = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            doc_id = d.get("sha256_clean") or d.get("metadata", {}).get("doc_id")
            if not doc_id:
                continue
            clean = tp.read_text(encoding="utf-8", errors="replace")
            for sec in d.get("segmentation", {}).get("sections", []):
                if not sec.get("is_target", False):
                    continue
                sid, rng = sec.get("section_id"), sec.get("char_range")
                if not sid or not rng or len(rng) != 2:
                    continue
                seg = clean[rng[0]:rng[1]].strip()
                if seg:
                    yield doc_id, sid, seg


def recorrer_ab(ext):
    """Devuelve dict (doc_id, section_id) -> (raw, n_inst) para A y B."""
    print("\n" + "=" * 70)
    print("CORPUS A / B (texto escrito)")
    print("=" * 70)
    res = {"A": {}, "B": {}}
    for nombre, dirs in [("A", [PROC_A]), ("B", PROC_B)]:
        n = 0
        for doc_id, sid, texto in iter_secciones(dirs):
            r = ext.extract(texto, doc_id=doc_id, section_id=sid, corpus_type=nombre)
            res[nombre][(doc_id, sid)] = (r.score_raw, r.n_instances)
            n += 1
            if n % 100 == 0:
                print(f"  {nombre}: {n} secciones...", flush=True)
        print(f"  {nombre}: {n} secciones procesadas, {len(res[nombre])} llaves unicas")
    return res


# ============================================================ CORPUS C =========
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


def recorrer_c(ext):
    print("\n" + "=" * 70)
    print("CORPUS C (audiencias orales, comparecientes)")
    print("=" * 70)
    filas = []
    for sc, pref in PREFIJO.items():
        hits = glob.glob(str(BASE / "corpus_c" / f"{pref}*segments*.json"))
        ruta_marc = BASE / "data" / "marcacion" / f"inventario_{sc}.csv"
        if not hits or not ruta_marc.exists():
            print(f"  [saltado] {sc}")
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
            r = ext.extract(texto, doc_id=str(ident), section_id="RECONOCIMIENTO", corpus_type="C")
            filas.append({"subcaso": sc, "compareciente": ident,
                          "raw": r.score_raw, "n_inst": r.n_instances})
        print(f"  {sc}: {sum(1 for f in filas if f['subcaso']==sc)} comparecientes")
    return pd.DataFrame(filas)


# ============================================================ MAIN =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from features.y10_rep_extractor import REPExtractor, REPScoreNormalizer
    ext = REPExtractor()

    # 1. re-extraer raw en A, B, C
    ab = recorrer_ab(ext)
    dfc = recorrer_c(ext)

    # 2. re-fitear normalizador sobre distribucion CONJUNTA de raw (A+B+C)
    print("\n" + "=" * 70)
    print("RE-FIT DEL NORMALIZADOR (distribucion conjunta A+B+C)")
    print("=" * 70)
    todos_raw = ([v[0] for v in ab["A"].values()] +
                 [v[0] for v in ab["B"].values()] +
                 list(dfc["raw"].values))
    norm = REPScoreNormalizer(method="percentile")
    norm.fit(todos_raw)
    print(f"  n scores para fit: {len(todos_raw)}")
    print(f"  p_low={norm._p_low:.4f}  p_high={norm._p_high:.4f}  "
          f"mean={norm._mean:.4f}  std={norm._std:.4f}")

    def norm_score(raw):
        return norm.normalize(raw)

    # 3. actualizar SOLO y10_rep en CSV A y B (preservando y7/y8/y9)
    print("\n" + "=" * 70)
    print("ACTUALIZACION DE y10_rep EN CSV (preservando y7/y8/y9)")
    print("=" * 70)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cambios = {}
    for nombre, csv_path in [("A", CSV_A), ("B", CSV_B)]:
        df = pd.read_csv(csv_path)
        antes = df["y10_rep"].copy() if "y10_rep" in df.columns else pd.Series([np.nan]*len(df))
        nuevos = []
        for _, row in df.iterrows():
            key = (row["doc_id"], row["section_id"])
            raw = ab[nombre].get(key, (np.nan, 0))[0]
            nuevos.append(norm_score(raw) if not np.isnan(raw) else np.nan)
        df["_y10_nuevo"] = nuevos
        n_cambian = int((df["_y10_nuevo"].round(4) != antes.round(4)).sum())
        subieron = int((df["_y10_nuevo"] > antes).sum())
        bajaron = int((df["_y10_nuevo"] < antes).sum())
        print(f"\n  Corpus {nombre}: {len(df)} filas")
        print(f"    y10_rep cambian: {n_cambian}  (suben {subieron}, bajan {bajaron})")
        print(f"    media antes: {antes.mean():.4f}  ->  despues: {df['_y10_nuevo'].mean():.4f}")
        cambios[nombre] = (df, antes, csv_path)

    # Corpus C: normalizar y guardar
    dfc["score"] = dfc["raw"].apply(norm_score)
    ceros_c = int((dfc["score"] == 0).sum())
    print(f"\n  Corpus C: {len(dfc)} comparecientes")
    print(f"    con REP=0: {ceros_c}  |  media: {dfc['score'].mean():.4f}  "
          f"max: {dfc['score'].max():.4f}  saturados(>=0.999): {int((dfc['score']>=0.999).sum())}")

    if args.dry_run:
        print("\n[dry-run] NO se escribe nada.")
        # mostrar los que antes eran 0 en C
        print("\n  Comparecientes C con REP>0 ahora (muestra):")
        for _, r in dfc.sort_values("score", ascending=False).head(10).iterrows():
            print(f"    {r['subcaso']:12s} {str(r['compareciente'])[:34]:34s} "
                  f"raw={r['raw']:.4f} score={r['score']:.4f}")
        return

    # escribir A y B (backup + solo y10_rep)
    for nombre, (df, antes, csv_path) in cambios.items():
        backup = csv_path.with_name(csv_path.stem + f"_BACKUP_pre_rep_{ts}.csv")
        shutil.copy2(csv_path, backup)
        df["y10_rep"] = df["_y10_nuevo"]
        df = df.drop(columns=["_y10_nuevo"])
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n  [{nombre}] backup -> {backup.name}")
        print(f"  [{nombre}] ESCRITO: {csv_path.name} (solo y10_rep; y7/y8/y9 intactos)")

    # guardar C
    out_c = BASE / "data" / "rep_corpus_c_corregido.csv"
    dfc.to_csv(out_c, index=False, encoding="utf-8")
    print(f"\n  [C] guardado: {out_c}")

    print("\n" + "=" * 70)
    print("LISTO. y10_rep re-calculado con metodo corregido en A, B y C.")
    print("y7/y8/y9 preservados en los CSV del SEM.")
    print("Siguiente: re-correr cfh_icm_tricanal_final.py para propagar al ICM.")
    print("=" * 70)


if __name__ == "__main__":
    main()
