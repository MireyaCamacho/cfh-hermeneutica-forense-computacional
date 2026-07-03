# -*- coding: utf-8 -*-
"""
cfh_recorrer_rep_ab.py
=======================
Re-corre y10_rep para Corpus A y B en el CSV que alimenta DIS/IEI:
    data/features/indicators_completo_conflibert.csv   (873 filas)

VERIFICADO (no supuesto):
  - Llaves del CSV: (doc_id, section_id). corpus_type en el CSV es
    'A-CE' / 'A-CSJ' / 'B'.
  - Texto fuente: data/processed/corpus_a/{hash16}.json + {hash16}.txt
    (secciones is_target=True cortadas por char_range) y
    data/processed/corpus_b_json/ con el mismo esquema.
  - El extractor corregido separa detectores con corpus_type in ("A","B")
    vs C. Por eso aqui se pasa corpus_type NORMALIZADO ("A" o "B") segun
    la carpeta de origen, NO el valor del CSV ('A-CE' caeria en la rama C).
  - El flujo historico escribe .score (normalizado del extractor), no raw.
    Se mantiene identico. Solo cambia que el extractor ya esta corregido
    (patrones + max_length + separacion por corpus).

Actualiza UNICAMENTE la columna y10_rep (y n_rep_instances) por
(doc_id, section_id). Todo lo demas queda intacto. Backup previo.

Uso:
    python cfh_recorrer_rep_ab.py --dry-run    # calcula y reporta, no escribe
    python cfh_recorrer_rep_ab.py              # escribe con backup
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
sys.path.insert(0, str(BASE / "code" / "src"))

CSV_OBJETIVO = BASE / "data" / "features" / "indicators_completo_conflibert.csv"
FUENTES = [
    ("A", BASE / "data" / "processed" / "corpus_a"),
    ("B", BASE / "data" / "processed" / "corpus_b_json"),
]


def iter_secciones(pdir: Path):
    """(doc_id, section_id, texto) por seccion is_target desde JSON+txt (verificado)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from features.y10_rep_extractor import REPExtractor
    ext = REPExtractor()

    if not CSV_OBJETIVO.exists():
        print(f"[ERROR] no existe {CSV_OBJETIVO}")
        sys.exit(1)

    print("=" * 70)
    print("RE-CORRIDO y10_rep  A+B  ->  indicators_completo_conflibert.csv")
    print("=" * 70)

    # 1. re-extraer por seccion, con corpus_type normalizado por carpeta
    resultados = {}   # (doc_id, section_id) -> (score, n_inst)
    for corpus, pdir in FUENTES:
        if not pdir.exists():
            print(f"  [AVISO] no existe {pdir}, se omite {corpus}")
            continue
        n = 0
        for doc_id, sid, texto in iter_secciones(pdir):
            key = (doc_id, sid)
            if key in resultados:
                continue
            r = ext.extract(texto, doc_id=doc_id, section_id=sid, corpus_type=corpus)
            resultados[key] = (float(r.score), int(r.n_instances))
            n += 1
            if n % 100 == 0:
                print(f"  {corpus}: {n} secciones...", flush=True)
        print(f"  {corpus}: {n} secciones procesadas")
    print(f"  llaves unicas totales: {len(resultados)}")

    # 2. actualizar el CSV solo en y10_rep / n_rep_instances
    df = pd.read_csv(CSV_OBJETIVO)
    print(f"\n  CSV objetivo: {len(df)} filas")
    antes = df["y10_rep"].copy()

    nuevos_score, nuevos_ninst, sin_match = [], [], 0
    for _, row in df.iterrows():
        key = (row["doc_id"], row["section_id"])
        if key in resultados:
            s, ni = resultados[key]
            nuevos_score.append(s)
            nuevos_ninst.append(ni)
        else:
            nuevos_score.append(np.nan)
            nuevos_ninst.append(np.nan)
            sin_match += 1

    df["_y10_nuevo"] = nuevos_score
    df["_ninst_nuevo"] = nuevos_ninst

    n_cambian = int((df["_y10_nuevo"].round(4) != antes.round(4)).sum())
    suben = int((df["_y10_nuevo"] > antes).sum())
    bajan = int((df["_y10_nuevo"] < antes).sum())
    print(f"  sin match (quedarian NaN): {sin_match}")
    print(f"  y10_rep cambian: {n_cambian}  (suben {suben}, bajan {bajan})")
    print(f"  media antes:   {antes.mean():.4f}")
    print(f"  media despues: {df['_y10_nuevo'].mean():.4f}")
    for ct in sorted(df["corpus_type"].dropna().unique()):
        sub = df[df["corpus_type"] == ct]
        print(f"    {ct:6s}: antes {sub['y10_rep'].mean():.4f} -> "
              f"despues {sub['_y10_nuevo'].mean():.4f}  (n={len(sub)})")

    if sin_match > 0:
        print(f"\n  [AVISO] {sin_match} filas del CSV sin seccion correspondiente en processed/.")
        faltan = df[df["_y10_nuevo"].isna()][["doc_id", "section_id"]].head(10)
        for _, r in faltan.iterrows():
            print(f"      {str(r['doc_id'])[:16]} / {r['section_id']}")
        print("  Esas filas NO se tocarian (conservan su y10_rep actual).")

    if args.dry_run:
        print("\n[dry-run] NO se escribe nada.")
        return

    # escribir: solo pisar donde hay valor nuevo; conservar el actual donde no
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV_OBJETIVO.with_name(CSV_OBJETIVO.stem + f"_BACKUP_pre_rep_{ts}.csv")
    shutil.copy2(CSV_OBJETIVO, backup)
    print(f"\n  backup -> {backup.name}")

    mask = df["_y10_nuevo"].notna()
    df.loc[mask, "y10_rep"] = df.loc[mask, "_y10_nuevo"]
    if "n_rep_instances" in df.columns:
        df.loc[mask, "n_rep_instances"] = df.loc[mask, "_ninst_nuevo"]
    df = df.drop(columns=["_y10_nuevo", "_ninst_nuevo"])
    df.to_csv(CSV_OBJETIVO, index=False, encoding="utf-8-sig")
    print(f"  ESCRITO: {CSV_OBJETIVO.name} (solo y10_rep/n_rep_instances; resto intacto)")
    print("\nSiguiente: python reproducible\\cfh_unificar_corpus_c.py")
    print("(regenera C y recalcula DIS/IEI con la distribucion conjunta actualizada)")


if __name__ == "__main__":
    main()
