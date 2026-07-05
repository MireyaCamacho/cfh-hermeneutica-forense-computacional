# -*- coding: utf-8 -*-
r"""
cfh_recalcular_y10_AB.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Recalcula y10 REP (v5, extractor del repo) sobre las 873 secciones de A+B,
para que el DIS/IEI tri-corpus use el y10 consistente con el tuneo de hoy
(y no el y10 viejo del archivo indicators_completo_conflibert.csv).

Usa DIRECTAMENTE el extractor del repo:
    code/src/features/y10_rep_extractor.py  -> REPExtractor.extract(
        text, doc_id, section_id, corpus_type)
El corpus_type se pasa por seccion para que el FILTRO PROCESAL de Corpus A
(excluir nombres en formulas RESUELVE/CONSIDERANDO) se active correctamente.

Texto: se lee de los JSON de A y B (segmentation.sections + .txt por char_range),
igual que en el recalculo de y1.

Salida: outputs/y10_rep_v5_AB_recalculado.csv
        (doc_id, section_id, corpus_type, y10_rep_v5)

Uso (raiz del repo, env cfh):
    python code\cfh_recalcular_y10_AB.py
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "src", "features"))

CSV = os.path.join(REPO, "data", "features", "indicators_completo_conflibert.csv")
DIR_A = os.path.join(REPO, "data", "processed", "corpus_a")
DIR_B = os.path.join(REPO, "data", "processed", "corpus_b")
OUT = os.path.join(REPO, "outputs", "y10_rep_v5_AB_recalculado.csv")


def cargar_textos(carpeta):
    """(doc16, section_id) -> texto, desde JSON segmentation.sections + .txt."""
    tex = {}
    for jp in glob.glob(os.path.join(carpeta, "*.json")):
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        seg = d.get("segmentation")
        if not isinstance(seg, dict) or "sections" not in seg:
            continue
        doc16 = os.path.splitext(os.path.basename(jp))[0]
        txt_path = os.path.join(carpeta, doc16 + ".txt")
        if not os.path.exists(txt_path):
            continue
        texto = open(txt_path, encoding="utf-8").read()
        for sec in seg["sections"]:
            cr = sec.get("char_range")
            if isinstance(cr, str):
                cr = json.loads(cr)
            if not cr or len(cr) != 2:
                continue
            tex[(doc16, str(sec.get("section_id", "")))] = texto[int(cr[0]):int(cr[1])]
    return tex


def sacar_score(result):
    """Extrae el score del REPExtractionResult probando atributos tipicos."""
    for attr in ["score", "rep_score", "normalized_score", "score_norm",
                 "value", "y10", "final_score"]:
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, (int, float)):
                return float(v)
    # si es dataclass, buscar el primer float
    if hasattr(result, "__dict__"):
        for k, v in vars(result).items():
            if isinstance(v, (int, float)) and "score" in k.lower():
                return float(v)
    raise AttributeError(f"No encuentro score en {type(result)}: "
                         f"{[a for a in dir(result) if not a.startswith('_')]}")


def main():
    print("=" * 66)
    print("Recalculo y10 REP v5 sobre A+B (extractor del repo)")
    print("=" * 66)

    import y10_rep_extractor as y10mod
    print("\nInstanciando REPExtractor (es_core_news_lg)...")
    extractor = y10mod.REPExtractor()

    df = pd.read_csv(CSV)
    df["corpus"] = df["corpus_type"].apply(lambda x: "A" if str(x).startswith("A") else "B")
    df["doc16"] = df["doc_id"].str[:16]
    print(f"Secciones A+B: {len(df)}  (A={sum(df.corpus=='A')}, B={sum(df.corpus=='B')})")

    print("\nCargando textos de los JSON...")
    tex = {**cargar_textos(DIR_A), **cargar_textos(DIR_B)}
    print(f"  Textos cargados: {len(tex)}")

    print("\nExtrayendo y10 v5 por seccion (con corpus_type para filtro A)...")
    scores, n_ok, n_miss = [], 0, 0
    diag = None
    for i, r in df.iterrows():
        key = (r["doc16"], str(r["section_id"]))
        t = tex.get(key)
        if t is None or not t.strip():
            scores.append(0.0)
            n_miss += 1
            continue
        try:
            res = extractor.extract(
                text=t, doc_id=str(r["doc_id"]),
                section_id=str(r["section_id"]),
                corpus_type=str(r["corpus_type"]),
            )
            if diag is None:
                diag = [a for a in dir(res) if not a.startswith("_")]
            scores.append(sacar_score(res))
            n_ok += 1
        except Exception as e:
            if n_ok == 0 and n_miss == 0:
                print(f"  [ERROR primer extract] {e}")
                if diag:
                    print(f"  atributos del resultado: {diag}")
                raise
            scores.append(0.0)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(df)}...")

    df["y10_rep_v5"] = scores
    print(f"\n  Procesadas OK: {n_ok}  |  sin texto: {n_miss}")
    print(f"  y10_rep_v5: media={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
          f"max={np.max(scores):.4f}")
    print(f"  con REP>0: {(np.array(scores)>0).sum()}/{len(scores)} "
          f"({100*(np.array(scores)>0).mean():.1f}%)")

    print("\n  Por corpus (media y10_rep_v5):")
    print(df.groupby("corpus_type")["y10_rep_v5"].agg(["mean", "std", "max"]).round(4).to_string())

    print("\n  Comparacion con y10 viejo (del archivo base):")
    print(f"    y10 viejo A+B: media={df['y10_rep'].mean():.4f}")
    print(f"    y10 v5    A+B: media={df['y10_rep_v5'].mean():.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[["doc_id", "section_id", "corpus_type", "y10_rep_v5"]].to_csv(
        OUT, index=False, encoding="utf-8")
    print(f"\n  Guardado -> {OUT}")
    print("\n  Con y1 (ya hecho) + y10 v5 (este) + los de C, armamos el Paso 2 (DIS/IEI).")


if __name__ == "__main__":
    main()
