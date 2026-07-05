# -*- coding: utf-8 -*-
"""
CFH — Analisis de errores de y4 (NV — Negacion de Victimizacion)
================================================================
Exporta los casos donde el extractor y4 falla contra el gold, para afinar
su diccionario/patrones:
  - FALSOS NEGATIVOS: gold marca NV, extractor da score bajo (no detecta)
  - FALSOS POSITIVOS: extractor marca NV, gold no lo tiene

Muestra el texto y los spans NV que los anotadores marcaron, para ver
que expresiones se le escapan al extractor.

Uso:
    conda activate cfh
    python code/cfh_errores_y4.py
"""
import sys, json
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

GOLD = REPO / "data" / "referencias" / "gold_consolidado_A1A2.json"
A1_JSON = REPO / "data" / "referencias" / "annotations_mireya_v1.json"
OUT = REPO / "data" / "referencias" / "errores_y4_nv.csv"
SPACY_MODEL = "es_core_news_lg"
UMBRAL = 0.4  # optimo hallado en la calibracion


def main():
    print("=" * 64)
    print("CFH — Errores de y4 (NV) contra gold")
    print("=" * 64)

    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    # Spans NV de A1 (para mostrar que marco el humano)
    with open(A1_JSON, encoding="utf-8") as f:
        a1 = json.load(f)
    a1_nv_spans = {}
    for d in a1:
        spans = [s["text"] for s in d.get("label", []) if "NV" in s.get("labels", [])]
        a1_nv_spans[d["id"]] = spans

    from features.y4_nv_extractor import NVExtractor
    print("  Cargando extractor NV...")
    nv = NVExtractor(model_name=SPACY_MODEL)
    if hasattr(nv, "_nlp"):
        nv._nlp.max_length = 3_000_000

    print("  Corriendo y4 sobre los 100 fragmentos...")
    filas = []
    for g in gold:
        ct = g["corpus"][0] if g.get("corpus") else "A"
        r = nv.extract(g["text"], doc_id=str(g["id"]), section_id="frag", corpus_type=ct)
        score = r.score
        pred = score >= UMBRAL
        gold_nv = g["NV_union"]  # usar union (NV que cualquiera marco)
        tipo = None
        if gold_nv and not pred:
            tipo = "FALSO_NEGATIVO"  # habia NV, no lo detecto
        elif not gold_nv and pred:
            tipo = "FALSO_POSITIVO"  # marco NV donde no habia
        if tipo:
            filas.append({
                "id": g["id"],
                "tipo": tipo,
                "score_y4": round(score, 3),
                "NV_A1": " | ".join(a1_nv_spans.get(g["id"], [])) or "(nada)",
                "NV_A2_marco": g["NV_A2"],
                "texto": g["text"][:220],
            })

    df = pd.DataFrame(filas)
    df = df.sort_values("tipo")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    fn = (df["tipo"] == "FALSO_NEGATIVO").sum()
    fp = (df["tipo"] == "FALSO_POSITIVO").sum()
    print(f"\n  Falsos negativos (NV no detectado): {fn}")
    print(f"  Falsos positivos (NV inventado):    {fp}")
    print(f"  Guardado: {OUT}")

    print("\n  --- FALSOS NEGATIVOS (lo que se le escapa a y4) ---")
    for _, r in df[df["tipo"] == "FALSO_NEGATIVO"].iterrows():
        print(f"   #{r['id']} score={r['score_y4']} | NV marcado: {r['NV_A1']}")

    print("\n  --- FALSOS POSITIVOS (lo que y4 marca de mas) ---")
    for _, r in df[df["tipo"] == "FALSO_POSITIVO"].iterrows():
        print(f"   #{r['id']} score={r['score_y4']} | texto: {r['texto'][:90]}")
    print("=" * 64)


if __name__ == "__main__":
    main()
