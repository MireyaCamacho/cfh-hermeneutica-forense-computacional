# -*- coding: utf-8 -*-
"""
CFH — Indicadores lexicos y2, y3, y4 POR COMPARECIENTE (n=47)
=============================================================
Para el SEM multimodal en Corpus C (unidad = compareciente).
Reusa los extractores de code/src/features/ (SA, Civil, NV).
Coherente con y8/y11/y12: trunca a 8000 chars.

Indicadores (bloque xi1 - Violencia Discursiva):
  y2_sa    - Supresion de agentividad (SAExtractor)
  y3_civil - Lexico civil vs militar (CivilLexiconExtractor)
  y4_nv    - Negacion de victimizacion (NVExtractor)

Corre LOCAL (spaCy es_core_news_lg, sin GPU).

Uso:
    conda activate cfh
    python code/cfh_indicadores_lexicos_compareciente.py
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))

CSV_IN = REPO / "data" / "texto_por_compareciente.csv"
CSV_OUT = REPO / "data" / "referencias" / "indicadores_lexicos_compareciente.csv"
MAX_CHARS = 8000  # coherencia con y8/y11/y12
SPACY_MODEL = "es_core_news_lg"


def main():
    print("=" * 64)
    print("CFH — Indicadores lexicos y2, y3, y4 por compareciente (n=47)")
    print("=" * 64)

    df = pd.read_csv(CSV_IN)
    print(f"\n  Comparecientes: {len(df)}")

    from features.y2_sa_extractor import SAExtractor
    from features.y3_civil_extractor import CivilLexiconExtractor
    from features.y4_nv_extractor import NVExtractor

    print("  Cargando extractores (spaCy es_core_news_lg)...")
    sa = SAExtractor(model_name=SPACY_MODEL)
    civ = CivilLexiconExtractor(spacy_model=SPACY_MODEL)
    nv = NVExtractor(model_name=SPACY_MODEL)
    print("  OK extractores cargados")

    # Textos largos: subir max_length de spaCy (aunque truncamos a 8000)
    for ext in [sa, nv]:
        if hasattr(ext, "_nlp"):
            ext._nlp.max_length = 3_000_000
    if hasattr(civ, "_nlp"):
        civ._nlp.max_length = 3_000_000

    filas = []
    for i, row in df.iterrows():
        texto = str(row["texto_completo"])[:MAX_CHARS]
        doc_id = f"{row['subcaso']}_{row['identidad']}"[:60]

        r_sa = sa.extract(texto, doc_id=doc_id, section_id="comp", corpus_type="C")
        r_civ = civ.extract(texto, doc_id=doc_id, section_id="comp", corpus_type="C")
        r_nv = nv.extract(texto, doc_id=doc_id, section_id="comp", corpus_type="C")

        filas.append({
            "subcaso": row["subcaso"],
            "identidad": row["identidad"],
            "n_tokens_aprox": row.get("n_tokens_aprox", None),
            "y2_sa": r_sa.score,
            "y3_civil": r_civ.score,
            "y4_nv": r_nv.score,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(df)}")

    out = pd.DataFrame(filas)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n  Guardado: {CSV_OUT}")
    print("\n  Estadisticas:")
    print(out[["y2_sa", "y3_civil", "y4_nv"]].describe().round(4).to_string())
    print("\n  Por subcaso (media):")
    print(out.groupby("subcaso")[["y2_sa","y3_civil","y4_nv"]].mean().round(4).to_string())
    print("=" * 64)


if __name__ == "__main__":
    main()
