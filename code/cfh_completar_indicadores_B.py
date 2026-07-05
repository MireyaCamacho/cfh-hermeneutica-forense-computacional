# -*- coding: utf-8 -*-
r"""
cfh_completar_indicadores_B.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Completa los indicadores faltantes del Corpus B fortalecido (80 secciones):
  - y2_sa   (SAExtractor,  lexico local)
  - y4_nv   (NVExtractor,  lexico local)
  - y8_mafapo (ConfliBERT CLS + distancia coseno al centroide MAFAPO v5)
  - y9_cidh   (ConfliBERT CLS + distancia coseno al centroide CIDH v3)

Ya tenia y1_ebi e y10_rep_v5 (de cfh_fortalecer_corpus_b.py). Con esto B queda
con los 6 indicadores que necesitan DIS e IEI.

METODO y8/y9 (identico al centroide v5, para comparabilidad):
  - Embedding CLS: last_hidden_state[:, 0, :] de ConfliBERT-Spanish-Beto-Cased-v1
  - y8/y9 = distancia coseno del embedding de la seccion al centroide
    (cs = 1 - cos_sim; y8_mafapo_cs, y9_cidh_cs) -> asi se llaman en el pipeline
  - Se trunca a 512 tokens (limite de BERT); para secciones largas se usa el
    CLS del primer chunk (consistente con como se calculo en A/B antes).

Entrada:
  outputs/corpus_b_secciones_texto.csv   (doc, seccion, chars, texto)
  outputs/corpus_b_indicadores_v2.csv    (y1_ebi, y10_rep_v5)
  data/referencias/centroide_mafapo_v5.npy
  data/referencias/centroide_cidh_v3.npy

Salida:
  outputs/corpus_b_indicadores_COMPLETO.csv
    (doc, seccion, corpus_type, chars, y1_ebi, y2_sa, y4_nv,
     y8_mafapo_cs, y9_cidh_cs, y10_rep_v5)

Uso (raiz del repo, env cfh):
    python code\cfh_completar_indicadores_B.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "src", "features"))

TEXTO = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
IND_V2 = os.path.join(REPO, "outputs", "corpus_b_indicadores_v2.csv")
CENT_MAFAPO = os.path.join(REPO, "data", "referencias", "centroide_mafapo_v5.npy")
CENT_CIDH = os.path.join(REPO, "data", "referencias", "centroide_cidh_v3.npy")
OUT = os.path.join(REPO, "outputs", "corpus_b_indicadores_COMPLETO.csv")

MODEL = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"


def sacar_score(result):
    for attr in ["score", "sa_score", "nv_score", "normalized_score",
                 "value", "final_score"]:
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, (int, float)):
                return float(v)
    if hasattr(result, "__dict__"):
        for k, v in vars(result).items():
            if isinstance(v, (int, float)) and "score" in k.lower():
                return float(v)
    return 0.0


def coseno_dist(emb, centroide):
    a = emb / (np.linalg.norm(emb) + 1e-9)
    b = centroide / (np.linalg.norm(centroide) + 1e-9)
    cos = float(np.dot(a, b))
    return 1.0 - cos  # distancia coseno (cs)


def main():
    print("=" * 66)
    print("Completar indicadores de Corpus B (80 secciones)")
    print("=" * 66)

    df_txt = pd.read_csv(TEXTO)
    df_v2 = pd.read_csv(IND_V2)
    print(f"\nSecciones: {len(df_txt)}")

    # --- 1. Lexicos: y2_sa, y4_nv ---
    print("\n[1] Extractores lexicos y2_sa, y4_nv (spaCy)...")
    import y2_sa_extractor as y2mod
    import y4_nv_extractor as y4mod
    sa_ext = y2mod.SAExtractor()
    nv_ext = y4mod.NVExtractor()

    y2_vals, y4_vals = [], []
    for i, r in df_txt.iterrows():
        t = str(r["texto"] or "")
        try:
            y2_vals.append(sacar_score(sa_ext.extract(t, "B", str(r["seccion"]), "B")))
        except Exception:
            y2_vals.append(0.0)
        try:
            y4_vals.append(sacar_score(nv_ext.extract(t, "B", str(r["seccion"]), "B")))
        except Exception:
            y4_vals.append(0.0)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(df_txt)}...")
    print(f"    y2_sa: media={np.mean(y2_vals):.3f}  y4_nv: media={np.mean(y4_vals):.3f}")

    # --- 2. ConfliBERT: y8, y9 ---
    print("\n[2] ConfliBERT CLS + distancia coseno a centroides...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    cent_mafapo = np.load(CENT_MAFAPO)
    cent_cidh = np.load(CENT_CIDH)
    print(f"    centroides: MAFAPO {cent_mafapo.shape}, CIDH {cent_cidh.shape}")

    y8_vals, y9_vals = [], []
    with torch.no_grad():
        for i, r in df_txt.iterrows():
            t = str(r["texto"] or "")
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS
            y8_vals.append(coseno_dist(cls, cent_mafapo))
            y9_vals.append(coseno_dist(cls, cent_cidh))
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(df_txt)}...")
    print(f"    y8_mafapo_cs: media={np.mean(y8_vals):.3f}  "
          f"y9_cidh_cs: media={np.mean(y9_vals):.3f}")

    # --- 3. Ensamblar ---
    out = df_txt[["doc", "seccion", "chars"]].copy()
    out["corpus_type"] = "B"
    out["y1_ebi"] = df_v2["y1_ebi"].values
    out["y2_sa"] = y2_vals
    out["y4_nv"] = y4_vals
    out["y8_mafapo_cs"] = y8_vals
    out["y9_cidh_cs"] = y9_vals
    out["y10_rep_v5"] = df_v2["y10_rep_v5"].values

    out.to_csv(OUT, index=False, encoding="utf-8")
    print("\n" + "-" * 66)
    print("RESUMEN B completo:")
    for c in ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep_v5"]:
        s = out[c]
        print(f"  {c:<14}: media={s.mean():.4f}  std={s.std():.4f}  "
              f"ceros={(s==0).sum()}/{len(s)}")
    print(f"\n  Guardado -> {OUT}")
    print("  B completo con los 6 indicadores. Listo para el Paso 2 (DIS/IEI).")


if __name__ == "__main__":
    main()
