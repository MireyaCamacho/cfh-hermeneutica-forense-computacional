# -*- coding: utf-8 -*-
r"""
cfh_parsimonia_dis_v2.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

PASO 1 del recalculo DIS/IEI: recalcula y1_ebi (gazetteer) sobre A+B y corre
la PARSIMONIA EXTENDIDA del DIS incluyendo y1.

CAMBIOS respecto a la parsimonia anterior (parsimonia_dis_ab.py):
  - Incluye y1_ebi (rescatado por gazetteer) como tercer componente del DIS.
  - y10_rep SALE del DIS (se reserva para el IEI, mas cercano a la injusticia
    epistemica - decision de Mireya).
  - DIS nuevo = w1*y1_ebi_z + w2*SA_z + w3*NV_z   (mecanismos de violencia
    discursiva de xi1; sin componente reparatorio).

FLUJO:
  1. Recalcula y1_ebi con el gazetteer sobre las 873 secciones de A+B,
     leyendo el texto de los JSON (segmentation.sections + .txt por char_range).
  2. Normaliza (z-score+sigmoide sobre A+B) y1, SA, NV.
  3. Grid search de pesos (w1,w2,w3) suma=1: para cada combinacion mide
     d de Cohen y p Mann-Whitney A vs B.
  4. Reporta teorico vs optimo, % combinaciones significativas, peso que mas
     importa. Guarda outputs/parsimonia_dis_v2.csv.

Uso (raiz del repo, env cfh):
    python code\cfh_parsimonia_dis_v2.py

Entradas:
  data\features\indicators_completo_conflibert.csv   (SA, NV; y1 se recalcula)
  data\processed\corpus_a\*.json + .txt              (texto de A)
  data\processed\corpus_b\*.json + .txt              (texto de B)
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "data", "features", "indicators_completo_conflibert.csv")
DIR_A = os.path.join(REPO, "data", "processed", "corpus_a")
DIR_B = os.path.join(REPO, "data", "processed", "corpus_b")
OUT = os.path.join(REPO, "outputs", "parsimonia_dis_v2.csv")


# ---- Gazetteer EBI (mismos patrones que cfh_y1_ebi_gazetteer.py) ----
EBI_PATRONES = [
    (r"\bbaja[s]?\s+(?:en\s+)?combate", 1.8), (r"\bdad[oa]s?\s+de\s+baja", 1.8),
    (r"\bdieron\s+de\s+baja", 1.8), (r"\bda(?:r|rle|rles|ndo)\s+de\s+baja", 1.8),
    (r"\bpresentad[oa]s?\s+como\s+baja[s]?", 1.8),
    (r"\breportar(?:on|se)?\s+como\s+(?:baja|muert|dad[oa]\s+de\s+baja)", 1.8),
    (r"\bbaja[s]?\s+del\s+enemigo", 1.8), (r"\bpresunto\s+combate", 1.8),
    (r"\bcombate\s+simulad[oa]", 1.8), (r"\bsimular?\s+(?:un\s+)?combate", 1.8),
    (r"\bsimulad[oa]\s+en\s+combate", 1.8),
    (r"\bmuert[eo]s?\s+en\s+(?:presunto\s+)?combate", 1.8),
    (r"\bfalsa\s+presentaci[oó]n\s+de\s+la\s+muerte", 1.8),
    (r"\bmuertes?\s+ileg[ií]timamente\s+presentad", 1.8),
    (r"\bresultad[oa]s?\s+operacional(?:es)?", 1.8), (r"\bmisi[oó]n\s+t[aá]ctica", 1.5),
    (r"\boperaci[oó]n\s+(?:militar|t[aá]ctica|fragmentaria)", 1.5),
    (r"\borden\s+de\s+operaci[oó]n", 1.2), (r"\bregistro\s+y\s+control\s+militar", 1.5),
    (r"\bdieron\s+muerte", 1.8), (r"\bdar(?:le|les)?\s+muerte", 1.8),
    (r"\bcausar(?:le|les)?\s+la\s+muerte", 1.5), (r"\bneutraliz(?:ar|ado|aron|acion)", 1.8),
    (r"\bacordaron\s+darle\s+muerte", 1.8), (r"\bfue\s+interceptad[oa]\s+y\s+retenid", 1.2),
    (r"\bfueron\s+abordad[oa]s", 1.2), (r"\bfue\s+reclutad[oa]", 1.2),
    (r"\bresultaron\s+muert[oa]s", 1.5), (r"\bhabr[ií]an\s+perdido\s+la\s+vida", 1.5),
    (r"\bpresentar\s+(?:este\s+tipo\s+de\s+)?bajas", 1.8),
    (r"\bpresi[oó]n\s+por\s+resultados", 1.5), (r"\bmuertes?\s+en\s+combate\b", 1.5),
]
_EBI = [(re.compile(p, re.IGNORECASE), w) for p, w in EBI_PATRONES]


def ebi_densidad(text):
    if not text or not text.strip():
        return 0.0
    n = max(1, len(text.split()))
    s = 0.0
    for rx, w in _EBI:
        h = len(rx.findall(text))
        if h:
            s += h * w
    return (s / n) * 100.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cargar_textos_secciones(carpeta):
    """doc_id16 + section_id -> texto (de JSON segmentation.sections + .txt)."""
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
            key = (doc16, str(sec.get("section_id", "")))
            tex[key] = texto[int(cr[0]):int(cr[1])]
    return tex


def main():
    print("=" * 66)
    print("PARSIMONIA DIS v2 - con y1_ebi (gazetteer) | y10 reservado al IEI")
    print("=" * 66)

    df = pd.read_csv(CSV)
    df["corpus"] = df["corpus_type"].apply(lambda x: "A" if str(x).startswith("A") else "B")
    print(f"\nSecciones A+B: {len(df)}  (A={sum(df.corpus=='A')}, B={sum(df.corpus=='B')})")

    # ---- 1. Recalcular y1_ebi con gazetteer ----
    print("\n[1] Recalculando y1_ebi con gazetteer sobre A+B...")
    tex_a = cargar_textos_secciones(DIR_A)
    tex_b = cargar_textos_secciones(DIR_B)
    tex = {**tex_a, **tex_b}
    print(f"    Textos de seccion cargados: {len(tex)} (A={len(tex_a)}, B={len(tex_b)})")

    df["doc16"] = df["doc_id"].str[:16]
    y1_new, n_match, n_miss = [], 0, 0
    for _, r in df.iterrows():
        key = (r["doc16"], str(r["section_id"]))
        t = tex.get(key)
        if t is None:
            y1_new.append(0.0)
            n_miss += 1
        else:
            y1_new.append(ebi_densidad(t))
            n_match += 1
    df["y1_ebi"] = y1_new
    print(f"    Emparejadas: {n_match}  |  sin texto: {n_miss}")
    print(f"    y1_ebi nuevo: media={np.mean(y1_new):.4f}  "
          f"con EBI>0: {(np.array(y1_new)>0).sum()}/{len(y1_new)} "
          f"({100*(np.array(y1_new)>0).mean():.1f}%)")

    # ---- 2. Normalizar (z-score+sigmoide sobre A+B) ----
    for col in ["y1_ebi", "y2_sa", "y4_nv"]:
        mu, sd = df[col].mean(), df[col].std() + 1e-9
        df[col + "_z"] = sigmoid((df[col] - mu) / sd)

    a_mask = df.corpus == "A"
    b_mask = df.corpus == "B"

    # ---- 3. Grid search de pesos (w1 y1, w2 SA, w3 NV) ----
    print("\n[2] Grid search de pesos DIS = w1*y1 + w2*SA + w3*NV ...")
    res = []
    for w1 in np.arange(0.05, 0.91, 0.05):
        for w2 in np.arange(0.05, 0.91, 0.05):
            w3 = round(1 - w1 - w2, 3)
            if not (0.05 <= w3 <= 0.90):
                continue
            dis = w1 * df["y1_ebi_z"] + w2 * df["y2_sa_z"] + w3 * df["y4_nv_z"]
            a, b = dis[a_mask].values, dis[b_mask].values
            d = abs(a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-9)
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            res.append({"w_y1": round(w1, 2), "w_SA": round(w2, 2), "w_NV": round(w3, 2),
                        "d_cohen": round(d, 4), "p_valor": round(p, 4),
                        "sig": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."})
    dr = pd.DataFrame(res)
    n = len(dr)

    # ---- 4. Reporte ----
    print(f"\n=== RESULTADOS ({n} combinaciones) ===")
    for u, lab in [(0.20, "pequeno"), (0.30, "medio"), (0.50, "grande")]:
        k = (dr.d_cohen >= u).sum()
        print(f"  d >= {u} ({lab:8}): {k}/{n} ({100*k/n:.0f}%)")
    ksig = (dr.p_valor < 0.05).sum()
    print(f"  p < 0.05: {ksig}/{n} ({100*ksig/n:.0f}%)")

    print("\nTop 8 combinaciones (mayor d de Cohen):")
    print(dr.nlargest(8, "d_cohen").to_string(index=False))

    # peso que mas importa
    print("\nCorrelacion peso -> d de Cohen:")
    for col in ["w_y1", "w_SA", "w_NV"]:
        print(f"  {col}: r={dr[col].corr(dr['d_cohen']):+.3f}")

    # comparar con teorico repartido (y1/SA/NV = 0.33/0.33/0.34 como neutro)
    for wq in [(0.33, 0.33, 0.34), (0.50, 0.25, 0.25), (0.40, 0.30, 0.30)]:
        row = dr[(dr.w_y1 == wq[0]) & (dr.w_SA == wq[1]) & (dr.w_NV == wq[2])]
        if len(row):
            print(f"\nPesos {wq}:")
            print(row.to_string(index=False))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dr.to_csv(OUT, index=False, encoding="utf-8-sig")
    # guardar tambien el y1 recalculado para el Paso 2
    y1out = os.path.join(REPO, "outputs", "y1_ebi_AB_recalculado.csv")
    df[["doc_id", "section_id", "corpus_type", "y1_ebi"]].to_csv(y1out, index=False, encoding="utf-8")
    print(f"\n  Parsimonia -> {OUT}")
    print(f"  y1_ebi A+B recalculado -> {y1out}")
    print("\n  Con estos pesos decidimos la formula final del DIS y pasamos al Paso 2.")


if __name__ == "__main__":
    main()
