# -*- coding: utf-8 -*-
"""
CFH — Centroide RESTAURATIVO v1 (para indicador y11)
=====================================================
Construye el centroide del POLO RESTAURATIVO a partir del corpus CURADO
manualmente por Mireya (corpus_restaurativo_curado_v1.json, 74 apartes).

y11 = convergencia semantica de cada compareciente a este centroide
      (1 - distancia coseno), indicador de eta2 (Transicion Epistemica).

CORPUS (externo a los 47 comparecientes del Macrocaso 03 -> sin circularidad):
  - Reconocimientos de FARC (JEP Caso 01), AUC (Justicia y Paz),
    EPL y agentes del Estado (Comision de la Verdad), + estandar Ley 975.
  - Multi-actor: demuestra que el lenguaje restaurativo del perdon es el
    mismo con independencia del perpetrador.
  - Sin voz de victimas (no se confunde con el centroide MAFAPO / y8).

PONDERACION (definida en la curaduria):
  nivel 1 (performativo, 1a persona)        -> w=1.8
  nivel 2 (institucional del reconocimiento) -> w=1.8
  nivel 3 (estandar normativo, Ley 975)      -> w=1.0

METODO (identico a MAFAPO v5, para comparabilidad):
  modelo eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1, embedding token CLS,
  deduplicacion por primeros 100 chars, promedio ponderado, np.save.

Ejecutar en COLAB (ConfliBERT necesita GPU):
  python cfh_centroide_restaurativo.py
"""
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent
REF_DIR = REPO / "data" / "referencias"
CORPUS = REF_DIR / "corpus_restaurativo_curado_v1.json"
OUT = REF_DIR / "centroide_restaurativo_v1.npy"
OUT_INV = REF_DIR / "inventario_centroide_restaurativo_v1.json"


def cargar_corpus():
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"No existe {CORPUS}. Mueve corpus_restaurativo_curado_v1.json "
            f"a data/referencias/ antes de correr.")
    with open(CORPUS, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segmentos", [])


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("=" * 60)
    print("CFH — Centroide RESTAURATIVO v1 (y11)")
    print("=" * 60)

    segmentos = cargar_corpus()
    print(f"\n  Corpus curado: {len(segmentos)} apartes")

    # ── Reunir textos con deduplicacion (mismo metodo que MAFAPO v5) ──
    textos_pesos = []
    vistos = set()
    for s in segmentos:
        texto = s["texto"].strip()
        peso = s.get("peso", 1.0)
        clave = texto[:100]
        if clave and clave not in vistos and len(texto) >= 10:
            vistos.add(clave)
            textos_pesos.append((texto, peso))

    n_total = len(textos_pesos)
    n_w18 = sum(1 for _, w in textos_pesos if w == 1.8)
    print(f"  Textos unicos (tras dedup): {n_total}")
    print(f"    peso 1.8 (reconocimiento):  {n_w18}")
    print(f"    peso 1.0 (normativo):       {n_total - n_w18}")

    # ── Cargar modelo (identico a MAFAPO v5) ─────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
    print(f"\n  Cargando ConfliBERT-Spanish en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  OK modelo cargado")

    def get_emb(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512,
                           truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # ── Embeddings ───────────────────────────────────────────────────
    print(f"\n  Calculando embeddings ({n_total} textos)...")
    embs, pesos = [], []
    for i, (texto, w) in enumerate(textos_pesos):
        emb = get_emb(texto)
        if np.linalg.norm(emb) > 0:
            embs.append(emb)
            pesos.append(w)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_total}")
    embs = np.array(embs)
    pesos = np.array(pesos)
    centroide = np.average(embs, axis=0, weights=pesos)
    n_eff = len(embs)
    margen = 100 / (n_eff ** 0.5)
    norma = float(np.linalg.norm(centroide))

    print(f"\n  OK Centroide RESTAURATIVO v1 calculado")
    print(f"    textos efectivos: {n_eff}")
    print(f"    norma: {norma:.4f}")
    print(f"    margen estimado: +-{margen:.1f}%")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT, centroide)
    inv = {
        "version": "restaurativo_v1",
        "n_textos_efectivos": n_eff,
        "n_peso_1_8": int(n_w18),
        "margen_estimado_pct": round(margen, 2),
        "norma": norma,
        "modelo": model_name,
        "metodo": "CLS token, promedio ponderado, dedup 100 chars",
        "corpus": "corpus_restaurativo_curado_v1.json (curado por Mireya, 74 apartes)",
        "nota": ("Corpus EXTERNO a comparecientes M03 (evita circularidad). "
                 "Sin voz de victimas. Multi-actor: FARC, AUC, EPL, Estado, normativo. "
                 "El lenguaje restaurativo del perdon es el mismo con independencia del actor."),
    }
    with open(OUT_INV, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {OUT}")
    print(f"  Inventario: {OUT_INV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
