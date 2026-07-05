# -*- coding: utf-8 -*-
r"""
cfh_fortalecer_corpus_b.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Fortalece el Corpus B: toma los 14 documentos .txt unicos, los limpia,
los segmenta por secciones tematicas (reusando la logica del ingestor) y
recalcula y1_ebi (gazetteer) + y10_rep (v5, extractor del repo) sobre cada
seccion. Deja B estructurado y comparable con A para el DIS/IEI tri-corpus.

Resuelve el problema detectado: el texto segmentado de las 54 secciones
originales de B no estaba persistido -> B salia en ceros. Aqui se regenera
desde los .txt reales, de forma reproducible.

MEJORAS sobre el ingestor original:
  - Salta la TABLA DE CONTENIDO (encabezados seguidos de "....." o numeros
    de pagina no cuentan como inicio de seccion).
  - Excluye documentos duplicados por hash (Auto_125/128, sentencias repetidas).
  - Filtra secciones muy cortas (< 400 chars = probable fragmento de indice).

Salida:
  outputs/corpus_b_secciones_texto.csv   (doc, seccion, chars, texto)
  outputs/corpus_b_indicadores_v2.csv    (doc, seccion, y1_ebi, y10_rep_v5)

Uso (raiz del repo, env cfh):
    python code\cfh_fortalecer_corpus_b.py
"""

import os
import re
import sys
import glob
import hashlib
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "src", "features"))
DIR_B = os.path.join(REPO, "data", "processed", "corpus_b")
OUT_TEXTO = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
OUT_IND = os.path.join(REPO, "outputs", "corpus_b_indicadores_v2.csv")

MIN_CHARS_SECCION = 400  # secciones mas cortas se descartan (ruido/indice)


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
    s = sum(len(rx.findall(text)) * w for rx, w in _EBI)
    return (s / n) * 100.0


# ---- Limpieza (misma logica que ingest_corpus_b.limpiar_texto) ----
def limpiar_texto(texto):
    texto = re.sub(r"\r\n|\r", "\n", texto)
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ---- Patrones de seccion (del ingestor) ----
PATRONES_SECCIONES = {
    "HECHOS_Y_CONDUCTAS": [r"HECHOS Y CONDUCTAS", r"I\.\s+HECHOS",
                           r"DETERMINACIÓN DE HECHOS", r"DESCRIPCIÓN DE LOS HECHOS"],
    "PATRONES_MACROCRIMINALES": [r"PATRONES MACROCRIMINALES", r"PATRÓN MACROCRIMINAL",
                                 r"MODUS OPERANDI"],
    "CALIFICACION_JURIDICA": [r"CALIFICACIÓN JURÍDICA", r"CALIFICACION JURIDICA",
                              r"TIPIFICACIÓN", r"CRÍMENES DE GUERRA",
                              r"CRÍMENES DE LESA HUMANIDAD"],
    "RECONOCIMIENTO": [r"RECONOCIMIENTO DE RESPONSABILIDAD", r"RECONOCIMIENTO DE VERDAD",
                       r"ACTOS DE RECONOCIMIENTO"],
    "SANCIONES_PROPIAS": [r"SANCIONES PROPIAS", r"SANCIÓN PROPIA",
                          r"PROYECTOS RESTAURATIVOS", r"TRABAJOS.*REPARACIÓN"],
    "CONSIDERACIONES": [r"^\s*(?:[IVX]+\.\s*)?CONSIDERACIONES\b", r"FUNDAMENTOS JURÍDICOS"],
    "RESUELVE": [r"^\s*RESUELVE\b", r"EN MÉRITO DE LO EXPUESTO", r"POR LO ANTERIOR.*RESUELVE"],
}


def es_linea_indice(linea):
    """True si la linea es de la tabla de contenido (encabezado + puntos/pagina)."""
    # muchos puntos suspensivos, o termina en numero de pagina tras puntos
    if re.search(r"\.{5,}", linea):
        return True
    if re.search(r"\.{3,}\s*\d+\s*$", linea):
        return True
    return False


def segmentar(texto):
    """Divide en secciones tematicas, saltando la tabla de contenido."""
    secciones = {}
    seccion_actual = "CUERPO"
    buffer = []
    for linea in texto.split("\n"):
        lu = linea.upper().strip()
        if es_linea_indice(linea):
            # es indice: no cambia de seccion, pero tampoco suma al cuerpo
            continue
        detectada = None
        for nombre, patrones in PATRONES_SECCIONES.items():
            for pat in patrones:
                if re.search(pat, lu):
                    detectada = nombre
                    break
            if detectada:
                break
        if detectada and detectada != seccion_actual:
            if buffer:
                t = "\n".join(buffer).strip()
                if t:
                    secciones.setdefault(seccion_actual, []).append(t)
            seccion_actual = detectada
            buffer = []
        else:
            buffer.append(linea)
    if buffer:
        t = "\n".join(buffer).strip()
        if t:
            secciones.setdefault(seccion_actual, []).append(t)
    # unir multiples apariciones de la misma seccion
    return {k: "\n".join(v) for k, v in secciones.items()}


def main():
    print("=" * 66)
    print("Fortalecer Corpus B - segmentacion + y1_ebi + y10_rep_v5")
    print("=" * 66)

    # 1. Documentos unicos (por hash de contenido)
    files = glob.glob(os.path.join(DIR_B, "*.txt"))
    unicos, vistos = [], set()
    for f in sorted(files):
        h = hashlib.sha256(open(f, encoding="utf-8").read().encode()).hexdigest()
        if h not in vistos:
            vistos.add(h)
            unicos.append(f)
    print(f"\n.txt totales: {len(files)}  |  unicos: {len(unicos)}")

    # 2. Cargar extractor y10 v5
    import y10_rep_extractor as y10mod
    print("Instanciando REPExtractor (es_core_news_lg)...")
    extractor = y10mod.REPExtractor()

    def y10_score(text, corpus_type="B"):
        try:
            res = extractor.extract(text=text, doc_id="B", section_id="s", corpus_type=corpus_type)
            for a in ["score", "rep_score", "normalized_score", "value", "final_score"]:
                if hasattr(res, a) and isinstance(getattr(res, a), (int, float)):
                    return float(getattr(res, a))
            for k, v in vars(res).items():
                if isinstance(v, (int, float)) and "score" in k.lower():
                    return float(v)
        except Exception:
            return 0.0
        return 0.0

    # 3. Segmentar cada documento y calcular indicadores
    filas_texto, filas_ind = [], []
    for f in unicos:
        doc = os.path.splitext(os.path.basename(f))[0]
        crudo = open(f, encoding="utf-8").read()
        limpio = limpiar_texto(crudo)
        secs = segmentar(limpio)
        n_val = 0
        for seccion, txt in secs.items():
            if len(txt) < MIN_CHARS_SECCION:
                continue
            n_val += 1
            filas_texto.append({"doc": doc, "seccion": seccion,
                                "chars": len(txt), "texto": txt})
            filas_ind.append({"doc": doc, "seccion": seccion,
                              "corpus_type": "B", "chars": len(txt),
                              "y1_ebi": ebi_densidad(txt),
                              "y10_rep_v5": y10_score(txt)})
        print(f"  {doc[:45]:<45} -> {n_val} secciones validas")

    df_txt = pd.DataFrame(filas_texto)
    df_ind = pd.DataFrame(filas_ind)

    print("\n" + "-" * 66)
    print(f"Total secciones de B (validas): {len(df_ind)}")
    print("\nSecciones por tipo:")
    print(df_txt["seccion"].value_counts().to_string())
    print("\nIndicadores B recalculados:")
    print(f"  y1_ebi:     media={df_ind['y1_ebi'].mean():.4f}  "
          f"con EBI>0: {(df_ind['y1_ebi']>0).sum()}/{len(df_ind)}")
    print(f"  y10_rep_v5: media={df_ind['y10_rep_v5'].mean():.4f}  "
          f"con REP>0: {(df_ind['y10_rep_v5']>0).sum()}/{len(df_ind)}")

    os.makedirs(os.path.dirname(OUT_TEXTO), exist_ok=True)
    df_txt.to_csv(OUT_TEXTO, index=False, encoding="utf-8")
    df_ind.to_csv(OUT_IND, index=False, encoding="utf-8")
    print(f"\n  Texto secciones -> {OUT_TEXTO}")
    print(f"  Indicadores     -> {OUT_IND}")
    print("\n  B fortalecido. Ahora si el DIS/IEI tri-corpus tendra un B valido.")


if __name__ == "__main__":
    main()
