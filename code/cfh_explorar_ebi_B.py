# -*- coding: utf-8 -*-
r"""
cfh_explorar_ebi_B.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Explora el hallazgo NUEVO: el Corpus B (JEP escrita) usa MAS eufemismo belico
(EBI, y1) que el Corpus A (justicia ordinaria) -- contraintuitivo. Verifica la
hipotesis de METALENGUAJE CITATIVO: la JEP cita/describe el vocabulario militar
para refutarlo, no para avalarlo.

Prueba de la hipotesis citativa: si el EBI en B esta concentrado en las
secciones que DESCRIBEN los hechos (HECHOS_Y_CONDUCTAS, PATRONES_MACROCRIMINALES,
CALIFICACION) y NO en las de reconocimiento/resuelve, apoya que es citativo
(describe el lenguaje militar) y no avalativo.

Analiza:
  1. EBI medio por SECCION tematica en B
  2. Que secciones concentran el EBI
  3. Ejemplos de spans EBI (si el CSV los tiene) para ver si son citativos
  4. Comparacion: en A, donde aparece el EBI?

Entrada: outputs/corpus_b_secciones_texto.csv (texto + seccion + y1_ebi)
         outputs/corpus_b_indicadores_COMPLETO.csv (indicadores por seccion)

Salida: outputs/exploracion_ebi_B.txt

Uso (raiz del repo, env cfh):
    python code\cfh_explorar_ebi_B.py
"""

import os
import sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_TEXTO = os.path.join(REPO, "outputs", "corpus_b_secciones_texto.csv")
F_IND = os.path.join(REPO, "outputs", "corpus_b_indicadores_COMPLETO.csv")
OUT = os.path.join(REPO, "outputs", "exploracion_ebi_B.txt")


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    print("=" * 66)
    print("EXPLORACION DEL EBI EN CORPUS B - hipotesis metalenguaje citativo")
    print("=" * 66)

    ind = pd.read_csv(F_IND)
    print(f"\nSecciones de B: {len(ind)}")
    print(f"Columnas: {list(ind.columns)}")

    # detectar columna de seccion
    col_sec = None
    for c in ["seccion", "section", "seccion_tipo", "tipo_seccion"]:
        if c in ind.columns:
            col_sec = c
            break
    if not col_sec:
        print("[AVISO] no encuentro columna de seccion; muestro solo global")
        print(f"EBI medio B: {ind['y1_ebi'].mean():.3f}")
        sys.stdout = sys.__stdout__; fh.close(); return

    # EBI por seccion tematica
    print("\n--- EBI (y1) medio por seccion tematica en B ---")
    g = ind.groupby(col_sec).agg(
        n=("y1_ebi", "size"),
        ebi_medio=("y1_ebi", "mean"),
        ebi_max=("y1_ebi", "max"),
        con_ebi=("y1_ebi", lambda s: (s > 0).sum()),
    ).round(3).sort_values("ebi_medio", ascending=False)
    print(g.to_string())

    # clasificar secciones: descriptivas (citativas esperadas) vs performativas
    DESCRIPTIVAS = ["HECHOS_Y_CONDUCTAS", "PATRONES_MACROCRIMINALES",
                    "CALIFICACION_JURIDICA", "CONSIDERACIONES", "CUERPO"]
    PERFORMATIVAS = ["RECONOCIMIENTO", "RESUELVE", "SANCIONES_PROPIAS"]

    ind["tipo"] = ind[col_sec].apply(
        lambda s: "descriptiva" if s in DESCRIPTIVAS else
                  ("performativa" if s in PERFORMATIVAS else "otra"))

    print("\n--- EBI por TIPO de seccion (test de hipotesis citativa) ---")
    t = ind.groupby("tipo").agg(
        n=("y1_ebi", "size"),
        ebi_medio=("y1_ebi", "mean"),
    ).round(3)
    print(t.to_string())

    desc = ind[ind["tipo"] == "descriptiva"]["y1_ebi"].mean()
    perf = ind[ind["tipo"] == "performativa"]["y1_ebi"].mean()
    print("\n--- VEREDICTO ---")
    if desc > perf * 1.3:
        print(f"  APOYA metalenguaje citativo: EBI en secciones descriptivas")
        print(f"  ({desc:.3f}) >> performativas ({perf:.3f}).")
        print(f"  La JEP usa vocabulario militar al DESCRIBIR los hechos")
        print(f"  (citando el lenguaje oficial), no al reconocer/sancionar.")
    elif perf > desc * 1.3:
        print(f"  NO apoya citativo: EBI mayor en performativas ({perf:.3f})")
        print(f"  que descriptivas ({desc:.3f}). Revisar interpretacion.")
    else:
        print(f"  MIXTO: EBI similar en descriptivas ({desc:.3f}) y")
        print(f"  performativas ({perf:.3f}). El patron citativo no es nitido;")
        print(f"  reportar con cautela.")

    # ejemplos de texto con EBI alto (si hay columna de texto)
    if os.path.exists(F_TEXTO):
        txt = pd.read_csv(F_TEXTO)
        col_txt = None
        for c in ["texto", "text", "contenido"]:
            if c in txt.columns:
                col_txt = c; break
        if col_txt and "y1_ebi" in txt.columns:
            print("\n--- Ejemplos de secciones con EBI alto (para ver si es citativo) ---")
            top = txt.nlargest(3, "y1_ebi")
            for _, r in top.iterrows():
                s = str(r[col_txt])[:280].replace("\n", " ")
                print(f"\n  EBI={r['y1_ebi']:.2f}: ...{s}...")

    print(f"\n  Reporte -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
