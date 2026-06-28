# -*- coding: utf-8 -*-
"""
cfh_revisar_rep_huila.py
================================================================================
CFH — Revisar el texto completo y el REP score de comparecientes con rep extremo

OBJETO:
    Harbey Sánchez (rep=1.0, 176 tokens) y Riveros (rep=0.769, texto dañado al
    inicio) tienen REP alto sospechoso. Antes de excluir/conservar, ver:
      1. El TEXTO COMPLETO real que se les atribuyó (todos los segmentos).
      2. Correr el REPExtractor sobre ese texto y ver el score + qué lo dispara.
      3. Juzgar si el rep alto es señal real (lenguaje reparatorio genuino) o
         artefacto (texto truncado, frases sueltas que el extractor sobrepondera).

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_revisar_rep_huila.py"

Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import json, sys
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
MARC = BASE / "data" / "marcacion" / "inventario_Huila.csv"
SEG = BASE / "corpus_c" / "huila_segments.json"

REVISAR = ["Harbey Sanchez Gomez", "Fernando Riveros Sarmiento"]


def hms(s):
    s = str(s).strip()
    if s in ("", "nan", "None") or ":" not in s:
        return None
    p = [int(x) for x in s.split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


def cargar_rep():
    """Carga el REPExtractor real del repo."""
    sys.path.insert(0, str(BASE / "code" / "src" / "features"))
    try:
        from y10_rep_extractor import REPExtractor
        return REPExtractor()
    except Exception as e:
        print(f"  [aviso] no se pudo cargar REPExtractor: {e}")
        return None


def main():
    print("CFH — Revisión de REP: Harbey y Riveros")
    print("="*72)

    marc = pd.read_csv(MARC)
    marc = marc[marc["uso"] == "ANALISIS"].copy()
    marc["ini_s"] = marc["inicio"].apply(hms)
    marc["fin_s"] = marc["fin"].apply(hms)

    segs = json.load(open(SEG, encoding="utf-8"))
    if isinstance(segs, dict):
        for v in segs.values():
            if isinstance(v, list):
                segs = v; break

    rep_ext = cargar_rep()

    for nombre in REVISAR:
        sub = marc[marc["identidad"] == nombre]
        if sub.empty:
            print(f"\n[{nombre}] no encontrado."); continue
        print(f"\n{'═'*72}\n{nombre}")
        textos = []
        for _, r in sub.iterrows():
            a0, a1 = r["ini_s"], r["fin_s"]
            if a0 is None:
                continue
            for s in segs:
                s0 = float(s.get("start",0)); s1 = float(s.get("end",0))
                if s1 > a0 and s0 < a1:
                    t = s.get("text","").strip()
                    if t:
                        textos.append(t)
        full = " ".join(textos)
        print(f"  Tokens: {len(full.split())}")
        print(f"\n  ── TEXTO COMPLETO ──")
        # imprimir en bloques legibles
        palabras = full.split()
        linea = "  "
        for w in palabras:
            if len(linea) + len(w) > 88:
                print(linea); linea = "  "
            linea += w + " "
        if linea.strip():
            print(linea)

        # REP score real
        if rep_ext is not None:
            try:
                res = rep_ext.extract(full, doc_id=nombre,
                                      section_id="RECONOCIMIENTO", corpus_type="C")
                print(f"\n  ── REP score: {res.score:.3f} ──")
                # intentar mostrar evidencia/spans si el extractor los expone
                for attr in ["matches","spans","evidence","hits","tokens_rep","detail"]:
                    if hasattr(res, attr):
                        val = getattr(res, attr)
                        if val:
                            print(f"     {attr}: {str(val)[:300]}")
            except Exception as e:
                print(f"  [aviso] REP no calculado: {e}")

    print(f"\n{'═'*72}")
    print("JUICIO:")
    print("  · Si el texto es un saludo protocolario corto → rep alto = ARTEFACTO.")
    print("  · Si el texto tiene frases reparatorias genuinas ('reconozco',")
    print("    'pido perdón', 'me arrepiento', nombres de víctimas) → rep alto REAL.")
    print("  · Texto truncado a mitad de frase → poco confiable, marcar.")


if __name__ == "__main__":
    main()
