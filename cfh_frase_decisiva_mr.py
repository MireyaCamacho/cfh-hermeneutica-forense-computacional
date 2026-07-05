# -*- coding: utf-8 -*-
"""
cfh_frase_decisiva_mr.py
=========================
Para los comparecientes marcados REVISAR en la pre-clasificacion, extrae la
FRASE DECISIVA que declara si son maximos responsables o no, buscando tanto
en los autos (Corpus B) como en las TRANSCRIPCIONES de audiencia (Corpus C),
donde con frecuencia se enuncia explicitamente la calidad
("...en calidad de maximo responsable...", "...no es maximo responsable...",
"...participe no determinante...", "...llamado a reconocer...").

A diferencia de la pre-clasificacion (que solo contaba marcadores), aqui se
EXTRAE la oracion completa que contiene el marcador MAS CERCANO al apellido,
para que Mireya lea solo esa frase y etiquete.

Entrada:  data/mr_preclasificado.csv   (usa la columna etiqueta_MR_sugerida)
Fuentes:  data/processed/corpus_b/*.txt   +   corpus_c/*.txt
Salida:   data/mr_frases_decisivas.csv
          (subcaso, compareciente, sugerida, frase_auto, frase_audiencia,
           etiqueta_MR vacia)

Uso:
    python cfh_frase_decisiva_mr.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(".")
PRECLAS = BASE / "data" / "mr_preclasificado.csv"
AUTOS_DIR = BASE / "data" / "processed" / "corpus_b"
AUD_DIR = BASE / "corpus_c"
OUT = BASE / "data" / "mr_frases_decisivas.csv"

MARC = re.compile(
    r"(m[aá]xim[oa]s?\s+responsab|no\s+es\s+m[aá]xim|no\s+m[aá]xim|"
    r"part[ií]cipe\s+no\s+determinante|no\s+determinante|autor[ií]a\s+mediata|"
    r"autor\s+mediato|coautor|en\s+calidad\s+de|llamad[oa]s?\s+a\s+reconocer|"
    r"responsabilidad\s+de\s+mando|le\s+asiste\s+la\s+condici[oó]n)",
    re.IGNORECASE)


def normalizar(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def variantes_grafia(palabra):
    vs = {palabra}
    for a, b in [("s", "z"), ("z", "s"), ("b", "v"), ("v", "b")]:
        for v in list(vs):
            if a in v:
                vs.add(v.replace(a, b))
    if palabra.startswith("h"):
        vs.add(palabra[1:])
    return vs


def apellidos(nombre):
    p = normalizar(nombre).split()
    base = p[-2:] if len(p) >= 2 else p
    out = set()
    for ap in base:
        if len(ap) >= 4:
            out |= variantes_grafia(ap)
    return out


def oracion_de(texto_orig, pos):
    """Devuelve la oracion (aprox) que contiene la posicion pos."""
    ini = texto_orig.rfind(".", 0, pos)
    ini = ini + 1 if ini != -1 else max(0, pos - 300)
    fin = texto_orig.find(".", pos)
    fin = fin + 1 if fin != -1 else min(len(texto_orig), pos + 300)
    return re.sub(r"\s+", " ", texto_orig[ini:fin]).strip()


def frase_decisiva(apes, textos):
    """Busca el marcador MR/noMR mas cercano a un apellido y devuelve su oracion."""
    mejor = None
    mejor_dist = 10**9
    for nombre_archivo, (orig, norm) in textos.items():
        # posiciones de apellidos
        pos_ape = []
        for ap in apes:
            pos_ape += [m.start() for m in re.finditer(re.escape(ap), norm)]
        if not pos_ape:
            continue
        # posiciones de marcadores
        for mm in MARC.finditer(norm):
            mp = mm.start()
            d = min(abs(mp - pa) for pa in pos_ape)
            if d < mejor_dist and d < 400:   # marcador a <400 chars del apellido
                mejor_dist = d
                mejor = (nombre_archivo, oracion_de(orig, mp))
    return mejor


def main():
    if not PRECLAS.exists():
        print(f"[ERROR] no existe {PRECLAS}")
        return
    df = pd.read_csv(PRECLAS)

    autos = {}
    for tp in sorted(AUTOS_DIR.glob("*.txt")):
        try:
            o = tp.read_text(encoding="utf-8", errors="replace")
            autos[tp.name] = (o, normalizar(o))
        except Exception:
            pass
    auds = {}
    for tp in sorted(AUD_DIR.glob("*.txt")):
        if tp.name.lower() == "readme.txt":
            continue
        try:
            o = tp.read_text(encoding="utf-8", errors="replace")
            auds[tp.name] = (o, normalizar(o))
        except Exception:
            pass
    print(f"Autos: {len(autos)}  Audiencias: {len(auds)}")

    filas = []
    for _, row in df.iterrows():
        apes = apellidos(row["compareciente"])
        fa = frase_decisiva(apes, autos)
        fc = frase_decisiva(apes, auds)
        filas.append({
            "subcaso": row["subcaso"],
            "compareciente": row["compareciente"],
            "sugerida": row.get("etiqueta_MR_sugerida", ""),
            "frase_auto": f"({fa[0]}) {fa[1]}" if fa else "",
            "frase_audiencia": f"({fc[0]}) {fc[1]}" if fc else "",
            "etiqueta_MR": "",
        })
    out = pd.DataFrame(filas).sort_values(["sugerida", "subcaso", "compareciente"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("FRASES DECISIVAS (revisa y etiqueta 'etiqueta_MR')")
    print("=" * 72)
    for _, r in out.iterrows():
        print(f"\n[{r['sugerida']}] {r['subcaso']} — {r['compareciente']}")
        if r["frase_auto"]:
            print(f"  AUTO: {r['frase_auto'][:240]}")
        if r["frase_audiencia"]:
            print(f"  AUD : {r['frase_audiencia'][:240]}")
        if not r["frase_auto"] and not r["frase_audiencia"]:
            print("  (sin marcador cercano — revisar contexto completo en el CSV doble)")

    print(f"\n  GUARDADO: {OUT}")


if __name__ == "__main__":
    main()
