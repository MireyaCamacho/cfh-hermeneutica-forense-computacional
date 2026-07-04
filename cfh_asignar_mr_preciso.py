# -*- coding: utf-8 -*-
"""
cfh_asignar_mr_preciso.py
==========================
Asignacion PRECISA de MR / NO_MR evitando el cruce entre personas.

CLAVE: solo cuenta un marcador ("maximo responsable" / "no maximo responsable")
cuando el NOMBRE del compareciente aparece EN LA MISMA FRASE/ventana breve que
el marcador (no el marcador mas cercano, que podia ser de otra persona de una
lista). Usa variantes de grafia ASR (Zamboni/Samboni, Arbeys/Harbey, etc.).

Prioridad de fuente:
  1) AUDIENCIA (corpus_c): el magistrado anuncia "el compareciente [no] maximo
     responsable, <Nombre>" — declaracion directa y explicita.
  2) AUTO (corpus_b): "le asiste la condicion de maximo responsable",
     "participe no determinante", etc., en frase con el apellido.

Resultado:
  - etiqueta_MR AUTO-ASIGNADA donde hay declaracion explicita (MR o NO_MR).
  - REVISAR solo donde NO hay frase que ligue nombre + calidad.
  - Cada asignacion lleva la frase-fuente para trazabilidad.

Salida: data/mr_asignacion_final.csv
        (subcaso, compareciente, etiqueta_MR, fuente, frase_evidencia)
Consola: distribucion y SOLO la lista de REVISAR (los que debes mirar).

Uso:
    python cfh_asignar_mr_preciso.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(".")
ICM_CSV = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
AUTOS_DIR = BASE / "data" / "processed" / "corpus_b"
AUD_DIR = BASE / "corpus_c"
OUT = BASE / "data" / "mr_asignacion_final.csv"

# ventana breve: nombre y marcador deben estar a <=120 chars
VENTANA = 120

RE_NOMR = re.compile(
    r"no\s+(?:fu(?:e|eron)\s+(?:determinad[oa]s?\s+como\s+)?)?m[aá]xim[oa]s?\s+responsab|"
    r"no\s+determinad[oa]s?\s+como\s+m[aá]xim|part[ií]cipe\s+no\s+determinante|"
    r"no\s+determinante", re.IGNORECASE)
RE_MR = re.compile(
    r"(?<!no\s)(?<!no\s\s)m[aá]xim[oa]s?\s+responsab|determinad[oa]\s+como\s+m[aá]ximo|"
    r"seleccionad[oa]\s+por\s+la\s+sala\s+como\s+m[aá]ximo|"
    r"le\s+asiste\s+la\s+condici[oó]n\s+de\s+m[aá]ximo|"
    r"soy\s+m[aá]ximo\s+responsable|"
    r"en\s+calidad\s+de\s+m[aá]ximo|autor[ií]a\s+mediata", re.IGNORECASE)


def normalizar(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def variantes(pal):
    vs = {pal}
    for a, b in [("s", "z"), ("z", "s"), ("b", "v"), ("v", "b"), ("ll", "y")]:
        for v in list(vs):
            if a in v:
                vs.add(v.replace(a, b))
    if pal.startswith("h"):
        vs.add(pal[1:])
    else:
        vs.add("h" + pal)
    return {v for v in vs if len(v) >= 4}


def claves_nombre(nombre):
    """apellidos con variantes de grafia."""
    p = [x for x in normalizar(nombre).split() if len(x) >= 4]
    base = p[-2:] if len(p) >= 2 else p
    out = set()
    for ap in base:
        out |= variantes(ap)
    return out


def evaluar(texto_orig, texto_norm, claves):
    """
    Busca ventanas donde aparece algun apellido y, dentro de +-VENTANA,
    un marcador MR o NO_MR. Devuelve (etiqueta, frase) o None.
    NO_MR tiene prioridad sobre MR si ambos aparecen ligados al nombre
    (porque 'no maximo responsable' contiene 'maximo responsable').
    """
    encontrados = []
    for ap in claves:
        for m in re.finditer(re.escape(ap), texto_norm):
            a = max(0, m.start() - VENTANA)
            b = min(len(texto_norm), m.end() + VENTANA)
            frag = texto_norm[a:b]
            if RE_NOMR.search(frag):
                encontrados.append(("NO_MR", texto_orig[a:b]))
            elif RE_MR.search(frag):
                encontrados.append(("MR", texto_orig[a:b]))
    if not encontrados:
        return None
    # si hay algun NO_MR explicito, gana NO_MR (declaracion mas especifica)
    for et, fr in encontrados:
        if et == "NO_MR":
            return ("NO_MR", re.sub(r"\s+", " ", fr).strip())
    return ("MR", re.sub(r"\s+", " ", encontrados[0][1]).strip())


def cargar(dir_path, excluir_readme=True):
    d = {}
    for tp in sorted(dir_path.glob("*.txt")):
        if excluir_readme and tp.name.lower() == "readme.txt":
            continue
        try:
            o = tp.read_text(encoding="utf-8", errors="replace")
            d[tp.name] = (o, normalizar(o))
        except Exception:
            pass
    return d


def buscar_en(fuente_dict, claves):
    """Evalua en cada archivo; devuelve (etiqueta, archivo, frase) o None.
    NO_MR prioritario."""
    hallazgos = []
    for nombre_archivo, (orig, norm) in fuente_dict.items():
        r = evaluar(orig, norm, claves)
        if r:
            hallazgos.append((r[0], nombre_archivo, r[1]))
    if not hallazgos:
        return None
    for et, arch, fr in hallazgos:
        if et == "NO_MR":
            return (et, arch, fr)
    return hallazgos[0]


def main():
    icm = pd.read_csv(ICM_CSV)
    col_i = "identidad" if "identidad" in icm.columns else icm.columns[1]
    col_s = "subcaso" if "subcaso" in icm.columns else icm.columns[0]
    personas = icm[[col_s, col_i]].drop_duplicates().values.tolist()

    autos = cargar(AUTOS_DIR)
    auds = cargar(AUD_DIR)
    print(f"Autos: {len(autos)}  Audiencias: {len(auds)}  Comparecientes: {len(personas)}")

    filas = []
    for sub, ident in personas:
        claves = claves_nombre(ident)
        # 1) audiencia (declaracion directa del magistrado)
        r = buscar_en(auds, claves)
        fuente = "audiencia"
        if r is None:
            # 2) auto
            r = buscar_en(autos, claves)
            fuente = "auto"
        if r is None:
            filas.append({"subcaso": sub, "compareciente": ident,
                          "etiqueta_MR": "REVISAR", "fuente": "",
                          "frase_evidencia": ""})
        else:
            et, arch, fr = r
            filas.append({"subcaso": sub, "compareciente": ident,
                          "etiqueta_MR": et, "fuente": f"{fuente}:{arch}",
                          "frase_evidencia": f"...{fr[:200]}..."})

    df = pd.DataFrame(filas).sort_values(["subcaso", "compareciente"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("ASIGNACION MR (automatica donde hay declaracion explicita)")
    print("=" * 64)
    vc = df["etiqueta_MR"].value_counts()
    for k in ["MR", "NO_MR", "REVISAR"]:
        print(f"  {k:8s}: {int(vc.get(k, 0))}")
    print(f"  TOTAL: {len(df)}")

    rev = df[df["etiqueta_MR"] == "REVISAR"]
    print("\n  --- SOLO ESTOS REQUIEREN TU REVISION ---")
    if rev.empty:
        print("    (ninguno: todos quedaron asignados por declaracion explicita)")
    else:
        for _, r in rev.iterrows():
            print(f"    {r['subcaso']:12s} {r['compareciente']}")

    print(f"\n  GUARDADO: {OUT}")
    print("  Revisa la columna frase_evidencia de los MR/NO_MR por si algun")
    print("  match de grafia quedo dudoso; corrige en etiqueta_MR si hace falta.")


if __name__ == "__main__":
    main()
