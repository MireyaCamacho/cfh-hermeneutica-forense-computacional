# -*- coding: utf-8 -*-
"""
cfh_preclasificar_mr.py
========================
Pre-clasifica MR / NO_MR para los 47 comparecientes analizando la EVIDENCIA
ya extraida (data/mr_evidencia_doble.csv). NO decide de forma definitiva:
propone una etiqueta_MR_sugerida que Mireya revisa y confirma/corrige.

REGLAS (terminos juridicos de la JEP en el Caso 003):
  MR  <- el contexto asocia al nombre con:
         "maximo responsable", "autor mediato", "coautor",
         "en calidad de autor", "responsabilidad de mando",
         "liderazgo", "llamado a reconocer" (en seccion de max. resp.)
  NO_MR <- el contexto asocia al nombre con:
         "participe no determinante", "no maximo responsable",
         "no determinante"
  REVISAR <- hay evidencia pero sin marcador decisivo claro, o
             marcadores en conflicto (ambos aparecen).

IMPORTANTE: la regla busca el marcador en una VENTANA cercana al apellido del
compareciente dentro del contexto, no en todo el texto (para no contaminar con
menciones de OTRAS personas que aparecen en el mismo parrafo, ej. listas).

Salida:
  data/mr_preclasificado.csv  con columnas de evidencia + etiqueta_MR_sugerida
  + justificacion (que marcador/regla disparo) + etiqueta_MR (vacia, para
  que Mireya confirme).
Consola: distribucion MR / NO_MR / REVISAR.

Uso:
    python cfh_preclasificar_mr.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(".")
DOBLE = BASE / "data" / "mr_evidencia_doble.csv"
OUT = BASE / "data" / "mr_preclasificado.csv"

# ventana de chars alrededor del apellido dentro de un contexto
VENTANA_LOCAL = 220

MARCADORES_MR = [
    r"m[aá]xim[oa]s?\s+responsab",
    r"autor[ií]a\s+mediata",
    r"autor\s+mediato",
    r"coautor",
    r"responsabilidad\s+de\s+mando",
    r"en\s+calidad\s+de\s+autor",
    r"le\s+asiste\s+la\s+condici[oó]n\s+de\s+m[aá]ximo",
    r"llamad[oa]s?\s+a\s+reconocer",
    r"liderazgo\s+y\s+dominio",
]
MARCADORES_NOMR = [
    r"part[ií]cipe\s+no\s+determinante",
    r"no\s+m[aá]xim[oa]s?\s+responsab",
    r"no\s+determinante",
    r"participaci[oó]n\s+no\s+determinante",
]

RE_MR = [re.compile(p, re.IGNORECASE) for p in MARCADORES_MR]
RE_NOMR = [re.compile(p, re.IGNORECASE) for p in MARCADORES_NOMR]


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def apellidos_de(nombre: str):
    partes = normalizar(nombre).split()
    return partes[-2:] if len(partes) >= 2 else partes


def marcadores_cerca(texto, apellidos):
    """Busca marcadores MR/NO_MR en ventanas cercanas a los apellidos."""
    tn = normalizar(texto)
    hits_mr, hits_nomr = set(), set()
    posiciones = []
    for ap in apellidos:
        for m in re.finditer(re.escape(ap), tn):
            posiciones.append(m.start())
    # si no encontramos el apellido en el contexto (por grafia), evaluamos todo el texto
    if not posiciones:
        ventanas = [(0, len(tn))]
    else:
        ventanas = [(max(0, p - VENTANA_LOCAL), min(len(tn), p + VENTANA_LOCAL))
                    for p in posiciones]
    for a, b in ventanas:
        frag = tn[a:b]
        for rx in RE_MR:
            if rx.search(frag):
                hits_mr.add(rx.pattern)
        for rx in RE_NOMR:
            if rx.search(frag):
                hits_nomr.add(rx.pattern)
    return hits_mr, hits_nomr


def main():
    if not DOBLE.exists():
        print(f"[ERROR] no existe {DOBLE}")
        return
    df = pd.read_csv(DOBLE)

    cols_ctx = [c for c in ["autos_contexto_1", "autos_contexto_2",
                            "aud_contexto_1", "aud_contexto_2"] if c in df.columns]

    sugeridas, justif = [], []
    for _, row in df.iterrows():
        apes = apellidos_de(row["compareciente"])
        texto_junto = " || ".join(str(row.get(c, "")) for c in cols_ctx)
        mr, nomr = marcadores_cerca(texto_junto, apes)

        if nomr and not mr:
            sug = "NO_MR"
            j = "NO_MR: " + "; ".join(sorted(nomr))
        elif mr and not nomr:
            sug = "MR"
            j = "MR: " + "; ".join(sorted(m.split("\\")[0][:25] for m in mr))
        elif mr and nomr:
            sug = "REVISAR"
            j = f"CONFLICTO -> MR:{len(mr)} NO_MR:{len(nomr)} (ambos presentes)"
        else:
            sug = "REVISAR"
            j = "sin marcador decisivo cerca del apellido"
        sugeridas.append(sug)
        justif.append(j)

    df["etiqueta_MR_sugerida"] = sugeridas
    df["justificacion"] = justif
    if "etiqueta_MR" not in df.columns:
        df["etiqueta_MR"] = ""
    else:
        df["etiqueta_MR"] = ""  # limpiar para confirmacion manual

    # ordenar columnas: identificacion + sugerida + justif + evidencia
    orden = ["subcaso", "compareciente", "etiqueta_MR_sugerida", "etiqueta_MR",
             "justificacion", "doble_clave",
             "autos_n_menciones", "n_menciones_audiencia",
             "autos_contexto_1", "autos_contexto_2",
             "aud_contexto_1", "aud_contexto_2"]
    orden = [c for c in orden if c in df.columns] + \
            [c for c in df.columns if c not in orden]
    df = df[orden].sort_values(["subcaso", "compareciente"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("PRE-CLASIFICACION MR (para revision de Mireya)")
    print("=" * 60)
    vc = df["etiqueta_MR_sugerida"].value_counts()
    for k in ["MR", "NO_MR", "REVISAR"]:
        print(f"  {k:8s}: {int(vc.get(k, 0))}")
    print(f"  TOTAL: {len(df)}")

    print("\n  --- REVISAR (requieren tu lectura) ---")
    for _, r in df[df["etiqueta_MR_sugerida"] == "REVISAR"].iterrows():
        print(f"    {r['subcaso']:12s} {str(r['compareciente'])[:32]:32s} {r['justificacion'][:50]}")

    print("\n  --- NO_MR sugeridos (verifica con cuidado) ---")
    for _, r in df[df["etiqueta_MR_sugerida"] == "NO_MR"].iterrows():
        print(f"    {r['subcaso']:12s} {str(r['compareciente'])[:32]:32s} {r['justificacion'][:50]}")

    print(f"\n  GUARDADO: {OUT}")
    print("  Revisa 'etiqueta_MR_sugerida', y confirma en la columna 'etiqueta_MR'")
    print("  (copiando la sugerida donde estes de acuerdo, corrigiendo donde no).")


if __name__ == "__main__":
    main()
