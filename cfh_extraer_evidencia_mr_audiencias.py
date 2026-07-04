# -*- coding: utf-8 -*-
"""
cfh_extraer_evidencia_mr_audiencias.py
=======================================
DOBLE VERIFICACION de la calidad MR / no-MR: busca a los 47 comparecientes
del ICM en las TRANSCRIPCIONES DE LAS AUDIENCIAS (Corpus C), donde los
magistrados presentan a cada compareciente indicando su calidad
("compareciente maximo responsable...", "no maximo responsable...").

Fuente 2 (audiencias): corpus_c/*.txt  (las 5 transcripciones canonicas)
Fuente 1 (autos):      data/mr_evidencia_comparecientes.csv (ya generado)

QUE HACE:
  1. Busca cada compareciente en las transcripciones (nombre completo y
     apellidos, normalizando tildes).
  2. Extrae hasta 3 contextos (+-250 chars) priorizando los que contienen
     "maximo responsable" / "no maximo" / "en calidad de" / "imputa".
  3. CRUZA con la evidencia de autos y genera un CSV DOBLE:
       data/mr_evidencia_doble.csv
     con columnas: subcaso, compareciente,
       autos_n_menciones, autos_contexto_1..2,
       audiencia_n_menciones, audiencia_contexto_1..2,
       etiqueta_MR (vacia -> MR / NO_MR / SIN_DATO)
  Donde AMBAS fuentes tengan contexto CLAVE, la confirmacion es directa.

Uso:
    python cfh_extraer_evidencia_mr_audiencias.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(".")
ICM_CSV = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
AUDIENCIAS_DIR = BASE / "corpus_c"
EVIDENCIA_AUTOS = BASE / "data" / "mr_evidencia_comparecientes.csv"
OUT_AUD = BASE / "data" / "mr_evidencia_audiencias.csv"
OUT_DOBLE = BASE / "data" / "mr_evidencia_doble.csv"

VENTANA = 250
MAX_CONTEXTOS = 3

CLAVE_RE = re.compile(
    r"(m[aá]xim[oa]s?\s+responsab|no\s+m[aá]xim|imputa|en\s+calidad\s+de|"
    r"part[ií]cipe|determinante|responsabilidad\s+de\s+mando|"
    r"autor[ií]a\s+mediata|coautor)", re.IGNORECASE)


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def variantes_busqueda(nombre: str):
    n = normalizar(nombre)
    partes = n.split()
    v = {n}
    if len(partes) >= 2:
        v.add(" ".join(partes[-2:]))
    return {x for x in v if len(x) >= 10}


def buscar_en_textos(personas, textos):
    """textos: dict nombre_archivo -> (original, normalizado).
    Devuelve dict (sub, ident) -> (n_total, archivos, [(prio, arch, ctx), ...])"""
    out = {}
    for sub, ident in personas:
        vs = variantes_busqueda(ident)
        contextos, n_total, archivos = [], 0, set()
        for nombre_archivo, (orig, norm) in textos.items():
            for v in vs:
                for m in re.finditer(re.escape(v), norm):
                    n_total += 1
                    archivos.add(nombre_archivo)
                    pos = int(m.start() / max(len(norm), 1) * len(orig))
                    a = max(0, pos - VENTANA - 80)
                    b = min(len(orig), pos + VENTANA + 80)
                    ctx = re.sub(r"\s+", " ", orig[a:b]).strip()
                    contextos.append((bool(CLAVE_RE.search(ctx)), nombre_archivo, ctx))
        contextos.sort(key=lambda x: (not x[0], -len(x[1])))
        out[(sub, ident)] = (n_total, archivos, contextos[:MAX_CONTEXTOS])
    return out


def main():
    icm = pd.read_csv(ICM_CSV)
    col_ident = "identidad" if "identidad" in icm.columns else icm.columns[1]
    col_sub = "subcaso" if "subcaso" in icm.columns else icm.columns[0]
    personas = icm[[col_sub, col_ident]].drop_duplicates().values.tolist()
    print(f"Comparecientes del ICM: {len(personas)}")

    # cargar transcripciones de audiencias
    textos = {}
    for tp in sorted(AUDIENCIAS_DIR.glob("*.txt")):
        try:
            orig = tp.read_text(encoding="utf-8", errors="replace")
            textos[tp.name] = (orig, normalizar(orig))
        except Exception as e:
            print(f"  [WARN] {tp.name}: {e}")
    print(f"Transcripciones cargadas: {len(textos)}  ({', '.join(textos.keys())})")

    res = buscar_en_textos(personas, textos)

    # CSV de audiencias
    filas = []
    for (sub, ident), (n, archivos, ctxs) in res.items():
        fila = {"subcaso": sub, "compareciente": ident,
                "n_menciones_audiencia": n,
                "archivos_audiencia": "; ".join(sorted(archivos))}
        for i in range(MAX_CONTEXTOS):
            if i < len(ctxs):
                prio, arch, ctx = ctxs[i]
                marca = "[CLAVE] " if prio else ""
                fila[f"aud_contexto_{i+1}"] = f"{marca}({arch}) ...{ctx}..."
            else:
                fila[f"aud_contexto_{i+1}"] = ""
        filas.append(fila)
    df_aud = pd.DataFrame(filas).sort_values(["subcaso", "compareciente"])
    df_aud.to_csv(OUT_AUD, index=False, encoding="utf-8-sig")

    # cruce con autos
    print("\n" + "=" * 70)
    print("COBERTURA EN AUDIENCIAS")
    print("=" * 70)
    con = (df_aud["n_menciones_audiencia"] > 0).sum()
    clave = df_aud["aud_contexto_1"].str.startswith("[CLAVE]").sum()
    print(f"  con mencion en audiencias: {con} / {len(df_aud)}")
    print(f"  con contexto CLAVE:        {clave}")
    sin = df_aud[df_aud["n_menciones_audiencia"] == 0]
    if len(sin):
        print("  SIN mencion en audiencias:")
        for _, r in sin.iterrows():
            print(f"      {r['subcaso']:12s} {r['compareciente']}")

    if EVIDENCIA_AUTOS.exists():
        df_autos = pd.read_csv(EVIDENCIA_AUTOS)
        doble = pd.merge(
            df_autos[["subcaso", "compareciente", "n_menciones_total",
                      "archivos_fuente", "contexto_1", "contexto_2"]].rename(
                columns={"n_menciones_total": "autos_n_menciones",
                         "archivos_fuente": "autos_archivos",
                         "contexto_1": "autos_contexto_1",
                         "contexto_2": "autos_contexto_2"}),
            df_aud[["subcaso", "compareciente", "n_menciones_audiencia",
                    "archivos_audiencia", "aud_contexto_1", "aud_contexto_2"]],
            on=["subcaso", "compareciente"], how="outer")
        doble["etiqueta_MR"] = ""
        # marcar confirmacion potencial: CLAVE en ambas fuentes
        doble["doble_clave"] = (
            doble["autos_contexto_1"].astype(str).str.startswith("[CLAVE]") &
            doble["aud_contexto_1"].astype(str).str.startswith("[CLAVE]"))
        doble = doble.sort_values(["subcaso", "compareciente"])
        doble.to_csv(OUT_DOBLE, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 70)
        print("CRUCE DOBLE (autos x audiencias)")
        print("=" * 70)
        print(f"  CLAVE en AMBAS fuentes (confirmacion directa): "
              f"{int(doble['doble_clave'].sum())} / {len(doble)}")
        solo_autos = ((doble['autos_n_menciones'].fillna(0) > 0) &
                      (doble['n_menciones_audiencia'].fillna(0) == 0)).sum()
        solo_aud = ((doble['autos_n_menciones'].fillna(0) == 0) &
                    (doble['n_menciones_audiencia'].fillna(0) > 0)).sum()
        ninguna = ((doble['autos_n_menciones'].fillna(0) == 0) &
                   (doble['n_menciones_audiencia'].fillna(0) == 0)).sum()
        print(f"  solo en autos:      {int(solo_autos)}")
        print(f"  solo en audiencias: {int(solo_aud)}")
        print(f"  en NINGUNA fuente:  {int(ninguna)}")
        print(f"\n  GUARDADO: {OUT_DOBLE}")
        print("  -> llenar etiqueta_MR leyendo ambas columnas de contexto.")
    else:
        print(f"\n  [AVISO] no existe {EVIDENCIA_AUTOS}; solo se genero el de audiencias.")
    print(f"  GUARDADO: {OUT_AUD}")


if __name__ == "__main__":
    main()
