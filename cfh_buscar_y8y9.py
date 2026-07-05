# -*- coding: utf-8 -*-
"""
cfh_buscar_y8y9.py
==================
Localiza en que CSV del proyecto estan los valores REALES (no-NaN) de
y8 (distancia MAFAPO) e y9 (distancia CIDH) para Corpus A y B.

Recorre todos los CSV bajo data/, detecta columnas y8/y9 (varios alias
posibles) y reporta cuantos valores reales tienen y que corpus cubren.

Uso:
    python cfh_buscar_y8y9.py
"""

import glob
import pandas as pd

# alias posibles de las columnas
ALIAS_Y8 = {"y8_mafapo", "y8", "y8_mafapo_v5", "dist_mafapo", "mafapo_dist"}
ALIAS_Y9 = {"y9_cidh", "y9", "y9_cidh_v5", "dist_cidh", "cidh_dist"}


def col_match(cols, alias):
    for c in cols:
        cl = c.lower()
        if c in alias or cl in alias:
            return c
        if "mafapo" in cl and alias is ALIAS_Y8:
            return c
        if "cidh" in cl and alias is ALIAS_Y9:
            return c
    return None


def main():
    archivos = sorted(glob.glob("data/**/*.csv", recursive=True))
    print(f"Revisando {len(archivos)} CSV...\n")
    print(f"{'ARCHIVO':70s} {'y8 no-NaN':>12s} {'y9 no-NaN':>12s}  CORPUS")
    print("=" * 120)

    candidatos_a = []

    for f in archivos:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        c8 = col_match(df.columns, ALIAS_Y8)
        c9 = col_match(df.columns, ALIAS_Y9)
        if not c8 and not c9:
            continue

        n8 = df[c8].notna().sum() if c8 else 0
        n9 = df[c9].notna().sum() if c9 else 0

        # solo mostrar si al menos uno tiene datos reales
        if n8 == 0 and n9 == 0:
            continue

        corpus = "?"
        if "corpus_type" in df.columns:
            corpus = ",".join(str(x) for x in df["corpus_type"].dropna().unique())
        elif "corpus" in df.columns:
            corpus = ",".join(str(x) for x in df["corpus"].dropna().unique())

        nombre = f if len(f) <= 68 else "..." + f[-65:]
        print(f"{nombre:70s} {n8:>12d} {n9:>12d}  [{corpus}]")

        # marcar candidatos que cubran Corpus A
        if ("A" in corpus or "CE" in corpus or "CSJ" in corpus) and n8 > 0:
            candidatos_a.append((f, c8, c9, n8, n9, corpus))

    print("\n" + "=" * 120)
    print("\nCANDIDATOS QUE CUBREN CORPUS A CON y8 REAL:")
    if candidatos_a:
        for f, c8, c9, n8, n9, corpus in candidatos_a:
            print(f"  {f}")
            print(f"      col y8={c8} ({n8} valores) | col y9={c9} ({n9} valores) | corpus=[{corpus}]")
    else:
        print("  NINGUNO. y8/y9 de Corpus A no estan calculados en ningun CSV.")
        print("  -> Habria que recalcularlos con el centroide v5 (Paso 2 de la guia de Julian).")

    print("\n" + "=" * 120)
    print("Tambien conviene revisar el doc_id/section_id de los candidatos")
    print("para ver si se pueden unir con indicators_corpus_a.csv por llave.")


if __name__ == "__main__":
    main()
