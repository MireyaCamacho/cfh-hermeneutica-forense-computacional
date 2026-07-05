# -*- coding: utf-8 -*-
"""
cfh_verificar_merge_y8y9.py
===========================
Verifica que indicators_completo_conflibert_v3b.csv (que tiene y8/y9 reales)
se pueda unir con indicators_corpus_a.csv / _b.csv (el CSV del SEM, que tiene
y7 recien calculado) por la llave (doc_id, section_id).

NO modifica nada. Solo diagnostica el emparejamiento.

Uso:
    python cfh_verificar_merge_y8y9.py
"""

import pandas as pd

SEM_A = "data/features/indicators_corpus_a.csv"
SEM_B = "data/features/indicators_corpus_b.csv"
FUENTE = "data/features/canonico/indicators_completo_conflibert_v3b.csv"


def main():
    print("=" * 70)
    print("VERIFICACION DE MERGE y8/y9  (v3b -> CSV del SEM)")
    print("=" * 70)

    src = pd.read_csv(FUENTE)
    print(f"\nFUENTE: {FUENTE}")
    print(f"  filas: {len(src)}")
    print(f"  columnas llave presentes: "
          f"doc_id={'doc_id' in src.columns} | section_id={'section_id' in src.columns}")
    print(f"  y8_mafapo no-NaN: {src['y8_mafapo'].notna().sum()}")
    print(f"  y9_cidh   no-NaN: {src['y9_cidh'].notna().sum()}")
    if 'corpus_type' in src.columns:
        print(f"  corpus_type: {src['corpus_type'].value_counts().to_dict()}")

    for nombre, sem_path in [("A", SEM_A), ("B", SEM_B)]:
        print("\n" + "-" * 70)
        print(f"CORPUS {nombre}: {sem_path}")
        sem = pd.read_csv(sem_path)
        print(f"  filas SEM: {len(sem)}")

        # llave compuesta
        sem_keys = set(zip(sem["doc_id"], sem["section_id"]))
        src_keys = set(zip(src["doc_id"], src["section_id"]))

        comunes = sem_keys & src_keys
        solo_sem = sem_keys - src_keys
        print(f"  llaves (doc_id, section_id) en SEM: {len(sem_keys)}")
        print(f"  llaves que TAMBIEN estan en la fuente v3b: {len(comunes)}")
        print(f"  llaves del SEM SIN match en v3b: {len(solo_sem)}")

        if solo_sem:
            print(f"  ejemplos sin match (hasta 10):")
            for k in list(solo_sem)[:10]:
                print(f"      {str(k[0])[:16]} / {k[1]}")

        pct = 100 * len(comunes) / len(sem_keys) if sem_keys else 0
        print(f"  >> cobertura del merge: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("Si la cobertura es ~100%, se puede traer y8/y9 al CSV del SEM")
    print("con un merge por (doc_id, section_id) sin perder filas.")


if __name__ == "__main__":
    main()
