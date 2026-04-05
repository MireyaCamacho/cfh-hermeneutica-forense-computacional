"""
CFH · Script de estimación del modelo SEM
Ejecutar desde la raíz: python run_sem.py --spec parcial
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))

import argparse
import pandas as pd
from sem.sem_model import CFHSEMModel

parser = argparse.ArgumentParser(description="CFH SEM Model")
parser.add_argument("--csv-a", type=Path, default=Path("data/features/indicators_corpus_a.csv"))
parser.add_argument("--csv-b", type=Path, default=Path("data/features/indicators_corpus_b.csv"))
parser.add_argument("--spec", choices=["completo", "parcial", "dos_factores"], default="parcial")
parser.add_argument("--bootstrap", type=int, default=0)
parser.add_argument("--multigroup", action="store_true")
args = parser.parse_args()

# Cargar datos
dfs = []
for csv_path in [args.csv_a, args.csv_b]:
    if csv_path.exists():
        dfs.append(pd.read_csv(csv_path))
        print(f"✓ {csv_path}: {len(dfs[-1])} filas")

if not dfs:
    print("Error: no se encontraron CSVs.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df)} secciones de {df['doc_id'].nunique()} documentos")
print(f"Distribución: {df['corpus_type'].value_counts().to_dict()}\n")

# Estimar modelo
model = CFHSEMModel(spec=args.spec)

if args.multigroup:
    results = model.fit_multigroup(df, group_col="corpus_type")
    for group, res in results.items():
        print(f"\n{'='*40}\nGrupo: {group}")
        print(res.summary())
else:
    results = model.fit(df)
    print(results.summary())

    if args.bootstrap > 0:
        print(f"\nEjecutando bootstrap ({args.bootstrap} muestras)...")
        beta_mean, ci_low, ci_high = model.bootstrap_beta23(
            df, n_samples=args.bootstrap
        )
        print(f"β₂₃ bootstrap: media={beta_mean:.3f} IC95%=[{ci_low:.3f}, {ci_high:.3f}]")
        print(f"H₃: {'APOYADA ✓' if ci_high < 0 else 'NO APOYADA ✗'} (IC excluye 0)")
