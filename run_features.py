"""
CFH · Script de extracción de features
Ejecutar desde la raíz del proyecto:
    python run_features.py --corpus B
    python run_features.py --corpus A
    python run_features.py --corpus all
"""
import sys
from pathlib import Path

# Agregar code/src al path para que los imports funcionen
sys.path.insert(0, str(Path(__file__).parent / "code" / "src"))

import argparse
from features.feature_pipeline import FeaturePipeline

parser = argparse.ArgumentParser(description="CFH Feature Pipeline")
parser.add_argument("--corpus", choices=["A", "B", "all"], default="B")
parser.add_argument("--corpus-a-dir", type=Path, default=Path("data/processed/corpus_a"))
parser.add_argument("--corpus-b-dir", type=Path, default=Path("data/processed/corpus_b_json"))
parser.add_argument("--output-dir", type=Path, default=Path("data/features"))
parser.add_argument("--max-docs", type=int, default=None)
parser.add_argument("--use-ebi", action="store_true")
args = parser.parse_args()

pipeline = FeaturePipeline(use_ebi=args.use_ebi)

if args.corpus == "A":
    df = pipeline.run_corpus_a(
        corpus_dir=args.corpus_a_dir,
        output_path=args.output_dir / "indicators_corpus_a.csv",
        max_docs=args.max_docs,
    )
    print(f"\n✓ Corpus A: {len(df)} secciones")

elif args.corpus == "B":
    df = pipeline.run_corpus_b(
        corpus_dir=args.corpus_b_dir,
        output_path=args.output_dir / "indicators_corpus_b.csv",
        max_docs=args.max_docs,
    )
    print(f"\n✓ Corpus B: {len(df)} secciones")
    if len(df) > 0:
        print(df[["corpus_type", "section_id", "y2_sa", "y4_nv", "y10_rep"]].to_string())

else:
    df = pipeline.run_all(
        corpus_a_dir=args.corpus_a_dir,
        corpus_b_dir=args.corpus_b_dir,
        output_path=args.output_dir / "indicators_all.csv",
    )
    print(f"\n✓ Corpus completo: {len(df)} secciones")
