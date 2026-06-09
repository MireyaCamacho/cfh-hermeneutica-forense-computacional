"""Explorar CSVs con indicadores semánticos"""
import pandas as pd

for path in [
    'data/features/indicators_corpus_c.csv',
    'data/features/indicators_completo_conflibert.csv',
    'data/features/indicators_corpus_b_v2.csv',
]:
    try:
        df = pd.read_csv(path)
        cols_sem = [c for c in df.columns if any(x in c.lower()
                    for x in ['y8','y9','mafapo','cidh','cs','confli','embed'])]
        print(f"\n{path}")
        print(f"  Shape: {df.shape}")
        if 'corpus_type' in df.columns:
            print(f"  corpus_type: {df['corpus_type'].value_counts().to_dict()}")
        elif 'audio' in df.columns:
            print(f"  audio: {df['audio'].value_counts().to_dict()}")
        print(f"  cols semánticas: {cols_sem}")
    except Exception as e:
        print(f"\n{path} → ERROR: {e}")
