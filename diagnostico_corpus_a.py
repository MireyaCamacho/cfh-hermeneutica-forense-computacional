"""
CFH · Diagnóstico — filas mal clasificadas en corpus A
Ejecutar: python diagnostico_corpus_a.py
"""
import pandas as pd

df = pd.read_csv("data/features/indicators_corpus_a.csv")

print(f"Total filas corpus A: {len(df)}")
print(f"\nDistribución corpus_type:")
print(df["corpus_type"].value_counts())

# Filas clasificadas como B dentro del corpus A
mal_clasificadas = df[df["corpus_type"] == "B"]
print(f"\nFilas clasificadas como 'B' dentro del corpus A: {len(mal_clasificadas)}")
print("\nDetalle:")
print(mal_clasificadas[["doc_id", "corpus_type", "section_id", "y4_nv", "y10_rep"]].to_string())
