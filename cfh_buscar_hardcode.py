"""Extrae el bloque con datos hardcodeados de cfh_parsimonia_pesos.py"""
from pathlib import Path

REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
script = REPO / "cfh_parsimonia_pesos.py"

if not script.exists():
    print(f"No encontrado: {script}")
    # Buscar en code/
    for p in REPO.rglob("*parsimonia*"):
        print(f"Encontrado: {p}")
else:
    content = script.read_text(encoding='utf-8')
    # Buscar el bloque con corpus_ab o datos hardcodeados
    for keyword in ['corpus_ab', 'hardcode', 'valores escritos', 'A=', 'B=', 'mean_a', 'mean_b']:
        idx = content.find(keyword)
        if idx >= 0:
            print(f"\n=== '{keyword}' en línea {content[:idx].count(chr(10))+1} ===")
            print(content[max(0,idx-100):idx+400])
            print("...")
            break
