"""
CFH — Auditoría source_file en JSONs Corpus B
"""
import json
from pathlib import Path

JDIR = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\data\processed\corpus_b_json")
TXT_DIR = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\data\processed\corpus_b")

print(f"{'JSON':<55} {'source_file':<60} {'secciones':>9} {'TXT existe':>10}")
print("-" * 140)

total_secciones = 0
sin_txt = []

for f in sorted(JDIR.glob("*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        sf = d.get("source_file", "")
        secs = d.get("segmentation", {}).get("sections", [])
        n_secs = len(secs)
        total_secciones += n_secs

        # Verificar si el TXT existe
        txt_exists = False
        txt_path = Path(sf) if sf else None
        if txt_path and txt_path.exists():
            txt_exists = True
        elif sf:
            # Buscar solo por nombre de archivo en TXT_DIR
            nombre = Path(sf).name
            if not nombre.endswith(".txt"):
                nombre += ".txt"
            candidato = TXT_DIR / nombre
            txt_exists = candidato.exists()

        estado = "✓" if txt_exists else "✗ FALTA"
        if not txt_exists:
            sin_txt.append((f.name, sf, n_secs))

        print(f"{f.name:<55} {sf:<60} {n_secs:>9} {estado:>10}")
    except Exception as e:
        print(f"{f.name:<55} ERROR: {e}")

print()
print(f"Total secciones en JSONs: {total_secciones}")
print(f"JSONs sin TXT fuente: {len(sin_txt)}")
if sin_txt:
    print("\nDetalle de JSONs sin TXT:")
    for nombre, sf, n in sin_txt:
        print(f"  {nombre}: source_file={sf!r} ({n} secciones)")
