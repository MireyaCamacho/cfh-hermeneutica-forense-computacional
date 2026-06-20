"""
CFH — Auditoría OpenFace vs MediaPipe en el código
"""
import os, re

BASE = r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional"
PATRON = re.compile(r'OpenFace|MediaPipe|mediapipe|FaceLandmarker|blendshape', re.IGNORECASE)

print("=== MENCIONES EN CÓDIGO PYTHON ===\n")
for root, dirs, files in os.walk(BASE):
    # Saltar carpetas de cache y node_modules
    dirs[:] = [d for d in dirs if d not in ['__pycache__', 'node_modules', '.git']]
    for f in files:
        if f.endswith('.py'):
            ruta = os.path.join(root, f)
            try:
                txt = open(ruta, encoding='utf-8', errors='ignore').read()
                matches = PATRON.findall(txt)
                if matches:
                    unicos = set(m.lower() for m in matches)
                    ruta_corta = ruta.replace(BASE, '').lstrip('\\')
                    print(f"{ruta_corta}")
                    print(f"  → {set(matches)}")
                    # Mostrar líneas relevantes
                    for i, linea in enumerate(txt.split('\n'), 1):
                        if PATRON.search(linea):
                            print(f"  L{i}: {linea.strip()[:100]}")
                    print()
            except:
                pass
