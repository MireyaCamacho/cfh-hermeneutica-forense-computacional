# -*- coding: utf-8 -*-
"""
cfh_diag_sospechosos_huila.py
================================================================================
CFH — Diagnóstico caso por caso de valores ICM sospechosos en Huila

OBJETO:
    Antes de decidir si subimos pisos, examinar los comparecientes con valores
    extremos para distinguir ruido (datos escasos) de señal real:
      · Harbey Sánchez: ICM=0.757, rep=1.0, f=0.947 → ¿artefacto?
      · Aris Ramírez / Aguilera: facial ~0.01 con muchos frames → ¿detección mala?
      · varios rep=0.000 → ¿real o extractor falló?

QUÉ MUESTRA por compareciente sospechoso:
    1. Texto real que dijo (primeras frases) → para juzgar si rep=1.0 o rep=0.0 es creíble
    2. Distribución de AUs faciales → si f≈0 con frames, ver si AUs son todos ~0
    3. Nº efectivo de datos por canal

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_diag_sospechosos_huila.py"

Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
MARC = BASE / "data" / "marcacion" / "inventario_Huila.csv"
SEG = BASE / "corpus_c" / "huila_segments.json"
EG = BASE / "outputs" / "capa3" / "egemap_huila.csv"

# Comparecientes a examinar (los de valores extremos).
SOSPECHOSOS = [
    "Harbey Sanchez Gomez",          # ICM=0.757 rep=1.0 f=0.947
    "Luis Carlos Aguilera Quintero", # f=0.006 con 547 frames
    "Aris Ramirez Campos",           # f=0.012 con 277 frames
    "Annuar Herrera Osorio",         # f=0.609 con solo 10 frames
    "Jair Arias Sanchez",            # rep=0.665
    "Fernando Riveros Sarmiento",    # rep=0.769
]


def hms(s):
    p = [int(x) for x in str(s).strip().split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


def main():
    print("CFH — Diagnóstico de sospechosos en Huila")
    print("="*72)

    marc = pd.read_csv(MARC)
    marc = marc[marc["uso"] == "ANALISIS"].copy()
    marc["ini_s"] = marc["inicio"].apply(hms)
    marc["fin_s"] = marc["fin"].apply(hms)

    segs = json.load(open(SEG, encoding="utf-8"))
    if isinstance(segs, dict):
        for v in segs.values():
            if isinstance(v, list):
                segs = v; break

    # AUs de Huila (todos los archivos).
    aus_files = glob.glob(str(BASE/"outputs"/"capa3"/"aus_huila*.csv"))
    aus = pd.concat([pd.read_csv(f) for f in aus_files], ignore_index=True)
    aus = aus.drop_duplicates(["speaker","start"]) if "speaker" in aus.columns else aus

    eg = pd.read_csv(EG) if EG.exists() else pd.DataFrame()

    for nombre in SOSPECHOSOS:
        sub_m = marc[marc["identidad"] == nombre]
        if sub_m.empty:
            print(f"\n[{nombre}] no está en la marcación."); continue
        print(f"\n{'─'*72}\n{nombre}")
        tramos = [(r["ini_s"], r["fin_s"]) for _, r in sub_m.iterrows()]
        dur_total = sum(b-a for a,b in tramos)
        print(f"  Tramos: {len(tramos)} | duración total: {dur_total/60:.1f} min")

        # 1) TEXTO real
        textos = []
        for a0, a1 in tramos:
            for s in segs:
                s0 = float(s.get("start",0)); s1 = float(s.get("end",0))
                if s1 > a0 and s0 < a1:
                    t = s.get("text","").strip()
                    if t: textos.append(t)
        full = " ".join(textos)
        ntok = len(full.split())
        print(f"  Texto: {ntok} tokens")
        print(f"    «{full[:240]}{'...' if len(full)>240 else ''}»")

        # 2) AUs faciales en esos tramos
        frames = []
        for a0, a1 in tramos:
            f = aus[(aus["end"] > a0) & (aus["start"] < a1)]
            frames.append(f)
        if frames:
            fr = pd.concat(frames)
            if len(fr):
                aucols = [c for c in ["AU1","AU4","AU12","AU15","AU17"] if c in fr.columns]
                medias = {c: fr[c].mean() for c in aucols}
                print(f"  Facial: {len(fr)} frames | AUs medios: " +
                      ", ".join(f"{c}={v:.3f}" for c,v in medias.items()))
                distress = np.mean([fr[c].mean() for c in ["AU1","AU4","AU15","AU17"] if c in fr.columns])
                sonrisa = fr["AU12"].mean() if "AU12" in fr.columns else 0
                print(f"    → distress medio={distress:.3f} | sonrisa AU12={sonrisa:.3f}")
                if distress < 0.02 and sonrisa < 0.02:
                    print(f"    ⚠ AUs casi todos ~0: rostro no detectado bien o totalmente neutro")
                elif sonrisa > 0.5:
                    print(f"    ⚠ sonrisa muy alta → empuja ICM facial arriba artificialmente")
            else:
                print(f"  Facial: 0 frames en los tramos")

        # 3) Ventanas vocales
        if not eg.empty and "start_s" in eg.columns:
            nv = 0
            for a0, a1 in tramos:
                nv += len(eg[(eg["end_s"] > a0) & (eg["start_s"] < a1)])
            print(f"  Vocal: {nv} ventanas eGeMAPS (~{nv*10}s)")

    print(f"\n{'='*72}")
    print("INTERPRETACIÓN:")
    print("  · rep=1.0 con texto corto/genérico → artefacto del extractor (texto breve)")
    print("  · facial ~0 con muchos frames → rostro neutro real O mala detección")
    print("  · facial alto con <20 frames → poco robusto, alta varianza")
    print("  · revisar si el TEXTO justifica el rep extremo (alto o cero)")


if __name__ == "__main__":
    main()
