# -*- coding: utf-8 -*-
"""
cfh_cuadrar_huila.py
================================================================================
CFH — Cuadrar la marcación detallada de Huila con la diarización

OBJETO:
    Mireya marcó por video (letreros JEP) los tramos exactos de cada interviniente
    en la audiencia de reconocimiento del subcaso Huila (día 3, ~10.5 h, dos
    batallones: Cacique Pigoanza y Magdalena). Este script cruza esos tramos con
    la diarización para:
      (1) Asignar a cada compareciente su SPEAKER_XX por solapamiento temporal.
      (2) Marcar víctimas (CENTROIDE) y magistrados/representantes (EXCLUIR).
      (3) Generar inventario_Huila.csv anclado a la diarización, listo para el ICM.

NOTA SOBRE NOMBRES REPETIDOS:
    Hay comparecientes con doble intervención (turno de reconocimiento + turno de
    planes reparadores). Sus tramos se acumulan bajo la misma identidad para que
    el ICM los una. Ej.: Ricardo Andrés López García y José David Restrepo Solarte.

USO:
    cd "C:\\PROYECTOS 2026\\...\\CFH_Hermeneutica_Forense_Computacional"
    python "%USERPROFILE%\\Downloads\\cfh_cuadrar_huila.py"

REQUISITO: confirmar el nombre del archivo de diarización en DIAR (abajo).
Entorno: Python 3.11, conda env cfh.
================================================================================
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
# Candidatos de nombre de diarización — el script prueba en orden.
DIAR_CANDIDATOS = [
    BASE / "corpus_c" / "huila_diarization.json",
    BASE / "corpus_c" / "huila_reconocimiento_diarization.json",
    BASE / "corpus_c" / "huila" / "huila_diarization.json",
]
OUT = BASE / "data" / "marcacion" / "inventario_Huila.csv"
VIDEO_URL = "https://www.youtube.com/watch?v=huila_reconocimiento"


def hms(s):
    """'1:09:03' o '07:48' → segundos."""
    p = [int(x) for x in s.strip().split(":")]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]


# ─────────────────────────────────────────────────────────────────────────
# MARCACIÓN DETALLADA DE MIREYA (Huila) — (inicio, fin, identidad, rol, batallón)
# Roles: COMPARECIENTE → ANALISIS (ICM) | VICTIMA → CENTROIDE | resto → EXCLUIR
# ─────────────────────────────────────────────────────────────────────────
TRAMOS = [
    # ── Apertura y primeras víctimas ──
    ("00:13:28", "00:15:30", "Fabian Camilo Martinez Guerrero", "MAGISTRADO", ""),
    ("00:15:30", "00:17:41", "Jose Ricardo Cifuentes", "VICTIMA", ""),       # acto simbólico
    ("00:17:41", "00:18:50", "Alejandro Ramelli Arteaga", "MAGISTRADO", ""),
    ("00:24:50", "00:25:34", "Magistrada Rueda", "MAGISTRADO", ""),
    ("00:25:34", "00:42:11", "William Adolfo Sanchez Sarria", "VICTIMA", ""),
    ("00:42:11", "00:46:38", "Magistrada Rueda", "MAGISTRADO", ""),
    ("00:46:38", "01:02:36", "Luz Marina Castillo Tovar", "VICTIMA", ""),
    ("01:02:36", "01:05:32", "Magistrada Rueda", "MAGISTRADO", ""),
    # ── Comparecientes bloque 1 ──
    ("01:05:32", "01:40:14", "Faiver Coronado Camero", "COMPARECIENTE", "Cacique Pigoanza"),
    ("01:40:14", "01:45:31", "Magistrada Rueda", "MAGISTRADO", ""),
    ("01:45:31", "02:08:15", "Jhon Jairo Buitrago Lopez", "COMPARECIENTE", "Magdalena"),
    ("02:08:15", "02:21:06", "Jair Fernandez", "COMPARECIENTE", "Cacique Pigoanza"),
    ("02:21:06", "02:26:00", "Magistrada Rueda", "MAGISTRADO", ""),
    # receso 2:26–2:58
    ("02:58:00", "02:59:10", "Fabian Camilo Martinez Guerrero", "MAGISTRADO", ""),
    ("02:59:10", "03:07:18", "Hilton Santiago Gomez Avila", "VICTIMA", ""),  # sobrino víctima
    ("03:07:18", "03:08:00", "Alejandro Ramelli Arteaga", "MAGISTRADO", ""),
    ("03:08:00", "03:34:28", "Claudia Rocio Lugo Moreno", "VICTIMA", ""),
    ("03:34:28", "03:39:44", "Alejandro Ramelli Arteaga", "MAGISTRADO", ""),
    # ── Comparecientes bloque 2 ──
    ("03:39:44", "04:17:03", "Luis Carlos Aguilera Quintero", "COMPARECIENTE", "Cacique Pigoanza"),
    ("04:17:29", "04:35:09", "Aris Ramirez Campos", "COMPARECIENTE", "Cacique Pigoanza"),
    ("04:35:55", "04:43:58", "Fabian Fermin Durango de la Cruz", "COMPARECIENTE", "Magdalena"),
    # receso 4:44–5:53
    ("05:53:16", "05:54:48", "Alejandro Ramelli Arteaga", "MAGISTRADO", ""),
    # ── Comparecientes bloque 3 ──
    ("05:54:48", "06:01:58", "Angel Fernando Carvajal Rojas", "COMPARECIENTE", "Magdalena"),
    ("06:02:12", "06:08:50", "Ricardo Andres Lopez Garcia", "COMPARECIENTE", "Cacique Pigoanza"),
    ("06:09:05", "06:18:31", "Jair Arias Sanchez", "COMPARECIENTE", "Cacique Pigoanza"),
    ("06:18:46", "06:28:15", "Julian Andres Calderon Motta", "COMPARECIENTE", "Magdalena"),
    ("06:28:43", "06:39:18", "Fernando Riveros Sarmiento", "COMPARECIENTE", "Magdalena"),
    ("06:39:36", "06:43:43", "Jose Yati Anacona Bueno", "COMPARECIENTE", "Magdalena"),
    ("06:44:00", "06:50:40", "Jose Roldan Lopez Ceron", "COMPARECIENTE", "Magdalena"),
    ("06:50:55", "06:58:44", "Francisco Javier Castañeda Alfaro", "COMPARECIENTE", "Magdalena"),
    ("06:59:14", "07:02:10", "Annuar Herrera Osorio", "COMPARECIENTE", "Cacique Pigoanza"),
    ("07:02:23", "07:06:49", "Jose Albeiro Bustos Aguilar", "COMPARECIENTE", "Cacique Pigoanza"),
    ("07:07:01", "07:09:21", "Hernando Mendez Rodriguez", "COMPARECIENTE", "Cacique Pigoanza"),
    ("07:09:32", "07:14:25", "Jose David Restrepo Solarte", "COMPARECIENTE", "Magdalena"),
    ("07:14:40", "07:21:33", "Luis Carlos Oyola Tapia", "COMPARECIENTE", "Magdalena"),
    ("07:21:46", "07:30:33", "Cesar Augusto Vasquez Ordoñez", "COMPARECIENTE", "Magdalena"),
    ("07:30:58", "07:33:16", "Harbey Sanchez Gomez", "COMPARECIENTE", "Magdalena"),
    ("07:33:37", "07:36:59", "Jose Gañan Tapasco", "COMPARECIENTE", "Magdalena"),
    ("07:37:14", "07:41:55", "Wilder Samboni Chanchi", "COMPARECIENTE", "Magdalena"),
    ("07:42:10", "07:48:12", "Fabio Nelson Rodriguez Barrera", "COMPARECIENTE", "Magdalena"),
    ("07:48:21", "07:51:35", "Jose Fabio Guzman Muñoz", "COMPARECIENTE", "Magdalena"),
    ("07:51:53", "07:56:27", "Divar Juspian Jimenez", "COMPARECIENTE", "Magdalena"),
    # ── Ministra / representantes (EXCLUIR) ──
    ("07:57:08", "08:03:48", "Angela Maria Buitrago", "MINISTRA", ""),
    # receso 8:04–8:35
    ("08:35:45", "08:36:44", "Fabian Camilo Martinez Guerrero", "MAGISTRADO", ""),
    ("08:36:44", "08:45:32", "Alejandra Ballesteros", "REPRESENTANTE_VICTIMAS", ""),  # Colectivo Abogados
    ("08:46:02", "08:50:01", "Oscar Fernando Sierra", "REPRESENTANTE_VICTIMAS", ""),  # Observatorio Sur
    ("08:51:07", "08:56:52", "Gloria Orcue", "REPRESENTANTE_VICTIMAS", ""),           # Comisión Intereclesial
    # ── Planes reparadores (comparecientes, 2º turno: se acumula) ──
    ("08:57:15", "09:07:20", "Ricardo Andres Lopez Garcia", "COMPARECIENTE", "Cacique Pigoanza"),
    ("09:07:48", "09:15:20", "Jose David Restrepo Solarte", "COMPARECIENTE", "Magdalena"),
    ("09:15:42", "09:25:25", "Claudia Rocio Saldaña Montoya", "MAGISTRADO", ""),
    ("09:25:40", "09:36:35", "Jose Miller Hormiga", "MAGISTRADO", ""),
    # ── Video víctimas + actos simbólicos finales (todas VICTIMA/CENTROIDE) ──
    ("09:36:56", "09:47:45", "Video victimas Huila", "VICTIMA", ""),
    ("09:48:29", "09:57:16", "Alejandro Ramelli Arteaga", "MAGISTRADO", ""),
    ("09:57:55", "10:02:24", "Acto simbolico poema victima", "VICTIMA", ""),
    ("10:02:50", "10:17:45", "Actos simbolicos victimas", "VICTIMA", ""),
    ("10:18:20", "10:30:48", "Entrega de cartas a magistratura", "VICTIMA", ""),
]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def cargar_diar():
    for p in DIAR_CANDIDATOS:
        if p.exists():
            print(f"  Diarización: {p.name}")
            diar = json.load(open(p, encoding="utf-8"))
            if isinstance(diar, dict):
                for v in diar.values():
                    if isinstance(v, list):
                        return v
            return diar
    print("  [ERROR] No encontré la diarización de Huila. Candidatos probados:")
    for p in DIAR_CANDIDATOS:
        print(f"     - {p}")
    print("  Edita DIAR_CANDIDATOS con el nombre correcto y vuelve a correr.")
    return None


def main():
    print("CFH — Cuadrar marcación de Huila con diarización")
    print("="*72)

    diar = cargar_diar()
    if diar is None:
        return

    USO = {"COMPARECIENTE":"ANALISIS", "VICTIMA":"CENTROIDE",
           "MAGISTRADO":"EXCLUIR", "MINISTRA":"EXCLUIR",
           "REPRESENTANTE_VICTIMAS":"EXCLUIR"}

    # Correspondencia identidad → SPEAKER (solo comparecientes interesan para ICM).
    ident_speaker = defaultdict(Counter)
    ident_tiempo = defaultdict(float)
    ident_rol = {}
    for ini, fin, ident, rol, bat in TRAMOS:
        a0, a1 = hms(ini), hms(fin)
        ident_tiempo[ident] += (a1 - a0)
        ident_rol[ident] = rol
        for d in diar:
            ov = overlap(a0, a1, float(d["start"]), float(d["end"]))
            if ov > 0:
                ident_speaker[ident][d["speaker"]] += ov

    print("\n── Comparecientes (ANALISIS) → SPEAKER por solapamiento ──")
    ident_to_spk = {}
    comparecientes = [(i,t) for i,t in ident_tiempo.items()
                      if USO.get(ident_rol[i]) == "ANALISIS"]
    for ident, t in sorted(comparecientes, key=lambda x:-x[1]):
        spks = ident_speaker[ident]
        if spks:
            top = spks.most_common(2)
            spk = top[0][0]; total = sum(spks.values()); pct = 100*top[0][1]/total
            ident_to_spk[ident] = spk
            flag = "" if t/60 >= 8 else "  ⚠ corto (posible cobertura insuficiente)"
            print(f"  {ident[:34]:34s} t={t/60:5.1f}min → {spk} ({pct:.0f}%){flag}")
        else:
            print(f"  {ident[:34]:34s} sin solapamiento")

    # Construir inventario (una fila por tramo).
    filas = []
    for ini, fin, ident, rol, bat in TRAMOS:
        a0, a1 = hms(ini), hms(fin)
        best, bestov = "", 0
        for d in diar:
            ov = overlap(a0, a1, float(d["start"]), float(d["end"]))
            if ov > bestov:
                bestov, best = ov, d["speaker"]
        filas.append({
            "audiencia":"Huila", "video_url":VIDEO_URL,
            "speaker_diar":best, "inicio":ini, "fin":fin,
            "tiempo_total_s":round(a1-a0,1),
            "identidad":ident, "rol":rol, "batallon":bat,
            "evidencia_identificacion":"letrero JEP + video",
            "confianza":"ALTA",
            "uso":USO.get(rol,"EXCLUIR"), "muestra_texto":"",
        })

    df = pd.DataFrame(filas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    # Resumen.
    n_comp = df[df["uso"]=="ANALISIS"]["identidad"].nunique()
    n_vict = df[df["uso"]=="CENTROIDE"]["identidad"].nunique()
    n_excl = df[df["uso"]=="EXCLUIR"]["identidad"].nunique()
    print(f"\n[GUARDADO] {OUT}")
    print(f"  {len(df)} tramos | comparecientes únicos (ANALISIS): {n_comp}")
    print(f"  víctimas (CENTROIDE): {n_vict} | excluidos: {n_excl}")

    # Comparecientes por tiempo (los <8 min probablemente queden fuera por pisos).
    tiempos = (df[df["uso"]=="ANALISIS"].groupby("identidad")["tiempo_total_s"]
               .sum().sort_values(ascending=False))
    solidos = (tiempos/60 >= 8).sum()
    print(f"\n  Comparecientes con ≥8 min (ICM sólido probable): {solidos}")
    print(f"  Comparecientes con <8 min (posible cobertura insuficiente): {len(tiempos)-solidos}")
    print(f"\n  Top por tiempo:")
    for ident, t in tiempos.head(8).items():
        print(f"    {ident[:34]:34s} {t/60:5.1f} min")


if __name__ == "__main__":
    main()
