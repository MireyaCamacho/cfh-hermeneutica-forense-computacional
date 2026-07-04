# -*- coding: utf-8 -*-
"""
cfh_texto_por_compareciente.py
================================
PASO 1 del pipeline SEM Corpus C por compareciente.

Reconstruye el TEXTO completo de cada compareciente cruzando:
  - los segmentos Whisper con texto  (corpus_c/*_segments.json: start, end, text)
  - las marcas de tiempo VALIDADAS a mano por Mireya en el inventario de
    marcacion (data/marcacion/inventario_*.csv: inicio, fin, identidad, uso)

NO usa el speaker generico de la diarizacion (SPEAKER_00..). Usa los tramos
inicio-fin que Mireya ajusto minuto por minuto, que son la fuente de verdad.

LOGICA:
  Para cada fila del inventario con uso=ANALISIS (los comparecientes):
    - convierte inicio/fin (HH:MM:SS) a segundos
    - recoge todos los segmentos cuyo CENTRO temporal cae dentro del tramo
    - concatena su texto -> texto de ese compareciente en ese tramo
  Un compareciente puede tener varios tramos (varias filas): se unen todos.

SALIDA:
  data/texto_por_compareciente.csv  con columnas:
    subcaso, identidad, n_tramos, n_segmentos, n_tokens_aprox, texto_completo
  (solo comparecientes uso=ANALISIS; magistrados/victimas EXCLUIR quedan fuera)

Es un paso LOCAL, no requiere GPU ni Colab. Es la base para:
  - y11 (embeddings del texto -> convergencia al centroide restaurativo)  [Colab]
  - y12 (appraisal: juicio sobre el texto por persona)                    [local]

USO:
    python cfh_texto_por_compareciente.py            # dry-run (no escribe)
    python cfh_texto_por_compareciente.py --escribir # genera el CSV
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(".")
MARCACION = BASE / "data" / "marcacion"
CORPUS_C = BASE / "corpus_c"
OUT = BASE / "data" / "texto_por_compareciente.csv"
ICM = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"


def norm_nombre(s):
    """Normaliza un nombre para cruce robusto (sin tildes, minusculas, espacios simples)."""
    import unicodedata
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s

# Mapeo subcaso -> (inventario, segments)
MAPEO = {
    "Casanare":    ("inventario_Casanare.csv",    "casanare_torres_segments.json"),
    "Catatumbo":   ("inventario_Catatumbo.csv",   "catatumbo_audiencia_reconocimiento_segments.json"),
    "CostaCaribe": ("inventario_CostaCaribe.csv", "costa_caribe_segments.json"),
    "Dabeiba":     ("inventario_Dabeiba.csv",     "dabeiba_antioquia_segments.json"),
    "Huila":       ("inventario_Huila.csv",       "huila_segments.json"),
}


def t_a_seg(v):
    """HH:MM:SS o MM:SS o segundos -> segundos (float)."""
    if pd.isna(v):
        return None
    s = str(v).strip().replace(",", ".")
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        if len(p) == 3:
            return p[0] * 3600 + p[1] * 60 + p[2]
        if len(p) == 2:
            return p[0] * 60 + p[1]
    try:
        return float(s)
    except ValueError:
        return None


def cargar_segments(path):
    d = json.load(open(path, encoding="utf-8"))
    segs = d if isinstance(d, list) else list(d.values())[0]
    out = []
    for s in segs:
        st = t_a_seg(s.get("start"))
        en = t_a_seg(s.get("end"))
        tx = str(s.get("text", "")).strip()
        if st is not None and en is not None and tx:
            out.append((st, en, tx))
    return out


def procesar_subcaso(subcaso, inv_file, seg_file):
    inv_path = MARCACION / inv_file
    seg_path = CORPUS_C / seg_file
    if not inv_path.exists():
        print(f"  [ERROR] no existe {inv_path}")
        return []
    if not seg_path.exists():
        print(f"  [ERROR] no existe {seg_path}")
        return []

    m = pd.read_csv(inv_path)
    segs = cargar_segments(seg_path)
    print(f"  {subcaso}: {len(m)} filas inventario | {len(segs)} segmentos con texto")

    # solo comparecientes (uso = ANALISIS)
    usoc = m.get("uso", pd.Series([""] * len(m))).astype(str).str.upper()
    comp = m[usoc.eq("ANALISIS")].copy()
    print(f"    comparecientes (ANALISIS): {comp['identidad'].nunique()} personas, {len(comp)} tramos")

    filas = []
    for ident in comp["identidad"].dropna().unique():
        tramos = []
        for _, r in comp[comp["identidad"] == ident].iterrows():
            ini = t_a_seg(r.get("inicio"))
            fin = t_a_seg(r.get("fin"))
            if ini is not None and fin is not None and fin > ini:
                tramos.append((ini, fin))
        if not tramos:
            print(f"    [aviso] {ident}: sin tramos validos")
            continue
        # recoger segmentos cuyo centro cae en algun tramo
        textos = []
        n_seg = 0
        for st, en, tx in segs:
            c = (st + en) / 2.0
            if any(a <= c <= b for a, b in tramos):
                textos.append(tx)
                n_seg += 1
        texto = " ".join(textos).strip()
        texto = re.sub(r"\s+", " ", texto)
        n_tok = len(texto.split())
        filas.append({
            "subcaso": subcaso,
            "identidad": ident,
            "n_tramos": len(tramos),
            "n_segmentos": n_seg,
            "n_tokens_aprox": n_tok,
            "texto_completo": texto,
        })
        print(f"    {ident[:34]:34s} tramos={len(tramos)} seg={n_seg:4d} tokens={n_tok}")
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escribir", action="store_true", help="genera el CSV (por defecto dry-run)")
    args = ap.parse_args()

    print("=" * 72)
    print("PASO 1 — Reconstruccion de texto por compareciente (marcas validadas)")
    print("=" * 72)

    todas = []
    for subcaso, (inv, seg) in MAPEO.items():
        todas.extend(procesar_subcaso(subcaso, inv, seg))

    df = pd.DataFrame(todas)
    print("\n" + "=" * 72)
    print(f"Comparecientes reconstruidos (bruto): {len(df)}")

    # --- Filtrar a los 47 del ICM tri-canal (unidad consolidada) ---
    if not ICM.exists():
        print(f"  [ERROR] no existe {ICM} — no puedo alinear con los 47 del ICM.")
        return
    icm = pd.read_csv(ICM)
    icm_ident = set(norm_nombre(x) for x in icm["identidad"].dropna())
    print(f"  ICM tri-canal: {len(icm_ident)} comparecientes")

    df["_key"] = df["identidad"].apply(norm_nombre)
    en_icm = df[df["_key"].isin(icm_ident)].copy()
    # comparecientes del ICM que NO empataron con texto (para revisar)
    keys_texto = set(df["_key"])
    faltan = [x for x in icm["identidad"].dropna() if norm_nombre(x) not in keys_texto]

    print(f"\n  Empatados texto <-> ICM: {len(en_icm)} / {len(icm_ident)}")
    if faltan:
        print(f"  [aviso] {len(faltan)} del ICM sin texto reconstruido:")
        for f in faltan:
            print(f"     - {f}")
    # texto reconstruido que NO esta en el ICM (quedan fuera, es lo esperado)
    fuera = df[~df["_key"].isin(icm_ident)]
    if not fuera.empty:
        print(f"  ({len(fuera)} reconstruidos fuera del ICM — se excluyen, correcto)")

    df = en_icm.drop(columns=["_key"])
    print("\n" + "=" * 72)
    print(f"TOTAL a escribir (alineado con ICM): {len(df)}")
    if not df.empty:
        print(f"  tokens: min={df['n_tokens_aprox'].min()} "
              f"mediana={int(df['n_tokens_aprox'].median())} "
              f"max={df['n_tokens_aprox'].max()}")
        cortos = df[df["n_tokens_aprox"] < 100]
        if not cortos.empty:
            print(f"  [aviso] {len(cortos)} con <100 tokens (texto escaso para embeddings):")
            for _, r in cortos.iterrows():
                print(f"     {r['subcaso']}/{r['identidad'][:30]}: {r['n_tokens_aprox']} tokens")
        print(f"\n  por subcaso:")
        print(df.groupby('subcaso')['identidad'].count().to_string())

    if args.escribir:
        df.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"\n  [OK] escrito: {OUT}  ({len(df)} filas)")
    else:
        print(f"\n  [DRY-RUN] no se escribio nada. Repite con --escribir para generar:")
        print(f"            {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
