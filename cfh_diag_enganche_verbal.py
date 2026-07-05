# -*- coding: utf-8 -*-
"""
cfh_diag_enganche_verbal.py
============================
Diagnostica POR QUE y10_rep=0.000 para algunos comparecientes.

NO hardcodea ningun compareciente. Recorre TODOS los subcasos y verifica
la cadena de enganche que usa cfh_icm_tricanal_final.py:

    segments (texto+tiempo)  x  marcacion (identidad+tiempo)  ->  texto por persona

El bug tipico: los segments.json usan un nombre de campo temporal distinto
al que espera el pipeline ('start'/'end'). Si el pipeline hace
s.get('start', 0) y el campo real es 'inicio' o 'start_time', TODOS los
segments quedan en 0 -> overlap 0 -> texto vacio -> REP 0.

Este script revela, por cada subcaso:
  - que claves tienen realmente los segments
  - si 'start'/'end' existen y estan poblados
  - cuanto texto recuperaria CADA compareciente con el enganche actual
  - cuantos comparecientes quedan con texto vacio (los falsos negativos)

Uso:
    python cfh_diag_enganche_verbal.py
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
PREFIJO = {"Catatumbo": "catatumbo", "Dabeiba": "dabeiba", "Casanare": "casanare",
           "Huila": "huila", "CostaCaribe": "costa_caribe"}
USOS_ICM = {"ANALISIS"}
ROLES_ICM = {"COMPARECIENTE"}
NO_PERSONAS = {"BLOQUE_COMPARECIENTES", "BLOQUE_COMPARECIENTES_NO_MR"}
MIN_OVERLAP_S = 0.5


def t_a_seg(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return p[0]*3600+p[1]*60+p[2] if len(p) == 3 else p[0]*60+p[1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def cargar_json_lista(ruta):
    d = json.load(open(ruta, encoding="utf-8"))
    if isinstance(d, dict):
        for k in ["segments", "segmentos", "results", "chunks"]:
            if k in d and isinstance(d[k], list):
                return d[k]
        for v in d.values():
            if isinstance(v, list):
                return v
    return d if isinstance(d, list) else []


def buscar_archivo(pref, sufijo):
    hits = glob.glob(str(BASE / "corpus_c" / f"{pref}*{sufijo}.json"))
    return Path(sorted(hits, key=len)[0]) if hits else None


def cargar_intervalos(ruta_marc):
    m = pd.read_csv(ruta_marc)
    m["ini_s"] = m["inicio"].apply(t_a_seg)
    m["fin_s"] = m["fin"].apply(t_a_seg)
    cond = (m["uso"].astype(str).str.upper().isin(USOS_ICM)) | \
           (m["rol"].astype(str).str.upper().isin(ROLES_ICM))
    m = m[cond & m["ini_s"].notna() & m["fin_s"].notna() & (m["fin_s"] > m["ini_s"])]
    m = m[~m["identidad"].astype(str).str.upper().isin(NO_PERSONAS)]
    return m.reset_index(drop=True)


# campos temporales alternativos que podrian usar los segments
CAMPOS_START = ["start", "inicio", "start_time", "from", "ini", "begin", "t0", "startTime"]
CAMPOS_END = ["end", "fin", "end_time", "to", "t1", "endTime"]
CAMPOS_TEXT = ["text", "texto", "transcript", "content", "value"]


def detectar_campo(seg, candidatos):
    for c in candidatos:
        if c in seg and seg[c] is not None:
            return c
    return None


def diag_subcaso(subcaso):
    pref = PREFIJO.get(subcaso, subcaso.lower())
    print(f"\n{'='*72}\nSUBCASO: {subcaso}\n{'='*72}")

    ruta_marc = BASE / "data" / "marcacion" / f"inventario_{subcaso}.csv"
    if not ruta_marc.exists():
        print("  [SALTADO] sin marcacion")
        return

    r_seg = buscar_archivo(pref, "segments")
    if r_seg is None:
        print("  [SALTADO] sin segments.json")
        return

    segs = cargar_json_lista(r_seg)
    print(f"  segments.json: {r_seg.name}  ({len(segs)} segmentos)")
    if not segs:
        print("  [!!] segments vacio")
        return

    # claves reales del primer segmento
    s0 = segs[0]
    print(f"  claves de un segmento: {list(s0.keys())}")

    # detectar que campos temporales/texto usa REALMENTE
    campo_start = detectar_campo(s0, CAMPOS_START)
    campo_end = detectar_campo(s0, CAMPOS_END)
    campo_text = detectar_campo(s0, CAMPOS_TEXT)
    print(f"  campo START detectado: {campo_start!r}  (pipeline espera 'start')")
    print(f"  campo END   detectado: {campo_end!r}  (pipeline espera 'end')")
    print(f"  campo TEXT  detectado: {campo_text!r}  (pipeline espera 'text')")

    # DIAGNOSTICO CLAVE: el pipeline hace s.get('start', 0).
    # Si el campo real NO es 'start', todos quedan en 0.
    usa_start_literal = "start" in s0 and s0.get("start") is not None
    usa_end_literal = "end" in s0 and s0.get("end") is not None
    usa_text_literal = "text" in s0 and s0.get("text") is not None
    print(f"\n  >> segments tienen 'start' literal poblado: {usa_start_literal}")
    print(f"  >> segments tienen 'end'   literal poblado: {usa_end_literal}")
    print(f"  >> segments tienen 'text'  literal poblado: {usa_text_literal}")

    if not (usa_start_literal and usa_end_literal and usa_text_literal):
        print("  [!! BUG] el pipeline usa s.get('start'/'end'/'text', 0) pero el")
        print("           campo real es otro -> overlap=0 -> texto vacio -> REP=0")

    # rango temporal de los segments segun campo real
    if campo_start and campo_end:
        starts = [float(s.get(campo_start, 0) or 0) for s in segs]
        ends = [float(s.get(campo_end, 0) or 0) for s in segs]
        print(f"\n  rango temporal segments (campo real): "
              f"{min(starts):.0f}s .. {max(ends):.0f}s")

    # marcacion
    intervalos = cargar_intervalos(ruta_marc)
    print(f"  comparecientes marcados: {intervalos['identidad'].nunique()}")
    print(f"  rango temporal marcacion: "
          f"{intervalos['ini_s'].min():.0f}s .. {intervalos['fin_s'].max():.0f}s")

    # SIMULAR el enganche de DOS formas: (A) como el pipeline actual (campo 'start'),
    # (B) con el campo real detectado. Comparar cuanto texto recupera cada persona.
    def recuperar_texto(campo_s, campo_e, campo_t):
        out = {}
        for _, r in intervalos.iterrows():
            a0, a1 = r["ini_s"], r["fin_s"]
            partes = []
            for s in segs:
                s0v = float(s.get(campo_s, 0) or 0) if campo_s else 0.0
                s1v = float(s.get(campo_e, 0) or 0) if campo_e else 0.0
                if s1v <= s0v:
                    continue
                if overlap(a0, a1, s0v, s1v) >= MIN_OVERLAP_S:
                    txt = str(s.get(campo_t, "") if campo_t else "").strip()
                    if txt:
                        partes.append(txt)
            ident = r["identidad"]
            out.setdefault(ident, []).append(" ".join(partes))
        return {k: " ".join(v).strip() for k, v in out.items()}

    print("\n  --- COMPARACION DE ENGANCHE (chars de texto por compareciente) ---")
    print(f"  {'compareciente':40s} {'pipeline(start)':>16s} {'campo_real':>16s}")
    texto_actual = recuperar_texto("start", "end", "text")
    texto_real = recuperar_texto(campo_start, campo_end, campo_text)
    ceros_actual = 0
    for ident in sorted(set(texto_actual) | set(texto_real)):
        na = len(texto_actual.get(ident, ""))
        nr = len(texto_real.get(ident, ""))
        if na == 0:
            ceros_actual += 1
        flag = "  <-- FN" if (na == 0 and nr > 0) else ""
        print(f"  {str(ident)[:40]:40s} {na:>16d} {nr:>16d}{flag}")

    print(f"\n  comparecientes con texto VACIO en pipeline actual: {ceros_actual}")
    if ceros_actual > 0 and campo_start != "start":
        print(f"  >> CONFIRMADO: el campo real es '{campo_start}', no 'start'.")
        print(f"     Ese es el bug del enganche verbal. Corrigiendolo, el texto")
        print(f"     llega y el REP deja de ser 0 para esos comparecientes.")


def main():
    print("DIAGNOSTICO DEL ENGANCHE VERBAL (segments x marcacion)")
    print("Reproducible, sin hardcodear comparecientes.\n")
    for sc in ["Catatumbo", "Dabeiba", "Casanare", "Huila", "CostaCaribe"]:
        try:
            diag_subcaso(sc)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR en {sc}] {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*72}")
    print("LECTURA: si algun subcaso muestra 'campo real' != 'start' y comparecientes")
    print("con texto vacio en el pipeline actual, ese es el bug. La correccion va en")
    print("cfh_icm_tricanal_final.py -> texto_por_compareciente (y en la lectura de")
    print("diarizacion/facial si usan el mismo patron s.get('start')).")


if __name__ == "__main__":
    main()
