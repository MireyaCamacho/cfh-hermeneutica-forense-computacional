# -*- coding: utf-8 -*-
r"""
cfh_y11_densidad_restaurativa.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

REDISENO del indicador y11 (Transicion Epistemica, eta2).

PROBLEMA que resuelve:
    El y11 anterior (convergencia al centroide restaurativo) era otra
    distancia coseno a un centroide, igual que y8 (MAFAPO) e y9 (CIDH).
    Resultado: corr(y11,y8)=0.952 y corr(y11,y9)=0.972 -> multicolinealidad
    severa en el SEM (tres indicadores midiendo el mismo eje geometrico).

SOLUCION:
    y11 pasa de "distancia semantica" a "DENSIDAD DE ACTOS RESTAURATIVOS"
    (frecuencia de marcadores restaurativos por 100 palabras). Esto lo hace
    ortogonal a y8/y9 (que miden geometria de embeddings) y distinto de y10
    (que mide reconocimiento/restitucion/reparacion individual del acto).

DOS COMPONENTES (trazables por separado; y11 = promedio):
    y11a - DENSIDAD RESTAURATIVA INSTITUCIONAL (marco JEP / SIVJRNR / Ley 975)
           Vocabulario derivado del corpus curado v1 (niveles 2 y 3):
           reparacion integral/simbolica, garantias de no repeticion, TOAR,
           sancion propia, medidas de satisfaccion, restitucion/indemnizacion/
           rehabilitacion, funcion restauradora, contribucion a la verdad,
           regimen de condicionalidad, SIVJRNR.
    y11b - DENSIDAD DIALOGICA / TESTIMONIAL (Fricker + Zehr; nivel 1 curado)
           Dirigirse a las victimas, aceptar su narrativa, responder a sus
           preguntas, aporte dialogico a la verdad, nombrar el dolor causado.

    y11 = (y11a_norm + y11b_norm) / 2

    Se guardan y11a, y11b y y11 en columnas propias. Si el SEM muestra que
    los dos componentes difuminan eta2, se pueden separar sin recalcular.

METODO: densidad = (n_marcadores_ponderados / n_palabras) * 100
    Cada marcador aporta su peso; se sensibiliza a la longitud del texto
    (por 100 palabras) para no premiar a quien simplemente habla mas.

NORMALIZACION: min-max sobre los 47 comparecientes (universo del SEM),
    coherente con la decision de normalizar y10 sobre C (no A+B+C).

Ejecutar (raiz del repo, env cfh):
    python code\cfh_y11_densidad_restaurativa.py            # dry-run por defecto
    python code\cfh_y11_densidad_restaurativa.py --escribir # escribe el CSV SEM

Rutas por defecto (ajustables por CLI):
    --texto_c   data\texto_por_compareciente.csv
    --sem_csv   data\referencias\indicadores_sem_compareciente.csv
"""

import os
import re
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# VOCABULARIO y11a - RESTAURATIVO INSTITUCIONAL (marco JEP / SIVJRNR / Ley 975)
# Derivado del corpus_restaurativo_curado_v1 (niveles 2 y 3).
# Pesos: nivel 2 institucional = 1.8 ; nivel 3 normativo Ley 975 = 1.0
# (misma ponderacion confirmada por Mireya en la curaduria).
# ---------------------------------------------------------------------------
Y11A_PATRONES = [
    # --- nivel institucional (peso 1.8) ---
    (r"\breconocimiento\s+(?:public[oa]\s+)?de\s+responsabilidad(?:es)?", 1.8),
    (r"\bpetici[oó]n\s+de\s+perd[oó]n", 1.8),
    (r"\bacto(?:s)?\s+(?:tempran[oa]s?\s+)?de\s+reconocimiento", 1.8),
    (r"\bcontribuci[oó]n\s+a\s+la\s+verdad", 1.8),
    (r"\baporte\s+(?:al\s+esclarecimiento\s+)?(?:de\s+|a\s+la\s+)?verdad", 1.8),
    (r"\bfunci[oó]n\s+restaurador[a]", 1.8),
    (r"\breparaci[oó]n\s+integral", 1.8),
    (r"\bcompromiso\s+de\s+no\s+(?:incurrir|repetir|repetici[oó]n)", 1.8),
    (r"\bdeclaraci[oó]n\s+(?:public[a]\s+)?de\s+arrepentimiento", 1.8),
    (r"\bsanci[oó]n\s+propia", 1.8),
    (r"\btrabajos?\s+y?\s*obras?\s+(?:de\s+)?(?:reparaci[oó]n|restaurad)", 1.8),  # TOAR
    (r"\bproyecto(?:s)?\s+restaurador(?:es)?", 1.8),
    (r"\brégimen\s+de\s+condicionalidad", 1.8),
    # --- nivel normativo Ley 975 / estandar (peso 1.0) ---
    (r"\breparaci[oó]n\s+simb[oó]lica", 1.0),
    (r"\bgarant[ií]as?\s+de\s+no\s+repetici[oó]n", 1.0),
    (r"\bmedidas?\s+de\s+(?:reparaci[oó]n|satisfacci[oó]n|restituci[oó]n)", 1.0),
    (r"\b(?:restituci[oó]n|indemnizaci[oó]n|rehabilitaci[oó]n)\b", 1.0),
    (r"\bderecho(?:s)?\s+(?:de\s+las\s+v[ií]ctimas\s+)?a\s+la\s+(?:verdad|reparaci[oó]n)", 1.0),
    (r"\bSistema\s+Integral\s+de\s+Verdad", 1.0),
    (r"\bSIVJRNR\b", 1.0),
    (r"\bdesmovilizaci[oó]n\s+y\s+(?:el\s+)?desmantelamiento", 1.0),
    (r"\breparad[oa]s?\s+de\s+manera\s+(?:adecuada|transformadora|diferenciada|efectiva)", 1.0),
]

# ---------------------------------------------------------------------------
# VOCABULARIO y11b - DIALOGICO / TESTIMONIAL (Fricker + Zehr; nivel 1 curado)
# Dirigirse a las victimas, aceptar su narrativa, nombrar el dolor causado,
# aporte dialogico. Peso 1.8 (performativo, 1a persona) salvo formulas mas
# genericas (1.0).
# ---------------------------------------------------------------------------
Y11B_PATRONES = [
    # --- dirigirse directamente a la victima (peso 1.8) ---
    (r"\b(?:señora|señor|doña|don)\s*,?\s+(?:lamento|lamentamos|le\s+pido|"
     r"les\s+pido|quiero\s+decirle|quiero\s+pedirle)", 1.8),
    (r"\b(?:a\s+)?ustedes\s+(?:las?\s+)?v[ií]ctimas", 1.8),
    (r"\bme\s+dirijo\s+a\s+(?:usted|ustedes|las?\s+v[ií]ctimas|la\s+familia)", 1.8),
    (r"\ba\s+(?:usted|ustedes|la\s+familia|las\s+madres)\s+"
     r"(?:les?\s+)?(?:pido|ofrezco|debo)", 1.8),
    # --- nombrar y aceptar el dano/dolor (peso 1.8) ---
    (r"\b(?:reconozco|reconocer|reconocemos)\s+(?:ese\s+|el\s+)?dolor(?:\s+causado)?", 1.8),
    (r"\b(?:el\s+)?daño\s+causado(?:\s+a\s+(?:usted|ustedes|las?\s+v[ií]ctimas|la\s+familia))?", 1.5),
    (r"\btodo\s+el\s+daño\s+causado\s+y\s+nada\s+más\s+que\s+el\s+daño", 1.8),
    (r"\blamento\s+(?:mucho\s+|profundamente\s+)?(?:lo\s+ocurrido|el\s+daño|la\s+situaci[oó]n)", 1.8),
    (r"\bs[eé]\s+(?:el\s+)?dolor\s+que\s+(?:les?\s+)?(?:cause|caus[eé]|ocasion[eé])", 1.8),
    # --- aceptar la narrativa / responder a la victima (peso 1.5) ---
    (r"\b(?:acepto|acepta|aceptamos)\s+(?:lo\s+que\s+)?(?:dice|dicen|narr|cuentan)\s+"
     r"(?:la\s+|las\s+)?v[ií]ctima", 1.5),
    (r"\bresponder(?:\s+a)?\s+(?:las?\s+)?(?:preguntas?|necesidades?)\s+de\s+(?:las?\s+)?v[ií]ctimas", 1.5),
    (r"\btienen\s+(?:todo\s+el\s+)?derecho\s+a\s+saber", 1.0),
    (r"\bles?\s+debo\s+(?:la\s+)?verdad", 1.5),
]

_Y11A = [(re.compile(p, re.IGNORECASE), w) for p, w in Y11A_PATRONES]
_Y11B = [(re.compile(p, re.IGNORECASE), w) for p, w in Y11B_PATRONES]


def densidad(text, patrones):
    """Densidad ponderada de marcadores por 100 palabras."""
    if not text or not text.strip():
        return 0.0, 0
    n_palabras = max(1, len(text.split()))
    suma = 0.0
    n_hits = 0
    for rx, w in patrones:
        hits = len(rx.findall(text))
        if hits:
            suma += hits * w
            n_hits += hits
    return (suma / n_palabras) * 100.0, n_hits


def minmax(arr):
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto_c", default=os.path.join("data", "texto_por_compareciente.csv"))
    ap.add_argument("--sem_csv", default=os.path.join("data", "referencias", "indicadores_sem_compareciente.csv"))
    ap.add_argument("--escribir", action="store_true",
                    help="Escribe y11a/y11b/y11 en el CSV del SEM (con backup). Sin esto: dry-run.")
    args = ap.parse_args()

    if not os.path.exists(args.texto_c):
        raise SystemExit(f"No encuentro {args.texto_c}")
    if not os.path.exists(args.sem_csv):
        raise SystemExit(f"No encuentro {args.sem_csv}")

    print("=" * 66)
    print("y11 REDISENADO - densidad de actos restaurativos (institucional + dialogico)")
    print("=" * 66)

    df_txt = pd.read_csv(args.texto_c)
    print(f"\nComparecientes: {len(df_txt)}")

    raw_a, raw_b = [], []
    hits_a, hits_b = [], []
    for _, row in df_txt.iterrows():
        t = str(row.get("texto_completo", "") or "")
        da, ha = densidad(t, _Y11A)
        db, hb = densidad(t, _Y11B)
        raw_a.append(da)
        raw_b.append(db)
        hits_a.append(ha)
        hits_b.append(hb)

    y11a = minmax(raw_a)
    y11b = minmax(raw_b)
    y11 = (y11a + y11b) / 2.0

    df_txt["y11a_inst_raw"] = raw_a
    df_txt["y11b_dial_raw"] = raw_b
    df_txt["y11a_inst"] = y11a
    df_txt["y11b_dial"] = y11b
    df_txt["y11_densidad"] = y11
    df_txt["n_hits_a"] = hits_a
    df_txt["n_hits_b"] = hits_b

    # --- Reporte ---
    print("\n" + "-" * 66)
    print(f"{'subcaso':<12}{'identidad':<30}{'y11a':>6}{'y11b':>6}{'y11':>6}{'ha':>4}{'hb':>4}")
    print("-" * 66)
    for _, r in df_txt.iterrows():
        print(f"{str(r['subcaso'])[:12]:<12}{str(r['identidad'])[:30]:<30}"
              f"{r['y11a_inst']:>6.3f}{r['y11b_dial']:>6.3f}{r['y11_densidad']:>6.3f}"
              f"{int(r['n_hits_a']):>4}{int(r['n_hits_b']):>4}")

    # --- Correlaciones con y8/y9/y10 (verificar que ya NO duplica) ---
    print("\n" + "-" * 66)
    print("VERIFICACION - correlacion del y11 nuevo con y8/y9/y10")
    print("-" * 66)
    df_sem = pd.read_csv(args.sem_csv)
    # emparejar por identidad
    m = df_txt.set_index("identidad")
    df_sem["_y11_new"] = df_sem["identidad"].map(m["y11_densidad"])
    df_sem["_y11a"] = df_sem["identidad"].map(m["y11a_inst"])
    df_sem["_y11b"] = df_sem["identidad"].map(m["y11b_dial"])
    for col in ["y8_mafapo", "y9_cidh", "y10_rep"]:
        if col in df_sem.columns:
            c_new = df_sem["_y11_new"].corr(df_sem[col])
            c_a = df_sem["_y11a"].corr(df_sem[col])
            c_b = df_sem["_y11b"].corr(df_sem[col])
            print(f"  y11 vs {col:<12}: {c_new:+.3f}   (y11a {c_a:+.3f} | y11b {c_b:+.3f})")
    print("\n  Antes (y11 viejo): y8=0.952  y9=0.972  -> duplicaba las distancias")
    print("  Objetivo: |corr| baja (< ~0.6) = y11 ya NO duplica y8/y9.")

    if not args.escribir:
        df_txt.to_csv("y11_densidad_detalle.csv", index=False, encoding="utf-8")
        print("\n[DRY-RUN] No se escribio el CSV del SEM.")
        print("Detalle -> y11_densidad_detalle.csv")
        return

    # --- Escribir en el CSV del SEM (con backup) ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.sem_csv.replace(".csv", f"_BACKUP_pre_y11_{ts}.csv")
    shutil.copy2(args.sem_csv, backup)
    print(f"\n  Backup del SEM -> {backup}")

    df_sem["y11_conv_rest_old"] = df_sem["y11_conv_rest"] if "y11_conv_rest" in df_sem.columns else np.nan
    df_sem["y11a_inst"] = df_sem["identidad"].map(m["y11a_inst"])
    df_sem["y11b_dial"] = df_sem["identidad"].map(m["y11b_dial"])
    df_sem["y11_conv_rest"] = df_sem["identidad"].map(m["y11_densidad"])
    df_sem = df_sem.drop(columns=["_y11_new", "_y11a", "_y11b"], errors="ignore")
    df_sem.to_csv(args.sem_csv, index=False, encoding="utf-8")
    print(f"  CSV del SEM actualizado -> {args.sem_csv}")
    print("  y11_conv_rest = densidad nueva | y11_conv_rest_old = viejo | y11a_inst, y11b_dial trazables")


if __name__ == "__main__":
    main()
