# -*- coding: utf-8 -*-
"""
cfh_extraer_evidencia_mr.py
============================
PASO 3 (guia de cierre, redefinido): etiqueta MAXIMO RESPONSABLE (MR) vs
NO MAXIMO RESPONSABLE (no-MR) para los 47 comparecientes del ICM tri-canal.

PRINCIPIO METODOLOGICO: la calidad de MR es una determinacion JURIDICA que
hace la Sala en los autos. Este script NO asigna la etiqueta automaticamente:
extrae la EVIDENCIA textual de los autos (Corpus B, en disco) para que Mireya
asigne cada etiqueta leyendo el contexto, con cita trazable al auto fuente.

QUE HACE:
  1. Toma los comparecientes del consolidado outputs/capa3/icm_tricanal_final.csv
     (el universo exacto del analisis).
  2. Busca el nombre de cada uno en TODOS los .txt de data/processed/corpus_b/
     (normalizando tildes en ambos lados; busca nombre completo y apellidos).
  3. Extrae ventanas de contexto (+-250 chars) por mencion, priorizando las
     que contienen terminos clave ("maximo responsable", "no maximo",
     "imputa", "en calidad de", "participe no determinante").
  4. Genera data/mr_evidencia_comparecientes.csv con columnas:
       subcaso, compareciente, n_menciones_total, archivos_fuente,
       contexto_1..contexto_3 (los mas relevantes), etiqueta_MR (VACIA)
  5. Reporta quienes NO aparecen en ningun auto (requeriran fuente externa).

Uso:
    python cfh_extraer_evidencia_mr.py
Salida:
    data/mr_evidencia_comparecientes.csv   (para llenar etiqueta_MR a mano)
    consola: resumen de cobertura
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(".")
ICM_CSV = BASE / "outputs" / "capa3" / "icm_tricanal_final.csv"
AUTOS_DIR = BASE / "data" / "processed" / "corpus_b"
OUT = BASE / "data" / "mr_evidencia_comparecientes.csv"

VENTANA = 250          # chars de contexto a cada lado
MAX_CONTEXTOS = 3      # contextos por compareciente en el CSV

# terminos que hacen un contexto PRIORITARIO (determinacion de calidad)
CLAVE_RE = re.compile(
    r"(m[aá]xim[oa]s?\s+responsab|no\s+m[aá]xim|imputa|en\s+calidad\s+de|"
    r"part[ií]cipe|determinante|responsabilidad\s+de\s+mando|"
    r"autor[ií]a\s+mediata|coautor)", re.IGNORECASE)


def normalizar(s: str) -> str:
    """minusculas sin tildes, espacios colapsados."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def variantes_busqueda(nombre: str):
    """nombre completo normalizado + apellidos (ultimas 2 palabras)."""
    n = normalizar(nombre)
    partes = n.split()
    v = {n}
    if len(partes) >= 2:
        v.add(" ".join(partes[-2:]))          # apellidos
    if len(partes) >= 3:
        v.add(" ".join(partes[:2]))           # dos primeros nombres (desambigua poco, se usa solo si largo)
    # descartar variantes demasiado cortas/ambiguas
    return {x for x in v if len(x) >= 10}


def main():
    if not ICM_CSV.exists():
        print(f"[ERROR] no existe {ICM_CSV}")
        return
    icm = pd.read_csv(ICM_CSV)
    col_ident = "identidad" if "identidad" in icm.columns else icm.columns[1]
    col_sub = "subcaso" if "subcaso" in icm.columns else icm.columns[0]
    personas = icm[[col_sub, col_ident]].drop_duplicates().values.tolist()
    print(f"Comparecientes del ICM: {len(personas)}")

    # cargar autos (texto normalizado para busqueda + original para contexto)
    autos = {}
    for tp in sorted(AUTOS_DIR.glob("*.txt")):
        try:
            original = tp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [WARN] no se pudo leer {tp.name}: {e}")
            continue
        autos[tp.name] = (original, normalizar(original))
    print(f"Autos/documentos B cargados: {len(autos)}")

    filas = []
    sin_evidencia = []
    for sub, ident in personas:
        vs = variantes_busqueda(ident)
        contextos = []   # (prioritario: bool, archivo, texto_contexto)
        n_total = 0
        archivos = set()
        for nombre_archivo, (orig, norm) in autos.items():
            for v in vs:
                for m in re.finditer(re.escape(v), norm):
                    n_total += 1
                    archivos.add(nombre_archivo)
                    # mapear posicion aproximada al original:
                    # norm y orig difieren solo en tildes/mayusculas/espacios;
                    # usamos proporcion de posicion como aproximacion y ampliamos margen
                    pos = int(m.start() / max(len(norm), 1) * len(orig))
                    a = max(0, pos - VENTANA - 80)
                    b = min(len(orig), pos + VENTANA + 80)
                    ctx = re.sub(r"\s+", " ", orig[a:b]).strip()
                    prio = bool(CLAVE_RE.search(ctx))
                    contextos.append((prio, nombre_archivo, ctx))
        # ordenar: prioritarios primero, luego por longitud informativa
        contextos.sort(key=lambda x: (not x[0], -len(x[1])))
        top = contextos[:MAX_CONTEXTOS]
        fila = {
            "subcaso": sub,
            "compareciente": ident,
            "n_menciones_total": n_total,
            "archivos_fuente": "; ".join(sorted(archivos)) if archivos else "",
            "etiqueta_MR": "",   # <- llenar a mano: MR / NO_MR / SIN_DATO
        }
        for i in range(MAX_CONTEXTOS):
            if i < len(top):
                prio, arch, ctx = top[i]
                marca = "[CLAVE] " if prio else ""
                fila[f"contexto_{i+1}"] = f"{marca}({arch}) ...{ctx}..."
            else:
                fila[f"contexto_{i+1}"] = ""
        filas.append(fila)
        if n_total == 0:
            sin_evidencia.append((sub, ident))

    df = pd.DataFrame(filas)
    df = df.sort_values(["subcaso", "compareciente"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("RESUMEN DE COBERTURA")
    print("=" * 70)
    con = (df["n_menciones_total"] > 0).sum()
    print(f"  con evidencia en autos:  {con} / {len(df)}")
    print(f"  con contexto CLAVE:      {df['contexto_1'].str.startswith('[CLAVE]').sum()}")
    print(f"  SIN evidencia (fuente externa necesaria): {len(sin_evidencia)}")
    for sub, ident in sin_evidencia:
        print(f"      {sub:12s} {ident}")

    print(f"\n  GUARDADO: {OUT}")
    print("  Siguiente: abrir el CSV, leer contextos y llenar etiqueta_MR")
    print("  con MR / NO_MR / SIN_DATO. Cada etiqueta queda trazada a su auto.")


if __name__ == "__main__":
    main()
