# -*- coding: utf-8 -*-
"""
cfh_mr_variantes_grafia.py
===========================
Los comparecientes "sin mencion en audiencias" casi seguro SI estan en las
transcripciones, pero Whisper los escribio con otra grafia (caso confirmado:
"Samboni" -> "Zamboni", 3 menciones). Este script reintenta la busqueda SOLO
para los que quedaron con n_menciones_audiencia == 0 en el cruce doble,
generando variantes foneticas tipicas del ASR en espanol:

    s <-> z      (Samboni/Zamboni)
    b <-> v      (Buelvas/Buelbas)
    i <-> y      (Yati/Iati; finales -is/-ys)
    ll -> y      (Calderon no aplica, pero Buesaquillo/Buesaquiyo si)
    j <-> g(e,i) (Juspian/Guspian)
    h inicial opcional (Herrera/Errera)

Busca bigramas de apellidos con variantes combinadas y, si no encuentra,
palabras individuales largas y distintivas (>=7 chars, excluyendo apellidos
muy comunes para evitar falsos positivos).

Actualiza data/mr_evidencia_doble.csv (solo las filas recuperadas, columnas
aud_*), con BACKUP previo. Tambien reporta que variante matcheo, para que la
verificacion quede documentada.

Uso:
    python cfh_mr_variantes_grafia.py
"""

import itertools
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(".")
AUDIENCIAS_DIR = BASE / "corpus_c"
DOBLE = BASE / "data" / "mr_evidencia_doble.csv"

VENTANA = 250
MAX_CONTEXTOS = 3

CLAVE_RE = re.compile(
    r"(m[aá]xim[oa]s?\s+responsab|no\s+m[aá]xim|imputa|en\s+calidad\s+de|"
    r"part[ií]cipe|determinante|responsabilidad\s+de\s+mando|"
    r"autor[ií]a\s+mediata|coautor)", re.IGNORECASE)

APELLIDOS_COMUNES = {
    "garcia", "rodriguez", "martinez", "lopez", "gonzalez", "hernandez",
    "perez", "sanchez", "ramirez", "torres", "gomez", "diaz", "morales",
    "ortiz", "gutierrez", "ramos", "ruiz", "mendoza", "jimenez", "alvarez",
    "castillo", "romero", "vargas", "moreno", "munoz", "rojas", "medina",
    "aguilar", "herrera", "castro", "fernandez", "arias", "mendez",
}


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def variantes_palabra(p: str) -> set:
    """Variantes foneticas ASR de una palabra (acotadas)."""
    vs = {p}
    subs = [("s", "z"), ("z", "s"), ("b", "v"), ("v", "b"),
            ("ll", "y"), ("j", "g")]
    for a, b in subs:
        nuevos = set()
        for v in vs:
            if a in v:
                nuevos.add(v.replace(a, b))
        vs |= nuevos
        if len(vs) > 12:
            break
    # i/y al final
    extra = set()
    for v in vs:
        if v.endswith("i"):
            extra.add(v[:-1] + "y")
        if v.endswith("y"):
            extra.add(v[:-1] + "i")
        if v.startswith("h"):
            extra.add(v[1:])
    vs |= extra
    return {v for v in vs if len(v) >= 4}


def variantes_busqueda(nombre: str):
    """Bigramas de palabras consecutivas con variantes + individuales largas."""
    n = normalizar(nombre)
    partes = [p for p in n.split() if len(p) >= 4]
    bigramas = set()
    for i in range(len(partes) - 1):
        va = variantes_palabra(partes[i])
        vb = variantes_palabra(partes[i + 1])
        for a, b in itertools.product(va, vb):
            bigramas.add(f"{a} {b}")
            if len(bigramas) > 200:
                break
    individuales = set()
    for p in partes:
        if len(p) >= 7 and p not in APELLIDOS_COMUNES:
            individuales |= {v for v in variantes_palabra(p) if len(v) >= 7}
    return bigramas, individuales


def main():
    if not DOBLE.exists():
        print(f"[ERROR] no existe {DOBLE} (correr primero el extractor doble)")
        return
    df = pd.read_csv(DOBLE)
    pendientes = df[df["n_menciones_audiencia"].fillna(0) == 0]
    print(f"Comparecientes sin mencion en audiencias: {len(pendientes)}")
    if pendientes.empty:
        print("Nada que reintentar.")
        return

    textos = {}
    for tp in sorted(AUDIENCIAS_DIR.glob("*.txt")):
        if tp.name.lower() in ("readme.txt",):
            continue
        try:
            orig = tp.read_text(encoding="utf-8", errors="replace")
            textos[tp.name] = (orig, normalizar(orig))
        except Exception:
            pass
    print(f"Transcripciones: {len(textos)}")

    recuperados = 0
    for idx, row in pendientes.iterrows():
        ident = row["compareciente"]
        bigramas, individuales = variantes_busqueda(ident)
        contextos, n_total, archivos, variantes_hit = [], 0, set(), set()

        def buscar(patrones, etiqueta):
            nonlocal n_total
            for nombre_archivo, (orig, norm) in textos.items():
                for v in patrones:
                    for m in re.finditer(re.escape(v), norm):
                        n_total += 1
                        archivos.add(nombre_archivo)
                        variantes_hit.add(v)
                        pos = int(m.start() / max(len(norm), 1) * len(orig))
                        a = max(0, pos - VENTANA - 80)
                        b = min(len(orig), pos + VENTANA + 80)
                        ctx = re.sub(r"\s+", " ", orig[a:b]).strip()
                        contextos.append((bool(CLAVE_RE.search(ctx)), nombre_archivo, ctx))

        buscar(bigramas, "bigrama")
        if n_total == 0:
            buscar(individuales, "individual")

        if n_total > 0:
            recuperados += 1
            contextos.sort(key=lambda x: (not x[0], -len(x[1])))
            top = contextos[:MAX_CONTEXTOS]
            df.at[idx, "n_menciones_audiencia"] = n_total
            df.at[idx, "archivos_audiencia"] = "; ".join(sorted(archivos))
            for i in range(2):  # el doble tiene aud_contexto_1 y _2
                col = f"aud_contexto_{i+1}"
                if col in df.columns and i < len(top):
                    prio, arch, ctx = top[i]
                    marca = "[CLAVE] " if prio else ""
                    df.at[idx, col] = (f"{marca}({arch}) [variante: "
                                       f"{', '.join(sorted(variantes_hit))}] ...{ctx}...")
            print(f"  RECUPERADO  {row['subcaso']:12s} {ident}")
            print(f"      variante(s): {', '.join(sorted(variantes_hit))}  "
                  f"({n_total} menciones en {', '.join(sorted(archivos))})")
        else:
            print(f"  sin match   {row['subcaso']:12s} {ident}")

    if recuperados:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DOBLE.with_name(DOBLE.stem + f"_BACKUP_{ts}.csv")
        shutil.copy2(DOBLE, backup)
        df.to_csv(DOBLE, index=False, encoding="utf-8-sig")
        print(f"\n  backup -> {backup.name}")
        print(f"  ACTUALIZADO: {DOBLE}  ({recuperados} recuperados)")
    else:
        print("\n  Ningun recuperado; el CSV no se modifica.")

    print("\n  Los contextos recuperados llevan la marca [variante: ...] para que")
    print("  la verificacion de identidad quede documentada al etiquetar.")


if __name__ == "__main__":
    main()
