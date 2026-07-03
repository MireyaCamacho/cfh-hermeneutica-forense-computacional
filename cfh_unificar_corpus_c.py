# -*- coding: utf-8 -*-
"""
CFH — Unificación del Corpus C sobre segmentación canónica única
================================================================
PROBLEMA QUE RESUELVE:
  Los indicadores y2/y4/y10 estaban en `indicators_corpus_c_capa1_v2.csv`
  con segmentación '_c' (parche Costa Caribe), mientras y8/y9 estaban en
  `indicators_corpus_c.csv` con segmentación '_b' (2000 chars, canónica).
  El merge por nombre de subcaso producía un PRODUCTO CARTESIANO (68.116 filas)
  en vez de unir bloque a bloque.

SOLUCIÓN:
  Re-segmentar las transcripciones con el MISMO algoritmo canónico (2000 chars,
  prefijo '_b'), recalcular y2/y4/y10 sobre esos bloques, y unir con y8/y9
  por bloque_id (que ahora SÍ coincide). Verifica con asserts que no haya
  duplicación ni merge inflado.

SALIDA:
  data/indicators_corpus_c_unificado.csv  (588 bloques, 5 indicadores, base única)

Entorno: env `cfh`, spaCy con es_core_news_lg, ejecutar desde la raíz del repo.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(".")
CORPUS_C_DIR = BASE / "corpus_c"
CANONICO_Y89 = BASE / "data/features/indicators_corpus_c.csv"   # y8/y9 base '_b'
OUT          = BASE / "data/indicators_corpus_c_unificado.csv"

BLOCK_SIZE = 2000  # IDÉNTICO al script canónico analisis_corpus_c.py

# Las 5 transcripciones canónicas (mismas que usó el script '_b')
AUDIOS = {
    "catatumbo":         "catatumbo_audiencia_reconocimiento.txt",
    "costa_caribe":      "costa_caribe.txt",
    "casanare_torres":   "casanare_torres.txt",
    "dabeiba_antioquia": "dabeiba_antioquia.txt",
    "huila":             "huila.txt",
}
SUBCASO_META = {
    "catatumbo":         {"subcaso": "Norte de Santander", "fecha": "2022-04-26", "tipo": "audiencia_reconocimiento"},
    "costa_caribe":      {"subcaso": "Costa Caribe",       "fecha": "2022-07-18", "tipo": "audiencia_reconocimiento"},
    "casanare_torres":   {"subcaso": "Casanare",           "fecha": "2020-02-06", "tipo": "version_voluntaria"},
    "dabeiba_antioquia": {"subcaso": "Antioquia",          "fecha": "2023-06-27", "tipo": "audiencia_reconocimiento"},
    "huila":             {"subcaso": "Huila",              "fecha": "2024-08-10", "tipo": "audiencia_reconocimiento"},
}

def segmentar_transcripcion(texto, nombre, block_size=BLOCK_SIZE):
    """IDÉNTICA a la del script canónico — garantiza los mismos bloque_id '_b'."""
    bloques = []
    palabras = texto.split()
    bloque_actual, char_count, bloque_id = [], 0, 0
    for palabra in palabras:
        bloque_actual.append(palabra)
        char_count += len(palabra) + 1
        if char_count >= block_size:
            tb = " ".join(bloque_actual)
            bloques.append({"audio": nombre, "bloque_id": f"{nombre}_b{bloque_id:04d}",
                            "texto": tb, "chars": len(tb), **SUBCASO_META.get(nombre, {})})
            bloque_actual, char_count = [], 0
            bloque_id += 1
    if bloque_actual and len(bloque_actual) > 20:
        tb = " ".join(bloque_actual)
        bloques.append({"audio": nombre, "bloque_id": f"{nombre}_b{bloque_id:04d}",
                        "texto": tb, "chars": len(tb), **SUBCASO_META.get(nombre, {})})
    return bloques

# ── PASO 1: re-segmentar idéntico al canónico ───────────────────────────────
print("== PASO 1: Re-segmentación canónica (2000 chars) ==")
bloques = []
for nombre, archivo in AUDIOS.items():
    path = CORPUS_C_DIR / archivo
    if not path.exists():
        print(f"  ⚠ No encontrado: {path}")
        continue
    texto = path.read_text(encoding="utf-8")
    bl = segmentar_transcripcion(texto, nombre)
    bloques.extend(bl)
    print(f"  ✓ {nombre}: {len(texto):,} chars → {len(bl)} bloques")
df = pd.DataFrame(bloques)
print(f"\nTotal bloques re-segmentados: {len(df)}")
print(df["audio"].value_counts().to_string())

# ── PASO 2: cargar extractores y calcular y2/y4/y10 ─────────────────────────
print("\n== PASO 2: Calculando y2/y4/y10 con extractores reales ==")
import importlib.util
def cargar(modfile, modname, clase):
    """Carga registrando el módulo en sys.modules (necesario para @dataclass)."""
    spec = importlib.util.spec_from_file_location(modname, BASE / "code/src/features" / modfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod          # <- clave: registrar ANTES de ejecutar
    spec.loader.exec_module(mod)
    return getattr(mod, clase)

SAExtractor  = cargar("y2_sa_extractor.py",  "y2_sa_extractor",  "SAExtractor")
NVExtractor  = cargar("y4_nv_extractor.py",  "y4_nv_extractor",  "NVExtractor")
REPExtractor = cargar("y10_rep_extractor.py","y10_rep_extractor","REPExtractor")
sa, nv, rep = SAExtractor(), NVExtractor(), REPExtractor()

def get_score(extractor, texto):
    """Maneja el objeto resultado: devuelve .score (float)."""
    try:
        r = extractor.extract(texto)
        # el resultado es un *ExtractionResult con atributo .score
        return float(getattr(r, "score", r))
    except Exception as e:
        return np.nan
def get_score_rep(extractor, texto):
    """REP sobre Corpus C: corpus_type="C" activa los detectores orales
    (reconocimiento 1a persona + reparacion/perdon + restitucion)."""
    try:
        r = extractor.extract(texto, corpus_type="C")
        return float(getattr(r, "score", r))
    except Exception as e:
        return np.nan

y2, y4, y10 = [], [], []
for i, row in df.iterrows():
    t = row["texto"][:8000]
    y2.append(get_score(sa, t))
    y4.append(get_score(nv, t))
    y10.append(get_score_rep(rep, t))
    if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(df)} bloques")
df["y2_sa"]  = y2
df["y4_nv"]  = y4
df["y10_rep"] = y10
print(f"✓ y2/y4/y10 calculados. NaN: y2={df.y2_sa.isna().sum()} y4={df.y4_nv.isna().sum()} y10={df.y10_rep.isna().sum()}")

# ── PASO 3: unir con y8/y9 por bloque_id (sin cartesiano) ────────────────────
print("\n== PASO 3: Uniendo con y8/y9 por bloque_id ==")
y89 = pd.read_csv(CANONICO_Y89)
print(f"  y8/y9 canónico: {len(y89)} bloques")

# VERIFICACIÓN CRÍTICA: bloque_id debe ser único en ambos lados
assert df["bloque_id"].is_unique, "✗ bloque_id duplicado en re-segmentación"
assert y89["bloque_id"].is_unique, "✗ bloque_id duplicado en y8/y9"

# Merge 1:1 por bloque_id
unif = pd.merge(df.drop(columns=["texto"]),
                y89[["bloque_id", "y8_mafapo_cs", "y9_cidh_cs"]],
                on="bloque_id", how="inner", validate="one_to_one")
print(f"  Merge 1:1 resultante: {len(unif)} bloques")

# VERIFICACIÓN: no debe inflarse
assert len(unif) <= len(df), f"✗ MERGE INFLADO: {len(unif)} > {len(df)}"
match_rate = len(unif) / len(df) * 100
print(f"  Tasa de match: {match_rate:.1f}% ({len(unif)}/{len(df)})")
if match_rate < 90:
    print(f"  ⚠ ATENCIÓN: {len(df)-len(unif)} bloques sin y8/y9 (posible desajuste de segmentación)")

# ── PASO 4: DIS e IEI con z-score+sigmoid sobre A+B+C ────────────────────────
print("\n== PASO 4: DIS e IEI (z-score+sigmoid A+B+C) ==")
def sigmoid(x): return 1/(1+np.exp(-x))

# Cargar A+B
df_ab = pd.read_csv(BASE / "data/features/indicators_completo_conflibert.csv")
df_ab["corpus"] = df_ab["corpus_type"].apply(lambda x: "A" if str(x).startswith("A") else "B")
unif["corpus"] = "C"

COLS = ["y2_sa","y4_nv","y10_rep","y8_mafapo_cs","y9_cidh_cs"]
allc = pd.concat([df_ab[["corpus"]+COLS], unif[["corpus"]+COLS]], ignore_index=True)
print(f"  Total A+B+C: {len(allc)}  (A={sum(allc.corpus=='A')} B={sum(allc.corpus=='B')} C={sum(allc.corpus=='C')})")

# VERIFICACIÓN: C debe ser ~588, NO 68116
assert sum(allc.corpus=="C") < 1000, f"✗ Corpus C inflado: {sum(allc.corpus=='C')} filas"

params = {}
for col in COLS:
    v = allc[col].dropna()
    mu, sd = v.mean(), v.std()+1e-9
    params[col] = (mu, sd)
    allc[col+"_z"] = sigmoid((allc[col]-mu)/sd)

allc["DIS"] = 0.35*allc["y2_sa_z"] + 0.35*allc["y4_nv_z"] + 0.30*(1-allc["y10_rep_z"])
allc["IEI"] = 0.35*allc["y8_mafapo_cs_z"] + 0.20*allc["y9_cidh_cs_z"] + 0.25*allc["y4_nv_z"] + 0.20*(1-allc["y10_rep_z"])

# Re-aplicar a unif para tener DIS/IEI por bloque del Corpus C
for col in COLS:
    mu, sd = params[col]
    unif[col+"_z"] = sigmoid((unif[col]-mu)/sd)
unif["DIS"] = 0.35*unif["y2_sa_z"] + 0.35*unif["y4_nv_z"] + 0.30*(1-unif["y10_rep_z"])
unif["IEI"] = 0.35*unif["y8_mafapo_cs_z"] + 0.20*unif["y9_cidh_cs_z"] + 0.25*unif["y4_nv_z"] + 0.20*(1-unif["y10_rep_z"])

# ── PASO 5: resultados por subcaso ──────────────────────────────────────────
print("\n== PASO 5: DIS/IEI por subcaso (base unificada) ==")
res = unif.groupby("audio").agg(
    n=("bloque_id","count"),
    DIS=("DIS","mean"), IEI=("IEI","mean"),
    y2=("y2_sa","mean"), y4=("y4_nv","mean"), y10=("y10_rep","mean"),
    y8=("y8_mafapo_cs","mean"), y9=("y9_cidh_cs","mean"),
).round(3)
print(res.to_string())

print("\n=== TESTS A vs B (DIS, IEI) ===")
from scipy.stats import mannwhitneyu
a = allc[allc.corpus=="A"]; b = allc[allc.corpus=="B"]
for idx in ["DIS","IEI"]:
    _,p = mannwhitneyu(a[idx].dropna(), b[idx].dropna(), alternative="two-sided")
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    print(f"  {idx}: A={a[idx].mean():.3f} vs B={b[idx].mean():.3f}  p={p:.4f} {sig}")

# ── PASO 6: guardar ─────────────────────────────────────────────────────────
unif.to_csv(OUT, index=False, encoding="utf-8-sig")
res.to_csv(BASE / "data/dis_iei_corpus_c_unificado.csv", encoding="utf-8-sig")
print(f"\n✓ {OUT}")
print(f"✓ data/dis_iei_corpus_c_unificado.csv")
print("\n=== VERIFICACIÓN FINAL ===")
print(f"  Bloques únicos: {unif['bloque_id'].nunique()}")
print(f"  Sin duplicación: {unif['bloque_id'].is_unique}")
print(f"  Corpus C N={len(unif)} (esperado ~588, NO 68116)")
