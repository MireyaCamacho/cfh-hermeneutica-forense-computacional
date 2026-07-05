"""
============================================================
CFH — Métricas de robustez DIS / IEI  (versión corregida)
============================================================
Corrige el bug que producía NaN en la separación A vs B:
la columna de tipo de corpus no venía codificada 0/1, por lo
que los grupos a/b quedaban vacíos y Mann-Whitney devolvía NaN.

Esta versión:
  - Detecta automáticamente la columna de tipo de corpus.
  - Normaliza cualquier codificación (A/B/C, 0/1, 'corpus_a'...)
    a una etiqueta canónica {'A','B','C'}.
  - Recalcula la separación A vs B con Mann-Whitney U + Cohen d.
  - Reporta el margen del centroide MAFAPO v5 (punto 3).

Ejecutar desde el env `cfh`:
    python cfh_metricas_robustez_fix.py

Requiere: pandas, numpy, scipy
============================================================
"""

import sqlite3
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
DB = BASE / "data" / "cfh.db"

# CSV definitivo con DIS/IEI por documento (ajusta el nombre si difiere)
CSV_DIS_IEI = BASE / "data" / "dis_iei_corpus_abc_definitivo.csv"

# Columnas de los componentes para el análisis de sensibilidad de pesos
COL_SA = "y2_sa_norm"      # Supresión de Agentividad normalizada
COL_NV = "y4_nv_norm"      # Negación de Victimización normalizada
COL_REP = "y10_rep_norm"   # REP normalizado (se usa 1 - REP en el DIS)

PASO_PESOS = 0.05          # paso del grid de pesos


# ------------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------------
def cohen_d(x, y):
    """Cohen d para dos muestras independientes (pooled std)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def detectar_columna_corpus(df):
    """Encuentra la columna que codifica el tipo de corpus."""
    candidatas = [
        "corpus_type", "corpus", "tipo_corpus", "corpus_tipo",
        "sistema", "fuente", "subsistema", "corpus_label", "grupo",
    ]
    for c in candidatas:
        if c in df.columns:
            return c
    # Heurística: alguna columna cuyos valores únicos parezcan A/B/C
    for c in df.columns:
        vals = df[c].astype(str).str.upper().str.strip().unique()
        if set(vals) & {"A", "B", "C"} or any("CORPUS" in v for v in vals):
            return c
    raise ValueError(
        "No se encontró columna de tipo de corpus. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def normalizar_corpus(serie):
    """
    Normaliza cualquier codificación a {'A','B','C'}.
    Maneja: 'A'/'B'/'C', 'corpus_a', 'A-CE', 'A-CSJ', 'B-JEP',
            'C-JEP oral', enteros 0/1/2, etc.
    """
    def _map(v):
        s = str(v).upper().strip()
        if s in {"0", "A", "CORPUS_A", "A-CE", "A-CSJ", "CE", "CSJ", "ORDINARIA"}:
            return "A"
        if s in {"1", "B", "CORPUS_B", "B-JEP", "JEP", "JEP_ESCRITO"}:
            return "B"
        if s in {"2", "C", "CORPUS_C", "C-JEP ORAL", "JEP_ORAL", "ORAL"}:
            return "C"
        # Prefijos
        if s.startswith("A"):
            return "A"
        if s.startswith("B"):
            return "B"
        if s.startswith("C"):
            return "C"
        return np.nan

    return serie.map(_map)


def cargar_dis_iei():
    """Carga el DataFrame con DIS, IEI y el tipo de corpus normalizado."""
    if CSV_DIS_IEI.exists():
        df = pd.read_csv(CSV_DIS_IEI)
        print(f"  Fuente: {CSV_DIS_IEI.name} ({len(df)} filas)")
    else:
        # Fallback: leer de cfh.db
        print(f"  CSV no encontrado, leyendo de {DB.name}")
        con = sqlite3.connect(DB)
        # Ajusta esta query a tu esquema real si difiere
        df = pd.read_sql_query(
            "SELECT * FROM documentos_dis_iei", con
        )
        con.close()

    col = detectar_columna_corpus(df)
    df["corpus_norm"] = normalizar_corpus(df[col])
    print(f"  Columna de corpus detectada: '{col}'")
    print(f"  Distribución corpus: {df['corpus_norm'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------------
# 4. SEPARACIÓN A vs B  (lo que estaba en NaN)
# ------------------------------------------------------------------
def separacion_AB(df, col_valor):
    a = df.loc[df["corpus_norm"] == "A", col_valor].dropna()
    b = df.loc[df["corpus_norm"] == "B", col_valor].dropna()

    if len(a) < 2 or len(b) < 2:
        print(f"   [!] {col_valor}: muestras insuficientes "
              f"(A n={len(a)}, B n={len(b)})")
        return dict(mean_a=np.nan, mean_b=np.nan, delta=np.nan,
                    d=np.nan, p=np.nan, n_a=len(a), n_b=len(b))

    u, p = mannwhitneyu(a, b, alternative="two-sided")
    d = cohen_d(a, b)
    return dict(
        mean_a=a.mean(), mean_b=b.mean(), delta=a.mean() - b.mean(),
        d=d, p=p, n_a=len(a), n_b=len(b),
    )


# ------------------------------------------------------------------
# 6. SENSIBILIDAD DE PESOS (PARSIMONIA)  —  DIS
# ------------------------------------------------------------------
def sensibilidad_pesos_dis(df):
    """
    Grid de combinaciones de pesos (w_SA, w_NV, w_REP) que suman 1.0.
    Para cada combinación recalcula el DIS y testea separación A vs B.
    """
    faltan = [c for c in (COL_SA, COL_NV, COL_REP) if c not in df.columns]
    if faltan:
        print(f"   [!] Faltan columnas de componentes: {faltan} "
              "— se omite sensibilidad de pesos.")
        return None

    pasos = np.round(np.arange(0.0, 1.0 + 1e-9, PASO_PESOS), 2)
    combos = [
        (w1, w2, round(1.0 - w1 - w2, 2))
        for w1, w2 in itertools.product(pasos, pasos)
        if 0.0 < (1.0 - w1 - w2) < 1.0 and w1 > 0 and w2 > 0
    ]

    n_sig = 0
    d_vals = []
    d_teorico = np.nan

    a_mask = df["corpus_norm"] == "A"
    b_mask = df["corpus_norm"] == "B"

    for w_sa, w_nv, w_rep in combos:
        dis = (w_sa * df[COL_SA]
               + w_nv * df[COL_NV]
               + w_rep * (1.0 - df[COL_REP]))
        a = dis[a_mask].dropna()
        b = dis[b_mask].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        d = abs(cohen_d(a, b))
        d_vals.append(d)
        if p < 0.05:
            n_sig += 1
        if (w_sa, w_nv, w_rep) == (0.35, 0.35, 0.30):
            d_teorico = d

    return dict(
        total=len(combos),
        n_sig=n_sig,
        pct_sig=100.0 * n_sig / len(combos) if combos else np.nan,
        d_max=max(d_vals) if d_vals else np.nan,
        d_teorico=d_teorico,
    )


# ------------------------------------------------------------------
# 3. MARGEN DEL CENTROIDE MAFAPO v5  (cómo se calcula)
# ------------------------------------------------------------------
def margen_centroide(npy_path=None, n_boot=1000, seed=42):
    """
    El 'margen ±X%' del centroide MAFAPO es la semi-amplitud relativa
    del intervalo de confianza bootstrap del propio centroide.

    Procedimiento:
      1. Se tienen N embeddings de textos de víctimas (N=293 en v5).
      2. Centroide = media de los N vectores.
      3. Bootstrap: se remuestrean N textos CON reemplazo, B veces,
         y se recalcula el centroide en cada réplica.
      4. Para cada réplica se mide la distancia coseno al centroide
         original. El margen es el percentil 95 de esa distribución,
         expresado como porcentaje  ->  ±margen%.

    Interpretación: cuánto se movería el polo semántico de las
    víctimas si la muestra de textos hubiera sido distinta.
    Margen bajo = centroide estable = referencia robusta.

    Si no se pasa el .npy, solo imprime la fórmula documental.
    """
    if npy_path is None:
        npy_path = BASE / "data" / "referencias" / "embeddings_mafapo_v5.npy"

    print("\n" + "=" * 60)
    print("3. MARGEN DEL CENTROIDE MAFAPO v5 (definición operativa)")
    print("=" * 60)
    print("  margen ±% = percentil 95 de la distancia coseno entre")
    print("  el centroide bootstrap y el centroide original, sobre")
    print("  B réplicas remuestreadas con reemplazo de los N textos.")

    npy_path = Path(npy_path)
    if not npy_path.exists():
        print(f"  [i] No se halló {npy_path.name} (matriz de embeddings).")
        print("      v5 FINAL reportado: N=293, margen=±5.8%, sim_v4=0.9888")
        return np.nan

    X = np.load(npy_path)                      # (N, d) embeddings de textos
    N = X.shape[0]
    centroide = X.mean(axis=0)
    centroide /= np.linalg.norm(centroide)

    rng = np.random.default_rng(seed)
    dists = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)       # remuestreo con reemplazo
        c_b = X[idx].mean(axis=0)
        c_b /= np.linalg.norm(c_b)
        cos_sim = float(np.dot(centroide, c_b))
        dists[i] = 1.0 - cos_sim               # distancia coseno

    margen = np.percentile(dists, 95) * 100.0
    print(f"  N textos        = {N}")
    print(f"  Réplicas boot   = {n_boot}")
    print(f"  Margen ±%       = ±{margen:.2f}%")
    return margen


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MÉTRICAS DE ROBUSTEZ — DIS e IEI  (FIX corpus_type)")
    print("=" * 60)

    df = cargar_dis_iei()

    # Detecta nombres de columnas DIS/IEI (tolerante a sufijos)
    col_dis = next((c for c in df.columns if c.upper().startswith("DIS")), "DIS")
    col_iei = next((c for c in df.columns if c.upper().startswith("IEI")), "IEI")

    print("\n4. Separación A vs B  (corregida)")
    for nombre, col in (("DIS", col_dis), ("IEI", col_iei)):
        r = separacion_AB(df, col)
        print(f"   {nombre}: A={r['mean_a']:.4f} (n={r['n_a']})  "
              f"B={r['mean_b']:.4f} (n={r['n_b']})  "
              f"Δ={r['delta']:.4f}  d={r['d']:.3f}  p={r['p']:.4g}")

    print("\n6. Sensibilidad de pesos DIS (parsimonia)")
    s = sensibilidad_pesos_dis(df)
    if s:
        print(f"   {s['n_sig']}/{s['total']} combinaciones significativas "
              f"({s['pct_sig']:.1f}%)")
        print(f"   d_cohen max = {s['d_max']:.3f}")
        print(f"   d_cohen pesos teóricos (0.35/0.35/0.30) = "
              f"{s['d_teorico']:.3f}")

    # Punto 3: margen del centroide
    margen_centroide()

    print("\n[CFH] Completado.")


if __name__ == "__main__":
    main()
