"""
Capa 2 — Visualización UMAP/t-SNE del espacio de embeddings CFH
================================================================
Genera visualizaciones del espacio semántico de ConfliBERT-Spanish
mostrando la separación entre Corpus A (justicia ordinaria),
Corpus B (JEP escrita) y Corpus C (JEP oral).

También calcula la convergencia temporal: ¿la JEP se acerca
semánticamente a las víctimas (polo MAFAPO) año a año?

Uso en Colab (requiere GPU):
    python code/src/visualizacion_embeddings.py \
        --indicators indicators_final_completo.csv \
        --corpus_c indicators_corpus_c.csv

Requiere:
    pip install umap-learn plotly pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Configuración visual CFH ──────────────────────────────────────────────────

COLORS = {
    "A-CE":  "#1E3A4C",   # navy oscuro — Consejo de Estado
    "A-CSJ": "#2E6DA4",   # azul — Corte Suprema
    "B":     "#0D9488",   # teal — JEP escrita
    "C":     "#F59E0B",   # ámbar — JEP oral
}

LABELS = {
    "A-CE":  "Corpus A — Consejo de Estado",
    "A-CSJ": "Corpus A — Corte Suprema",
    "B":     "Corpus B — JEP escrita",
    "C":     "Corpus C — JEP oral",
}

# ── Cargador de datos ─────────────────────────────────────────────────────────

def cargar_datos(path_indicators, path_corpus_c=None):
    """Carga y combina los indicadores de A, B y C."""

    df_ab = pd.read_csv(path_indicators)
    print(f"✓ Corpus A+B cargado: {len(df_ab)} secciones")

    if path_corpus_c and Path(path_corpus_c).exists():
        df_c = pd.read_csv(path_corpus_c)
        df_c["corpus_type"] = "C"
        df = pd.concat([df_ab, df_c], ignore_index=True)
        print(f"✓ Corpus C agregado: {len(df_c)} bloques")
    else:
        df = df_ab
        print("⚠ Corpus C no disponible")

    print(f"  Total: {len(df)} registros")
    print(f"  Distribución: {df['corpus_type'].value_counts().to_dict()}")
    return df


# ── UMAP ─────────────────────────────────────────────────────────────────────

def generar_umap(df, output_path="umap_cfh.png"):
    """Genera visualización UMAP del espacio de embeddings."""
    try:
        import umap
    except ImportError:
        print("⚠ umap-learn no disponible. Instala: pip install umap-learn")
        return None

    # Features para UMAP — indicadores disponibles
    feature_cols = [
        col for col in ["y2_sa", "y3_civil", "y4_nv", "y10_rep",
                        "y8_mafapo_cs", "y9_cidh_cs"]
        if col in df.columns
    ]

    if len(feature_cols) < 2:
        print("⚠ Insuficientes features para UMAP")
        return None

    df_clean = df[feature_cols + ["corpus_type"]].dropna()
    X = df_clean[feature_cols].values

    print(f"\nGenerando UMAP con {len(X)} puntos y {len(feature_cols)} features...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(X)

    # Graficar
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0D2137')
    ax.set_facecolor('#0D2137')

    for corpus_type, color in COLORS.items():
        mask = df_clean["corpus_type"] == corpus_type
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=LABELS[corpus_type],
            alpha=0.6, s=8, edgecolors='none'
        )

    ax.set_title("Espacio semántico CFH — UMAP\nSeparación entre sistemas de justicia",
                 color='white', fontsize=14, pad=15)
    ax.set_xlabel("UMAP 1", color='white')
    ax.set_ylabel("UMAP 2", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E3A4C')

    legend = ax.legend(
        facecolor='#1E3A4C', edgecolor='#0D9488',
        labelcolor='white', fontsize=10, markerscale=3
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D2137')
    plt.close()
    print(f"✓ UMAP guardado: {output_path}")
    return embedding


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def generar_tsne(df, output_path="tsne_cfh.png"):
    """Genera visualización t-SNE del espacio de embeddings."""
    from sklearn.manifold import TSNE

    feature_cols = [
        col for col in ["y2_sa", "y3_civil", "y4_nv", "y10_rep",
                        "y8_mafapo_cs", "y9_cidh_cs"]
        if col in df.columns
    ]

    if len(feature_cols) < 2:
        print("⚠ Insuficientes features para t-SNE")
        return None

    # Muestra para t-SNE (lento con N grande)
    df_clean = df[feature_cols + ["corpus_type"]].dropna()
    if len(df_clean) > 2000:
        df_sample = df_clean.groupby("corpus_type").apply(
            lambda x: x.sample(min(500, len(x)), random_state=42)
        ).reset_index(drop=True)
    else:
        df_sample = df_clean

    X = df_sample[feature_cols].values
    print(f"\nGenerando t-SNE con {len(X)} puntos...")

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedding = tsne.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0D2137')
    ax.set_facecolor('#0D2137')

    for corpus_type, color in COLORS.items():
        mask = df_sample["corpus_type"] == corpus_type
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=LABELS[corpus_type],
            alpha=0.6, s=12, edgecolors='none'
        )

    ax.set_title("Espacio semántico CFH — t-SNE\nSeparación entre sistemas de justicia",
                 color='white', fontsize=14, pad=15)
    ax.set_xlabel("t-SNE 1", color='white')
    ax.set_ylabel("t-SNE 2", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E3A4C')

    ax.legend(facecolor='#1E3A4C', edgecolor='#0D9488',
              labelcolor='white', fontsize=10, markerscale=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D2137')
    plt.close()
    print(f"✓ t-SNE guardado: {output_path}")
    return embedding


# ── Convergencia temporal ─────────────────────────────────────────────────────

def analizar_convergencia_temporal(df, output_path="convergencia_temporal.png"):
    """
    Mide si la JEP se acerca semánticamente a las víctimas año a año.
    Responde: ¿la distancia MAFAPO disminuye con el tiempo en el Corpus B?
    """
    if "y8_mafapo_cs" not in df.columns or "año" not in df.columns:
        print("⚠ Columnas y8_mafapo_cs o año no disponibles")
        return None

    df_temporal = df[df["año"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0D2137')

    for ax in axes:
        ax.set_facecolor('#1E3A4C')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0D9488')

    # ── Panel 1: Dist MAFAPO por año y corpus ──
    for corpus_type, color in COLORS.items():
        df_corp = df_temporal[df_temporal["corpus_type"] == corpus_type]
        if len(df_corp) == 0:
            continue
        media_año = df_corp.groupby("año")["y8_mafapo_cs"].mean()
        axes[0].plot(media_año.index, media_año.values,
                     color=color, marker='o', linewidth=2,
                     label=LABELS[corpus_type], markersize=5)

    axes[0].set_title("Distancia MAFAPO por año\n(menor = más cercano a víctimas)",
                      color='white', fontsize=11)
    axes[0].set_xlabel("Año", color='white')
    axes[0].set_ylabel("Distancia coseno MAFAPO (y₈)", color='white')
    axes[0].legend(facecolor='#0D2137', labelcolor='white', fontsize=8)
    axes[0].axhline(y=df_temporal[df_temporal["corpus_type"]=="B"]["y8_mafapo_cs"].mean(),
                    color='#0D9488', linestyle='--', alpha=0.5, label='Media JEP')

    # ── Panel 2: Dist CIDH por año y corpus ──
    for corpus_type, color in COLORS.items():
        df_corp = df_temporal[df_temporal["corpus_type"] == corpus_type]
        if len(df_corp) == 0 or "y9_cidh_cs" not in df_corp.columns:
            continue
        media_año = df_corp.groupby("año")["y9_cidh_cs"].mean()
        axes[1].plot(media_año.index, media_año.values,
                     color=color, marker='s', linewidth=2,
                     label=LABELS[corpus_type], markersize=5)

    axes[1].set_title("Distancia CIDH por año\n(menor = más cercano al DIH)",
                      color='white', fontsize=11)
    axes[1].set_xlabel("Año", color='white')
    axes[1].set_ylabel("Distancia coseno CIDH (y₉)", color='white')
    axes[1].legend(facecolor='#0D2137', labelcolor='white', fontsize=8)

    plt.suptitle("Convergencia temporal — ¿La JEP se acerca a las víctimas?",
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D2137')
    plt.close()
    print(f"✓ Convergencia temporal guardada: {output_path}")

    # Estadísticas de tendencia para Corpus B
    df_b = df_temporal[df_temporal["corpus_type"] == "B"]
    if len(df_b) > 0 and "y8_mafapo_cs" in df_b.columns:
        from scipy.stats import pearsonr
        media_b = df_b.groupby("año")["y8_mafapo_cs"].mean()
        if len(media_b) > 2:
            r, p = pearsonr(media_b.index, media_b.values)
            print(f"\nCorrelación año-distancia MAFAPO en Corpus B:")
            print(f"  r={r:.3f}, p={p:.4f} — {'↓ Convergencia' if r < 0 else '↑ Divergencia'}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualización embeddings CFH")
    parser.add_argument("--indicators", default="indicators_final_completo.csv")
    parser.add_argument("--corpus_c",   default="indicators_corpus_c.csv")
    parser.add_argument("--output_dir", default="outputs/visualizaciones")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("== VISUALIZACIÓN ESPACIO SEMÁNTICO CFH ==\n")
    df = cargar_datos(args.indicators, args.corpus_c)

    # UMAP
    generar_umap(df, f"{args.output_dir}/umap_cfh.png")

    # t-SNE
    generar_tsne(df, f"{args.output_dir}/tsne_cfh.png")

    # Convergencia temporal
    analizar_convergencia_temporal(df, f"{args.output_dir}/convergencia_temporal.png")

    print("\n✓ Todas las visualizaciones generadas en:", args.output_dir)
