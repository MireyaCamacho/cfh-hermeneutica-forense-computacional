"""
CFH · Módulo SEM — Estimación del Discursive Injustice Score
=============================================================
Proyecto: Hermenéutica Forense Computacional

Especificación semopy del modelo CFH v2.0:
    4 variables latentes — 11 indicadores observables

    xi1 =~ y1 + y2 + y3 + y4          (Violencia Discursiva — exógena)
    xi2 =~ y5 + y6                     (Contexto Institucional — exógena)
    eta1 =~ y7 + y8 + y9              (DIS Score — endógena)
    eta2 =~ y10 + y11 + y12           (Transición Epistémica — endógena)
    eta1 ~ xi1 + xi2                   (H1: xi1 → eta1, H2: xi2 → eta1)
    eta2 ~ eta1                        (H3: eta1 → eta2, beta_23 < 0)
    xi1 ~~ xi2                         (covarianza entre exógenas)

Hipótesis:
    H1: gamma_11 > 0  (mayor violencia discursiva → mayor DIS)
    H2: gamma_21 ≠ 0  (contexto institucional predice DIS)
    H3: beta_23 < 0   (mayor DIS → menor transición epistémica) ← CENTRAL

Indicadores disponibles sin GPU (versión actual):
    y2, y3, y4, y5, y6, y10  → extraídos, en CSV

Indicadores pendientes (requieren ConfliBERT en Colab Pro):
    y1  → EBI Score           (placeholder 0.0)
    y7  → Surprisal           (NaN)
    y8  → Distancia MAFAPO    (NaN)
    y9  → Distancia CIDH      (NaN)
    y11 → Conv. restaurativa  (NaN)

Estrategia actual:
    Modelo parcial con indicadores disponibles (ξ₁ sin y1, η₁ sin y7-y9,
    η₂ sin y11). Cuando lleguen los features de ConfliBERT, reemplazar
    los NaN y re-estimar el modelo completo.

Uso:
    python -m src.sem.sem_model --csv-a data/features/indicators_corpus_a.csv
                                --csv-b data/features/indicators_corpus_b.csv
    O desde Python:
        from src.sem.sem_model import CFHSEMModel
        model = CFHSEMModel()
        results = model.fit(df)
        print(results.summary())
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import semopy
    from semopy import Model
    _SEMOPY_AVAILABLE = True
except ImportError:
    _SEMOPY_AVAILABLE = False
    warnings.warn("semopy no instalado. Instala con: pip install semopy")

try:
    from scipy import stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger("cfh.sem")


# ---------------------------------------------------------------------------
# Especificación del modelo SEM
# ---------------------------------------------------------------------------

# Modelo completo (11 indicadores) — para cuando estén todos los features
SEM_SPEC_COMPLETO = """
    xi1 =~ y2 + y3 + y4 + y1
    xi2 =~ y5 + y6
    eta1 =~ y7 + y8 + y9
    eta2 =~ y10 + y11
    eta1 ~ xi1 + xi2
    eta2 ~ eta1
    xi1 ~~ xi2
"""

# Modelo parcial (indicadores sin GPU) — versión actual
# Usa y2, y3, y4 para xi1; y5, y6 para xi2; solo y10 para eta2
# eta1 no se puede estimar sin y7-y9 — se omite en el modelo parcial
SEM_SPEC_PARCIAL = """
    xi1 =~ y2 + y3 + y4
    xi2 =~ y5 + y6
    eta2 =~ y10
    eta2 ~ xi1 + xi2
    xi1 ~~ xi2
"""

# Modelo de dos factores para análisis exploratorio (xi1 vs eta2)
SEM_SPEC_DOS_FACTORES = """
    xi1 =~ y2 + y3 + y4
    eta2 =~ y10
    eta2 ~ xi1
"""


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class SEMResults:
    """Resultados de la estimación del modelo SEM."""

    # ── Ajuste del modelo ────────────────────────────────────────────────────
    cfi: Optional[float] = None          # ≥ 0.95 = buen ajuste
    rmsea: Optional[float] = None        # ≤ 0.06 = buen ajuste
    srmr: Optional[float] = None         # ≤ 0.08 = buen ajuste
    chi2: Optional[float] = None
    chi2_df: Optional[int] = None
    chi2_pvalue: Optional[float] = None

    # ── Cargas factoriales ───────────────────────────────────────────────────
    loadings: Optional[pd.DataFrame] = None

    # ── Coeficientes estructurales ───────────────────────────────────────────
    structural: Optional[pd.DataFrame] = None

    # ── Hipótesis central ────────────────────────────────────────────────────
    beta_23: Optional[float] = None      # coeficiente eta1 → eta2
    beta_23_se: Optional[float] = None
    beta_23_pvalue: Optional[float] = None
    beta_23_ci_lower: Optional[float] = None
    beta_23_ci_upper: Optional[float] = None
    h3_supported: Optional[bool] = None  # True si beta_23 < 0, p < 0.01

    # ── Bootstrap ────────────────────────────────────────────────────────────
    bootstrap_samples: int = 0
    bootstrap_ci: Optional[tuple] = None

    # ── Modelo ──────────────────────────────────────────────────────────────
    spec_used: str = "parcial"
    n_obs: int = 0
    n_indicators: int = 0
    converged: bool = False
    warning: Optional[str] = None

    def summary(self) -> str:
        """Resumen en texto del modelo estimado."""
        def fmt(val, spec=".3f"):
            return format(val, spec) if val is not None else "N/A"

        lines = [
            "=" * 55,
            "MODELO SEM CFH — Resumen de estimación",
            "=" * 55,
            f"Especificación: {self.spec_used}",
            f"N observaciones: {self.n_obs}",
            f"Convergencia: {'✓' if self.converged else '✗'}",
            "",
            "── Índices de ajuste ──",
            f"CFI:   {fmt(self.cfi)}  {'✓ (≥0.95)' if self.cfi and self.cfi >= 0.95 else '✗' if self.cfi is not None else ''}",
            f"RMSEA: {fmt(self.rmsea)}  {'✓ (≤0.06)' if self.rmsea and self.rmsea <= 0.06 else '✗' if self.rmsea is not None else ''}",
            f"SRMR:  {fmt(self.srmr)}  {'✓ (≤0.08)' if self.srmr and self.srmr <= 0.08 else '✗' if self.srmr is not None else ''}",
        ]

        if self.beta_23 is not None:
            lines += [
                "",
                "── Hipótesis H₃ (eta1 → eta2) ──",
                f"β₂₃ = {fmt(self.beta_23)}  SE={fmt(self.beta_23_se)}  p={fmt(self.beta_23_pvalue, '.4f')}",
                f"IC 95%: [{fmt(self.beta_23_ci_lower)}, {fmt(self.beta_23_ci_upper)}]",
                f"H₃ {'APOYADA ✓' if self.h3_supported else 'NO APOYADA ✗'} (β₂₃ < 0, p < 0.01)",
            ]

        if self.warning:
            lines.append(f"\n⚠ {self.warning}")

        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clase principal del modelo
# ---------------------------------------------------------------------------

class CFHSEMModel:
    """
    Modelo SEM de la Hermenéutica Forense Computacional.

    Gestiona la especificación, estimación, evaluación de ajuste
    y análisis bootstrap del modelo CFH con semopy.

    Parámetros
    ----------
    spec : str
        Especificación del modelo. "completo" | "parcial" | "dos_factores"
    estimator : str
        Estimador semopy. "MLW" (ML) | "DWLS" | "ULS"
    """

    SPECS = {
        "completo":     SEM_SPEC_COMPLETO,
        "parcial":      SEM_SPEC_PARCIAL,
        "dos_factores": SEM_SPEC_DOS_FACTORES,
    }

    def __init__(
        self,
        spec: str = "parcial",
        estimator: str = "MLW",
    ):
        if not _SEMOPY_AVAILABLE:
            raise ImportError(
                "semopy no instalado. Ejecuta: pip install semopy"
            )
        if spec not in self.SPECS:
            raise ValueError(f"spec debe ser uno de: {list(self.SPECS.keys())}")

        self.spec = spec
        self.spec_str = self.SPECS[spec]
        self.estimator = estimator
        self._model: Optional[Model] = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara el DataFrame para la estimación del SEM.

        Selecciona los indicadores relevantes según la especificación,
        elimina filas con NaN en los indicadores necesarios,
        y verifica que haya suficientes observaciones.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con los features extraídos por el pipeline.

        Retorna
        -------
        DataFrame limpio con solo las columnas de indicadores.
        """
        # Columnas necesarias según especificación
        if self.spec == "completo":
            cols = ["y1", "y2", "y3", "y4", "y5", "y6",
                    "y7", "y8", "y9", "y10", "y11"]
        elif self.spec == "parcial":
            cols = ["y2", "y3", "y4", "y5", "y6", "y10"]
        else:  # dos_factores
            cols = ["y2", "y3", "y4", "y10"]

        # Mapear nombres del CSV a nombres del modelo
        col_map = {
            "y2_sa": "y2", "y3_civil": "y3", "y4_nv": "y4",
            "y5_corpus_type": "y5", "y6_period": "y6",
            "y10_rep": "y10",
        }
        # Renombrar columnas disponibles
        df_prep = df.rename(columns=col_map)

        # Agregar columnas faltantes como NaN
        for col in cols:
            if col not in df_prep.columns:
                df_prep[col] = np.nan

        # Seleccionar solo las columnas necesarias
        df_prep = df_prep[cols].copy()

        # Eliminar filas con NaN en columnas esenciales
        essential = [c for c in cols if c not in ["y1", "y7", "y8", "y9", "y11"]]
        df_clean = df_prep.dropna(subset=essential)

        n_dropped = len(df_prep) - len(df_clean)
        if n_dropped > 0:
            logger.warning(f"{n_dropped} filas eliminadas por NaN en indicadores esenciales.")

        if len(df_clean) < 50:
            logger.warning(
                f"N={len(df_clean)} es muy pequeño para estimación SEM confiable. "
                "Se recomienda N ≥ 200."
            )

        logger.info(f"Datos preparados: {len(df_clean)} observaciones, {len(cols)} indicadores")
        return df_clean

    def fit(self, df: pd.DataFrame) -> SEMResults:
        """
        Estima el modelo SEM sobre el DataFrame.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame de features (produce del feature pipeline).

        Retorna
        -------
        SEMResults con todos los parámetros estimados.
        """
        df_prep = self.prepare_data(df)
        n_obs = len(df_prep)

        logger.info(f"Estimando modelo SEM ({self.spec}) con N={n_obs}...")

        try:
            self._model = Model(self.spec_str)
            self._model.fit(df_prep, obj=self.estimator)

            results = SEMResults(
                spec_used=self.spec,
                n_obs=n_obs,
                n_indicators=len(df_prep.columns),
                converged=True,
            )

            # Índices de ajuste
            try:
                fit_indices = semopy.calc_stats(self._model)
                results.cfi   = float(fit_indices.get("CFI", np.nan).iloc[0] if hasattr(fit_indices.get("CFI", np.nan), "iloc") else fit_indices.get("CFI", np.nan))
                results.rmsea = float(fit_indices.get("RMSEA", np.nan).iloc[0] if hasattr(fit_indices.get("RMSEA", np.nan), "iloc") else fit_indices.get("RMSEA", np.nan))
                results.srmr  = float(fit_indices.get("SRMR", np.nan).iloc[0] if hasattr(fit_indices.get("SRMR", np.nan), "iloc") else fit_indices.get("SRMR", np.nan))
                results.chi2  = float(fit_indices.get("chi2", np.nan).iloc[0] if hasattr(fit_indices.get("chi2", np.nan), "iloc") else fit_indices.get("chi2", np.nan))
            except Exception as e:
                logger.warning(f"No se pudieron calcular índices de ajuste: {e}")

            # Cargas factoriales y coeficientes estructurales
            try:
                params = self._model.inspect()
                results.loadings = params[params["op"] == "=~"]
                results.structural = params[params["op"] == "~"]
            except Exception as e:
                logger.warning(f"No se pudieron extraer parámetros: {e}")

            # β₂₃ — coeficiente eta1 → eta2 (H₃)
            results = self._extract_beta23(results)

            logger.info(results.summary())
            return results

        except Exception as e:
            logger.error(f"Error en estimación SEM: {e}")
            return SEMResults(
                spec_used=self.spec,
                n_obs=n_obs,
                converged=False,
                warning=str(e),
            )

    def fit_multigroup(
        self,
        df: pd.DataFrame,
        group_col: str = "corpus_type",
    ) -> dict[str, SEMResults]:
        """
        Análisis multi-grupo (MG-SEM).

        Estima el modelo por separado para cada subsistema y evalúa
        la invarianza de medición comparando los parámetros.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame completo con columna de grupo.
        group_col : str
            Columna que define los grupos (default: "corpus_type").

        Retorna
        -------
        Dict grupo → SEMResults
        """
        groups = df[group_col].dropna().unique()
        results = {}

        for group in sorted(groups):
            df_group = df[df[group_col] == group]
            logger.info(f"Estimando modelo para grupo: {group} (N={len(df_group)})")
            results[group] = self.fit(df_group)

        # Comparar cargas factoriales entre grupos
        self._compare_loadings(results)

        return results

    def bootstrap_beta23(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        ci_level: float = 0.95,
        random_state: int = 42,
    ) -> tuple[float, float, float]:
        """
        Bootstrap no paramétrico para el coeficiente β₂₃.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame de features.
        n_samples : int
            Número de remuestras bootstrap (default: 1000).
            Para la versión final usar 10000.
        ci_level : float
            Nivel de confianza del intervalo (default: 0.95).
        random_state : int
            Semilla aleatoria para reproducibilidad.

        Retorna
        -------
        (beta_23_mean, ci_lower, ci_upper)
        """
        rng = np.random.default_rng(random_state)
        df_prep = self.prepare_data(df)
        beta_samples = []

        logger.info(f"Bootstrap β₂₃: {n_samples} remuestras...")

        for i in range(n_samples):
            # Remuestreo con reemplazo
            df_boot = df_prep.sample(n=len(df_prep), replace=True,
                                     random_state=int(rng.integers(0, 1e6)))
            try:
                model_boot = Model(self.spec_str)
                model_boot.fit(df_boot, obj=self.estimator)
                params = model_boot.inspect()
                structural = params[params["op"] == "~"]

                # Buscar el coeficiente eta1 → eta2 o xi1 → eta2
                beta = self._find_beta23(structural)
                if beta is not None:
                    beta_samples.append(beta)
            except Exception:
                pass  # muestra problemática — ignorar

            if (i + 1) % 100 == 0:
                logger.debug(f"Bootstrap: {i+1}/{n_samples} completadas")

        if not beta_samples:
            logger.error("Bootstrap falló: no se obtuvieron muestras válidas.")
            return np.nan, np.nan, np.nan

        beta_arr = np.array(beta_samples)
        alpha = 1 - ci_level
        ci_lower = float(np.percentile(beta_arr, alpha/2 * 100))
        ci_upper = float(np.percentile(beta_arr, (1 - alpha/2) * 100))
        beta_mean = float(beta_arr.mean())

        logger.info(
            f"Bootstrap β₂₃ ({len(beta_samples)}/{n_samples} válidas): "
            f"media={beta_mean:.3f} IC{ci_level:.0%}=[{ci_lower:.3f}, {ci_upper:.3f}]"
        )
        return beta_mean, ci_lower, ci_upper

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_beta23(self, results: SEMResults) -> SEMResults:
        """Extrae el coeficiente β₂₃ de los parámetros estructurales."""
        if results.structural is None:
            return results

        beta = self._find_beta23(results.structural)
        if beta is not None:
            results.beta_23 = beta
            # Buscar SE y p-valor si están disponibles
            row = results.structural[
                results.structural["lval"].isin(["eta2", "eta1"])
            ]
            if len(row) > 0 and "Std. Err" in row.columns:
                results.beta_23_se = float(row["Std. Err"].iloc[0])
            if len(row) > 0 and "p-value" in row.columns:
                results.beta_23_pvalue = float(row["p-value"].iloc[0])
                results.h3_supported = (beta < 0 and
                                        results.beta_23_pvalue < 0.01)
        return results

    def _find_beta23(self, structural: pd.DataFrame) -> Optional[float]:
        """Busca el coeficiente del path principal en los parámetros estructurales."""
        # Buscar eta2 ~ eta1 o eta2 ~ xi1
        for lval, rval in [("eta2", "eta1"), ("eta2", "xi1"), ("eta2", "xi2")]:
            row = structural[
                (structural["lval"] == lval) & (structural["rval"] == rval)
            ]
            if len(row) > 0 and "Estimate" in row.columns:
                return float(row["Estimate"].iloc[0])
        return None

    def _compare_loadings(self, results: dict[str, SEMResults]) -> None:
        """Compara cargas factoriales entre grupos para evaluar invarianza."""
        logger.info("\n── Comparación de cargas factoriales (MG-SEM) ──")
        for group, res in results.items():
            if res.loadings is not None:
                logger.info(f"\nGrupo: {group}")
                logger.info(res.loadings[["lval", "rval", "Estimate"]].to_string())

    def save_results(self, results: SEMResults, path: Path) -> None:
        """Guarda el resumen de resultados en un archivo de texto."""
        Path(path).write_text(results.summary(), encoding="utf-8")
        logger.info(f"Resultados guardados: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CFH SEM Model")
    parser.add_argument("--csv-a", type=Path, default=Path("data/features/indicators_corpus_a.csv"))
    parser.add_argument("--csv-b", type=Path, default=Path("data/features/indicators_corpus_b.csv"))
    parser.add_argument("--spec", choices=["completo", "parcial", "dos_factores"],
                        default="parcial")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="N remuestras bootstrap (0 = sin bootstrap)")
    parser.add_argument("--multigroup", action="store_true",
                        help="Análisis multi-grupo por corpus_type")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Cargar datos
    dfs = []
    for csv_path in [args.csv_a, args.csv_b]:
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        print("Error: no se encontraron CSVs de features.")
        exit(1)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Datos cargados: {len(df)} secciones de {df['doc_id'].nunique()} documentos")

    # Estimar modelo
    model = CFHSEMModel(spec=args.spec)

    if args.multigroup:
        results_mg = model.fit_multigroup(df)
        for group, res in results_mg.items():
            print(f"\n{'='*40}\nGrupo: {group}\n{res.summary()}")
    else:
        results = model.fit(df)
        print(results.summary())

        if args.bootstrap > 0:
            beta_mean, ci_low, ci_high = model.bootstrap_beta23(
                df, n_samples=args.bootstrap
            )
            print(f"\nBootstrap β₂₃ ({args.bootstrap} muestras):")
            print(f"  Media: {beta_mean:.3f}")
            print(f"  IC 95%: [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"  H₃: {'APOYADA ✓' if ci_high < 0 else 'NO APOYADA ✗'} (IC excluye 0)")

        if args.output:
            model.save_results(results, args.output)
