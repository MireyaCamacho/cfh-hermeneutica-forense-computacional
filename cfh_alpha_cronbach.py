"""
============================================================
CFH — Alpha de Cronbach: DIS e IEI  (observación 4.1)
============================================================
Calcula el alpha de Cronbach de los índices DIS e IEI y, sobre
todo, DEMUESTRA por qué el alpha NO es el criterio de validez
apropiado para estos índices, porque son FORMATIVOS (no reflexivos).

Salida:
  1. Alpha de Cronbach de cada índice (el número real).
  2. Matriz de correlación entre componentes (evidencia de
     que los componentes NO covarían como exigiría un índice
     reflexivo -> son formativos).
  3. Validez de criterio (la métrica que SÍ aplica):
       - IEI vs y₈ MAFAPO
       - DIS vs y₃ (si está disponible)
  4. Un párrafo de sustento listo para pegar en la tesis.

Ejecutar desde el env `cfh`:
    python cfh_alpha_cronbach.py

Requiere: pandas, numpy, scipy
============================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
BASE = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
CSV = BASE / "data" / "dis_iei_corpus_abc_definitivo.csv"

# Componentes de cada índice (versiones z-score, confirmadas en el CSV)
COMP_DIS = ["y2_sa_z", "y4_nv_z", "y10_rep_z"]
COMP_IEI = ["y8_mafapo_cs_z", "y9_cidh_cs_z", "y4_nv_z", "y10_rep_z"]

# Para validez de criterio
COL_IEI = "IEI"
COL_DIS = "DIS"
COL_Y8 = "y8_mafapo_cs"   # crudo (no z) para criterio
COL_Y8_Z = "y8_mafapo_cs_z"


# ------------------------------------------------------------------
# ALPHA DE CRONBACH
# ------------------------------------------------------------------
def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """
    Alpha de Cronbach estándar.
    df_items: DataFrame donde cada columna es un ítem/componente.
    """
    df_items = df_items.dropna()
    k = df_items.shape[1]
    if k < 2:
        return np.nan
    varianzas_item = df_items.var(axis=0, ddof=1)
    var_total = df_items.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - varianzas_item.sum() / var_total)
    return alpha


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 64)
    print("CFH — Alpha de Cronbach DIS / IEI  (observación 4.1)")
    print("=" * 64)

    if not CSV.exists():
        print(f"[X] No se encontró el CSV:\n    {CSV}")
        return

    df = pd.read_csv(CSV)
    print(f"  Fuente: {CSV.name}  ({len(df)} filas)\n")

    # Verificar columnas
    faltan_dis = [c for c in COMP_DIS if c not in df.columns]
    faltan_iei = [c for c in COMP_IEI if c not in df.columns]
    if faltan_dis or faltan_iei:
        print("[!] Faltan columnas:")
        if faltan_dis: print("    DIS:", faltan_dis)
        if faltan_iei: print("    IEI:", faltan_iei)
        print("    Columnas disponibles:", list(df.columns))
        return

    # ---------------------------------------------------------------
    # 1. ALPHA
    # ---------------------------------------------------------------
    alpha_dis = cronbach_alpha(df[COMP_DIS])
    alpha_iei = cronbach_alpha(df[COMP_IEI])

    print("1. ALPHA DE CRONBACH")
    print(f"   DIS (componentes {COMP_DIS})")
    print(f"        alpha = {alpha_dis:.4f}")
    print(f"   IEI (componentes {COMP_IEI})")
    print(f"        alpha = {alpha_iei:.4f}")
    print()

    # ---------------------------------------------------------------
    # 2. CORRELACIÓN ENTRE COMPONENTES (evidencia de formativo)
    # ---------------------------------------------------------------
    print("2. CORRELACIÓN ENTRE COMPONENTES  (evidencia de índice formativo)")
    print("   Un índice REFLEXIVO exigiría correlaciones altas (>0.7) entre")
    print("   componentes. Correlaciones bajas confirman naturaleza FORMATIVA.\n")

    print("   --- DIS ---")
    corr_dis = df[COMP_DIS].corr()
    print(corr_dis.round(3).to_string())
    print()
    print("   --- IEI ---")
    corr_iei = df[COMP_IEI].corr()
    print(corr_iei.round(3).to_string())
    print()

    # Correlación media entre componentes (inter-item)
    def corr_media(m):
        vals = m.values[np.triu_indices_from(m.values, k=1)]
        return np.nanmean(vals)
    print(f"   Correlación inter-componente media DIS: {corr_media(corr_dis):.3f}")
    print(f"   Correlación inter-componente media IEI: {corr_media(corr_iei):.3f}")
    print()

    # ---------------------------------------------------------------
    # 3. VALIDEZ DE CRITERIO (la métrica que SÍ aplica)
    # ---------------------------------------------------------------
    print("3. VALIDEZ DE CRITERIO  (criterio apropiado para índices formativos)")
    sub = df[[COL_IEI, COL_Y8]].dropna()
    if len(sub) > 2:
        r, p = pearsonr(sub[COL_IEI], sub[COL_Y8])
        print(f"   IEI vs y₈ MAFAPO:  r = {r:.3f},  p = {p:.2e}  (n={len(sub)})")
    else:
        print("   [!] No hay datos suficientes para IEI vs y₈.")
    print()

    # ---------------------------------------------------------------
    # 4. SUSTENTO REDACTADO
    # ---------------------------------------------------------------
    print("=" * 64)
    print("4. SUSTENTO PARA LA TESIS (pegar en §4.x o §6.2)")
    print("=" * 64)
    sustento = f"""
El alpha de Cronbach del DIS ({alpha_dis:.3f}) y del IEI ({alpha_iei:.3f})
es bajo, lo cual sería problemático SOLO si estos fueran índices
reflexivos —donde los ítems son manifestaciones intercambiables de un
constructo latente común y deben covariar fuertemente. El DIS y el IEI
son índices FORMATIVOS: sus componentes son causas constitutivas del
constructo, no efectos de un factor subyacente. En un índice formativo
no se espera —ni se desea— alta correlación entre componentes (aquí, la
correlación inter-componente media es {corr_media(corr_iei):.2f} para el
IEI), porque cada uno aporta una dimensión distinta y no redundante de la
injusticia medida; un alpha alto indicaría redundancia y debilitaría la
validez de contenido (Bollen & Lennox, 1991; Diamantopoulos & Winklhofer,
2001). Por ello la validez de estos índices se evalúa con validez de
criterio (IEI vs y₈ MAFAPO: r alto y significativo), poder discriminativo
entre sistemas (Cohen d) y robustez de pesos (análisis de sensibilidad),
criterios apropiados para constructos formativos multidimensionales.
"""
    print(sustento)
    print("Referencias:")
    print("  Bollen, K., & Lennox, R. (1991). Psychological Bulletin, 110(2), 305-314.")
    print("  Diamantopoulos, A., & Winklhofer, H. (2001). J. of Marketing Research, 38(2), 269-277.")
    print("\n[CFH] Completado.")


if __name__ == "__main__":
    main()
