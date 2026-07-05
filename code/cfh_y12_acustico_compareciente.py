# -*- coding: utf-8 -*-
"""
CFH — y12: Score acustico de reconocimiento POR COMPARECIENTE (n=47)
====================================================================
Indicador de eta2 (Transicion Epistemica), canal ACUSTICO.
Definicion (modelo_medicion_cfh_v2): "rasgos prosodicos asociados a la
expresion de carga emocional y reconocimiento en testimonios de trauma:
tasa de habla reducida, mayor proporcion de pausas largas, energia vocal
moderada." Extraido con OpenSMILE eGeMAPS v02.

METODO DE COMPOSICION: identico a icm_vocal (cfh_vocal_corregido.py):
  z-score a nivel subcaso -> tanh(z*0.5) -> promedio -> sigmoide [0,1].
Coherencia metodologica con la capa vocal existente.

RASGOS (segun definicion documentada de y12):
  - tasa de habla reducida  -> VoicedSegmentsPerSec        (INVERTIDO: menos = mas reconocimiento)
  - pausas largas           -> MeanUnvoicedSegmentLength   (directo: mas = mas reconocimiento)
  - energia moderada        -> loudness_sma3_amean         (INVERTIDO |z|: penaliza extremos)

Corre LOCAL (solo pandas/numpy, sin GPU).

Uso:
    python code/cfh_y12_acustico_compareciente.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAPA3 = REPO / "outputs" / "capa3"
OUT = REPO / "data" / "referencias" / "y12_acustico_compareciente.csv"

# Rasgos eGeMAPS para y12 (segun definicion documentada)
FEAT_HABLA   = "VoicedSegmentsPerSec"        # tasa de habla (invertir: menos = reconocimiento)
FEAT_PAUSAS  = "MeanUnvoicedSegmentLength"   # pausas largas (directo)
FEAT_ENERGIA = "loudness_sma3_amean"         # energia (moderada = -|z|)

SUBCASOS = ["casanare", "catatumbo", "costa_caribe", "dabeiba", "huila"]
MIN_VENTANAS = 3  # igual que icm_vocal

# Mapa al formato canonico del SEM (icm_tricanal_final.csv)
SUBCASO_CANONICO = {
    "casanare": "Casanare",
    "catatumbo": "Catatumbo",
    "costa_caribe": "CostaCaribe",
    "dabeiba": "Dabeiba",
    "huila": "Huila",
}


def y12_score_subcaso(eg):
    """z-score a nivel subcaso -> tanh(z*0.5) -> promedio -> sigmoide. Igual
    metodo que icm_vocal, con los 3 rasgos de reconocimiento acustico."""
    feats = [FEAT_HABLA, FEAT_PAUSAS, FEAT_ENERGIA]
    feats_ok = [f for f in feats if f in eg.columns]
    if not feats_ok:
        return pd.DataFrame()

    mu = {f: eg[f].mean() for f in feats_ok}
    sd = {f: eg[f].std() + 1e-9 for f in feats_ok}

    def score_persona(sub):
        vals = []
        for f in feats_ok:
            z = (sub[f] - mu[f]) / sd[f]
            if f == FEAT_HABLA:
                z = -z                 # habla reducida = mas reconocimiento
            elif f == FEAT_ENERGIA:
                z = -np.abs(z)         # energia moderada: penaliza extremos
            # FEAT_PAUSAS queda directo (mas pausas = mas reconocimiento)
            vals.append(np.tanh(z * 0.5))
        sc = np.mean(np.column_stack(vals), axis=1)
        return float(1 / (1 + np.exp(-np.mean(sc) * 2)))

    filas = []
    for persona, sub in eg.groupby("identidad"):
        if len(sub) < MIN_VENTANAS:
            filas.append({"identidad": persona, "n_ventanas": len(sub),
                          "y12_acustico": None, "estado": "pocas_ventanas"})
            continue
        filas.append({"identidad": persona, "n_ventanas": len(sub),
                      "y12_acustico": round(score_persona(sub), 4),
                      "estado": "ok"})
    return pd.DataFrame(filas)


def main():
    print("=" * 64)
    print("CFH — y12: Score acustico de reconocimiento por compareciente")
    print("=" * 64)

    todo = []
    for sub in SUBCASOS:
        path = CAPA3 / f"egemap_{sub}_compareciente.csv"
        if not path.exists():
            print(f"  [SALTADO] no existe {path.name}")
            continue
        eg = pd.read_csv(path)
        res = y12_score_subcaso(eg)
        res["subcaso"] = SUBCASO_CANONICO.get(sub, sub)
        todo.append(res)
        ok = res[res["y12_acustico"].notna()]
        print(f"  {sub:14s}: {len(ok)} comparecientes con y12 "
              f"(rango {ok['y12_acustico'].min():.3f}-{ok['y12_acustico'].max():.3f})"
              if len(ok) else f"  {sub:14s}: sin datos suficientes")

    if not todo:
        print("  [ERROR] no se genero ningun score.")
        return

    df = pd.concat(todo, ignore_index=True)
    df = df[["subcaso", "identidad", "n_ventanas", "y12_acustico", "estado"]]

    # ── Filtrar a los 47 comparecientes definitivos del SEM ──────────
    base = pd.read_csv(REPO / "outputs" / "capa3" / "icm_tricanal_final.csv")
    base_keys = base[["subcaso", "identidad"]].copy()
    n_antes = len(df)
    df_47 = base_keys.merge(df, on=["subcaso", "identidad"], how="left")
    con_y12 = df_47["y12_acustico"].notna().sum()
    print(f"\n  Filtrado a los 47 del SEM: {n_antes} -> {len(df_47)} "
          f"(con y12: {con_y12}, sin y12: {len(df_47)-con_y12})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_47.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"  Guardado: {OUT}")
    print("\n  Estadisticas y12 (47 comparecientes):")
    print(df_47["y12_acustico"].describe().round(4).to_string())
    print("\n  Por subcaso:")
    print(df_47.groupby("subcaso")["y12_acustico"].agg(["mean","min","max","count"]).round(4).to_string())
    print("=" * 64)


if __name__ == "__main__":
    main()
