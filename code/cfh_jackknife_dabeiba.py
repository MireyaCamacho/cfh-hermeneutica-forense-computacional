# -*- coding: utf-8 -*-
r"""
cfh_jackknife_dabeiba.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

CHEQUEO COMPLEMENTARIO (no modifica nada previo): jackknife del hallazgo
exploratorio de Dabeiba. El SEM por subcaso mostro en Dabeiba (n=12) una
correlacion fuerte injust~transic (aprox -0.748). Como n=12 es pequeno, se
verifica si esa correlacion es ROBUSTA o depende de 1-2 comparecientes.

METODO: recalcula la correlacion quitando un compareciente a la vez (n
iteraciones). Reporta el rango, la media y si algun caso la desploma.

Reusa la MISMA logica de cfh_heterogeneidad_H3.py:
  viol    = media z(y1_ebi, y2_sa, y4_nv)
  injust  = media z(1-y8_mafapo, 1-y9_cidh)   [invertido = injusticia]
  transic = media z(y10_rep, y11_conv_rest)
  H (clave) = corr(injust, transic)  reestandarizando dentro del subgrupo

NO altera archivos previos. Salida propia:
  outputs/jackknife_dabeiba_reporte.txt

Uso (raiz del repo, env cfh):
    python code\cfh_jackknife_dabeiba.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_SEM = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")
CSV_MR = os.path.join(REPO, "data", "mr_asignacion_final.csv")
OUT = os.path.join(REPO, "outputs", "jackknife_dabeiba_reporte.txt")

SUBCASO = "Dabeiba"


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def z(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 1e-9 else 1.0)


def corr_seg(sub, a, b):
    if len(sub) < 3:
        return np.nan
    return z(sub[a]).corr(z(sub[b]))


def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)

    df = pd.read_csv(CSV_SEM)

    # construir indices observados (misma logica que heterogeneidad_H3)
    df["viol"] = (z(df["y1_ebi"]) + z(df["y2_sa"]) + z(df["y4_nv"])) / 3
    inj8 = 1.0 - df["y8_mafapo"]
    inj9 = 1.0 - df["y9_cidh"]
    df["injust"] = (z(inj8) + z(inj9)) / 2
    df["transic"] = (z(df["y10_rep"]) + z(df["y11_conv_rest"])) / 2

    # subcaso Dabeiba
    if "subcaso" not in df.columns:
        print("[ERROR] no hay columna subcaso en el CSV del SEM")
        return
    dab = df[df["subcaso"] == SUBCASO].copy().reset_index(drop=True)

    print("=" * 66)
    print(f"JACKKNIFE - subcaso {SUBCASO}")
    print("=" * 66)
    print(f"n = {len(dab)} comparecientes")

    if len(dab) < 4:
        print("n demasiado bajo para jackknife interpretable.")
        return

    # correlacion completa
    h_full = corr_seg(dab, "injust", "transic")
    print(f"\nCorrelacion COMPLETA injust~transic: {h_full:+.3f}")

    # jackknife: quitar uno a la vez
    print("\n--- Jackknife (quitando un compareciente por vez) ---")
    vals = []
    ident_col = "identidad" if "identidad" in dab.columns else dab.columns[0]
    for i in range(len(dab)):
        sub = dab.drop(index=i)
        h = corr_seg(sub, "injust", "transic")
        vals.append(h)
        quitado = str(dab.loc[i, ident_col])[:35]
        delta = h - h_full
        flag = "  <- cambia mucho" if abs(delta) > 0.20 else ""
        print(f"  sin {quitado:<37}: {h:+.3f}  (delta {delta:+.3f}){flag}")

    vals = np.array(vals)
    print("\n--- Resumen jackknife ---")
    print(f"  correlacion completa : {h_full:+.3f}")
    print(f"  media jackknife      : {vals.mean():+.3f}")
    print(f"  rango                : [{vals.min():+.3f}, {vals.max():+.3f}]")
    print(f"  desv. estandar       : {vals.std():.3f}")

    # veredicto
    rango = vals.max() - vals.min()
    cambia_signo = (vals.min() < 0) != (vals.max() < 0)
    todas_fuertes = np.all(np.abs(vals) > 0.4)
    print("\n--- Veredicto de robustez ---")
    if cambia_signo:
        print("  [FRAGIL] la correlacion CAMBIA DE SIGNO al quitar algun caso.")
        print("  -> depende de casos puntuales; reportar con mucha cautela.")
    elif todas_fuertes and rango < 0.35:
        print("  [ROBUSTA] la correlacion se mantiene fuerte y mismo signo en")
        print("  todas las iteraciones -> no depende de un solo compareciente.")
        print("  Defendible como hallazgo exploratorio solido (dado n bajo).")
    else:
        print("  [MODERADA] mismo signo pero magnitud sensible a algun caso.")
        print("  -> reportar como sugerente, con la salvedad del n pequeno.")

    print("""
  NOTA: con n bajo, el jackknife es un chequeo de sensibilidad, no una
  prueba de significancia. Sea cual sea el resultado, Dabeiba se reporta
  como EXPLORATORIO y complementa (no sustituye) la evidencia principal
  (relacion y8->y10 robusta y contraste inter-corpus del IEI).""")

    print(f"\n  Reporte -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
