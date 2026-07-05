# -*- coding: utf-8 -*-
r"""
cfh_revisar_ceros_y11.py
CFH | Mireya Camacho Celis

Revisa los textos de los comparecientes que dieron y11 = 0, para decidir
si el cero es legitimo (habla puramente operativa, sin contenido restaurativo)
o si hay reparacion real que los patrones actuales no capturan.

Uso (raiz del repo, env cfh):
  python code\cfh_revisar_ceros_y11.py
  python code\cfh_revisar_ceros_y11.py --chars 500   # ver mas texto por compareciente
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detalle", default="y11_densidad_detalle.csv")
    ap.add_argument("--chars", type=int, default=350,
                    help="Cuantos caracteres de texto mostrar por compareciente")
    args = ap.parse_args()

    d = pd.read_csv(args.detalle)
    ceros = d[d["y11_densidad"] == 0].copy()
    con_señal = d[d["y11_densidad"] > 0].copy()

    print("=" * 70)
    print(f"COMPARECIENTES EN CERO: {len(ceros)} / {len(d)}")
    print("=" * 70)
    print("Revisa cada texto: ¿hay reparacion/perdon/dirigirse a victimas que")
    print("los patrones no captaron, o es habla puramente operativa (cero legitimo)?")
    print()

    for _, r in ceros.iterrows():
        txt = str(r.get("texto_completo", "") or "")
        n_tok = len(txt.split())
        print("-" * 70)
        print(f"[{r['identidad']} | {r['subcaso']}]  ({n_tok} palabras)")
        print(f"  {txt[:args.chars]}")
        print()

    # Resumen de los que SI tienen señal, para contraste
    print("=" * 70)
    print(f"CON SEÑAL (y11>0): {len(con_señal)} comparecientes")
    print("=" * 70)
    for _, r in con_señal.sort_values("y11_densidad", ascending=False).iterrows():
        print(f"  {r['identidad'][:34]:<34} y11={r['y11_densidad']:.3f} "
              f"(a={r['y11a_inst']:.2f} b={r['y11b_dial']:.2f} | ha={int(r['n_hits_a'])} hb={int(r['n_hits_b'])})")


if __name__ == "__main__":
    main()
