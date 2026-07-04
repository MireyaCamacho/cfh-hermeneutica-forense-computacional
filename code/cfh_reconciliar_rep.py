# -*- coding: utf-8 -*-
"""
cfh_reconciliar_rep.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Reconcilia los desacuerdos REP entre A1 y A2, recalcula Cohen's kappa
y construye el gold REP por CONSENSO ESTRICTO.

Regla de reconciliacion (Opcion 1):
  - Se parte del gold_consolidado_A1A2.json (100 fragmentos, campos REP_A1/REP_A2 bool).
  - Para cada fragmento en REP_diferencias_A1_A2.csv (34 desacuerdos):
      * A1 mantiene su marca REP -> REP_A1_recon = 1
      * A2 queda 1 salvo que su celda sea "(nada)" -> REP_A2_recon = 0
  - GOLD estricto: REP_gold = 1 solo si (REP_A1_recon AND REP_A2_recon).
  - Los 6 desacuerdos remanentes (A2 = nada) quedan NO REP, coherente con
    haber eliminado el mecanismo DIH del extractor y10.

Entradas (misma carpeta o rutas por CLI):
  - gold_consolidado_A1A2.json
  - REP_diferencias_A1_A2.csv

Salidas:
  - gold_REP_reconciliado.json   (100 fragmentos con REP_gold + trazas)
  - reporte impreso de kappa antes/despues por categoria

Uso:
  python cfh_reconciliar_rep.py
  python cfh_reconciliar_rep.py --gold ruta.json --difs ruta.csv --out salida.json
"""

import json
import csv
import argparse
import os


def cohen_kappa(pairs):
    """Cohen's kappa binario a partir de lista de pares (a, b) booleanos."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, (0, 0, 0, 0)
    n11 = sum(1 for a, b in pairs if a and b)
    n00 = sum(1 for a, b in pairs if not a and not b)
    n10 = sum(1 for a, b in pairs if a and not b)
    n01 = sum(1 for a, b in pairs if not a and b)
    po = (n11 + n00) / n
    pa1 = (n11 + n10) / n
    pb1 = (n11 + n01) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    return kappa, po, (n11, n10, n01, n00)


def a2_final(valor):
    """A2 queda 0 (no REP) si su celda es '(nada)' o vacia; 1 en otro caso."""
    return 0 if (valor or "").strip() in ("", "(nada)") else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_consolidado_A1A2.json")
    ap.add_argument("--difs", default="REP_diferencias_A1_A2.csv")
    ap.add_argument("--out", default="gold_REP_reconciliado.json")
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        raise SystemExit(f"No encuentro el gold: {args.gold}")
    if not os.path.exists(args.difs):
        raise SystemExit(f"No encuentro el CSV de diferencias: {args.difs}")

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    with open(args.difs, encoding="utf-8-sig") as f:
        difs = list(csv.DictReader(f))

    gmap = {str(r["id"]): r for r in gold}
    disp = set(str(r["fragmento"]) for r in difs)

    # Validacion: todos los ids del CSV existen en el gold
    faltantes = [i for i in disp if i not in gmap]
    if faltantes:
        raise SystemExit(f"IDs del CSV ausentes en gold: {faltantes}")

    # --- Aplicar reconciliacion a los fragmentos en disputa ---
    for r in difs:
        g = gmap[str(r["fragmento"])]
        g["REP_A1_recon"] = 1  # A1 mantiene su marca en los 34
        g["REP_A2_recon"] = a2_final(r["A2_REP"])

    # --- Fragmentos NO en disputa: recon = original ---
    for g in gold:
        if str(g["id"]) not in disp:
            g["REP_A1_recon"] = int(bool(g["REP_A1"]))
            g["REP_A2_recon"] = int(bool(g["REP_A2"]))

    # --- GOLD consenso estricto ---
    for g in gold:
        g["REP_gold"] = int(bool(g["REP_A1_recon"]) and bool(g["REP_A2_recon"]))

    # --- Kappa antes / despues (todas las categorias para contexto) ---
    print("=" * 64)
    print("KAPPA POR CATEGORIA (base original)")
    print("=" * 64)
    for cat in ["EBI", "SA", "NV", "REP"]:
        pairs = [(bool(r[f"{cat}_A1"]), bool(r[f"{cat}_A2"])) for r in gold]
        k, po, t = cohen_kappa(pairs)
        print(f"  {cat:4}: kappa={k:.3f}  po={po:.3f}  [11={t[0]} 10={t[1]} 01={t[2]} 00={t[3]}]")

    print()
    print("=" * 64)
    print("REP: ORIGINAL vs RECONCILIADO")
    print("=" * 64)
    p_old = [(bool(r["REP_A1"]), bool(r["REP_A2"])) for r in gold]
    p_new = [(bool(r["REP_A1_recon"]), bool(r["REP_A2_recon"])) for r in gold]
    ko, poo, to = cohen_kappa(p_old)
    kn, pon, tn = cohen_kappa(p_new)
    print(f"  ORIGINAL     : kappa={ko:.3f}  po={poo:.3f}  {to}")
    print(f"  RECONCILIADO : kappa={kn:.3f}  po={pon:.3f}  {tn}")
    print(f"  Delta kappa  : +{kn - ko:.3f}")

    pos = sum(g["REP_gold"] for g in gold)
    print()
    print(f"GOLD REP consenso estricto: {pos} positivos / {len(gold)} fragmentos")
    disp6 = sorted(
        [str(r["fragmento"]) for r in difs if a2_final(r["A2_REP"]) == 0],
        key=lambda x: int(x),
    )
    print(f"Desacuerdos remanentes (NO REP en gold): {disp6}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    print(f"\nGold escrito en: {args.out}")


if __name__ == "__main__":
    main()
