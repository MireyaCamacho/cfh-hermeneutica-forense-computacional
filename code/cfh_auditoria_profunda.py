# -*- coding: utf-8 -*-
r"""
cfh_auditoria_profunda.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

AUDITORIA PROFUNDA antes de escribir los capitulos. Re-ejecuta los calculos
clave DESDE CERO y los compara contra los valores reportados. Semaforo por
punto: [OK] / [REVISAR] / [ERROR]. No modifica ningun archivo; solo valida.

Bloques:
  1. Integridad de datos (conteos, ceros anomalos, rangos, NaN/inf)
  2. Reproducibilidad (recalcula DIS/IEI, correlaciones, IAA, y8->y10)
  3. Consistencia de formulas (pesos suman 1.0, normalizacion)
  4. Alineacion (orden base vs recalculos, duplicados de clave)
  5. Reporte final + archivos listos para reproducible/

Salida: outputs/AUDITORIA_PROFUNDA_reporte.txt

Uso (raiz del repo, env cfh):
    python code\cfh_auditoria_profunda.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(REPO, *a)
OUT = P("outputs", "AUDITORIA_PROFUNDA_reporte.txt")

# valores reportados (los que iran a la tesis) - tolerancia +-0.003
ESPERADO = {
    "DIS_A": 0.510, "DIS_B": 0.498, "DIS_C": 0.394,
    "IEI_A": 0.513, "IEI_B": 0.353, "IEI_C": 0.370,
    "corr_DIS_IEI": 0.164, "corr_DIS_ICM": -0.19, "corr_IEI_ICM": -0.11,
    "iaa_global": 0.722, "iaa_rep": 0.841,
    "beta_y8_y10_C": -0.679, "beta_y8_y10_ABC": -0.597,
    "n_A": 819, "n_B": 80, "n_C": 47, "n_total": 946,
}
TOL = 0.005

resultados = []  # (nivel, mensaje)


class Tee:
    def __init__(self, fh): self.fh = fh
    def write(self, s): sys.__stdout__.write(s); self.fh.write(s)
    def flush(self): sys.__stdout__.flush(); self.fh.flush()


def chk(cond, msg_ok, msg_fail, nivel_fail="ERROR"):
    nivel = "OK" if cond else nivel_fail
    resultados.append(nivel)
    print(f"  [{nivel:<7}] {msg_ok if cond else msg_fail}")
    return cond


def cerca(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


# ============ BLOQUE 1 - INTEGRIDAD ============
def bloque1():
    print("\n" + "=" * 66)
    print("BLOQUE 1 - INTEGRIDAD DE DATOS")
    print("=" * 66)
    df = pd.read_csv(P("outputs", "dis_iei_corpus_abc_v2.csv"))
    vc = df["corpus"].value_counts().to_dict()
    chk(vc.get("A") == ESPERADO["n_A"], f"A tiene {ESPERADO['n_A']} filas",
        f"A tiene {vc.get('A')} (esperado {ESPERADO['n_A']}) - posible explosion de merge")
    chk(vc.get("B") == ESPERADO["n_B"], f"B tiene {ESPERADO['n_B']} filas",
        f"B tiene {vc.get('B')} (esperado {ESPERADO['n_B']})")
    chk(vc.get("C") == ESPERADO["n_C"], f"C tiene {ESPERADO['n_C']} filas",
        f"C tiene {vc.get('C')} (esperado {ESPERADO['n_C']})")
    chk(len(df) == ESPERADO["n_total"], f"total {ESPERADO['n_total']} unidades",
        f"total {len(df)} (esperado {ESPERADO['n_total']})")

    # B no en ceros anomalos (el bug historico)
    b = pd.read_csv(P("outputs", "corpus_b_indicadores_COMPLETO.csv"))
    for col in ["y8_mafapo_cs", "y9_cidh_cs"]:
        ceros = (b[col] == 0).sum()
        chk(ceros == 0, f"B {col}: 0 ceros (correcto)",
            f"B {col}: {ceros} ceros - REGRESION del bug de Corpus B", "ERROR")
    # y1, y10 pueden tener algunos ceros legitimos, pero no todos
    for col in ["y1_ebi", "y10_rep_v5", "y2_sa", "y4_nv"]:
        todos_cero = (b[col] == 0).all()
        chk(not todos_cero, f"B {col}: tiene variacion (no todo cero)",
            f"B {col}: TODO EN CERO - indicador roto", "ERROR")

    # rangos y8/y9 sanos (distancias coseno, deben estar en [0,1] aprox)
    for col in ["y8_mafapo_cs", "y9_cidh_cs"]:
        vals = df[col].dropna()
        ok = (vals.min() >= -0.01) and (vals.max() <= 1.01)
        chk(ok, f"{col} en rango sano [{vals.min():.3f}, {vals.max():.3f}]",
            f"{col} FUERA de rango [{vals.min():.3f}, {vals.max():.3f}]", "REVISAR")

    # NaN / inf en indices finales
    for col in ["DIS", "IEI_A"]:
        nnan = df[col].isna().sum()
        ninf = np.isinf(df[col]).sum()
        chk(nnan == 0 and ninf == 0, f"{col}: sin NaN ni inf",
            f"{col}: {nnan} NaN, {ninf} inf", "ERROR")
    return df


# ============ BLOQUE 2 - REPRODUCIBILIDAD ============
def bloque2(df):
    print("\n" + "=" * 66)
    print("BLOQUE 2 - REPRODUCIBILIDAD (recalcula desde cero)")
    print("=" * 66)

    # --- DIS/IEI recalculado ---
    d = df.copy()
    IND = ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep"]
    for col in IND:
        d[col + "_zz"] = sigmoid(z(d[col].astype(float)))
    d["DIS_re"] = 0.40*d["y1_ebi_zz"] + 0.30*d["y2_sa_zz"] + 0.30*(1-d["y10_rep_zz"])
    d["IEI_re"] = 0.40*d["y8_mafapo_cs_zz"] + 0.30*d["y9_cidh_cs_zz"] + 0.30*d["y4_nv_zz"]

    print("\n  -- DIS/IEI por corpus (recalculado vs reportado) --")
    for corp in ["A", "B", "C"]:
        sub = d[d["corpus"] == corp]
        dis_re, iei_re = sub["DIS_re"].mean(), sub["IEI_re"].mean()
        chk(cerca(dis_re, ESPERADO[f"DIS_{corp}"]),
            f"DIS_{corp}={dis_re:.3f} == reportado {ESPERADO[f'DIS_{corp}']}",
            f"DIS_{corp}={dis_re:.3f} != reportado {ESPERADO[f'DIS_{corp}']}", "REVISAR")
        chk(cerca(iei_re, ESPERADO[f"IEI_{corp}"]),
            f"IEI_{corp}={iei_re:.3f} == reportado {ESPERADO[f'IEI_{corp}']}",
            f"IEI_{corp}={iei_re:.3f} != reportado {ESPERADO[f'IEI_{corp}']}", "REVISAR")

    # tambien comparar contra la columna DIS/IEI_A ya guardada
    print("\n  -- coherencia con columnas guardadas en el CSV --")
    dif_dis = (d["DIS_re"] - df["DIS"]).abs().max()
    dif_iei = (d["IEI_re"] - df["IEI_A"]).abs().max()
    chk(dif_dis < 0.01, f"DIS recalc == DIS guardado (dif max {dif_dis:.4f})",
        f"DIS recalc != guardado (dif max {dif_dis:.4f}) - output desactualizado", "REVISAR")
    chk(dif_iei < 0.01, f"IEI recalc == IEI guardado (dif max {dif_iei:.4f})",
        f"IEI recalc != guardado (dif max {dif_iei:.4f}) - output desactualizado", "REVISAR")

    # --- correlaciones ---
    print("\n  -- correlaciones DIS/IEI/ICM --")
    cor = df[["DIS", "IEI_A"]].corr(method="spearman").iloc[0, 1]
    chk(abs(cor) < 0.33, f"corr DIS-IEI={cor:.3f} (<0.33, dimensiones distintas)",
        f"corr DIS-IEI={cor:.3f} (>=0.33)", "REVISAR")
    # ICM (solo C)
    csem = pd.read_csv(P("data", "referencias", "indicadores_sem_compareciente.csv"))
    dc = df[df["corpus"] == "C"].merge(
        csem[["identidad", "icm_tricanal"]], left_on="unidad",
        right_on="identidad", how="inner")
    if len(dc) > 10:
        c1 = dc[["DIS", "icm_tricanal"]].corr(method="spearman").iloc[0, 1]
        c2 = dc[["IEI_A", "icm_tricanal"]].corr(method="spearman").iloc[0, 1]
        chk(abs(c1) < 0.33 and abs(c2) < 0.33,
            f"corr DIS-ICM={c1:.3f}, IEI-ICM={c2:.3f} (<0.33)",
            f"corr con ICM alta: DIS-ICM={c1:.3f}, IEI-ICM={c2:.3f}", "REVISAR")

    # --- IAA ---
    print("\n  -- IAA (dos rondas) --")
    try:
        from sklearn.metrics import cohen_kappa_score
        g = json.load(open(P("data", "referencias", "gold_consolidado_A1A2.json"),
                           encoding="utf-8"))
        ids = sorted(x["id"] for x in g)
        CATS = ["EBI", "SA", "NV", "REP"]
        k1 = {}
        for c in CATS:
            k1[c] = cohen_kappa_score([int(x[f"{c}_A1"]) for x in g],
                                      [int(x[f"{c}_A2"]) for x in g])
        # aplicar revision REP
        dif = pd.read_csv(P("data", "referencias", "REP_diferencias_A1_A2.csv"))
        marc = lambda v: 0 if (pd.isna(v) or str(v).strip().lower() in
                               ("(nada)", "nada", "", "none")) else 1
        ra1 = {x["id"]: int(x["REP_A1"]) for x in g}
        ra2 = {x["id"]: int(x["REP_A2"]) for x in g}
        for _, r in dif.iterrows():
            fid = int(r["fragmento"])
            if fid in ra1:
                ra1[fid] = marc(r["A1_REP"]); ra2[fid] = marc(r["A2_REP"])
        krep2 = cohen_kappa_score([ra1[i] for i in ids], [ra2[i] for i in ids])
        glob2 = np.mean([k1["EBI"], k1["SA"], k1["NV"], krep2])
        chk(cerca(krep2, ESPERADO["iaa_rep"], 0.01),
            f"REP kappa ronda2={krep2:.3f} == {ESPERADO['iaa_rep']}",
            f"REP kappa ronda2={krep2:.3f} != {ESPERADO['iaa_rep']}", "REVISAR")
        chk(cerca(glob2, ESPERADO["iaa_global"], 0.01),
            f"IAA global={glob2:.3f} == {ESPERADO['iaa_global']}",
            f"IAA global={glob2:.3f} != {ESPERADO['iaa_global']}", "REVISAR")
    except Exception as e:
        chk(False, "", f"IAA no reproducible: {e}", "ERROR")

    # --- y8 -> y10 ---
    print("\n  -- hallazgo y8->y10 (regresion) --")
    try:
        from semopy import Model
        for universo, sub in [("C", df[df["corpus"] == "C"]), ("ABC", df)]:
            dd = sub.copy()
            for col in ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep"]:
                dd[col] = z(dd[col].astype(float))
            m = Model("y10_rep ~ y1_ebi + y2_sa + y4_nv + y8_mafapo_cs + y9_cidh_cs")
            m.fit(dd[["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo_cs", "y9_cidh_cs", "y10_rep"]])
            est = m.inspect()
            b = est[(est["lval"] == "y10_rep") & (est["rval"] == "y8_mafapo_cs")]["Estimate"].values[0]
            key = f"beta_y8_y10_{universo}"
            chk(cerca(b, ESPERADO[key], 0.05),
                f"beta y8->y10 ({universo})={b:+.3f} == {ESPERADO[key]:+.3f}",
                f"beta y8->y10 ({universo})={b:+.3f} != {ESPERADO[key]:+.3f}", "REVISAR")
    except Exception as e:
        chk(False, "", f"y8->y10 no reproducible: {e}", "REVISAR")


# ============ BLOQUE 3 - FORMULAS ============
def bloque3():
    print("\n" + "=" * 66)
    print("BLOQUE 3 - CONSISTENCIA DE FORMULAS")
    print("=" * 66)
    dis_w = [0.40, 0.30, 0.30]
    iei_w = [0.40, 0.30, 0.30]
    chk(cerca(sum(dis_w), 1.0, 1e-9), "pesos DIS suman 1.0",
        f"pesos DIS suman {sum(dis_w)}")
    chk(cerca(sum(iei_w), 1.0, 1e-9), "pesos IEI (opcion A) suman 1.0",
        f"pesos IEI suman {sum(iei_w)}")
    # DIS e IEI no comparten indicador (opcion A)
    dis_ind = {"y1_ebi", "y2_sa", "y10_rep"}
    iei_ind = {"y8_mafapo_cs", "y9_cidh_cs", "y4_nv"}
    comp = dis_ind & iei_ind
    chk(len(comp) == 0, "DIS e IEI NO comparten indicadores (opcion A, dimensiones limpias)",
        f"DIS e IEI comparten: {comp}", "REVISAR")


# ============ BLOQUE 4 - ALINEACION ============
def bloque4():
    print("\n" + "=" * 66)
    print("BLOQUE 4 - ALINEACION DE ARCHIVOS")
    print("=" * 66)
    base = pd.read_csv(P("data", "features", "indicators_completo_conflibert.csv"))
    y1 = pd.read_csv(P("outputs", "y1_ebi_AB_recalculado.csv"))
    y10 = pd.read_csv(P("outputs", "y10_rep_v5_AB_recalculado.csv"))
    chk(len(base) == len(y1) == len(y10) == 873,
        "base, y1, y10 tienen 873 filas",
        f"desajuste: base={len(base)}, y1={len(y1)}, y10={len(y10)}", "ERROR")
    ok_orden = (base["doc_id"].values == y1["doc_id"].values).all() and \
               (base["doc_id"].values == y10["doc_id"].values).all()
    chk(ok_orden, "orden doc_id identico entre base y recalculos (merge por posicion OK)",
        "orden doc_id DIFERENTE - el merge por posicion seria incorrecto", "ERROR")
    # duplicados de clave en A (documentar, no es error)
    a = base[base["corpus_type"].str.startswith("A")]
    claves = a.groupby(["doc_id", "section_id"]).ngroups
    print(f"  [INFO   ] A: {len(a)} filas, {claves} claves unicas "
          f"({len(a)-claves} comparten clave -> por eso se usa merge por posicion)")


# ============ MAIN ============
def main():
    fh = open(OUT, "w", encoding="utf-8")
    sys.stdout = Tee(fh)
    print("=" * 66)
    print("AUDITORIA PROFUNDA CFH - antes de escribir capitulos")
    print("=" * 66)

    df = bloque1()
    bloque2(df)
    bloque3()
    bloque4()

    # resumen
    print("\n" + "=" * 66)
    print("RESUMEN DE LA AUDITORIA")
    print("=" * 66)
    n_ok = resultados.count("OK")
    n_rev = resultados.count("REVISAR")
    n_err = resultados.count("ERROR")
    print(f"  [OK]      {n_ok}")
    print(f"  [REVISAR] {n_rev}")
    print(f"  [ERROR]   {n_err}")
    print()
    if n_err == 0 and n_rev == 0:
        print("  >> TODO OK. Los numeros son reproducibles y consistentes.")
        print("     Listo para escribir los capitulos y llevar a reproducible/.")
    elif n_err == 0:
        print("  >> Sin errores criticos, pero hay puntos a REVISAR (arriba).")
        print("     Revisar cada [REVISAR] antes de escribir esos numeros.")
    else:
        print("  >> HAY ERRORES CRITICOS. Resolver los [ERROR] antes de escribir.")
    print(f"\n  Reporte -> {OUT}")
    sys.stdout = sys.__stdout__
    fh.close()


if __name__ == "__main__":
    main()
