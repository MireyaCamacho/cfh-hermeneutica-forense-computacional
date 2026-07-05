# -*- coding: utf-8 -*-
r"""
cfh_auditoria_estado.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Auditoria exhaustiva del estado de los indicadores en los tres corpus,
para saber -verificando en disco, no de memoria- que quedo arreglado tras
los ajustes de ayer y que falta.

Revisa, para A, B y C:
  - Que indicadores existen (y1..y12) y en que archivo.
  - Cuales estan en cero o con NaN (senal de que no se calcularon).
  - Version de y1_ebi (gazetteer nuevo vs 0.0 viejo), y10 (v5 vs viejo),
    y11 (dialogico vs centroide).
  - Estado del gold reconciliado A1+A2 y del CSV del SEM de C.
  - Estado de DIS/IEI (si existen y con que version se calcularon).

NO modifica nada: solo reporta.

Uso (raiz del repo, env cfh):
    python code\cfh_auditoria_estado.py
"""

import os
import glob
import json
import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def existe(*parts):
    return os.path.exists(os.path.join(REPO, *parts))


def head(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def col_estado(df, col):
    if col not in df.columns:
        return "AUSENTE"
    s = df[col]
    nn = s.notna().sum()
    if nn == 0:
        return "TODO NaN"
    ceros = (s.fillna(0) == 0).sum()
    return (f"media={s.mean():.3f} std={s.std():.3f} "
            f"ceros={ceros}/{len(s)} nan={s.isna().sum()}")


def main():
    print("=" * 68)
    print("AUDITORIA DE ESTADO CFH - verificacion en disco")
    print("=" * 68)

    # ---------- CORPUS C: SEM por compareciente ----------
    head("CORPUS C - CSV del SEM (indicadores_sem_compareciente.csv)")
    p = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"  filas: {len(df)}  columnas: {len(df.columns)}")
        for c in ["y1_ebi", "y2_sa", "y4_nv", "y8_mafapo", "y9_cidh",
                  "y10_rep", "y11_conv_rest", "y12_acustico", "y3_civil",
                  "icm_tricanal", "icm_facial", "icm_vocal"]:
            print(f"    {c:<16}: {col_estado(df, c)}")
        # trazas de version
        print("  --- trazas de version ---")
        for c in ["y10_rep_old", "y10_rep_v4_norm", "y11_conv_rest_old"]:
            print(f"    {c:<20}: {'presente' if c in df.columns else 'ausente'}")
    else:
        print("  [FALTA] no existe el CSV del SEM")

    # ---------- CORPUS A+B: indicadores por seccion ----------
    head("CORPUS A+B - indicators_completo_conflibert.csv")
    p = os.path.join(REPO, "data", "features", "indicators_completo_conflibert.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"  filas: {len(df)}")
        if "corpus_type" in df.columns:
            print(f"  por corpus: {df['corpus_type'].value_counts().to_dict()}")
        for c in ["y1_ebi", "y2_sa", "y3_civil", "y4_nv", "y10_rep",
                  "y8_mafapo", "y9_cidh", "y11_conv_rest"]:
            print(f"    {c:<16}: {col_estado(df, c)}")
        # y1 viejo?
        if "y1_ebi" in df.columns and df["y1_ebi"].fillna(0).eq(0).all():
            print("    >> y1_ebi es el VIEJO (0.0 en todo) - pendiente recalcular en el CSV base")
    else:
        print("  [FALTA] no existe")

    # ---------- Recalculos de ayer (outputs) ----------
    head("RECALCULOS DE AYER (outputs/)")
    for f, desc in [
        ("outputs/y1_ebi_AB_recalculado.csv", "y1 EBI gazetteer sobre A+B"),
        ("outputs/y10_rep_v5_AB_recalculado.csv", "y10 v5 sobre A+B"),
        ("outputs/corpus_b_indicadores_v2.csv", "B fortalecido (y1+y10 v5)"),
        ("outputs/corpus_b_secciones_texto.csv", "B texto segmentado"),
        ("outputs/parsimonia_dis_v2.csv", "parsimonia DIS con y1"),
    ]:
        pp = os.path.join(REPO, f)
        if os.path.exists(pp):
            try:
                d = pd.read_csv(pp)
                extra = ""
                if "y1_ebi" in d.columns:
                    extra += f" y1>0:{(d['y1_ebi']>0).sum()}/{len(d)}"
                if "y10_rep_v5" in d.columns:
                    extra += f" y10>0:{(d['y10_rep_v5']>0).sum()}/{len(d)}"
                print(f"  [OK] {f}  ({len(d)} filas){extra}")
            except Exception as e:
                print(f"  [OK] {f}  (no CSV tabular: {e})")
        else:
            print(f"  [FALTA] {f}  <- {desc}")

    # ---------- Gold reconciliado A1+A2 ----------
    head("GOLD RECONCILIADO (segundo anotador / IAA)")
    for f in ["data/referencias/gold_consolidado_A1A2.json",
              "data/referencias/annotations_mireya_v1.json"]:
        pp = os.path.join(REPO, f)
        if os.path.exists(pp):
            try:
                d = json.load(open(pp, encoding="utf-8"))
                n = len(d) if isinstance(d, list) else len(d.get("items", d))
                print(f"  [OK] {f}  ({n} entradas)")
            except Exception as e:
                print(f"  [OK] {f}  (error leyendo: {e})")
        else:
            print(f"  [FALTA] {f}")
    # reporte kappa
    for f in ["outputs/iaa_kappa_resultados.csv", "outputs/iaa_kappa_reporte.txt"]:
        print(f"  {'[OK]' if existe(f) else '[FALTA]'} {f}")

    # ---------- DIS/IEI ----------
    head("DIS / IEI (indices tri-corpus)")
    p = os.path.join(REPO, "data", "dis_iei_corpus_abc_definitivo.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"  [OK] dis_iei_corpus_abc_definitivo.csv ({len(df)} filas)")
        print("  >> OJO: este es el VIEJO (y1=0, y10 sin v5, formulas viejas)")
        if "corpus_type" in df.columns:
            print(f"  por corpus: {df['corpus_type'].value_counts().to_dict()}")
    else:
        print("  [FALTA] dis_iei_corpus_abc_definitivo.csv")
    print("  >> DIS/IEI con formulas nuevas (opciones A/B) + y1/y10 v5: PENDIENTE (Paso 2)")

    # ---------- RESUMEN ----------
    head("RESUMEN - que falta para cerrar")
    print("""  Checklist tras ajustes:
   [ ] y1 EBI recalculado en CSV base A+B (esta en outputs/, falta integrarlo)
   [ ] y10 v5 recalculado en CSV base A+B (idem)
   [ ] Corpus B fortalecido (correr cfh_fortalecer_corpus_b.py si falta)
   [ ] Rehacer parsimonia con B fortalecido (opcional)
   [ ] Paso 2: DIS/IEI opciones A y B sobre A+B+C con y1/y10 v5
   [ ] Matriz de correlaciones DIS/IEI/ICM (demostrar dimensiones distintas)
   [ ] IAA kappa del segundo anotador (verificar si esta completo)
  El detalle real lo marcan los [OK]/[FALTA] de arriba.""")


if __name__ == "__main__":
    main()
