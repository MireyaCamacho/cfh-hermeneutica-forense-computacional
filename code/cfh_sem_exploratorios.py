# -*- coding: utf-8 -*-
r"""
cfh_sem_exploratorios.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Corre DOS analisis SEM como EXPLORATORIOS complementarios a la evidencia
principal (DIS/IEI descriptivo), reportando el ajuste y -para cada uno- el
CHECKLIST de que faltaria para que fuera CONFIRMATORIO.

  MODELO 1 - SEM de C (n=47, latentes)
    eta1 =~ y8            (un solo indicador semantico: evita colapso y8~y9=0.91)
    eta2 =~ y10 + y11     (transicion epistemica)
    eta1 ~ eta2 ?  -> aqui usamos H2 (injusticia -> transicion)
    NOTA: con 1 indicador la latente = observada; el SEM de C es exploratorio.
    Para simplificar y ser honestos, se corre el PATH sobre indices observados.

  MODELO 2 - MG-SEM con indices construidos (DIS, IEI) como observadas
    Estructura: IEI ~ DIS   (la injusticia discursiva predice la epistemica)
    Multigrupo por corpus (A/B/C).
    v2a - TODOS: A=819, B=80, C=47 (desbalanceado)
    v2b - MUESTRA: A submuestreado a ~80 para equilibrar

Criterios de ajuste (Hu & Bentler; Kline):
    CFI >= 0.90  aceptable / >= 0.95 bueno
    RMSEA <= 0.08 aceptable / <= 0.06 bueno
    SRMR <= 0.08

Uso (raiz del repo, env cfh):
    python code\cfh_sem_exploratorios.py

Entrada: outputs/dis_iei_corpus_abc_v2.csv
         data/referencias/indicadores_sem_compareciente.csv (para SEM de C)
"""

import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_ABC = os.path.join(REPO, "outputs", "dis_iei_corpus_abc_v2.csv")
F_C = os.path.join(REPO, "data", "referencias", "indicadores_sem_compareciente.csv")
SEED = 42


def fit_stats(model):
    try:
        from semopy import calc_stats
        s = calc_stats(model).T
        g = lambda k: float(s.loc[k].iloc[0]) if k in s.index else np.nan
        return {"CFI": g("CFI"), "RMSEA": g("RMSEA"), "SRMR": g("SRMR"),
                "chi2": g("chi2"), "DoF": g("DoF")}
    except Exception as e:
        return {"error": str(e)}


def checklist(nombre, stats, extra):
    print(f"\n  --- CHECKLIST confirmatorio ({nombre}) ---")
    cfi = stats.get("CFI", np.nan)
    rmsea = stats.get("RMSEA", np.nan)
    ok_cfi = (not np.isnan(cfi)) and cfi >= 0.90
    ok_rmsea = (not np.isnan(rmsea)) and rmsea <= 0.08
    print(f"    [{'OK' if ok_cfi else 'NO'}] CFI>=0.90     (obs: {cfi:.3f})")
    print(f"    [{'OK' if ok_rmsea else 'NO'}] RMSEA<=0.08   (obs: {rmsea:.3f})")
    for txt in extra:
        print(f"    {txt}")


def modelo1_C():
    print("=" * 66)
    print("MODELO 1 - SEM de C (n=47) EXPLORATORIO")
    print("=" * 66)
    from semopy import Model
    df = pd.read_csv(F_C)

    # η1 con UN indicador (y8) para evitar colapso y8~y9
    # η2 = transicion (y10 + y11)
    # Para n=47, path sobre observadas es lo honesto; se estandariza.
    for c in ["y8_mafapo", "y10_rep", "y11_conv_rest"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

    desc = """
    eta2 =~ y10_rep_z + y11_conv_rest_z
    eta2 ~ y8_mafapo_z
    """
    m = Model(desc)
    try:
        m.fit(df[["y8_mafapo_z", "y10_rep_z", "y11_conv_rest_z"]])
        est = m.inspect()
        print("\n  Parametros:")
        print(est.to_string())
        st = fit_stats(m)
        print(f"\n  Ajuste: CFI={st.get('CFI',np.nan):.3f} "
              f"RMSEA={st.get('RMSEA',np.nan):.3f} SRMR={st.get('SRMR',np.nan):.3f}")
        # path observado H2: y8 -> (y10+y11)/2
        df["transic"] = (df["y10_rep_z"] + df["y11_conv_rest_z"]) / 2
        r = np.corrcoef(df["y8_mafapo_z"], df["transic"])[0, 1]
        checklist("SEM de C", st, [
            f"[{'OK' if abs(r)>0.3 else 'NO'}] path y8->transic observado: r={r:+.3f}",
            "[NO] n>=150 comparecientes (obs: 47) <- requiere mas audiencias",
            "[info] con 1 indicador en eta1, la latente = observada (n insuf.)",
        ])
    except Exception as e:
        print(f"  [no converge] {e}")
        print("  -> confirma que el SEM de C es exploratorio (n=47 insuficiente).")


def mg_sem(df, etiqueta):
    from semopy import Model
    print(f"\n{'='*66}\nMG-SEM ({etiqueta}): IEI ~ DIS por corpus\n{'='*66}")
    print(f"  n por grupo: {df['corpus'].value_counts().to_dict()}")
    resultados = {}
    for corp in ["A", "B", "C"]:
        sub = df[df["corpus"] == corp].copy()
        if len(sub) < 20:
            print(f"  [{corp}] n={len(sub)} <20, omitido")
            continue
        sub["DIS_z"] = (sub["DIS"] - sub["DIS"].mean()) / (sub["DIS"].std() + 1e-9)
        sub["IEI_z"] = (sub["IEI_A"] - sub["IEI_A"].mean()) / (sub["IEI_A"].std() + 1e-9)
        m = Model("IEI_z ~ DIS_z")
        try:
            m.fit(sub[["DIS_z", "IEI_z"]])
            est = m.inspect()
            beta = est[(est["lval"] == "IEI_z") & (est["rval"] == "DIS_z")]["Estimate"].values
            pval = est[(est["lval"] == "IEI_z") & (est["rval"] == "DIS_z")]["p-value"].values
            b = float(beta[0]) if len(beta) else np.nan
            p = pval[0] if len(pval) else "NA"
            resultados[corp] = (b, p, len(sub))
            print(f"  [{corp}] n={len(sub):>4}  beta(DIS->IEI)={b:+.3f}  p={p}")
        except Exception as e:
            print(f"  [{corp}] no converge: {e}")
    # comparacion
    print("\n  Comparacion de beta entre corpus:")
    for c, (b, p, n) in resultados.items():
        print(f"    {c}: beta={b:+.3f} (p={p}, n={n})")
    if len(resultados) >= 2:
        betas = [v[0] for v in resultados.values()]
        rango = max(betas) - min(betas)
        print(f"  rango de beta entre grupos: {rango:.3f}")
        print("  >> si los beta difieren mucho -> la estructura NO es invariante")
        print("     (la relacion DIS-IEI opera distinto por corpus)")
    return resultados


def main():
    # ---- MODELO 1 ----
    modelo1_C()

    # ---- MODELO 2 ----
    df = pd.read_csv(F_ABC)

    print("\n\n" + "#" * 66)
    print("MODELO 2 - MG-SEM con indices construidos (DIS, IEI_A)")
    print("#" * 66)

    # v2a: TODOS
    mg_sem(df, "v2a - TODOS")

    # v2b: MUESTRA equilibrada (A submuestreado a ~80)
    a = df[df["corpus"] == "A"].sample(n=80, random_state=SEED)
    resto = df[df["corpus"] != "A"]
    df_bal = pd.concat([a, resto], ignore_index=True)
    mg_sem(df_bal, "v2b - MUESTRA equilibrada (A=80)")

    # ---- Checklist MG ----
    print("\n" + "=" * 66)
    print("CHECKLIST confirmatorio (MG-SEM)")
    print("=" * 66)
    print("""  Para que el MG-SEM fuera CONFIRMATORIO harian falta:
    [ ] Invarianza configural: misma estructura de medicion en A/B/C
    [ ] Invarianza metrica: cargas equivalentes entre grupos
    [ ] n por grupo suficiente y balanceado (B=80, C=47 son justos)
    [ ] Misma unidad de analisis (hoy: A/B=seccion, C=compareciente)
  Con estos datos, el MG-SEM es EXPLORATORIO: complementa la evidencia
  principal (DIS/IEI descriptivo), no la sustituye.""")


if __name__ == "__main__":
    main()
