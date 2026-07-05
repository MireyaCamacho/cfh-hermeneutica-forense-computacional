# -*- coding: utf-8 -*-
"""
CFH — Centroide RESTAURATIVO v1 (para indicador y11)
=====================================================
Construye el centroide del POLO RESTAURATIVO: el lenguaje explicito de
RECONOCIMIENTO DE RESPONSABILIDAD, PERDON y REPARACION.

y11 = convergencia semantica de cada compareciente a este centroide
      (1 - distancia coseno), indicador de eta2 (Transicion Epistemica).

DECISIONES METODOLOGICAS (de Mireya):
  - Corpus EXTERNO a los comparecientes -> evita CIRCULARIDAD (el referente
    no puede construirse con los mismos datos que evalua; critica de jurado).
  - SIN lenguaje de victimas -> no se confunde con el centroide MAFAPO (y8).
    Aqui solo va el lenguaje de QUIEN reconoce y repara, no de quien sufrio.
  - Fuentes: actos formales de reconocimiento/perdon ya fallados en otras
    jurisdicciones (Corte IDH) y actos del Estado colombiano + estandares.

METODO (identico a MAFAPO v5, para comparabilidad):
  modelo eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1, embedding token CLS,
  deduplicacion por primeros 100 chars, promedio ponderado, np.save.
  Ponderacion: w=1.8 reconocimiento PLENO (responsabilidad+perdon+reparacion),
               w=1.0 reconocimiento parcial / estandar normativo.

>>> IMPORTANTE: los textos de abajo son APARTES a CURAR por Mireya.
>>> Reemplaza/completa con los fragmentos textuales reales de cada acto.
>>> Manten cada texto CORTO (2-5 frases), como los de MAFAPO, para que el
>>> centroide sea comparable. Extrae SOLO el aparte de reconocimiento/perdon,
>>> no la sentencia completa.

Ejecutar en COLAB (ConfliBERT necesita GPU):
  python cfh_centroide_restaurativo.py
"""
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent
# en Colab, ajustar a la ruta donde esten las referencias
REF_DIR = REPO / "data" / "referencias"
OUT = REF_DIR / "centroide_restaurativo_v1.npy"
OUT_INV = REF_DIR / "inventario_centroide_restaurativo_v1.json"

# =====================================================================
# COMPOSICION DEL CORPUS RESTAURATIVO (externo puro, decision de Mireya):
#   A. Estandar normativo del reconocimiento/reparacion (el "deber ser")
#      - Ley 975/2005 (Justicia y Paz), arts. 45.3 y 49.4
#      - Estandares Corte IDH / ICTJ
#   B. Actos de reconocimiento de OTRO perpetrador (FARC, Macrocaso 01)
#      para mostrar que el lenguaje restaurativo del perdon es el mismo
#      independientemente de quien lo emite.
#   NINGUN texto proviene de los 47 comparecientes del Macrocaso 03
#   (agentes del Estado) -> evita circularidad.
# =====================================================================

# NIVEL 1-2 (w=1.8) — lenguaje performativo de reconocimiento/perdon
#   Fuente: actos de reconocimiento FARC (Macrocaso 01) + definicion legal
#   >>> Mireya: verifica literalidad contra la fuente y amplia <<<
TEXTOS_RESTAURATIVOS_PLENO = [
    # FARC ante la JEP, reconocimiento de secuestro (Macrocaso 01, 2021).
    # Documento colectivo de reconocimiento:
    "Tales conductas nunca debieron ocurrir y por eso pedimos perdon. Hoy reconocemos que estos hechos constituyen crimenes de guerra y de lesa humanidad.",

    # FARC (Rodrigo Londono), reconocimiento de responsabilidad de mando:
    "Reconozco mi responsabilidad por la politica de retener a personas capturadas. Estas conductas nunca debieron ocurrir. Honramos la valentia y la generosidad de las victimas que se han acercado confiando en la implementacion del Acuerdo Final.",

    # FARC, audiencia de reconocimiento (Caso 01):
    "Lamentamos lo que sucedio, entendemos que el secuestro genero graves afectaciones, no solo a las victimas directas, sino a sus familiares. La muerte de ninguna persona es justificable, pedimos perdon. Estamos convencidos de que este es el camino a la no repeticion.",

    # Ley 975/2005 (Justicia y Paz), art. 45.3 — definicion legal del acto:
    "El reconocimiento publico de haber causado danos a las victimas, la declaracion publica de arrepentimiento, la solicitud de perdon dirigida a las victimas y la promesa de no repetir tales conductas punibles.",

    # >>> AGREGAR MAS APARTES PERFORMATIVOS (FARC u otros actores externos) <<<
]

# NIVEL 3 (w=1.0) — estandar normativo del reconocimiento/reparacion
TEXTOS_RESTAURATIVOS_ESTANDAR = [
    # Ley 975/2005, art. 49.4 — contenido de la disculpa:
    "La disculpa debe incluir el reconocimiento publico de los hechos y la aceptacion de responsabilidades.",

    # Ley 975/2005, reparacion simbolica:
    "La reparacion simbolica tiende a asegurar la preservacion de la memoria historica, la no repeticion de los hechos victimizantes, la aceptacion publica de los hechos, el perdon publico y el restablecimiento de la dignidad de las victimas.",

    # Estandar Corte IDH / ICTJ sobre disculpa efectiva:
    "Una disculpa efectiva reconoce las injusticias especificas que ocurrieron, reconoce que las victimas sufrieron un dano grave como resultado, acepta la responsabilidad por lo sucedido, y asegura a las victimas que no tuvieron culpa, con garantias de no repeticion.",

    # Corte IDH, contenido del acto de reconocimiento (Radilla Pacheco, parrs. 351-354):
    "El Estado debe realizar un acto publico de reconocimiento de responsabilidad en desagravio a la memoria de la victima, haciendo referencia a las violaciones de derechos humanos declaradas, en presencia de altas autoridades y de los familiares.",

    # >>> AGREGAR MAS ESTANDARES NORMATIVOS AQUI <<<
]


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("=" * 60)
    print("CFH — Centroide RESTAURATIVO v1 (y11)")
    print("=" * 60)

    # ── Reunir textos con deduplicacion (mismo metodo que MAFAPO v5) ──
    textos_pesos = []
    vistos = set()

    def agregar(texto, peso):
        clave = texto.strip()[:100]
        if clave and clave not in vistos and len(texto.strip()) >= 10:
            vistos.add(clave)
            textos_pesos.append((texto, peso))
            return True
        return False

    n_pleno = sum(agregar(t, 1.8) for t in TEXTOS_RESTAURATIVOS_PLENO)
    n_estandar = sum(agregar(t, 1.0) for t in TEXTOS_RESTAURATIVOS_ESTANDAR)
    n_total = len(textos_pesos)
    n_w18 = sum(1 for _, w in textos_pesos if w == 1.8)

    print(f"\n  Textos unicos reunidos: {n_total}")
    print(f"    Reconocimiento pleno (w=1.8): {n_pleno}")
    print(f"    Estandar normativo   (w=1.0): {n_estandar}")

    if n_total < 15:
        print(f"\n  [AVISO] Solo {n_total} textos. El centroide MAFAPO v5 uso ~150+.")
        print(f"          Para un centroide robusto (margen <10%), cura MAS apartes.")
        print(f"          margen estimado actual: ~{100/(n_total**0.5):.1f}%")

    # ── Cargar modelo (identico a MAFAPO v5) ─────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
    print(f"\n  Cargando ConfliBERT-Spanish en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  OK modelo cargado")

    def get_emb(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512,
                           truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # ── Embeddings ───────────────────────────────────────────────────
    print(f"\n  Calculando embeddings ({n_total} textos)...")
    embs, pesos = [], []
    for i, (texto, w) in enumerate(textos_pesos):
        emb = get_emb(texto)
        if np.linalg.norm(emb) > 0:
            embs.append(emb)
            pesos.append(w)
    embs = np.array(embs)
    pesos = np.array(pesos)
    centroide = np.average(embs, axis=0, weights=pesos)
    n_eff = len(embs)
    margen = 100 / (n_eff ** 0.5)
    norma = float(np.linalg.norm(centroide))

    print(f"\n  OK Centroide RESTAURATIVO v1 calculado")
    print(f"    textos efectivos: {n_eff}")
    print(f"    norma: {norma:.4f}")
    print(f"    margen estimado: +-{margen:.1f}%")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT, centroide)
    inv = {
        "version": "restaurativo_v1",
        "n_textos_efectivos": n_eff,
        "n_reconocimiento_pleno_w18": n_w18,
        "margen_estimado_pct": round(margen, 2),
        "norma": norma,
        "modelo": model_name,
        "metodo": "CLS token, promedio ponderado w=1.8 pleno / w=1.0 estandar, dedup 100 chars",
        "nota": "Corpus EXTERNO a comparecientes M03 (evita circularidad). Sin voz de victimas.",
        "fuentes": ["Ley 975/2005 (Justicia y Paz) arts. 45.3 y 49.4 - estandar normativo",
                    "Estandares Corte IDH / ICTJ de reparacion y disculpa efectiva",
                    "Actos de reconocimiento FARC-EP ante la JEP (Macrocaso 01, secuestro)",
                    "Nota teorica: el lenguaje restaurativo del perdon es el mismo",
                    "independientemente del perpetrador (militar, paramilitar, guerrilla)"],
    }
    with open(OUT_INV, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado: {OUT}")
    print(f"  Inventario: {OUT_INV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
