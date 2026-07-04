# -*- coding: utf-8 -*-
r"""
cfh_revisar_ebi_solapamiento.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Revisa si los comparecientes con EBI alto USAN los eufemismos (encubrimiento
activo) o los CITAN para desmontarlos (reconocimiento del encubrimiento pasado).

Para cada compareciente objetivo, muestra CADA hit del gazetteer EBI con su
CONTEXTO (ventana de +-120 caracteres), para que se vea a simple vista si el
eufemismo aparece:
  - USADO   : "los di de baja en combate" (marco belico activo)
  - CITADO  : "los reporte como 'baja en combate', pero eran civiles" (desmonte)

Uso (raiz del repo, env cfh):
  python code\cfh_revisar_ebi_solapamiento.py
  python code\cfh_revisar_ebi_solapamiento.py --quien "Henry Torres"
  python code\cfh_revisar_ebi_solapamiento.py --ventana 160
"""

import os
import re
import argparse
import pandas as pd


# Mismo gazetteer que el extractor (para localizar los hits en contexto)
EBI_PATRONES = [
    r"\bbaja[s]?\s+(?:en\s+)?combate", r"\bdad[oa]s?\s+de\s+baja",
    r"\bdieron\s+de\s+baja", r"\bda(?:r|rle|rles|ndo)\s+de\s+baja",
    r"\bpresentad[oa]s?\s+como\s+baja[s]?",
    r"\breportar(?:on|se)?\s+como\s+(?:baja|muert|dad[oa]\s+de\s+baja)",
    r"\bbaja[s]?\s+del\s+enemigo", r"\bpresunto\s+combate",
    r"\bcombate\s+simulad[oa]", r"\bsimular?\s+(?:un\s+)?combate",
    r"\bsimulad[oa]\s+en\s+combate", r"\bmuert[eo]s?\s+en\s+(?:presunto\s+)?combate",
    r"\bfalsa\s+presentaci[oó]n\s+de\s+la\s+muerte",
    r"\bmuertes?\s+ileg[ií]timamente\s+presentad",
    r"\bresultad[oa]s?\s+operacional(?:es)?", r"\bmisi[oó]n\s+t[aá]ctica",
    r"\boperaci[oó]n\s+(?:militar|t[aá]ctica|fragmentaria)",
    r"\borden\s+de\s+operaci[oó]n", r"\bregistro\s+y\s+control\s+militar",
    r"\bdieron\s+muerte", r"\bdar(?:le|les)?\s+muerte",
    r"\bcausar(?:le|les)?\s+la\s+muerte", r"\bneutraliz(?:ar|ado|aron|acion)",
    r"\bacordaron\s+darle\s+muerte", r"\bfue\s+interceptad[oa]\s+y\s+retenid",
    r"\bfueron\s+abordad[oa]s", r"\bfue\s+reclutad[oa]",
    r"\bresultaron\s+muert[oa]s", r"\bhabr[ií]an\s+perdido\s+la\s+vida",
    r"\bpresentar\s+(?:este\s+tipo\s+de\s+)?bajas",
    r"\bpresi[oó]n\s+por\s+resultados", r"\bmuertes?\s+en\s+combate\b",
]
_EBI = [re.compile(p, re.IGNORECASE) for p in EBI_PATRONES]

# Marcadores de CITA/DESMONTE cerca del eufemismo (senales de que lo cita, no lo usa)
CITA_SIGNS = re.compile(
    r"\b(?:pero|aunque|falsamente|supuest|mal\s+llamad|"
    r"present[eé]\s+como|report[eé]\s+como|hice\s+pasar|"
    r"en\s+realidad|no\s+era|eran\s+civil|comill|entre\s+comillas|"
    r"llamad[oa]s?\s+as[ií]|mal\s+llamada|reconozco\s+que|"
    r"me\s+equivoqu|estuvo\s+mal|fue\s+un\s+error|"
    r"cuando\s+dije|dijimos\s+que|reportamos)\b",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto_c", default=os.path.join("data", "texto_por_compareciente.csv"))
    ap.add_argument("--quien", default=None,
                    help="Filtra por nombre (subcadena). Por defecto: Torres, Aguilera, Herrera.")
    ap.add_argument("--ventana", type=int, default=120,
                    help="Caracteres de contexto a cada lado del hit.")
    args = ap.parse_args()

    df = pd.read_csv(args.texto_c)

    if args.quien:
        objetivos = [args.quien]
    else:
        objetivos = ["Torres", "Aguilera", "Herrera", "Santiago Herrera"]

    V = args.ventana

    for patron_nombre in objetivos:
        subset = df[df["identidad"].str.contains(patron_nombre, case=False, na=False)]
        for _, row in subset.iterrows():
            ident = row["identidad"]
            txt = str(row.get("texto_completo", "") or "")
            print("=" * 74)
            print(f"COMPARECIENTE: {ident}  ({row.get('subcaso')})")
            print("=" * 74)

            hits = []
            for rx in _EBI:
                for mm in rx.finditer(txt):
                    hits.append((mm.start(), mm.end(), mm.group()))
            hits.sort()

            if not hits:
                print("  (sin hits EBI)\n")
                continue

            n_citados = 0
            print(f"  Total hits EBI: {len(hits)}\n")
            # Mostrar hasta 25 hits con contexto (para no saturar)
            for i, (s, e, g) in enumerate(hits[:25], 1):
                ini = max(0, s - V)
                fin = min(len(txt), e + V)
                ctx = txt[ini:fin].replace("\n", " ")
                # marcar el eufemismo con <<>>
                rel_s = s - ini
                rel_e = e - ini
                ctx_marcado = ctx[:rel_s] + ">>" + ctx[rel_s:rel_e] + "<<" + ctx[rel_e:]
                es_cita = bool(CITA_SIGNS.search(ctx))
                if es_cita:
                    n_citados += 1
                tag = "[CITA?]" if es_cita else "[USO? ]"
                print(f"  {i:>2}. {tag} ...{ctx_marcado}...")
                print()

            if len(hits) > 25:
                print(f"  (... {len(hits)-25} hits mas no mostrados)\n")

            print(f"  RESUMEN {ident}: {len(hits)} hits | "
                  f"~{n_citados} con senal de CITA/desmonte cerca "
                  f"(de los primeros {min(25,len(hits))} mostrados)")
            print()


if __name__ == "__main__":
    main()
