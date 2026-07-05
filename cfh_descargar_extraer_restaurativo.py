# -*- coding: utf-8 -*-
"""
CFH — Descarga sentencias Corte IDH y extrae candidatos RESTAURATIVOS
======================================================================
Replica el metodo del centroide MAFAPO (descarga + extraccion por lexicon +
niveles), pero sobre SENTENCIAS de la Corte IDH (PDF externos), para construir
el corpus del polo RESTAURATIVO (reconocimiento de responsabilidad + perdon +
reparacion).

Corpus EXTERNO a los comparecientes -> evita circularidad.
SIN voz de victimas -> no se confunde con MAFAPO.

FLUJO (identico en espiritu a cfh_descargar_y_extraer_v5.py):
  1. Descarga los PDF de sentencias Corte IDH (corteidh.or.cr)
  2. Extrae el texto con PyMuPDF (fitz)
  3. Segmenta en oraciones/parrafos
  4. Filtra por LEXICON RESTAURATIVO y clasifica por nivel (score)
  5. Guarda candidatos en JSON con el MISMO formato que MAFAPO
     (segmentos: texto, fuente, nivel, score, longitud)

Luego, cfh_centroide_restaurativo.py lee este JSON y calcula el centroide
(ConfliBERT CLS, promedio ponderado w=1.8 nivel 1-2 / w=1.0 nivel 3).

Ejecutar:
  conda activate cfh
  python cfh_descargar_extraer_restaurativo.py

NOTA: la Corte IDH publica los PDF en corteidh.or.cr. Si alguna URL cambia,
el script avisa y continua con las demas. Las sentencias descargadas quedan
en data/fuentes_restaurativas/ para trazabilidad (citar en la tesis).
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent
FUENTES_DIR = REPO / "data" / "fuentes_restaurativas"
OUT_DIR = REPO / "data" / "referencias"
OUT_JSON = OUT_DIR / "corpus_restaurativo_candidatos.json"
FUENTES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sentencias Corte IDH (PDF publicos). Fuente: corteidh.or.cr ──────
# Cada una: (nombre_archivo, url, etiqueta_para_cita)
SENTENCIAS = [
    ("seriec_372_esp.pdf",
     "https://www.corteidh.or.cr/docs/casos/articulos/seriec_372_esp.pdf",
     "Ordenes Guerra y otros vs. Chile (2018)"),
    ("seriec_209_esp.pdf",
     "https://www.corteidh.or.cr/docs/casos/articulos/seriec_209_esp.pdf",
     "Radilla Pacheco vs. Mexico (2009)"),
    ("seriec_140_esp.pdf",
     "https://www.corteidh.or.cr/docs/casos/articulos/seriec_140_esp.pdf",
     "Masacre de Pueblo Bello vs. Colombia (2006)"),
    ("seriec_205_esp.pdf",
     "https://www.corteidh.or.cr/docs/casos/articulos/seriec_205_esp.pdf",
     "Gonzalez y otras (Campo Algodonero) vs. Mexico (2009)"),
]

# ── Lexicon restaurativo por nivel (analogo a LEXICON_VICTIMAS) ──────
# Nivel 1 (w=1.8): perdon / responsabilidad en 1a persona (acto directo)
LEX_NIVEL1 = [
    "pido perdon", "pedir perdon", "pedimos perdon", "pide perdon",
    "reconozco la responsabilidad", "reconocemos la responsabilidad",
    "asumo la responsabilidad", "asumimos la responsabilidad",
    "en nombre del estado", "en nombre de la nacion",
]
# Nivel 2 (w=1.8): reconocimiento/aceptacion formal de responsabilidad
LEX_NIVEL2 = [
    "reconocimiento de responsabilidad", "aceptacion de responsabilidad",
    "reconocio su responsabilidad", "acepto la responsabilidad",
    "acepta la responsabilidad", "disculpa publica", "acto de disculpa",
    "acto de perdon", "reconocimiento de los hechos",
    "reconoce las violaciones", "desagravio",
]
# Nivel 3 (w=1.0): reparacion / no repeticion / dignificacion
LEX_NIVEL3 = [
    "reparacion integral", "garantias de no repeticion", "no repeticion",
    "medida de reparacion", "reparar el dano", "dignidad de las victimas",
    "dignificacion", "satisfaccion", "reivindicar", "memoria de las victimas",
    "reconciliacion",
]

# Ruido a excluir (texto procesal que NO es restaurativo)
RUIDO = [
    "excepcion preliminar", "competencia de la corte", "voto razonado",
    "articulo del reglamento", "notificacion", "costas y gastos",
    "supervision de cumplimiento",
]


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def descargar():
    try:
        import requests
    except ImportError:
        print("Instalando requests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"], check=True)
        import requests
    print("=" * 64)
    print("CFH — Descarga sentencias Corte IDH (corpus restaurativo)")
    print("=" * 64)
    ok = []
    for nombre, url, etiqueta in SENTENCIAS:
        ruta = FUENTES_DIR / nombre
        if ruta.exists() and ruta.stat().st_size > 10000:
            print(f"  = Ya existe: {nombre} ({ruta.stat().st_size//1024} KB)")
            ok.append((ruta, etiqueta))
            continue
        try:
            print(f"  Descargando {nombre} ...")
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 10000:
                ruta.write_bytes(r.content)
                print(f"  + OK: {nombre} ({len(r.content)//1024} KB)")
                ok.append((ruta, etiqueta))
            else:
                print(f"  x Error {r.status_code} en {nombre} (o archivo muy pequeno)")
        except Exception as e:
            print(f"  x Fallo {nombre}: {e}")
    return ok


def extraer_texto(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    partes = []
    for pag in doc:
        partes.append(pag.get_text())
    doc.close()
    return "\n".join(partes)


def segmentar(texto):
    # limpiar saltos, unir guiones de corte de linea
    t = texto.replace("-\n", "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    # partir en oraciones (por punto seguido de mayuscula o fin)
    oraciones = re.split(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚ«\"])", t)
    return [o.strip() for o in oraciones if len(o.strip()) >= 40]


def clasificar(oracion):
    """Devuelve (nivel, score) segun lexicon; None si no es restaurativa o es ruido."""
    o = norm(oracion)
    if any(r in o for r in RUIDO):
        return None
    n1 = sum(1 for k in LEX_NIVEL1 if k in o)
    n2 = sum(1 for k in LEX_NIVEL2 if k in o)
    n3 = sum(1 for k in LEX_NIVEL3 if k in o)
    if n1 > 0:
        return (1, n1 * 3 + n2 + n3)
    if n2 > 0:
        return (2, n2 * 2 + n3)
    if n3 >= 2:   # nivel 3 exige al menos 2 marcadores (evita falsos positivos)
        return (3, n3)
    return None


def main():
    fuentes = descargar()
    if not fuentes:
        print("\n  [ERROR] no se descargo ninguna sentencia. Revisa conexion o URLs.")
        return

    print("\n" + "=" * 64)
    print("EXTRACCION de apartes restaurativos")
    print("=" * 64)

    segmentos = []
    vistos = set()
    for ruta, etiqueta in fuentes:
        try:
            texto = extraer_texto(ruta)
        except Exception as e:
            print(f"  x No pude leer {ruta.name}: {e}")
            continue
        oraciones = segmentar(texto)
        n_fuente = 0
        for o in oraciones:
            clave = norm(o)[:100]
            if clave in vistos:
                continue
            cl = clasificar(o)
            if cl is None:
                continue
            nivel, score = cl
            # recortar oraciones larguisimas (mantener 2-4 frases ~ <600 chars)
            texto_seg = o if len(o) <= 600 else o[:600].rsplit(".", 1)[0] + "."
            vistos.add(clave)
            segmentos.append({
                "texto": texto_seg,
                "fuente": etiqueta,
                "nivel": nivel,
                "score": score,
                "longitud": len(texto_seg),
            })
            n_fuente += 1
        print(f"  {etiqueta[:45]:45s} -> {n_fuente} apartes")

    # ordenar por nivel y score
    segmentos.sort(key=lambda s: (s["nivel"], -s["score"]))
    por_nivel = {}
    for s in segmentos:
        por_nivel[s["nivel"]] = por_nivel.get(s["nivel"], 0) + 1

    out = {
        "version": "restaurativo_candidatos_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": len(segmentos),
        "por_nivel": por_nivel,
        "fuentes": [e for _, e in fuentes],
        "nota": "Corpus externo (sentencias Corte IDH). Sin voz de victimas. Evita circularidad.",
        "segmentos": segmentos,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"TOTAL apartes restaurativos: {len(segmentos)}")
    print(f"  por nivel: {por_nivel}")
    print(f"  nivel 1-2 (w=1.8): {por_nivel.get(1,0)+por_nivel.get(2,0)}")
    print(f"  nivel 3   (w=1.0): {por_nivel.get(3,0)}")
    print(f"\n  Guardado: {OUT_JSON}")
    print(f"  PDFs fuente: {FUENTES_DIR}  (para citar en la tesis)")
    if len(segmentos) < 30:
        print(f"\n  [AVISO] {len(segmentos)} apartes. Para centroide robusto conviene 50+.")
        print(f"          Agrega mas sentencias en la lista SENTENCIAS, o ajusta el lexicon.")
    print("\n  Revisa los apartes en el JSON antes de correr el centroide.")
    print("  Luego adapta cfh_centroide_restaurativo.py para leer este JSON.")
    print("=" * 64)


if __name__ == "__main__":
    main()
