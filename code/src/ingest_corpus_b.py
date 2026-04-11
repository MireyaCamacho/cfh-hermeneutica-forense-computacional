"""
Ingesta Corpus B ampliado — PDFs JEP Macrocaso 003
===================================================
Procesa los nuevos documentos descargados del portal JEP y los
convierte al formato JSON estándar del pipeline CFH.

Documentos pendientes de ingesta:
- adhc-062-2023-antioquia.pdf
- adhc-081-2023-huila.pdf
- RC-04-2024-dabeiba.pdf
- (+ cualquier otro que se agregue)

Uso:
    python code/src/ingest_corpus_b.py

Requiere:
    pip install pdfminer.six spacy
    python -m spacy download es_core_news_lg
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────────────────────

CORPUS_B_RAW  = Path("data/raw/corpus_b")
CORPUS_B_JSON = Path("data/processed/corpus_b")
CORPUS_B_JSON.mkdir(parents=True, exist_ok=True)

# Metadatos de cada documento — actualizar cuando se agreguen nuevos
METADATA = {
    "adhc-062-2023-antioquia.pdf": {
        "doc_id":    "ADHC-062-2023",
        "tipo":      "auto_determinacion_hechos",
        "subcaso":   "Antioquia",
        "año":       2023,
        "tribunal":  "SRVR",
        "macrocaso": "003",
        "comparecientes_principales": ["General Mario Montoya Uribe"],
        "secciones_target": [
            "HECHOS_Y_CONDUCTAS",
            "PATRONES_MACROCRIMINALES",
            "CALIFICACION_JURIDICA",
            "RECONOCIMIENTO",
            "RESUELVE"
        ]
    },
    "adhc-081-2023-huila.pdf": {
        "doc_id":    "ADHC-081-2023",
        "tipo":      "auto_determinacion_hechos",
        "subcaso":   "Huila",
        "año":       2023,
        "tribunal":  "SRVR",
        "macrocaso": "003",
        "comparecientes_principales": [],
        "secciones_target": [
            "HECHOS_Y_CONDUCTAS",
            "PATRONES_MACROCRIMINALES",
            "CALIFICACION_JURIDICA",
            "RESUELVE"
        ]
    },
    "RC-04-2024-dabeiba.pdf": {
        "doc_id":    "RC-04-2024",
        "tipo":      "resolucion_conclusiones",
        "subcaso":   "Dabeiba-Antioquia",
        "año":       2024,
        "tribunal":  "SRVR",
        "macrocaso": "003-004",
        "comparecientes_principales": [],
        "secciones_target": [
            "HECHOS_Y_CONDUCTAS",
            "PATRONES_MACROCRIMINALES",
            "CALIFICACION_JURIDICA",
            "RECONOCIMIENTO",
            "SANCIONES_PROPIAS",
            "RESUELVE"
        ]
    },
}

# ── Extractor de texto PDF ─────────────────────────────────────────────────────

def extraer_texto_pdf(path_pdf):
    """Extrae texto de PDF con pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text
        texto = extract_text(str(path_pdf))
        print(f"  ✓ Texto extraído: {len(texto):,} chars")
        return texto
    except Exception as e:
        print(f"  ✗ Error pdfminer: {e}")
        # Fallback: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path_pdf))
            texto = "\n".join(p.extract_text() or "" for p in reader.pages)
            print(f"  ✓ Texto extraído (pypdf): {len(texto):,} chars")
            return texto
        except Exception as e2:
            print(f"  ✗ Error pypdf: {e2}")
            return ""

# ── Limpiador de texto ─────────────────────────────────────────────────────────

def limpiar_texto(texto):
    """Aplica limpieza estándar del pipeline CFH."""
    # Normalizar saltos de línea
    texto = re.sub(r'\r\n|\r', '\n', texto)
    # Eliminar caracteres de control excepto saltos de línea
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    # Normalizar espacios múltiples
    texto = re.sub(r'[ \t]+', ' ', texto)
    # Eliminar líneas con solo números (páginas)
    texto = re.sub(r'^\s*\d+\s*$', '', texto, flags=re.MULTILINE)
    # Eliminar encabezados repetitivos JEP
    texto = re.sub(r'Jurisdicción Especial para la Paz\s*', '', texto)
    texto = re.sub(r'Sala de Reconocimiento.*?\n', '', texto)
    # Normalizar múltiples saltos de línea
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()

# ── Segmentador por secciones ─────────────────────────────────────────────────

PATRONES_SECCIONES = {
    "HECHOS_Y_CONDUCTAS": [
        r"HECHOS Y CONDUCTAS",
        r"I\.\s+HECHOS",
        r"DETERMINACIÓN DE HECHOS",
        r"DESCRIPCIÓN DE LOS HECHOS",
    ],
    "PATRONES_MACROCRIMINALES": [
        r"PATRONES MACROCRIMINALES",
        r"PATRÓN MACROCRIMINAL",
        r"MODUS OPERANDI",
    ],
    "CALIFICACION_JURIDICA": [
        r"CALIFICACIÓN JURÍDICA",
        r"CALIFICACION JURIDICA",
        r"TIPIFICACIÓN",
        r"CRÍMENES DE GUERRA",
        r"CRÍMENES DE LESA HUMANIDAD",
    ],
    "RECONOCIMIENTO": [
        r"RECONOCIMIENTO DE RESPONSABILIDAD",
        r"RECONOCIMIENTO DE VERDAD",
        r"ACTOS DE RECONOCIMIENTO",
    ],
    "SANCIONES_PROPIAS": [
        r"SANCIONES PROPIAS",
        r"SANCIÓN PROPIA",
        r"PROYECTOS RESTAURATIVOS",
        r"TRABAJOS.*REPARACIÓN",
    ],
    "RESUELVE": [
        r"^RESUELVE",
        r"EN MÉRITO DE LO EXPUESTO",
        r"POR LO ANTERIOR.*RESUELVE",
    ],
}

def segmentar_documento(texto, secciones_target):
    """Divide el documento en secciones usando patrones."""
    secciones = {}
    lineas = texto.split('\n')
    seccion_actual = "CUERPO"
    buffer = []

    for linea in lineas:
        linea_upper = linea.upper().strip()
        seccion_detectada = None

        for nombre_seccion, patrones in PATRONES_SECCIONES.items():
            if nombre_seccion in secciones_target:
                for patron in patrones:
                    if re.search(patron, linea_upper):
                        seccion_detectada = nombre_seccion
                        break
            if seccion_detectada:
                break

        if seccion_detectada and seccion_detectada != seccion_actual:
            if buffer:
                texto_seccion = '\n'.join(buffer).strip()
                if texto_seccion:
                    if seccion_actual not in secciones:
                        secciones[seccion_actual] = []
                    secciones[seccion_actual].append(texto_seccion)
            seccion_actual = seccion_detectada
            buffer = []
        else:
            buffer.append(linea)

    # Último buffer
    if buffer:
        texto_seccion = '\n'.join(buffer).strip()
        if texto_seccion:
            if seccion_actual not in secciones:
                secciones[seccion_actual] = []
            secciones[seccion_actual].append(texto_seccion)

    return secciones

# ── Procesador principal ───────────────────────────────────────────────────────

def procesar_documento(nombre_pdf, meta):
    """Procesa un PDF y genera el JSON estándar CFH."""
    path_pdf = CORPUS_B_RAW / nombre_pdf
    if not path_pdf.exists():
        print(f"  ✗ No encontrado: {path_pdf}")
        return None

    print(f"\nProcesando: {nombre_pdf}")

    # 1. Extraer texto
    texto_crudo = extraer_texto_pdf(path_pdf)
    if not texto_crudo:
        return None

    # 2. Limpiar
    texto_limpio = limpiar_texto(texto_crudo)

    # 3. SHA-256 para cadena de custodia
    sha256 = hashlib.sha256(texto_limpio.encode('utf-8')).hexdigest()

    # 4. Segmentar
    secciones = segmentar_documento(texto_limpio, meta["secciones_target"])
    print(f"  ✓ Secciones detectadas: {list(secciones.keys())}")

    # 5. Construir JSON
    doc_json = {
        "doc_id":      meta["doc_id"],
        "tipo":        meta["tipo"],
        "subcaso":     meta["subcaso"],
        "año":         meta["año"],
        "tribunal":    meta["tribunal"],
        "macrocaso":   meta["macrocaso"],
        "corpus_type": "B",
        "sha256":      sha256,
        "chars_total": len(texto_limpio),
        "fecha_ingesta": datetime.now().isoformat(),
        "secciones":   {},
        "comparecientes": meta.get("comparecientes_principales", [])
    }

    # 6. Agregar secciones target
    for nombre_sec, bloques in secciones.items():
        if nombre_sec in meta["secciones_target"] or nombre_sec == "CUERPO":
            doc_json["secciones"][nombre_sec] = {
                "texto":  '\n\n'.join(bloques),
                "chars":  sum(len(b) for b in bloques),
                "bloques": len(bloques)
            }

    secciones_target_encontradas = [
        s for s in meta["secciones_target"] if s in doc_json["secciones"]
    ]
    print(f"  ✓ Secciones target encontradas: {len(secciones_target_encontradas)}/{len(meta['secciones_target'])}")

    return doc_json

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("== INGESTA CORPUS B AMPLIADO ==\n")

    resultados = []
    for nombre_pdf, meta in METADATA.items():
        doc = procesar_documento(nombre_pdf, meta)
        if doc:
            # Guardar JSON
            path_out = CORPUS_B_JSON / f"{meta['doc_id']}.json"
            with open(path_out, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"  ✓ JSON guardado: {path_out}")
            resultados.append({
                "doc_id":   meta["doc_id"],
                "subcaso":  meta["subcaso"],
                "chars":    doc["chars_total"],
                "secciones": len([s for s in doc["secciones"] if s != "CUERPO"])
            })

    print("\n== RESUMEN ==")
    total_secciones = 0
    for r in resultados:
        print(f"  {r['doc_id']} ({r['subcaso']}): {r['chars']:,} chars | {r['secciones']} secciones target")
        total_secciones += r["secciones"]

    print(f"\nTotal secciones nuevas: {total_secciones}")
    print(f"Total Corpus B estimado: 54 + {total_secciones} = {54 + total_secciones} secciones")
    print(f"Objetivo ≥200: {'✓ CUMPLIDO' if 54 + total_secciones >= 200 else '⚠ PENDIENTE'}")
