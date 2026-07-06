# -*- coding: utf-8 -*-
r"""
cfh_subir_versiones_finales.py
CFH - Hermeneutica Forense Computacional | Mireya Camacho Celis

Copia las versiones FINALES de los documentos de tesis desde la carpeta de
descargas a la carpeta textos\ del proyecto, con nombres limpios y verificacion.

Los archivos descargados pueden tener sufijos (1), (2)... del navegador; el
script toma el mas reciente de cada tipo y lo copia SIN el numero.

Uso (raiz del repo, env cfh):
    python code\cfh_subir_versiones_finales.py

Si tus descargas estan en otra carpeta, edita DOWNLOADS abajo.
"""

import os
import re
import shutil
import glob
from datetime import datetime

# --- CONFIGURACION (editar si hace falta) ---
DOWNLOADS = os.path.join(os.environ.get("USERPROFILE", ""), "Downloads")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(REPO, "textos")

# Mapeo: patron de busqueda -> nombre final limpio
DOCUMENTOS = {
    "CFH_Tesis_Capitulo1_Introduccion_V6":       "CFH_Tesis_Capitulo1_Introduccion_FINAL.docx",
    "CFH_Tesis_Capitulo2_EstadoDelArte_V9":      "CFH_Tesis_Capitulo2_EstadoDelArte_FINAL.docx",
    "CFH_Tesis_Capitulo3_MarcoTeorico_V14":      "CFH_Tesis_Capitulo3_MarcoTeorico_FINAL.docx",
    "CFH_Tesis_Capitulo4_Metodologia_V18":       "CFH_Tesis_Capitulo4_Metodologia_FINAL.docx",
    "CFH_Tesis_Capitulo5_Resultados_V27":        "CFH_Tesis_Capitulo5_Resultados_FINAL.docx",
    "CFH_Tesis_Capitulo6_Discusion_V23":         "CFH_Tesis_Capitulo6_Discusion_FINAL.docx",
    "CFH_Apendice_Matematico_Experimentos_v2":   "CFH_Apendice_Matematico_Experimentos_FINAL.docx",
    "CFH_Referencias_Consolidadas_APA7_v2":      "CFH_Referencias_Consolidadas_APA7_FINAL.docx",
}


def mas_reciente(patron_base):
    """Encuentra el archivo mas reciente que coincide con el patron base,
    ignorando sufijos (1),(2) del navegador. Devuelve el de mtime mas nuevo."""
    # buscar variantes: base.docx, base (1).docx, base (2).docx
    candidatos = glob.glob(os.path.join(DOWNLOADS, patron_base + "*.docx"))
    # filtrar los que empiezan exactamente por el patron (evita falsos positivos)
    candidatos = [c for c in candidatos if os.path.basename(c).startswith(patron_base)]
    if not candidatos:
        return None
    # el mas reciente por fecha de modificacion
    return max(candidatos, key=os.path.getmtime)


def main():
    print("=" * 64)
    print("SUBIR VERSIONES FINALES A textos\\")
    print("=" * 64)
    print(f"Origen:  {DOWNLOADS}")
    print(f"Destino: {DESTINO}")

    if not os.path.isdir(DOWNLOADS):
        print(f"\n[ERROR] No existe la carpeta de descargas: {DOWNLOADS}")
        print("Edita la variable DOWNLOADS en el script con la ruta correcta.")
        return

    os.makedirs(DESTINO, exist_ok=True)

    # backup de textos existentes que se vayan a sobrescribir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(DESTINO, f"_backup_previos_{ts}")

    resumen = []
    for patron, nombre_final in DOCUMENTOS.items():
        origen = mas_reciente(patron)
        destino = os.path.join(DESTINO, nombre_final)
        if origen is None:
            print(f"\n[FALTA]  {patron}*.docx no encontrado en Downloads")
            resumen.append((nombre_final, "FALTA en Downloads"))
            continue
        # backup si ya existe en destino
        if os.path.exists(destino):
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(destino, os.path.join(backup_dir, nombre_final))
        # copiar
        shutil.copy2(origen, destino)
        tam_kb = os.path.getsize(destino) / 1024
        origen_base = os.path.basename(origen)
        print(f"\n[OK]     {nombre_final}")
        print(f"         desde: {origen_base}  ({tam_kb:.0f} KB)")
        resumen.append((nombre_final, "copiado"))

    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)
    ok = sum(1 for _, e in resumen if e == "copiado")
    for nombre, estado in resumen:
        marca = "OK  " if estado == "copiado" else "----"
        print(f"  [{marca}] {nombre}  ({estado})")
    print(f"\n  {ok}/{len(DOCUMENTOS)} documentos copiados a {DESTINO}")
    if os.path.isdir(backup_dir):
        print(f"  Versiones previas respaldadas en: {backup_dir}")
    if ok < len(DOCUMENTOS):
        print("\n  Los que faltan: descargalos primero desde el chat, o revisa")
        print("  que el nombre en Downloads empiece por el patron esperado.")


if __name__ == "__main__":
    main()
