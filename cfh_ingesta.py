#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cfh_ingesta.py  (v3)
====================
Ingesta canónica de los datos del proyecto CFH a SQLite.

CAMBIOS RESPECTO A v2:
  - Los documentos se crean a partir de los CSVs de indicadores (que tienen
    doc_id completo de 64 chars), no desde el inventario_documentos_completo_v2
    que está truncado a 20 chars + ellipsis.
  - El inventario se usa solo para ENRIQUECER (UPDATE) por match de prefijo.
  - Identificador único de bloque ahora es {doc_id}__{section_id}__{n}, donde
    n es el contador intra-(doc_id, section_id). Esto soporta los 211 casos
    de filas con la misma (doc_id, section_id) en Corpus A.
  - Granularidad correcta por corpus:
      A: bloque_granular (cada fila del CSV es un bloque)
      B v1/v2 (CSV): seccion (filas agregadas a nivel sección)
      B (.txt): bloque_granular (con sección padre)
      C: bloque_granular

Uso:
    python cfh_ingesta.py --listar
    python cfh_ingesta.py --reset-db --solo todo
    python cfh_ingesta.py --solo documentos

Autor: Mireya Camacho Celis (CFH)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURACIÓN                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DB_DEFAULT = "cfh.db"
SCHEMA_DEFAULT = "cfh_schema.sql"

AUDIENCIAS_CANONICAS = [
    {
        "subcaso": "Catatumbo", "fecha_inicio": "2022-04-26", "duracion_horas": 9.5,
        "magistrada": "Catalina Díaz",
        "materialidad": "Primera audiencia de reconocimiento — Norte de Santander",
        "ruta_audio": "corpus_c/Caso 03： Audiencia de Reconocimiento por 'falsos positivos' en el Catatumbo.mp3",
        "ruta_video": None, "drm_bloqueado": 0,
        "audio_id_csv": "catatumbo",
        "audio_aliases": ["catatumbo", "catatumbo_norte_santander", "catatumbo_chaparro"],
        "notas": "Capitán Juan Carlos Chaparro como compareciente principal",
    },
    {
        "subcaso": "Costa Caribe", "fecha_inicio": "2022-07-12", "duracion_horas": 10.5,
        "magistrada": "Catalina Díaz",
        "materialidad": "Batallón La Popa, 12 comparecientes",
        "ruta_audio": "corpus_c/costa_caribe.mp3",
        "ruta_video": None, "drm_bloqueado": 1,
        "audio_id_csv": "costa_caribe",
        "audio_aliases": ["costa_caribe", "la_popa", "costa_caribe_la_popa", "popa"],
        "notas": "Video bloqueado por DRM en YouTube. ICM facial pendiente.",
    },
    {
        "subcaso": "Casanare", "fecha_inicio": "2022-04-27", "duracion_horas": 5.3,
        "magistrada": "Catalina Díaz",
        "materialidad": "Casanare — General Henry Torres Escalante",
        "ruta_audio": "corpus_c/casanare_torres.mp3",
        "ruta_video": None, "drm_bloqueado": 0,
        "audio_id_csv": "casanare_torres",
        "audio_aliases": ["casanare_torres", "casanare", "torres_escalante", "casanare_torres_escalante"],
        "notas": None,
    },
    {
        "subcaso": "Dabeiba", "fecha_inicio": "2022-09-26", "duracion_horas": 2.3,
        "magistrada": "Catalina Díaz",
        "materialidad": "49 fosas Las Mercedes — Dabeiba, Antioquia",
        "ruta_audio": "corpus_c/dabeiba_antioquia.mp3",
        "ruta_video": None, "drm_bloqueado": 0,
        "audio_id_csv": "dabeiba",
        "audio_aliases": ["dabeiba", "dabeiba_antioquia", "antioquia_dabeiba", "las_mercedes"],
        "notas": "Comparecientes guiaron a investigadores a fosas físicas",
    },
    {
        "subcaso": "Huila", "fecha_inicio": "2024-04-22", "duracion_horas": 10.5,
        "magistrada": "Catalina Díaz",
        "materialidad": "Huila — Soldados y suboficiales (incluye Ollo La Tapia)",
        "ruta_audio": "corpus_c/huila.mp3",
        "ruta_video": None, "drm_bloqueado": 0,
        "audio_id_csv": "huila",
        "audio_aliases": ["huila", "huila_neiva", "neiva"],
        "notas": None,
    },
]

COMPARECIENTES_CANONICOS = [
    ("Catatumbo",   "SPEAKER_03", "Cap. Juan Carlos Chaparro Chaparro",        "Capitán/Mayor"),
    ("Catatumbo",   "SPEAKER_01", "Compareciente Catatumbo (no identificado)", "Oficial"),
    ("Casanare",    "SPEAKER_03", "Gral. Henry Torres Escalante",              "General"),
    ("Dabeiba",     "SPEAKER_01", "Compareciente Dabeiba (coronel/oficial)",   "Oficial"),
    ("Dabeiba",     "SPEAKER_03", "Compareciente Dabeiba 2",                   "Oficial"),
    ("Huila",       "SPEAKER_01", "Compareciente Huila 1 (Ollo La Tapia+)",    "Soldado/Suboficial"),
    ("Huila",       "SPEAKER_02", "Compareciente Huila 2",                     "Soldado/Suboficial"),
    ("Huila",       "SPEAKER_06", "Compareciente Huila 3",                     "Soldado/Suboficial"),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UTILIDADES                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def log(msg: str, level: str = "info") -> None:
    prefix = {"info": "[·]", "ok": "[✓]", "warn": "[!]", "error": "[✗]", "step": "[»]"}.get(level, "")
    print(f"{prefix} {msg}", flush=True)


def safe(value, default=None):
    if pd.isna(value):
        return default
    return value


def parsear_orden(s: str) -> int:
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else 0


def construir_id_bloque(doc_id: str, section_id: str, n_intra: int) -> str:
    """Identificador único: <doc_id>__<section_id>__<n_intra>."""
    return f"{doc_id}__{section_id}__{n_intra}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLASE PRINCIPAL                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFHIngesta:

    def __init__(self, db_path: str, raiz: Path, dry_run: bool = False):
        self.db_path = db_path
        self.raiz = raiz
        self.dry_run = dry_run
        self.con: Optional[sqlite3.Connection] = None

    def conectar(self) -> None:
        self.con = sqlite3.connect(self.db_path)
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.row_factory = sqlite3.Row

    def cerrar(self) -> None:
        if self.con:
            self.con.close()
            self.con = None

    def commit(self) -> None:
        if self.dry_run:
            log("dry-run: rollback", "warn")
            self.con.rollback()
        else:
            self.con.commit()

    def crear_run(self, descripcion: str, params: dict | None = None) -> int:
        cur = self.con.execute(
            "INSERT INTO runs (script, descripcion, parametros_json) VALUES (?, ?, ?)",
            ("cfh_ingesta.py", descripcion, json.dumps(params or {}, ensure_ascii=False)),
        )
        return cur.lastrowid

    def modelo_id(self, nombre: str, version: str) -> int:
        row = self.con.execute(
            "SELECT id FROM modelos WHERE nombre = ? AND version = ?",
            (nombre, version),
        ).fetchone()
        if not row:
            raise ValueError(f"Modelo no encontrado: {nombre} {version}")
        return row["id"]

    def buscar_documento_por_doc_id(self, doc_id_externo: str) -> Optional[int]:
        if not doc_id_externo:
            return None
        row = self.con.execute(
            "SELECT id FROM documentos WHERE doc_id_externo = ?",
            (doc_id_externo,),
        ).fetchone()
        return row["id"] if row else None

    def buscar_bloque_por_id_externo(self, identificador_externo: str) -> Optional[int]:
        row = self.con.execute(
            "SELECT id FROM bloques WHERE identificador_externo = ?",
            (identificador_externo,),
        ).fetchone()
        return row["id"] if row else None

    def buscar_audiencia_por_subcaso(self, subcaso: str) -> Optional[int]:
        row = self.con.execute(
            "SELECT id FROM audiencias WHERE subcaso = ?", (subcaso,)
        ).fetchone()
        return row["id"] if row else None

    def buscar_audiencia_por_audio_csv(self, audio_id: str) -> Optional[tuple[int, str]]:
        """Resuelve un nombre de audio (o alias) a (audiencia_id, subcaso)."""
        if not audio_id:
            return None
        audio_norm = str(audio_id).strip().lower()
        for au in AUDIENCIAS_CANONICAS:
            aliases = [au["audio_id_csv"]] + au.get("audio_aliases", [])
            if audio_norm in [a.lower() for a in aliases]:
                aud_id = self.buscar_audiencia_por_subcaso(au["subcaso"])
                return (aud_id, au["subcaso"]) if aud_id else None
        return None

    def buscar_compareciente(self, audiencia_id: int, speaker_id: str) -> Optional[int]:
        row = self.con.execute(
            "SELECT id FROM comparecientes WHERE audiencia_id = ? AND speaker_id = ?",
            (audiencia_id, speaker_id),
        ).fetchone()
        return row["id"] if row else None

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  HELPERS DE DOCUMENTO                                                 ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def _insertar_documento(self, p: dict) -> int:
        cols = [k for k, v in p.items() if v is not None]
        vals = [p[k] for k in cols]
        ph = ",".join(["?"] * len(cols))
        sql = f"INSERT INTO documentos ({','.join(cols)}) VALUES ({ph})"
        return self.con.execute(sql, vals).lastrowid

    def _actualizar_documento_si_vacio(self, doc_id: int, p: dict) -> None:
        actuales = self.con.execute("SELECT * FROM documentos WHERE id = ?", (doc_id,)).fetchone()
        actualizar = {}
        for k, v in p.items():
            if v is not None and (actuales[k] is None or actuales[k] == ""):
                actualizar[k] = v
        if not actualizar:
            return
        set_clause = ", ".join(f"{k} = ?" for k in actualizar)
        self.con.execute(
            f"UPDATE documentos SET {set_clause} WHERE id = ?",
            list(actualizar.values()) + [doc_id],
        )

    def _normalizar_corpus(self, corpus_csv: str) -> str:
        c = (corpus_csv or "").upper().strip()
        if "CSJ" in c: return "A-CSJ"
        if c == "A-CE" or c == "CE" or "CONSEJO" in c: return "A-CE"
        if c == "A": return "A-CE"  # fallback más probable para A genérico
        if c == "B" or "B-JEP" in c or "JEP" in c: return "B-JEP"
        if c == "C" or "C-JEP" in c: return "C-JEP-oral"
        return "A-CE"

    def _inferir_tipo_doc(self, tipo_accion: Optional[str]) -> Optional[str]:
        if not tipo_accion:
            return None
        t = tipo_accion.lower()
        if "interpretativa" in t or "senit" in t: return "sentencia_interpretativa"
        if "casacion" in t or "casación" in t: return "sentencia"
        if "auto" in t: return "auto"
        if "resolucion" in t or "resolución" in t: return "resolucion"
        if "reparacion" in t or "reparación" in t: return "sentencia"
        if "sentencia" in t: return "sentencia"
        return tipo_accion[:50]

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 1: DOCUMENTOS                                                 ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_documentos(self) -> dict:
        log("Módulo 1: documentos", "step")
        insertados = 0
        enriquecidos = 0

        # 1. Crear documentos desde CSVs de indicadores (tienen doc_id completo)
        csvs_canonicos = [
            "data/features/indicators_corpus_a.csv",
            "data/features/indicators_corpus_b.csv",
            "data/features/indicators_corpus_b_v2.csv",
            "data/features/indicators_corpus_c.csv",
        ]
        doc_ids_vistos: set[str] = set()
        for ruta_rel in csvs_canonicos:
            path = self.raiz / ruta_rel
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception as e:
                log(f"  WARN: no pude leer {ruta_rel}: {e}", "warn")
                continue
            if "doc_id" not in df.columns:
                continue

            for doc_id in df["doc_id"].dropna().unique():
                if doc_id in doc_ids_vistos:
                    continue
                doc_ids_vistos.add(doc_id)
                if self.buscar_documento_por_doc_id(doc_id):
                    continue
                # Inferir corpus desde la misma fila
                sub = df[df["doc_id"] == doc_id]
                corpus_csv = sub["corpus_type"].iloc[0] if "corpus_type" in sub.columns else ""
                corpus_norm = self._normalizar_corpus(str(corpus_csv) if corpus_csv else "")
                year_val = None
                if "year" in sub.columns:
                    nn = sub["year"].dropna()
                    if not nn.empty:
                        year_val = int(nn.iloc[0])
                self._insertar_documento({
                    "corpus": corpus_norm,
                    "doc_id_externo": doc_id,
                    "titulo": doc_id[:50],
                    "año": year_val,
                })
                insertados += 1
        log(f"  documentos creados desde CSVs de indicadores: {insertados}")

        # 2. Enriquecer con inventario maestro (doc_ids truncados, match por prefijo)
        path_inv = self.raiz / "data" / "inventario_documentos_completo_v2.csv"
        if path_inv.exists():
            df_inv = pd.read_csv(path_inv)
            log(f"  inventario maestro: {len(df_inv)} filas — enriqueciendo por prefijo de 20 chars")
            no_match = 0
            ambiguos = 0
            for _, row in df_inv.iterrows():
                doc_id_truncado = safe(row.get("doc_id"))
                if not doc_id_truncado:
                    continue
                prefijo = doc_id_truncado.replace("…", "").replace("...", "").rstrip(".").strip()
                if len(prefijo) < 16:
                    continue
                matches = self.con.execute(
                    "SELECT id FROM documentos WHERE doc_id_externo LIKE ?",
                    (prefijo + "%",),
                ).fetchall()
                if not matches:
                    no_match += 1
                    continue
                if len(matches) > 1:
                    ambiguos += 1
                    continue
                doc_db_id = matches[0]["id"]
                params = {
                    "tipo_documento": self._inferir_tipo_doc(safe(row.get("tipo_accion"))),
                    "radicado": safe(row.get("radicado")),
                    "fecha": safe(row.get("fecha_sentencia")),
                    "fuente_org": safe(row.get("subcorpus")),
                    "magistrado_ponente": safe(row.get("magistrado_ponente")),
                    "departamento": safe(row.get("departamento_hechos")) or safe(row.get("lugar_departamento")),
                    "municipio": safe(row.get("municipio_hechos")) or safe(row.get("lugar_municipio")),
                    "batallon": safe(row.get("batallon")),
                    "n_chars": int(row["n_chars"]) if not pd.isna(row.get("n_chars")) else None,
                }
                self._actualizar_documento_si_vacio(doc_db_id, params)
                enriquecidos += 1
            if no_match:
                log(f"  inventario: {no_match} filas sin match (no se encontró doc en BD)", "warn")
            if ambiguos:
                log(f"  inventario: {ambiguos} filas con match ambiguo (>1 documento)", "warn")

        # 3. Audiencias del Corpus C
        for au in AUDIENCIAS_CANONICAS:
            doc_id_aud = f"audiencia_{au['audio_id_csv']}"
            if not self.buscar_documento_por_doc_id(doc_id_aud):
                self._insertar_documento({
                    "corpus": "C-JEP-oral",
                    "doc_id_externo": doc_id_aud,
                    "titulo": f"Audiencia JEP — {au['subcaso']}",
                    "tipo_documento": "audiencia",
                    "fecha": au["fecha_inicio"],
                    "año": int(au["fecha_inicio"][:4]),
                    "fuente_org": "SRVR-JEP",
                    "ruta_original": au["ruta_audio"],
                })
                insertados += 1

        self.commit()
        log(f"  documentos — insertados: {insertados}, enriquecidos: {enriquecidos}", "ok")
        return {"insertados": insertados, "enriquecidos": enriquecidos}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 2: AUDIENCIAS Y COMPARECIENTES                                ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_audiencias_y_comparecientes(self) -> dict:
        log("Módulo 2: audiencias y comparecientes", "step")
        n_aud = 0
        n_comp = 0

        for au in AUDIENCIAS_CANONICAS:
            doc_id_aud = f"audiencia_{au['audio_id_csv']}"
            doc_db_id = self.buscar_documento_por_doc_id(doc_id_aud)
            if not doc_db_id:
                log(f"  documento no encontrado para {au['subcaso']} — corra primero documentos", "warn")
                continue
            existente = self.buscar_audiencia_por_subcaso(au["subcaso"])
            if not existente:
                self.con.execute(
                    "INSERT INTO audiencias (documento_id, subcaso, fecha_inicio, duracion_horas, "
                    "magistrada, materialidad, ruta_audio, ruta_video, drm_bloqueado, notas) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_db_id, au["subcaso"], au["fecha_inicio"], au["duracion_horas"],
                     au["magistrada"], au["materialidad"], au["ruta_audio"], au["ruta_video"],
                     au["drm_bloqueado"], au["notas"]),
                )
                n_aud += 1

        for subcaso, speaker_id, nombre, rango in COMPARECIENTES_CANONICOS:
            aud_id = self.buscar_audiencia_por_subcaso(subcaso)
            if not aud_id:
                continue
            if not self.buscar_compareciente(aud_id, speaker_id):
                self.con.execute(
                    "INSERT INTO comparecientes (audiencia_id, nombre, rango, rol_jep, speaker_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (aud_id, nombre, rango, "compareciente", speaker_id),
                )
                n_comp += 1

        self.commit()
        log(f"  audiencias insertadas: {n_aud}, comparecientes: {n_comp}", "ok")
        return {"audiencias": n_aud, "comparecientes": n_comp}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 3: BLOQUES GRANULARES — Corpus A (819 filas)                  ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_bloques_a(self) -> dict:
        log("Módulo 3: bloques granulares Corpus A (819 filas esperadas)", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_a.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas")
        n = self._cargar_bloques_granulares_csv(df)
        self.commit()
        log(f"  bloques A: {n}", "ok")
        return {"insertados": n}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 4: SECCIONES — Corpus B v2 (145 filas a nivel sección)        ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_secciones_b(self) -> dict:
        log("Módulo 4: secciones agregadas Corpus B v2 (145 filas esperadas)", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_b_v2.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas")
        n = self._cargar_secciones_csv(df)
        self.commit()
        log(f"  secciones B: {n}", "ok")
        return {"insertados": n}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  Helpers genéricos para bloques/secciones                             ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def _cargar_bloques_granulares_csv(self, df: pd.DataFrame) -> int:
        """Cada fila del CSV es un bloque granular. Identificador con contador intra-(doc_id, section_id)."""
        insertados = 0
        contador: dict = {}
        orden_doc: dict = {}
        sin_doc = 0

        for _, row in df.iterrows():
            doc_id = safe(row.get("doc_id"))
            section_id = safe(row.get("section_id"))
            if not doc_id or not section_id:
                continue
            doc_db_id = self.buscar_documento_por_doc_id(doc_id)
            if not doc_db_id:
                sin_doc += 1
                continue

            clave = (doc_id, section_id)
            contador[clave] = contador.get(clave, 0) + 1
            n_intra = contador[clave]

            orden_doc[doc_id] = orden_doc.get(doc_id, 0) + 1
            orden_global = orden_doc[doc_id]

            id_externo = construir_id_bloque(doc_id, section_id, n_intra)
            if self.buscar_bloque_por_id_externo(id_externo):
                continue

            n_chars = int(row["text_length_chars"]) if not pd.isna(row.get("text_length_chars")) else None

            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, identificador_externo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "bloque_granular", section_id, orden_global,
                 f"[bloque granular #{n_intra} de {section_id} — texto en CSV]",
                 n_chars, id_externo),
            )
            insertados += 1
        if sin_doc:
            log(f"  WARN: {sin_doc} filas sin documento en BD (correr módulo documentos primero)", "warn")
        return insertados

    def _cargar_secciones_csv(self, df: pd.DataFrame) -> int:
        """Cada fila del CSV es una sección agregada. Identificador con contador para robustez."""
        insertados = 0
        contador: dict = {}
        orden_doc: dict = {}
        sin_doc = 0

        for _, row in df.iterrows():
            doc_id = safe(row.get("doc_id"))
            section_id = safe(row.get("section_id"))
            if not doc_id or not section_id:
                continue
            doc_db_id = self.buscar_documento_por_doc_id(doc_id)
            if not doc_db_id:
                sin_doc += 1
                continue

            clave = (doc_id, section_id)
            contador[clave] = contador.get(clave, 0) + 1
            n_intra = contador[clave]

            orden_doc[doc_id] = orden_doc.get(doc_id, 0) + 1

            id_externo = construir_id_bloque(doc_id, section_id, n_intra)
            if self.buscar_bloque_por_id_externo(id_externo):
                continue

            n_chars = int(row["text_length_chars"]) if not pd.isna(row.get("text_length_chars")) else None

            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, identificador_externo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "seccion", section_id, orden_doc[doc_id],
                 f"[sección agregada {section_id} — calculada en CSV]",
                 n_chars, id_externo),
            )
            insertados += 1
        if sin_doc:
            log(f"  WARN: {sin_doc} filas sin documento en BD", "warn")
        return insertados

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 5: BLOQUES GRANULARES B — los 2.641 .txt                      ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_bloques_granulares_b_txt(self) -> dict:
        log("Módulo 5: bloques granulares Corpus B (2.641 .txt)", "step")
        carpeta = self.raiz / "data" / "processed" / "corpus_b_sentencias_secciones"
        if not carpeta.exists():
            log(f"  no existe {carpeta}", "warn")
            return {"insertados": 0}
        archivos = sorted(carpeta.glob("*.txt"))
        log(f"  archivos encontrados: {len(archivos)}")

        insertados = 0
        documentos_creados = 0
        secciones_creadas = 0

        for i, archivo in enumerate(archivos, 1):
            stem = archivo.stem
            partes = stem.split("_")
            if len(partes) < 2:
                continue
            orden = parsear_orden(partes[-1])
            doc_id_legible = partes[0]
            seccion = "_".join(partes[1:-1]) if len(partes) > 2 else partes[1]

            # Buscar documento por nombre legible
            doc_db_id = self.buscar_documento_por_doc_id(doc_id_legible)
            if not doc_db_id:
                doc_db_id = self._insertar_documento({
                    "corpus": "B-JEP",
                    "doc_id_externo": doc_id_legible,
                    "titulo": doc_id_legible,
                    "tipo_documento": "auto",
                    "ruta_original": doc_id_legible + ".pdf",
                })
                documentos_creados += 1

            # Buscar sección padre
            id_seccion_padre = f"{doc_id_legible}__{seccion}__padre"
            seccion_db_id = self.buscar_bloque_por_id_externo(id_seccion_padre)
            if not seccion_db_id:
                cur = self.con.execute(
                    "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                    "identificador_externo) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_db_id, "seccion", seccion, 0,
                     f"[contenedor de sección {seccion} — agrupa bloques granulares .txt]",
                     id_seccion_padre),
                )
                seccion_db_id = cur.lastrowid
                secciones_creadas += 1

            id_externo_bloque = stem  # nombre del archivo sin .txt
            if self.buscar_bloque_por_id_externo(id_externo_bloque):
                continue

            try:
                texto = archivo.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                texto = archivo.read_text(encoding="latin-1")

            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, bloque_padre_id, identificador_externo, ruta_archivo) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "bloque_granular", seccion, orden, texto,
                 len(texto), seccion_db_id, id_externo_bloque,
                 str(archivo.relative_to(self.raiz)).replace("\\", "/")),
            )
            insertados += 1

            if i % 500 == 0:
                log(f"  ...procesados {i}/{len(archivos)}")
                self.con.commit()

        self.commit()
        log(f"  granulares B: {insertados} | secciones nuevas: {secciones_creadas} | docs nuevos: {documentos_creados}", "ok")
        return {"insertados": insertados, "secciones_creadas": secciones_creadas,
                "documentos_creados": documentos_creados}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 6: BLOQUES Corpus C                                           ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_bloques_c(self) -> dict:
        log("Módulo 6: bloques Corpus C (588 filas esperadas)", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_c.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas")

        insertados = 0
        sin_audiencia = 0
        for _, row in df.iterrows():
            audio = safe(row.get("audio"))
            bloque_id = safe(row.get("bloque_id"))
            if not audio or not bloque_id:
                continue
            res = self.buscar_audiencia_por_audio_csv(audio)
            if not res:
                sin_audiencia += 1
                continue
            aud_id, subcaso = res
            # Obtener documento_id desde la audiencia (no construirlo desde el audio crudo,
            # que puede ser un alias como 'dabeiba_antioquia')
            doc_row = self.con.execute(
                "SELECT documento_id FROM audiencias WHERE id = ?", (aud_id,)
            ).fetchone()
            if not doc_row:
                continue
            doc_db_id = doc_row["documento_id"]
            if self.buscar_bloque_por_id_externo(bloque_id):
                continue

            n_chars = int(row["chars"]) if not pd.isna(row.get("chars")) else None
            m = re.search(r"_b(\d+)$|_c(\d+)$", bloque_id)
            orden = int(m.group(1) or m.group(2)) if m else 0

            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, identificador_externo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "bloque_granular", "audiencia_reconocimiento", orden,
                 f"[bloque temporal de audiencia {subcaso}]",
                 n_chars, bloque_id),
            )
            insertados += 1
        if sin_audiencia:
            log(f"  WARN: {sin_audiencia} filas con audio sin audiencia resuelta", "warn")
        self.commit()
        log(f"  bloques C: {insertados}", "ok")
        return {"insertados": insertados}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULOS 7-10: INDICADORES                                            ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_indicadores_a(self) -> dict:
        log("Módulo 7: indicadores Corpus A", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_a.csv"
        run_id = self.crear_run("Indicadores Corpus A — pipeline cap.5",
                                {"fuente": str(path.name)})
        modelo = self.modelo_id("Pipeline-Capa1", "cap5-v15")
        n = self._cargar_indicadores_csv(path, modelo, run_id)
        self.commit()
        log(f"  indicadores A: {n}", "ok")
        return {"insertados": n}

    def cargar_indicadores_b_v1(self) -> dict:
        log("Módulo 8a: indicadores Corpus B v1 (cap.5, n=54)", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_b.csv"
        run_id = self.crear_run("Indicadores Corpus B v1 — versión cap.5",
                                {"fuente": str(path.name), "version": "v1"})
        modelo = self.modelo_id("Pipeline-Capa1", "cap5-v15")
        n = self._cargar_indicadores_csv(path, modelo, run_id)
        self.commit()
        log(f"  indicadores B v1: {n}", "ok")
        return {"insertados": n}

    def cargar_indicadores_b_v2(self) -> dict:
        log("Módulo 8b: indicadores Corpus B v2 (n=145)", "step")
        path = self.raiz / "data" / "features" / "indicators_corpus_b_v2.csv"
        run_id = self.crear_run("Indicadores Corpus B v2 — versión actualizada",
                                {"fuente": str(path.name), "version": "v2"})
        modelo = self.modelo_id("Pipeline-Capa1", "cap5-v15")
        n = self._cargar_indicadores_csv(path, modelo, run_id)
        self.commit()
        log(f"  indicadores B v2: {n}", "ok")
        return {"insertados": n}

    def cargar_indicadores_c(self) -> dict:
        log("Módulo 9: indicadores Corpus C Capa 2 (y8/y9 ConfliBERT)", "step")
        run_id = self.crear_run("Indicadores Corpus C Capa 2 — y8/y9 ConfliBERT-Spanish",
                                {"fuente": "indicators_corpus_c.csv"})
        modelo_emb = self.modelo_id("ConfliBERT-Spanish", "beto-cased-v1")
        n = 0

        path_c2 = self.raiz / "data" / "features" / "indicators_corpus_c.csv"
        if path_c2.exists():
            df = pd.read_csv(path_c2)
            for _, row in df.iterrows():
                bloque_id = safe(row.get("bloque_id"))
                bloque_db_id = self.buscar_bloque_por_id_externo(bloque_id) if bloque_id else None
                if not bloque_db_id:
                    continue
                for codigo, col in [("y8_mafapo", "y8_mafapo_cs"), ("y9_cidh", "y9_cidh_cs")]:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    self._insertar_indicador(bloque_db_id, codigo, float(val), modelo_emb, run_id)
                    n += 1

        self.commit()
        log(f"  indicadores C Capa 2: {n}", "ok")
        return {"insertados": n}

    def cargar_bloques_c_capa1(self) -> dict:
        """Bloques granulares Capa 1 del Corpus C (538 con sufijo _c).
        Pipeline distinto al de Capa 2 (588 con sufijo _b). Ambos coexisten."""
        log("Módulo 6b: bloques Corpus C Capa 1 (538 bloques con sufijo _c)", "step")
        path = self.raiz / "data" / "indicators_corpus_c_capa1.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas")

        insertados = 0
        sin_audiencia = 0
        for _, row in df.iterrows():
            audio = safe(row.get("audio"))
            bloque_id = safe(row.get("bloque_id"))
            if not audio or not bloque_id:
                continue
            res = self.buscar_audiencia_por_audio_csv(audio)
            if not res:
                sin_audiencia += 1
                continue
            aud_id, subcaso = res
            doc_row = self.con.execute(
                "SELECT documento_id FROM audiencias WHERE id = ?", (aud_id,)
            ).fetchone()
            if not doc_row:
                continue
            doc_db_id = doc_row["documento_id"]
            if self.buscar_bloque_por_id_externo(bloque_id):
                continue
            n_chars = int(row["chars"]) if not pd.isna(row.get("chars")) else None
            m = re.search(r"_c(\d+)$|_b(\d+)$", bloque_id)
            orden = int(m.group(1) or m.group(2)) if m else 0
            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, identificador_externo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "bloque_granular", "audiencia_reconocimiento_capa1", orden,
                 f"[bloque Capa 1 de audiencia {subcaso}]",
                 n_chars, bloque_id),
            )
            insertados += 1
        if sin_audiencia:
            log(f"  WARN: {sin_audiencia} filas sin audiencia", "warn")
        self.commit()
        log(f"  bloques C Capa 1: {insertados}", "ok")
        return {"insertados": insertados}

    def cargar_indicadores_c_capa1(self) -> dict:
        """Indicadores y2/y4/y10 desde indicators_corpus_c_capa1.csv (538 bloques _c)."""
        log("Módulo 9b: indicadores Corpus C Capa 1 (y2/y4/y10)", "step")
        path = self.raiz / "data" / "indicators_corpus_c_capa1.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        run_id = self.crear_run("Indicadores Corpus C Capa 1 — y2/y4/y10",
                                {"fuente": str(path.name)})
        modelo = self.modelo_id("Pipeline-Capa1", "cap5-v15")
        n = 0
        sin_bloque = 0
        for _, row in df.iterrows():
            bloque_id = safe(row.get("bloque_id"))
            if not bloque_id:
                continue
            bloque_db_id = self.buscar_bloque_por_id_externo(bloque_id)
            if not bloque_db_id:
                sin_bloque += 1
                continue
            for codigo, col in [("y2_sa", "y2_sa"), ("y4_nv", "y4_nv"), ("y10_rep", "y10_rep")]:
                val = row.get(col)
                if pd.isna(val):
                    continue
                self._insertar_indicador(bloque_db_id, codigo, float(val), modelo, run_id)
                n += 1
        if sin_bloque:
            log(f"  WARN: {sin_bloque} bloques no encontrados (correr bloques_c_capa1 primero)", "warn")
        self.commit()
        log(f"  indicadores C Capa 1: {n}", "ok")
        return {"insertados": n}

    def cargar_indicadores_conflibert(self) -> dict:
        """Carga indicadores ConfliBERT-Spanish desde indicators_completo_conflibert.csv.
        Este CSV tiene los y8/y9 con sufijo _cs (cosine ConfliBERT) que reproducen
        las distancias semánticas de la Tabla 5.5, y también y7_surprisal (BETO)
        y y11_conv_rest. Estos valores NO están en indicators_corpus_a/b.csv."""
        log("Módulo 11b: indicadores ConfliBERT (y8/y9 cosine, y7 surprisal, y11 conv_rest)", "step")
        path = self.raiz / "data" / "features" / "indicators_completo_conflibert.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas (esperado: 873 = 819 A + 54 B v1)")

        run_id = self.crear_run(
            "Indicadores ConfliBERT-Spanish — distancias semánticas + surprisal",
            {"fuente": str(path.name)},
        )
        modelo_conflibert = self.modelo_id("ConfliBERT-Spanish", "beto-cased-v1")
        modelo_beto = self.modelo_id("BETO", "cased")
        modelo_pipe = self.modelo_id("Pipeline-Capa1", "cap5-v15")

        n = 0
        contador: dict = {}
        sin_bloque = 0

        for _, row in df.iterrows():
            doc_id = safe(row.get("doc_id"))
            section_id = safe(row.get("section_id"))
            if not doc_id or not section_id:
                continue
            clave = (doc_id, section_id)
            contador[clave] = contador.get(clave, 0) + 1
            n_intra = contador[clave]
            id_externo = construir_id_bloque(doc_id, section_id, n_intra)
            bloque_db_id = self.buscar_bloque_por_id_externo(id_externo)
            if not bloque_db_id:
                sin_bloque += 1
                continue

            # Distancias ConfliBERT (las que reproducen el cap. 5)
            for codigo, col in [("y8_mafapo_cs", "y8_mafapo_cs"),
                                 ("y9_cidh_cs", "y9_cidh_cs"),
                                 ("y8_cs", "y8_cs"),
                                 ("y9_cs", "y9_cs")]:
                val = row.get(col)
                if pd.isna(val):
                    continue
                self._insertar_indicador(bloque_db_id, codigo, float(val),
                                          modelo_conflibert, run_id)
                n += 1

            # Surprisal con BETO
            val = row.get("y7_surprisal")
            if not pd.isna(val):
                self._insertar_indicador(bloque_db_id, "y7_surprisal", float(val),
                                          modelo_beto, run_id)
                n += 1

            # Convergencia restaurativa
            val = row.get("y11_conv_rest")
            if not pd.isna(val):
                self._insertar_indicador(bloque_db_id, "y11_conv_rest", float(val),
                                          modelo_pipe, run_id)
                n += 1

        if sin_bloque:
            log(f"  WARN: {sin_bloque} filas sin bloque encontrado (B v1 puede usar contador distinto a B v2)", "warn")
        self.commit()
        log(f"  indicadores ConfliBERT: {n}", "ok")
        return {"insertados": n}

    def cargar_bloques_beach(self) -> dict:
        """Crea bloques específicos del pipeline Beach (y11/y12/y13).
        Estos bloques tienen identificadores propios tipo '01807c73506b8128_b000'
        que no coinciden con los de los otros pipelines."""
        log("Módulo 10a: bloques del pipeline Beach (y11/y12/y13)", "step")
        path = self.raiz / "data" / "indicators_y11_y12_y13.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        log(f"  CSV: {len(df)} filas")

        n = 0
        sin_doc = 0
        ambiguos = 0

        for _, row in df.iterrows():
            bloque_id = safe(row.get("bloque_id"))
            doc_id_corto = safe(row.get("doc_id"))
            if not bloque_id or not doc_id_corto:
                continue

            if self.buscar_bloque_por_id_externo(bloque_id):
                continue

            # Buscar documento por prefijo
            doc_matches = self.con.execute(
                "SELECT id FROM documentos WHERE doc_id_externo LIKE ?",
                (str(doc_id_corto) + "%",),
            ).fetchall()
            if not doc_matches:
                sin_doc += 1
                continue
            if len(doc_matches) > 1:
                ambiguos += 1
                continue
            doc_db_id = doc_matches[0]["id"]

            m = re.search(r"_b(\d+)$|_c(\d+)$", bloque_id)
            orden = int(m.group(1) or m.group(2)) if m else 0
            n_chars = int(row["chars"]) if not pd.isna(row.get("chars")) else None

            self.con.execute(
                "INSERT INTO bloques (documento_id, granularidad, seccion, orden, texto, "
                "n_chars, identificador_externo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_db_id, "bloque_granular", "bloque_beach", orden,
                 "[bloque del pipeline Beach — y11/y12/y13 adaptado]",
                 n_chars, bloque_id),
            )
            n += 1

        if sin_doc:
            log(f"  WARN: {sin_doc} filas sin documento", "warn")
        if ambiguos:
            log(f"  WARN: {ambiguos} filas con prefijo ambiguo", "warn")
        self.commit()
        log(f"  bloques Beach: {n}", "ok")
        return {"insertados": n}

    def cargar_indicadores_extras(self) -> dict:
        """Carga léxico emocional (capa1_nuevos_*.csv) y Beach (y11/y12/y13).

        Léxico: matchea por prefijo de doc_id (16 chars) + sección + contador intra.
        Beach: requiere que se haya corrido `bloques_beach` antes para que existan
        los bloques con identificadores propios.
        """
        log("Módulo 10: indicadores extras (léxico emocional + Beach)", "step")
        run_id = self.crear_run(
            "Indicadores extras — léxico emocional + Beach",
            {"fuentes": ["capa1_nuevos_corpus_a.csv", "capa1_nuevos_corpus_b.csv",
                          "indicators_y11_y12_y13.csv"]},
        )
        modelo_pipe = self.modelo_id("Pipeline-Capa1", "cap5-v15")
        modelo_beach = self.modelo_id("Pipeline-Beach", "y11-y12-y13")
        n = 0

        # ── Léxico emocional ──
        for fname in ["capa1_nuevos_corpus_a.csv", "capa1_nuevos_corpus_b.csv"]:
            path = self.raiz / "data" / "features" / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            log(f"  léxico emocional desde {fname}: {len(df)} filas")
            sin_match = 0
            cargados_archivo = 0
            contador: dict = {}

            for _, row in df.iterrows():
                doc_id_csv = safe(row.get("doc_id"))
                seccion = safe(row.get("seccion"))
                if not doc_id_csv or not seccion:
                    continue
                # Buscar bloques que matcheen por (prefijo, sección)
                # en orden por id (que respeta orden de inserción)
                clave = (doc_id_csv, seccion)
                contador[clave] = contador.get(clave, 0) + 1
                n_intra = contador[clave]

                matches = self.con.execute(
                    "SELECT id FROM bloques WHERE seccion = ? AND identificador_externo LIKE ? "
                    "ORDER BY id",
                    (seccion, str(doc_id_csv) + "%"),
                ).fetchall()
                if not matches or n_intra > len(matches):
                    sin_match += 1
                    continue
                bloque_db_id = matches[n_intra - 1]["id"]

                for col in ["accountability_score", "hedging_density",
                            "primera_persona_ratio", "sa_ratio",
                            "emo_culpa", "emo_tristeza", "emo_miedo",
                            "emo_ira", "emo_confianza", "emo_anticipacion",
                            "emo_violencia_institucional",
                            "emo_reconocimiento_victimas",
                            "emo_balance_victimas"]:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    self._insertar_indicador(bloque_db_id, col, float(val),
                                              modelo_pipe, run_id)
                    n += 1
                    cargados_archivo += 1
            if sin_match:
                log(f"    {sin_match} filas sin match (sección no existe en BD)", "warn")
            log(f"    indicadores cargados desde {fname}: {cargados_archivo}")

        # ── Beach y11/y12/y13 ──
        path_beach = self.raiz / "data" / "indicators_y11_y12_y13.csv"
        if path_beach.exists():
            df = pd.read_csv(path_beach)
            log(f"  Beach desde {path_beach.name}: {len(df)} filas")
            sin_bloque = 0
            cargados_archivo = 0
            for _, row in df.iterrows():
                bloque_id = safe(row.get("bloque_id"))
                if not bloque_id:
                    continue
                bloque_db_id = self.buscar_bloque_por_id_externo(bloque_id)
                if not bloque_db_id:
                    sin_bloque += 1
                    continue
                for codigo in ["y11_quotes", "y12_judgment", "y13_evidential",
                               "y11_oral_score", "y11_oral_v2", "y11_prop_mafapo",
                               "y11_n_dq", "y12_n_alta", "y12_n_baja",
                               "y13_n_inst", "y13_n_vic"]:
                    val = row.get(codigo)
                    if pd.isna(val):
                        continue
                    self._insertar_indicador(bloque_db_id, codigo, float(val),
                                              modelo_beach, run_id)
                    n += 1
                    cargados_archivo += 1
            if sin_bloque:
                log(f"    {sin_bloque} filas sin bloque Beach (corra `bloques_beach` primero)", "warn")
            log(f"    indicadores Beach cargados: {cargados_archivo}")

        self.commit()
        log(f"  indicadores extras: {n}", "ok")
        return {"insertados": n}

    def _cargar_indicadores_csv(self, path: Path, modelo_id: int, run_id: int) -> int:
        if not path.exists():
            return 0
        df = pd.read_csv(path)
        n = 0
        contador: dict = {}
        codigos_y = [c for c in df.columns if re.match(r"^y\d+_", c)]
        for _, row in df.iterrows():
            doc_id = safe(row.get("doc_id"))
            section_id = safe(row.get("section_id"))
            if not doc_id or not section_id:
                continue
            clave = (doc_id, section_id)
            contador[clave] = contador.get(clave, 0) + 1
            n_intra = contador[clave]
            id_externo = construir_id_bloque(doc_id, section_id, n_intra)
            bloque_db_id = self.buscar_bloque_por_id_externo(id_externo)
            if not bloque_db_id:
                continue
            for col in codigos_y:
                val = row.get(col)
                if pd.isna(val):
                    continue
                self._insertar_indicador(bloque_db_id, col, float(val), modelo_id, run_id)
                n += 1
        return n

    def _insertar_indicador(self, bloque_id: int, codigo: str, valor: float,
                             modelo_id: int, run_id: int) -> None:
        try:
            self.con.execute(
                "INSERT OR IGNORE INTO indicadores (bloque_id, codigo, valor, modelo_id, run_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (bloque_id, codigo, valor, modelo_id, run_id),
            )
        except sqlite3.IntegrityError:
            pass

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 11: SEGMENTOS FACIAL                                          ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_segmentos_facial(self) -> dict:
        log("Módulo 11: segmentos orales — facial (AUs)", "step")
        carpeta = self.raiz / "outputs" / "capa3"
        if not carpeta.exists():
            return {"insertados": 0}
        run_id = self.crear_run("Segmentos faciales — MediaPipe FaceLandmarker",
                                {"fuente": "outputs/capa3/aus_*.csv"})
        n = 0
        for path in sorted(carpeta.glob("aus_*.csv")):
            df = pd.read_csv(path)
            audio_csv = self._inferir_audio_de_archivo(path.name)
            res = self.buscar_audiencia_por_audio_csv(audio_csv) if audio_csv else None
            if not res:
                continue
            aud_id, _ = res
            for _, row in df.iterrows():
                speaker = safe(row.get("speaker"))
                comp_id = self.buscar_compareciente(aud_id, speaker) if speaker else None
                self.con.execute(
                    "INSERT INTO segmentos_orales (audiencia_id, compareciente_id, t_inicio, "
                    "t_fin, duracion, speaker_diarizacion, au1, au4, au6, au12, au15, au17, "
                    "fuente_csv, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (aud_id, comp_id,
                     float(row["start"]) if not pd.isna(row.get("start")) else 0.0,
                     float(row["end"]) if not pd.isna(row.get("end")) else 0.0,
                     float(row["duracion"]) if not pd.isna(row.get("duracion")) else None,
                     speaker,
                     None if pd.isna(row.get("AU1")) else float(row["AU1"]),
                     None if pd.isna(row.get("AU4")) else float(row["AU4"]),
                     None if pd.isna(row.get("AU6")) else float(row["AU6"]),
                     None if pd.isna(row.get("AU12")) else float(row["AU12"]),
                     None if pd.isna(row.get("AU15")) else float(row["AU15"]),
                     None if pd.isna(row.get("AU17")) else float(row["AU17"]),
                     path.name, run_id),
                )
                n += 1
            self.con.commit()
            log(f"  cargado: {path.name} → {len(df)} segmentos")
        log(f"  segmentos faciales: {n}", "ok")
        return {"insertados": n}

    def cargar_segmentos_vocal(self) -> dict:
        log("Módulo 12: segmentos orales — vocal (eGeMAPS comparecientes)", "step")
        carpeta = self.raiz / "outputs" / "capa3"
        if not carpeta.exists():
            return {"insertados": 0}
        run_id = self.crear_run("Segmentos vocales — eGeMAPS comparecientes",
                                {"fuente": "outputs/capa3/egemap_*_compareciente.csv"})
        n = 0
        for path in sorted(carpeta.glob("egemap_*_compareciente.csv")):
            audio_csv = path.stem.replace("egemap_", "").replace("_compareciente", "")
            res = self.buscar_audiencia_por_audio_csv(audio_csv)
            if not res:
                continue
            aud_id, _ = res
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                features_full = {c: (None if pd.isna(row[c]) else float(row[c]))
                                 for c in df.columns}
                self.con.execute(
                    "INSERT INTO segmentos_orales (audiencia_id, t_inicio, t_fin, "
                    "f0_mean, f0_stddev, loudness_mean, jitter, shimmer, hnr, "
                    "egemaps_full_json, fuente_csv, run_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (aud_id,
                     float(row["start_s"]),
                     float(row["end_s"]),
                     float(row["F0semitoneFrom27.5Hz_sma3nz_amean"]),
                     float(row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"]),
                     float(row["loudness_sma3_amean"]),
                     float(row["jitterLocal_sma3nz_amean"]),
                     float(row["shimmerLocaldB_sma3nz_amean"]),
                     float(row["HNRdBACF_sma3nz_amean"]),
                     json.dumps(features_full),
                     path.name, run_id),
                )
                n += 1
            self.con.commit()
            log(f"  cargado: {path.name} → {len(df)} ventanas")
        log(f"  segmentos vocales: {n}", "ok")
        return {"insertados": n}

    def _inferir_audio_de_archivo(self, fname: str) -> Optional[str]:
        if not fname.startswith("aus_"):
            return None
        s = fname[4:].replace(".csv", "")
        s = re.sub(r"_SPEAKER_\d+$", "", s)
        s = re.sub(r"_v\d+$", "", s)
        s = s.replace("_comparecientes", "")
        return s

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  MÓDULO 13: ANOTACIONES                                               ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def cargar_anotaciones_iaa(self) -> dict:
        log("Módulo 13: anotaciones IAA (resumen)", "step")
        path = self.raiz / "data" / "IAA_anotaciones_mireya.csv"
        if not path.exists():
            log(f"  no existe {path}", "warn")
            return {"insertados": 0}
        df = pd.read_csv(path)
        n = 0
        for _, row in df.iterrows():
            self.con.execute(
                "INSERT INTO anotaciones (label, anotador, label_studio_id, inner_id, "
                "n_spans, etiquetas_combinadas, es_resumen) VALUES (?, ?, ?, ?, ?, ?, 1)",
                (safe(row.get("etiqueta_principal")), "mireya",
                 int(row["id"]) if not pd.isna(row.get("id")) else None,
                 int(row["inner_id"]) if not pd.isna(row.get("inner_id")) else None,
                 int(row["n_spans"]) if not pd.isna(row.get("n_spans")) else None,
                 safe(row.get("etiquetas_mireya"))),
            )
            n += 1
        self.commit()
        log(f"  anotaciones: {n}", "ok")
        log(f"  NOTA: detalle de spans en data/annotations/label_studio.sqlite3", "info")
        return {"insertados": n}

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  REPORTE FINAL                                                        ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    def reporte_final(self) -> None:
        log("=" * 70)
        log("REPORTE FINAL", "step")
        log("=" * 70)
        for tabla in ["corpora", "documentos", "audiencias", "comparecientes",
                       "bloques", "indicadores", "anotaciones",
                       "segmentos_orales", "modelos", "runs", "centroides"]:
            n = self.con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
            log(f"  {tabla:25s}  {n:>10,}")

        log("")
        log("Tabla 5.1 (vista):", "step")
        for r in self.con.execute("SELECT * FROM v_tabla_5_1"):
            log(f"  {dict(r)}")

        log("")
        log("Audiencias:", "step")
        for r in self.con.execute("SELECT * FROM v_audiencias_resumen"):
            log(f"  {dict(r)}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ORQUESTACIÓN                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MODULOS = [
    ("documentos",          "cargar_documentos",                    "Documentos: desde CSVs de indicadores + enriquecimiento de inventario"),
    ("audiencias",          "cargar_audiencias_y_comparecientes",   "Audiencias y comparecientes Corpus C"),
    ("bloques_a",           "cargar_bloques_a",                     "Bloques granulares Corpus A (819 filas)"),
    ("secciones_b",         "cargar_secciones_b",                   "Secciones agregadas Corpus B v2 (145 filas)"),
    ("bloques_b_txt",       "cargar_bloques_granulares_b_txt",      "Bloques granulares Corpus B desde 2.641 .txt"),
    ("bloques_c",           "cargar_bloques_c",                     "Bloques Corpus C Capa 2 (588 con sufijo _b)"),
    ("bloques_c_capa1",     "cargar_bloques_c_capa1",               "Bloques Corpus C Capa 1 (538 con sufijo _c)"),
    ("indicadores_a",       "cargar_indicadores_a",                 "Indicadores y₁..y₁₃ Corpus A"),
    ("indicadores_b_v1",    "cargar_indicadores_b_v1",              "Indicadores Corpus B v1 (cap.5)"),
    ("indicadores_b_v2",    "cargar_indicadores_b_v2",              "Indicadores Corpus B v2 (actual)"),
    ("indicadores_c",       "cargar_indicadores_c",                 "Indicadores Corpus C Capa 2 (y8/y9 ConfliBERT)"),
    ("indicadores_c_capa1", "cargar_indicadores_c_capa1",           "Indicadores Corpus C Capa 1 (y2/y4/y10)"),
    ("indicadores_conflibert","cargar_indicadores_conflibert",      "Indicadores ConfliBERT (y8/y9 cs, y7 surprisal, y11 conv_rest)"),
    ("bloques_beach",       "cargar_bloques_beach",                 "Bloques específicos del pipeline Beach (y11/y12/y13)"),
    ("indicadores_extras",  "cargar_indicadores_extras",            "Léxico emocional + Beach (y₁₁/y₁₂/y₁₃)"),
    ("segmentos_facial",    "cargar_segmentos_facial",              "AUs faciales (MediaPipe)"),
    ("segmentos_vocal",     "cargar_segmentos_vocal",               "eGeMAPS vocal comparecientes"),
    ("anotaciones",         "cargar_anotaciones_iaa",               "Anotaciones IAA (resumen)"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta canónica CFH a SQLite (v3).")
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--schema", default=SCHEMA_DEFAULT)
    parser.add_argument("--raiz", default=".")
    parser.add_argument("--solo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--listar", action="store_true")
    args = parser.parse_args()

    if args.listar:
        print("Módulos disponibles:")
        for clave, _, desc in MODULOS:
            print(f"  {clave:22s}  {desc}")
        print(f"  {'todo':22s}  Ejecutar TODOS los módulos en orden")
        return 0

    if not args.solo:
        log("Falta --solo <modulo>. Use --listar para ver opciones.", "error")
        return 1

    raiz = Path(args.raiz).resolve()
    if not raiz.exists():
        log(f"Raíz no existe: {raiz}", "error")
        return 2

    if args.reset_db:
        if Path(args.db).exists():
            log(f"--reset-db: borrando {args.db}", "warn")
            Path(args.db).unlink()
        log(f"Recreando BD desde {args.schema}", "step")
        con = sqlite3.connect(args.db)
        with open(args.schema, encoding="utf-8") as f:
            con.executescript(f.read())
        con.close()
        log("BD recreada", "ok")

    if not Path(args.db).exists():
        log(f"BD no existe: {args.db}. Use --reset-db.", "error")
        return 3

    ing = CFHIngesta(args.db, raiz, dry_run=args.dry_run)
    ing.conectar()

    nombres = [m[0] for m in MODULOS]
    if args.solo == "todo":
        a_ejecutar = MODULOS
    elif args.solo in nombres:
        a_ejecutar = [m for m in MODULOS if m[0] == args.solo]
    else:
        log(f"Módulo desconocido: {args.solo}. Use --listar.", "error")
        ing.cerrar()
        return 4

    log(f"Modo dry-run: {args.dry_run}", "info")
    log(f"BD: {args.db}", "info")
    log(f"Raíz: {raiz}", "info")
    log(f"Módulos a ejecutar: {[m[0] for m in a_ejecutar]}", "info")
    log("")

    for clave, metodo, desc in a_ejecutar:
        try:
            getattr(ing, metodo)()
        except Exception as e:
            log(f"ERROR en módulo '{clave}': {e}", "error")
            import traceback
            traceback.print_exc()

    ing.reporte_final()
    ing.cerrar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
