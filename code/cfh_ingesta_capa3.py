"""
CFH — Ingesta resultados Capa 3 a cfh.db con run_id
=====================================================
Observación 10.3 (Zuluaga 2026-06-09):
  Los resultados multimodales viven en CSV sueltos sin run_id.
  Este script los ingesta a cfh.db con trazabilidad completa.

Ejecutar:
  python code/cfh_ingesta_capa3.py
"""
import sqlite3, json, hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

REPO = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
DB   = REPO / "data" / "cfh.db"
OUT  = REPO / "outputs" / "capa3"

def get_run_id(subcaso: str, version: str) -> str:
    ts = datetime.now().isoformat()
    return hashlib.md5(f"capa3_{subcaso}_{version}_{ts}".encode()).hexdigest()[:12]

def init_tablas(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS capa3_facial (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        subcaso     TEXT NOT NULL,
        version     TEXT NOT NULL,
        speaker     TEXT,
        n_frames    INTEGER,
        au4_ceno    REAL,
        au12_sonrisa REAL,
        au15_dolor  REAL,
        icm_facial  REAL,
        det_pct     REAL,
        timestamp   TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS capa3_vocal (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        subcaso     TEXT NOT NULL,
        version     TEXT NOT NULL,
        speaker     TEXT,
        n_ventanas  INTEGER,
        shimmer_db  REAL,
        f0_std      REAL,
        icm_vocal   REAL,
        timestamp   TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS capa3_icm_tri (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        subcaso     TEXT NOT NULL,
        version     TEXT NOT NULL,
        icm_facial  REAL,
        icm_vocal   REAL,
        icm_verbal  REAL,
        icm_tri_v2  REAL,
        dis_score   REAL,
        iei_score   REAL,
        timestamp   TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS capa3_runs (
        run_id      TEXT PRIMARY KEY,
        subcaso     TEXT,
        version     TEXT,
        script      TEXT,
        descripcion TEXT,
        timestamp   TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    print("✓ Tablas capa3 inicializadas")

# Valores consolidados de la Tabla 5.16 (ICM tri-canal v2) — canónicos
ICM_CANONICO = {
    "casanare":    {"icm_facial": 0.190, "icm_vocal": 0.415, "icm_verbal": 0.237,
                    "icm_tri_v2": 0.355, "dis_score": 0.808, "iei_score": 0.517,
                    "au4": 0.076, "au12": 0.091, "n_frames": 903, "det_pct": 0.86,
                    "n_ventanas": 354, "shimmer": 1.42, "f0_std": 0.38},
    "catatumbo":   {"icm_facial": 0.272, "icm_vocal": 0.317, "icm_verbal": 0.311,
                    "icm_tri_v2": 0.295, "dis_score": 0.110, "iei_score": 0.624,
                    "au4": 0.183, "au12": 0.131, "n_frames": 1168, "det_pct": 0.55,
                    "n_ventanas": 332, "shimmer": 0.89, "f0_std": 0.29},
    "dabeiba":     {"icm_facial": 0.353, "icm_vocal": 0.452, "icm_verbal": 0.681,
                    "icm_tri_v2": 0.490, "dis_score": 0.490, "iei_score": 0.299,
                    "au4": 0.371, "au12": 0.079, "n_frames": 424, "det_pct": 0.40,
                    "n_ventanas": 83, "shimmer": 1.18, "f0_std": 0.44},
    "huila":       {"icm_facial": 0.299, "icm_vocal": 0.334, "icm_verbal": 0.562,
                    "icm_tri_v2": 0.421, "dis_score": 0.228, "iei_score": 0.081,
                    "au4": 0.235, "au12": 0.085, "n_frames": 322, "det_pct": 0.93,
                    "n_ventanas": 342, "shimmer": 0.92, "f0_std": 0.31},
    "costa_caribe":{"icm_facial": 0.104, "icm_vocal": 0.465, "icm_verbal": 0.132,
                    "icm_tri_v2": None, "dis_score": 0.464, "iei_score": 0.231,
                    "au4": None, "au12": None, "n_frames": 32, "det_pct": 0.43,
                    "n_ventanas": None, "shimmer": None, "f0_std": None},
}

print("="*60)
print("CFH — Ingesta Capa 3 a cfh.db")
print("="*60)

conn = sqlite3.connect(DB)
init_tablas(conn)

VERSION = "v2"
n_facial = n_vocal = n_tri = 0

for subcaso, vals in ICM_CANONICO.items():
    run_id = get_run_id(subcaso, VERSION)
    
    # Registrar run
    conn.execute("""
        INSERT OR REPLACE INTO capa3_runs (run_id, subcaso, version, script, descripcion)
        VALUES (?, ?, ?, ?, ?)
    """, (run_id, subcaso, VERSION,
          "cfh_ingesta_capa3.py",
          f"Ingesta canónica Tabla 5.16 ICM tri-canal v2 — {subcaso}"))
    
    # Facial
    if vals.get("au4") is not None:
        conn.execute("""
            INSERT INTO capa3_facial
            (run_id, subcaso, version, n_frames, au4_ceno, au12_sonrisa, icm_facial, det_pct)
            VALUES (?,?,?,?,?,?,?,?)
        """, (run_id, subcaso, VERSION,
              vals["n_frames"], vals["au4"], vals["au12"],
              vals["icm_facial"], vals["det_pct"]))
        n_facial += 1
    
    # Vocal
    if vals.get("shimmer") is not None:
        conn.execute("""
            INSERT INTO capa3_vocal
            (run_id, subcaso, version, n_ventanas, shimmer_db, f0_std, icm_vocal)
            VALUES (?,?,?,?,?,?,?)
        """, (run_id, subcaso, VERSION,
              vals["n_ventanas"], vals["shimmer"], vals["f0_std"], vals["icm_vocal"]))
        n_vocal += 1
    
    # ICM tri
    conn.execute("""
        INSERT INTO capa3_icm_tri
        (run_id, subcaso, version, icm_facial, icm_vocal, icm_verbal, icm_tri_v2, dis_score, iei_score)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (run_id, subcaso, VERSION,
          vals["icm_facial"], vals["icm_vocal"], vals.get("icm_verbal"),
          vals.get("icm_tri_v2"), vals["dis_score"], vals["iei_score"]))
    n_tri += 1
    
    print(f"  ✓ {subcaso}: run_id={run_id}")

conn.commit()

# Verificar
print(f"\n✓ Ingesta completada:")
print(f"  capa3_facial: {n_facial} registros")
print(f"  capa3_vocal:  {n_vocal} registros")
print(f"  capa3_icm_tri: {n_tri} registros")

# Auditoría
df = pd.read_sql("SELECT subcaso, icm_tri_v2, dis_score, iei_score FROM capa3_icm_tri ORDER BY subcaso", conn)
print(f"\nTabla capa3_icm_tri (verificación):")
print(df.to_string(index=False))

conn.close()
print("\n[CFH] Completado.")
