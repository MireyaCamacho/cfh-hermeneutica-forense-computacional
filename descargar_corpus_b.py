"""
CFH · Descarga automatizada del Corpus B — versión 2
Autos + Resoluciones de Conclusiones JEP Macrocaso 003
"""
import requests
import time
from pathlib import Path

DESTINO = Path("data/raw/corpus_b")
DESTINO.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/pdf,*/*",
}

DOCUMENTOS = [
    # ── Resoluciones de Conclusiones (HTML — texto completo) ──────────────
    {
        "id": "rc_01_2022_norte_santander",
        "nombre": "RC_01_2022_Norte_Santander.html",
        "url": "https://jurinfo.jep.gov.co/normograma/compilacion/docs/resoluci%C3%B3n_srvr-01_20-octubre-2022.htm",
        "tipo": "html",
        "descripcion": "RC No. 01/2022 — Norte de Santander (11 imputados, 120 víctimas)"
    },
    # ── Auto 055/2022 — Subcaso Casanare ──────────────────────────────────
    {
        "id": "auto_055_2022_casanare",
        "nombre": "Auto_055_2022_Subcaso_Casanare.pdf",
        "url": "https://www.jep.gov.co/Notificaciones/Estado%20No.%20686.2022%20SRVR%20CASO%2003%20AUTO%20055%20DE%202022.pdf",
        "tipo": "pdf",
        "descripcion": "Auto 055/2022 — Subcaso Casanare (25 imputados)"
    },
    # ── Auto SUB D-081/2023 — Subcaso Huila ───────────────────────────────
    {
        "id": "auto_subd_081_2023_huila",
        "nombre": "Auto_SUBD_081_2023_Subcaso_Huila.pdf",
        "url": "https://relatoria.jep.gov.co/documentos/providencias/1/1/Auto_SRVR-SUBD-HUILA-081_20-noviembre-2023.pdf",
        "tipo": "pdf",
        "descripcion": "Auto SUB D-081/2023 — Subcaso Huila"
    },
    # ── Auto 033/2021 ya descargado — se omite ────────────────────────────
]

def descargar(doc: dict) -> bool:
    destino = DESTINO / doc["nombre"]
    if destino.exists():
        print(f"  ⏭ Ya existe: {doc['nombre']}")
        return True

    try:
        print(f"  ↓ {doc['descripcion']}...")
        resp = requests.get(doc["url"], headers=HEADERS, timeout=90, stream=True)
        if resp.status_code == 200:
            destino.write_bytes(resp.content)
            kb = len(resp.content) / 1024
            print(f"  ✓ {doc['nombre']} ({kb:.0f} KB)")
            return True
        else:
            print(f"  ✗ HTTP {resp.status_code}: {doc['nombre']}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

print("=" * 60)
print("CFH · Corpus B v2 — Resoluciones de Conclusiones + Autos")
print("=" * 60)

ok = sum(descargar(d) for d in DOCUMENTOS if not time.sleep(2))
print(f"\nDescargados: {ok}/{len(DOCUMENTOS)}")