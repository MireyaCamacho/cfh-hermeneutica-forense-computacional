"""
CFH · Descarga automatizada del Corpus B
Autos JEP Macrocaso 003 — Falsos Positivos
"""
import requests
import time
from pathlib import Path

DESTINO = Path("data/raw/corpus_b")
DESTINO.mkdir(parents=True, exist_ok=True)

# URLs directas verificadas de los autos del Macrocaso 003
AUTOS = [
    {
        "id": "auto_005_2018",
        "nombre": "Auto_005_2018_Apertura_Caso003.pdf",
        "url": "https://www.jep.gov.co/Sala-de-Prensa/Documents/Auto%20005%20-%20Apertura%20Caso%20003%20Muertes%20ileg%C3%ADtimamente%20presentadas%20como%20baja%20en%20combate%20SRVR%20(1).pdf",
        "descripcion": "Auto 005/2018 — Apertura Caso 003"
    },
    {
        "id": "auto_033_2021",
        "nombre": "Auto_033_2021_Priorizacion_Subcasos.pdf",
        "url": "http://www.lapluma.net/wp-content/uploads/2021/02/Auto_SRVR-033_12-febrero-2021.pdf",
        "descripcion": "Auto 033/2021 — Priorización 6 subcasos y 6.402 víctimas"
    },
    {
        "id": "auto_subd_062_2023",
        "nombre": "Auto_SUBD_062_2023_Antioquia_Montoya.pdf",
        "url": "https://www.jep.gov.co/Notificaciones/Estado%20No.1014.2023%20SRVR%20Caso%2003%20Auto%20SUB%20D-SUBCASO%20ANTIOQUIA-062%20de%202023.pdf",
        "descripcion": "Auto SUB D-062/2023 — Subcaso Antioquia (General Montoya)"
    },
    {
        "id": "auto_srvr_julio_2023",
        "nombre": "Auto_SRVR_ADHC_03_julio_2023.pdf",
        "url": "https://relatoria.jep.gov.co/documentos/providencias/1/1/Auto_SRVR-ADHC-03_05-julio-2023.pdf",
        "descripcion": "Auto SRVR-ADHC-03 julio 2023"
    },
    {
        "id": "auto_subd_081_2023",
        "nombre": "Auto_SUBD_081_2023_Subcaso_Huila.pdf",
        "url": "https://www.jep.gov.co/Notificaciones/Estado%20No.1229.2023%20SRVR%20CASO%2003%20A.%20OPV%20437%20de%202023.pdf",
        "descripcion": "Auto SUB D-081/2023 — Subcaso Huila"
    },
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/pdf,*/*",
}

def descargar(auto: dict) -> bool:
    destino = DESTINO / auto["nombre"]
    if destino.exists():
        print(f"  ⏭ Ya existe: {auto['nombre']}")
        return True

    try:
        print(f"  ↓ Descargando: {auto['descripcion']}...")
        resp = requests.get(
            auto["url"],
            headers=HEADERS,
            timeout=60,
            stream=True
        )
        if resp.status_code == 200:
            destino.write_bytes(resp.content)
            kb = len(resp.content) / 1024
            print(f"  ✓ {auto['nombre']} ({kb:.0f} KB)")
            return True
        else:
            print(f"  ✗ HTTP {resp.status_code}: {auto['nombre']}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {auto['nombre']}: {e}")
        return False

print("=" * 60)
print("CFH · Descarga Corpus B — Autos JEP Macrocaso 003")
print("=" * 60)

ok = 0
for auto in AUTOS:
    if descargar(auto):
        ok += 1
    time.sleep(2)  # pausa entre descargas

print()
print(f"Descargados: {ok}/{len(AUTOS)}")
print(f"Guardados en: {DESTINO}")
print()
print("Siguiente paso: convertir PDFs a TXT")
print("  python convertir_corpus_b.py")