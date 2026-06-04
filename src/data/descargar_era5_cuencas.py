"""
HidroCaribe·IA — Descarga ERA5-Land para cuencas sin cobertura IDEAM
Ariguaní (SZH 2804) y Bajo Magdalena (SZH 2904)
Mayo/Junio 2026 · IP: Mireya Camacho Celis

Variables descargadas:
  - total_precipitation: precipitación diaria (m → mm)
  - 2m_temperature: temperatura del aire (K → °C)
  - volumetric_soil_water_layer_1: humedad suelo capa 1 (m³/m³)
  - potential_evaporation: ETP (m → mm)

Resolución: 0.1° (~9 km) · Período: 2000-01-01 → 2024-12-31
Referencia: Muñoz-Sabater et al. (2021) doi:10.24381/cds.68d2bb30
"""

import cdsapi
import numpy as np
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("C:/PROYECTOS 2026/AMERICANA/PROYECTO/hidrocaribe-ia/data/raw/era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bounding boxes de las cuencas (lat_min, lat_max, lon_min, lon_max)
# Área extendida 0.5° para capturar contexto de cuenca
CUENCAS_ERA5 = {
    "Ariguani": {
        "bbox": [9.0, 11.5, -74.5, -72.5],  # SZH 2804 Río Ariguaní
        "descripcion": "Río Ariguaní (SZH 2804) · Cesar colombiano"
    },
    "BajoMagdalena": {
        "bbox": [8.5, 11.5, -75.5, -73.5],  # SZH 2904 Bajo Magdalena-Calamar
        "descripcion": "Bajo Magdalena-Calamar (SZH 2904)"
    },
}

VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "potential_evaporation",
]

# Años a descargar (por bloques de 5 años para no saturar la API)
PERIODOS = [
    (2000, 2004), (2005, 2009), (2010, 2014),
    (2015, 2019), (2020, 2024),
]

def descargar_era5_cuenca(nombre, bbox, variables, periodo, client):
    """Descarga ERA5-Land para una cuenca y período."""
    anio_ini, anio_fin = periodo
    lat_max, lat_min, lon_min, lon_max = bbox[1], bbox[0], bbox[2], bbox[3]

    # Nombre del archivo de salida
    out_file = OUTPUT_DIR / f"era5_{nombre}_{anio_ini}_{anio_fin}.nc"

    if out_file.exists():
        print(f"    Ya existe: {out_file.name} — saltando")
        return str(out_file)

    print(f"    Descargando {nombre} {anio_ini}-{anio_fin}...")

    years  = [str(y) for y in range(anio_ini, anio_fin+1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days   = [f"{d:02d}" for d in range(1, 32)]

    try:
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable"    : variables,
                "year"        : years,
                "month"       : months,
                "day"         : days,
                "time"        : ["00:00", "06:00", "12:00", "18:00"],
                "area"        : [lat_max, lon_min, lat_min, lon_max],
                "format"      : "netcdf",
                "download_format": "unarchived",
            },
            str(out_file)
        )
        print(f"    OK → {out_file.name}")
        return str(out_file)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def main():
    print("="*65)
    print("HidroCaribe·IA — Descarga ERA5-Land cuencas NaN")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*65)

    client = cdsapi.Client()

    for nombre, info in CUENCAS_ERA5.items():
        print(f"\nCuenca: {nombre}")
        print(f"  {info['descripcion']}")
        print(f"  BBox: {info['bbox']}")

        for periodo in PERIODOS:
            descargar_era5_cuenca(
                nombre, info['bbox'], VARIABLES, periodo, client)

    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Archivos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
