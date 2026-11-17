
import sys
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_STATIONS_DIRECTORY_PATH,
    MEDAN_STATIONS_CLASSIFIED_PATH
)


def determine_access(row):
    text = (
        str(row.get("name", "")) + " "
        + str(row.get("address", ""))
    ).lower()

    restricted_keywords = [
        "khusus kendaraan",
        "khusus toyota",
        "khusus lexus",
        "khusus mercedes-benz",
        "khusus hyundai"
    ]

    if any(keyword in text for keyword in restricted_keywords):
        return "RESTRICTED"

    return "UNKNOWN"


def classify_stations(
    input_path=MEDAN_STATIONS_DIRECTORY_PATH,
    output_path=MEDAN_STATIONS_CLASSIFIED_PATH
):
    stations = gpd.read_file(input_path)

    stations["access_status"] = stations.apply(
        determine_access,
        axis=1
    )

    stations["access_note"] = stations[
        "access_status"
    ].map({
        "RESTRICTED": "Terdapat keterangan akses khusus pada sumber data",
        "UNKNOWN": "Akses publik belum terverifikasi"
    })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stations.to_file(
        output_path,
        driver="GeoJSON"
    )

    print("\nKLASIFIKASI SPKLU BERHASIL!")
    print("\nJumlah berdasarkan status akses:")
    print(stations["access_status"].value_counts())

    print("\nDaftar lokasi:")
    print(
        stations[
            ["name", "access_status", "access_note"]
        ].to_string(index=False)
    )

    print(f"\nFile tersimpan: {output_path}")


if __name__ == "__main__":
    classify_stations()
