import json
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classify_stations import classify_stations
from src.project_paths import (
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_DIRECTORY_PATH,
    REGIONAL_STATIONS_CLASSIFIED_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH,
    SPKLU_SOURCE_PATH
)
from src.utils.graph import connect_stations_to_roads


def create_regional_directory():
    if not REGIONAL_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{REGIONAL_GRAPH_PATH} tidak ditemukan. "
            "Jalankan src/build_regional_graph.py terlebih dahulu."
        )

    graph = ox.load_graphml(REGIONAL_GRAPH_PATH)
    graph_bounds = (
        min(float(data["x"]) for _, data in graph.nodes(data=True)),
        min(float(data["y"]) for _, data in graph.nodes(data=True)),
        max(float(data["x"]) for _, data in graph.nodes(data=True)),
        max(float(data["y"]) for _, data in graph.nodes(data=True))
    )

    with SPKLU_SOURCE_PATH.open(encoding="utf-8") as source_file:
        dataset = json.load(source_file)

    rows = []

    for item in dataset.get("data", []):
        if len(item) < 8:
            continue

        try:
            latitude = float(item[3])
            longitude = float(item[4])
        except (TypeError, ValueError):
            continue

        west, south, east, north = graph_bounds
        if not (
            west <= longitude <= east
            and south <= latitude <= north
        ):
            continue

        try:
            power_kw = float(item[5]) if item[5] is not None else None
        except (TypeError, ValueError):
            power_kw = None

        rows.append({
            "name": item[0],
            "address": item[1],
            "city": item[2],
            "latitude": latitude,
            "longitude": longitude,
            "power_kw": power_kw,
            "operator": item[7],
            "source": "spklu.web.id"
        })

    if not rows:
        raise ValueError(
            "Tidak ada data SPKLU di dalam cakupan graf regional."
        )

    stations = gpd.GeoDataFrame(
        rows,
        geometry=[
            Point(row["longitude"], row["latitude"])
            for row in rows
        ],
        crs="EPSG:4326"
    )

    REGIONAL_STATIONS_DIRECTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    stations.to_file(
        REGIONAL_STATIONS_DIRECTORY_PATH,
        driver="GeoJSON"
    )

    print(
        f"Data SPKLU regional tersimpan: "
        f"{REGIONAL_STATIONS_DIRECTORY_PATH} ({len(stations)} lokasi)"
    )

    return stations


def build_regional_stations():
    create_regional_directory()
    classify_stations(
        input_path=REGIONAL_STATIONS_DIRECTORY_PATH,
        output_path=REGIONAL_STATIONS_CLASSIFIED_PATH
    )
    connect_stations_to_roads(
        graph_path=REGIONAL_GRAPH_PATH,
        stations_path=REGIONAL_STATIONS_CLASSIFIED_PATH,
        output_path=REGIONAL_STATIONS_CONNECTED_PATH
    )

    stations = gpd.read_file(REGIONAL_STATIONS_CONNECTED_PATH)
    print(
        f"\nSelesai: {len(stations)} SPKLU dipetakan ke graf regional."
    )
    print(
        "Jarak snap maksimum: "
        f"{stations['snap_distance_m'].max():.1f} meter"
    )

    return stations


if __name__ == "__main__":
    build_regional_stations()
