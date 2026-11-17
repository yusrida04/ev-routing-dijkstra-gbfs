
import sys
from pathlib import Path

import osmnx as ox
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_CLASSIFIED_PATH,
    MEDAN_STATIONS_CONNECTED_PATH
)


def connect_stations_to_roads(
    graph_path=MEDAN_GRAPH_PATH,
    stations_path=MEDAN_STATIONS_CLASSIFIED_PATH,
    output_path=MEDAN_STATIONS_CONNECTED_PATH
):
    print("Menghubungkan SPKLU ke jaringan jalan...")

    # Memuat jaringan jalan dan data SPKLU
    graph = ox.load_graphml(graph_path)
    stations = gpd.read_file(stations_path)

    # Menggunakan sistem koordinat meter untuk menghitung jarak
    graph_projected = ox.project_graph(
        graph,
        to_crs="EPSG:32647"
    )

    stations_projected = stations.to_crs("EPSG:32647")

    # Mencari simpul jalan terdekat untuk setiap SPKLU
    nearest_nodes = ox.distance.nearest_nodes(
        graph_projected,
        X=stations_projected.geometry.x,
        Y=stations_projected.geometry.y
    )

    snap_distances = []

    for point, node_id in zip(
        stations_projected.geometry,
        nearest_nodes
    ):
        node = graph_projected.nodes[node_id]

        distance = point.distance(
            gpd.points_from_xy(
                [node["x"]],
                [node["y"]]
            )[0]
        )

        snap_distances.append(distance)

    # Menyimpan ID simpul jalan dan jarak ke jalan
    stations["nearest_node"] = nearest_nodes
    stations["snap_distance_m"] = snap_distances

    # Menyimpan hasil dalam koordinat geografis
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stations.to_file(
        output_path,
        driver="GeoJSON"
    )

    print("\nPENGHUBUNGAN SPKLU BERHASIL!")
    print(f"Jumlah SPKLU: {len(stations)}")
    print(
        f"Jarak terdekat: {min(snap_distances):.2f} meter"
    )
    print(
        f"Jarak terjauh: {max(snap_distances):.2f} meter"
    )

    print("\nHASIL PENGHUBUNGAN:")
    print(
        stations[
            ["name", "nearest_node", "snap_distance_m"]
        ].to_string(index=False)
    )

    print(f"\nFile tersimpan: {output_path}")


if __name__ == "__main__":
    connect_stations_to_roads()
