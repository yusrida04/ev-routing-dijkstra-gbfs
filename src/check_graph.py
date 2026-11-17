import sys
from pathlib import Path

import osmnx as ox
import networkx as nx
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_CONNECTED_PATH,
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH
)


def check_profile(label, graph_path, stations_path):
    graph = ox.load_graphml(graph_path)
    stations = gpd.read_file(stations_path)
    station_nodes = set(
        int(node) for node in stations["nearest_node"]
    )
    graph_nodes = set(graph.nodes)

    components = (
        nx.strongly_connected_components(graph)
        if graph.is_directed()
        else nx.connected_components(graph)
    )
    largest_component = max(components, key=len)

    print(label)
    print(f"  Graf: {graph_path}")
    print(f"  Node: {len(graph.nodes)} | edge: {len(graph.edges)}")
    print(
        "  SPKLU node valid: "
        f"{len(station_nodes & graph_nodes)}/{len(station_nodes)}"
    )
    print(
        "  SPKLU di komponen terhubung terbesar: "
        f"{len(station_nodes & largest_component)}/{len(station_nodes)}"
    )


if __name__ == "__main__":
    check_profile(
        "Medan",
        MEDAN_GRAPH_PATH,
        MEDAN_STATIONS_CONNECTED_PATH
    )
    check_profile(
        "Regional",
        REGIONAL_GRAPH_PATH,
        REGIONAL_STATIONS_CONNECTED_PATH
    )