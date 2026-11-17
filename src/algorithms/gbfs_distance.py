
import argparse
import heapq
import itertools
import math
import sys
from pathlib import Path
from functools import lru_cache

import networkx as nx
import osmnx as ox
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_CONNECTED_PATH,
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH
)


def haversine_distance(graph, node_a, node_b):
    """
    Menghitung jarak garis lurus antara dua simpul dalam km.
    """
    a = graph.nodes[node_a]
    b = graph.nodes[node_b]

    lat1 = math.radians(a["y"])
    lon1 = math.radians(a["x"])
    lat2 = math.radians(b["y"])
    lon2 = math.radians(b["x"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * math.asin(math.sqrt(value))


def get_edge_length(graph, node_a, node_b):
    """
    Mengambil panjang ruas jalan terpendek
    jika terdapat beberapa edge antara dua simpul.
    """
    edges = graph.get_edge_data(node_a, node_b)

    return min(
        float(data.get("length", 0))
        for data in edges.values()
    )


@lru_cache(maxsize=16)
def get_road_distances_to_goal(graph, goal_node):
    """
    Menghitung jarak jalan terpendek dari setiap simpul menuju tujuan.
    Graf dibalik agar satu pencarian dari tujuan mencakup semua simpul.
    """
    road_graph = get_simple_road_graph(graph)
    reverse_graph = (
        road_graph.reverse(copy=False)
        if road_graph.is_directed()
        else road_graph
    )

    return nx.single_source_dijkstra_path_length(
        reverse_graph,
        goal_node,
        weight="length"
    )


@lru_cache(maxsize=1)
def get_simple_road_graph(graph):
    if not graph.is_multigraph():
        return graph

    road_graph = (
        nx.DiGraph() if graph.is_directed() else nx.Graph()
    )
    road_graph.add_nodes_from(graph.nodes(data=True))

    for source, target, edge_data in graph.edges(data=True):
        length = float(edge_data.get("length", 1))
        if (
            not road_graph.has_edge(source, target)
            or length < road_graph[source][target]["length"]
        ):
            road_graph.add_edge(source, target, length=length)

    return road_graph


def clear_road_distance_cache():
    get_road_distances_to_goal.cache_clear()
    get_simple_road_graph.cache_clear()


def greedy_best_first_search(graph, start_node, goal_node):
    """
    GBFS menggunakan heuristik jarak jalan terpendek menuju tujuan.
    """

    distances_to_goal = get_road_distances_to_goal(
        graph,
        goal_node
    )

    if start_node not in distances_to_goal:
        return {
            "success": False,
            "route": [],
            "distance_km": 0,
            "expanded": 0
        }

    counter = itertools.count()

    frontier = [
        (
            distances_to_goal[start_node],
            next(counter),
            start_node
        )
    ]

    parent = {start_node: None}
    visited = set()
    expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal_node:
            break

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue

            heuristic = distances_to_goal.get(neighbor)

            if heuristic is None:
                continue

            if neighbor not in parent:
                parent[neighbor] = current

            heapq.heappush(
                frontier,
                (heuristic, next(counter), neighbor)
            )

    if goal_node not in visited:
        return {
            "success": False,
            "route": [],
            "distance_km": 0,
            "expanded": expanded
        }

    # Menyusun kembali jalur dari tujuan ke awal
    route = []
    current = goal_node

    while current is not None:
        route.append(current)
        current = parent[current]

    route.reverse()

    # Menghitung total panjang ruas jalan
    total_distance = sum(
        get_edge_length(graph, route[i], route[i + 1])
        for i in range(len(route) - 1)
    )

    return {
        "success": True,
        "route": route,
        "distance_km": total_distance / 1000,
        "expanded": expanded
    }


if __name__ == "__main__":
    print("PENGUJIAN GBFS DISTANCE-BASED")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("medan", "regional"),
        default="medan"
    )
    profile = parser.parse_args().profile
    graph_path = (
        REGIONAL_GRAPH_PATH if profile == "regional"
        else MEDAN_GRAPH_PATH
    )
    stations_path = (
        REGIONAL_STATIONS_CONNECTED_PATH if profile == "regional"
        else MEDAN_STATIONS_CONNECTED_PATH
    )

    graph = ox.load_graphml(graph_path)
    stations = gpd.read_file(stations_path)

    # Dua SPKLU pertama digunakan sebagai contoh pengujian.
    start_node = int(stations.iloc[0]["nearest_node"])
    goal_node = int(stations.iloc[1]["nearest_node"])

    result = greedy_best_first_search(
        graph,
        start_node,
        goal_node
    )

    if result["success"]:
        print("\nPENCARIAN RUTE BERHASIL!")
        print(f"Simpul awal: {start_node}")
        print(f"Simpul tujuan: {goal_node}")
        print(f"Jumlah simpul dalam rute: {len(result['route'])}")
        print(f"Total jarak: {result['distance_km']:.2f} km")
        print(f"Simpul dikembangkan: {result['expanded']}")
    else:
        print("\nRUTE TIDAK DITEMUKAN.")
        print(f"Simpul dikembangkan: {result['expanded']}")
