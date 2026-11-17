import argparse
import heapq
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.gbfs_distance import (
    haversine_distance,
    get_edge_length,
    get_road_distances_to_goal
)

from src.utils.energy import (
    BATTERY_CAPACITY,
    calculate_energy
)
from src.project_paths import (
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_CONNECTED_PATH,
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH
)

# Bobot penalti (satuan: km per kWh). Dengan konsumsi 0,1218 kWh/km,
# 1 kWh setara sekitar 8,2 km, sehingga nilai ~10 menormalkan satuan.
# Lakukan analisis sensitivitas, misalnya lam = 1, 5, 10, 20.
DEFAULT_LAMBDA = 10.0

# Faktor jalan baku untuk memperkirakan kebutuhan energi dari jarak lurus.
DEFAULT_DETOUR_FACTOR = 1.3
DEFAULT_MAX_EXPANSIONS = 100_000


def battery_aware_heuristic(
    graph,
    current_node,
    goal_node,
    usable_battery,
    lam=DEFAULT_LAMBDA,
    detour_factor=DEFAULT_DETOUR_FACTOR,
    remaining_distance_km=None,
    remaining_straight_line_distance_km=None,
    energy_consumption_kwh_per_km=None
):
    """
    h2(n, b) = d(n, g) + lambda * P(n, b)

    d(n, g): jarak jalan terpendek ke tujuan (km), jika tersedia;
            selain itu estimasi jarak garis lurus x faktor jalan
    P(n, b): max(0, E_perlu - b), kekurangan energi (kWh)
    E_perlu: energi untuk jarak tersisa yang diperkirakan
    """
    straight_line_distance = (
        remaining_straight_line_distance_km
        if remaining_straight_line_distance_km is not None
        else haversine_distance(
            graph, current_node, goal_node
        )
    )

    distance = (
        remaining_distance_km
        if remaining_distance_km is not None
        else straight_line_distance * detour_factor
    )

    required_energy = calculate_energy(
        straight_line_distance * detour_factor,
        energy_consumption_kwh_per_km
    )

    penalty = max(0.0, required_energy - usable_battery)

    return distance + lam * penalty


def greedy_best_first_search_battery(
    graph,
    start_node,
    goal_node,
    initial_battery_percentage,
    lam=DEFAULT_LAMBDA,
    detour_factor=DEFAULT_DETOUR_FACTOR,
    reserve_percentage=0,
    energy_consumption_kwh_per_km=None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS
):
    if max_expansions < 1:
        raise ValueError("max_expansions harus lebih besar dari 0.")

    counter = itertools.count()

    initial_battery = (
        BATTERY_CAPACITY
        * initial_battery_percentage / 100
    )

    reserve = BATTERY_CAPACITY * reserve_percentage / 100
    straight_line_distances = {}

    def straight_line_distance_to_goal(node):
        if node not in straight_line_distances:
            straight_line_distances[node] = haversine_distance(
                graph, node, goal_node
            )
        return straight_line_distances[node]

    def failed_result(reason, expanded=0, limit_reached=False):
        return {
            "success": False,
            "route": [],
            "distance_km": 0,
            "energy_used_kwh": 0,
            "remaining_battery_kwh": initial_battery,
            "expanded": expanded,
            "max_expansions_reached": limit_reached,
            "failure_reason": reason
        }

    if start_node not in graph or goal_node not in graph:
        return failed_result("tidak_ada_rute_layak")

    minimum_goal_energy = calculate_energy(
        straight_line_distance_to_goal(start_node),
        energy_consumption_kwh_per_km
    )
    if minimum_goal_energy > initial_battery - reserve:
        return failed_result("tidak_ada_rute_layak")

    distances_to_goal = get_road_distances_to_goal(
        graph,
        goal_node
    )

    if start_node not in distances_to_goal:
        return failed_result("tidak_ada_rute_layak")

    frontier = [
        (
            battery_aware_heuristic(
                graph,
                start_node,
                goal_node,
                usable_battery=initial_battery - reserve,
                lam=lam,
                detour_factor=detour_factor,
                remaining_distance_km=(
                    distances_to_goal[start_node] / 1000
                ),
                remaining_straight_line_distance_km=(
                    straight_line_distance_to_goal(start_node)
                ),
                energy_consumption_kwh_per_km=(
                    energy_consumption_kwh_per_km
                )
            ),
            next(counter),
            start_node,
            initial_battery
        )
    ]

    # Menyimpan sisa baterai terbaik untuk setiap simpul.
    best_battery = {
        start_node: initial_battery
    }

    parent = {
        start_node: None
    }

    expanded_battery = {}
    expanded = 0
    max_expansions_reached = False

    while frontier:
        if expanded >= max_expansions:
            max_expansions_reached = True
            break

        _, _, current, queued_battery = heapq.heappop(frontier)

        current_battery = best_battery[current]
        if queued_battery < current_battery:
            continue

        if expanded_battery.get(current, -1) >= current_battery:
            continue

        expanded_battery[current] = current_battery
        expanded += 1

        if current == goal_node:
            break

        for neighbor in graph.neighbors(current):
            remaining_distance_m = distances_to_goal.get(neighbor)

            if remaining_distance_m is None:
                continue

            distance_m = get_edge_length(
                graph, current, neighbor
            )

            if distance_m <= 0:
                continue

            energy_used = calculate_energy(
                distance_m / 1000,
                energy_consumption_kwh_per_km
            )

            new_battery = current_battery - energy_used

            # Batasan kelayakan (hard constraint): baterai tidak
            # boleh turun di bawah cadangan minimum.
            if new_battery < reserve:
                continue

            minimum_goal_energy = calculate_energy(
                straight_line_distance_to_goal(neighbor),
                energy_consumption_kwh_per_km
            )
            if new_battery - reserve < minimum_goal_energy:
                continue

            # Abaikan jalur jika baterainya tidak lebih baik.
            if new_battery <= best_battery.get(neighbor, -1):
                continue

            best_battery[neighbor] = new_battery
            parent[neighbor] = current

            heuristic = battery_aware_heuristic(
                graph,
                neighbor,
                goal_node,
                usable_battery=new_battery - reserve,
                lam=lam,
                detour_factor=detour_factor,
                remaining_distance_km=remaining_distance_m / 1000,
                remaining_straight_line_distance_km=(
                    straight_line_distance_to_goal(neighbor)
                ),
                energy_consumption_kwh_per_km=(
                    energy_consumption_kwh_per_km
                )
            )

            heapq.heappush(
                frontier,
                (heuristic, next(counter), neighbor, new_battery)
            )

    if goal_node not in expanded_battery:
        reason = (
            "batas_simpul"
            if max_expansions_reached
            else "tidak_ada_rute_layak"
        )
        return failed_result(
            reason,
            expanded=expanded,
            limit_reached=max_expansions_reached
        )

    # Menyusun kembali rute.
    route = []
    current = goal_node

    while current is not None:
        route.append(current)
        current = parent[current]

    route.reverse()

    total_distance = sum(
        get_edge_length(graph, route[i], route[i + 1])
        for i in range(len(route) - 1)
    ) / 1000

    remaining_battery = best_battery[goal_node]

    return {
        "success": True,
        "route": route,
        "distance_km": total_distance,
        "energy_used_kwh": initial_battery - remaining_battery,
        "remaining_battery_kwh": remaining_battery,
        "expanded": expanded,
        "max_expansions_reached": max_expansions_reached,
        "failure_reason": None
    }


if __name__ == "__main__":

    import geopandas as gpd
    import osmnx as ox

    print("PENGUJIAN SKENARIO GBFS BATTERY-AWARE")

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

    start_node = int(stations.iloc[0]["nearest_node"])
    goal_node = int(stations.iloc[1]["nearest_node"])

    for battery_percentage in (80, 50, 20):
        print(f"\n{'=' * 40}")
        print(f"BATERAI AWAL: {battery_percentage}%")
        print(f"{'=' * 40}")

        result = greedy_best_first_search_battery(
            graph,
            start_node,
            goal_node,
            initial_battery_percentage=battery_percentage
        )

        if result["success"]:
            print("Status: BERHASIL")
            print(f"Jarak: {result['distance_km']:.2f} km")
            print(
                f"Energi terpakai: "
                f"{result['energy_used_kwh']:.2f} kWh"
            )
            print(
                f"Sisa baterai: "
                f"{result['remaining_battery_kwh']:.2f} kWh"
            )
            print(f"Simpul dikembangkan: {result['expanded']}")
        else:
            print("Status: RUTE TIDAK DITEMUKAN")
            print(f"Simpul dikembangkan: {result['expanded']}")
