import heapq
import itertools
import time

from src.utils.energy import (
    BATTERY_CAPACITY,
    calculate_energy
)


def dijkstra_search(graph, start, goal, battery_kwh=None):
    """
    Rute terpendek berdasarkan panjang jalan (Dijkstra).
    Format output dibuat sama dengan greedy_best_first_search
    sehingga bisa dipakai sebagai search_fn di simulate_ev_trip.
    Parameter battery_kwh diabaikan (Dijkstra tidak sadar baterai).
    """
    if start not in graph or goal not in graph:
        return {
            "success": False,
            "route": [start],
            "distance_km": 0,
            "expanded": 0
        }

    distances = {start: 0.0}
    parents = {start: None}
    counter = itertools.count()
    frontier = [(0.0, next(counter), start)]
    expanded = 0

    while frontier:
        distance_m, _, node = heapq.heappop(frontier)

        if distance_m != distances.get(node):
            continue

        expanded += 1
        if node == goal:
            break

        for neighbor, edge_data in graph[node].items():
            if graph.is_multigraph():
                edge_length = min(
                    float(data.get("length", 0))
                    for data in edge_data.values()
                )
            else:
                edge_length = float(edge_data.get("length", 0))

            next_distance = distance_m + edge_length
            if next_distance >= distances.get(neighbor, float("inf")):
                continue

            distances[neighbor] = next_distance
            parents[neighbor] = node
            heapq.heappush(
                frontier,
                (next_distance, next(counter), neighbor)
            )

    if goal not in distances:
        return {
            "success": False,
            "route": [start],
            "distance_km": 0,
            "expanded": expanded
        }

    route = []
    node = goal
    while node is not None:
        route.append(node)
        node = parents[node]
    route.reverse()

    return {
        "success": True,
        "route": route,
        "distance_km": distances[goal] / 1000,
        "expanded": expanded
    }


def simulate_baseline(
    graph,
    start_node,
    goal_node,
    initial_battery_percentage,
    start_snap_distance_m: float = 0.0,
    goal_snap_distance_m: float = 0.0,
    reserve_percentage=0,
    energy_consumption_kwh_per_km=None
):
    """
    Baseline: rute terpendek tanpa pengisian daya.
    Dipakai sebagai acuan panjang rute minimum, plus pengecekan
    apakah baterai awal cukup untuk rute tersebut.
    """
    start_node = int(start_node)
    goal_node = int(goal_node)
    battery = BATTERY_CAPACITY * initial_battery_percentage / 100
    reserve = BATTERY_CAPACITY * reserve_percentage / 100

    started = time.perf_counter()
    result = dijkstra_search(
        graph, start_node, goal_node
    )

    elapsed = time.perf_counter() - started

    if not result["success"]:
        return {
            "algorithm": "Dijkstra (tanpa pengisian)",
            "success": False,
            "route": [int(start_node)],
            "distance_km": 0,
            "energy_used_kwh": 0,
            "charged_kwh": 0,
            "remaining_battery_kwh": battery,
            "charging_stops": [],
            "expanded": result["expanded"],
            "computation_time_s": elapsed,
            "failure_reason": "tidak_ada_rute_layak"
        }

    distance = (
        result["distance_km"]
        + start_snap_distance_m / 1000
        + goal_snap_distance_m / 1000
    )

    energy = calculate_energy(
        distance,
        energy_consumption_kwh_per_km
    )

    success = energy <= battery - reserve
    return {
        "algorithm": "Dijkstra (tanpa pengisian)",
        # Layak hanya jika baterai awal cukup (tanpa isi daya).
        "success": success,
        "route": result["route"],
        "distance_km": distance,
        "energy_used_kwh": energy,
        "charged_kwh": 0,
        "remaining_battery_kwh": battery - energy,
        "charging_stops": [],
        "expanded": result["expanded"],
        "computation_time_s": elapsed,
        "failure_reason": (
            None if success else "tidak_ada_rute_layak"
        )
    }
