import sys
import time
import argparse
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.gbfs_distance import (
    clear_road_distance_cache,
    greedy_best_first_search,
    haversine_distance
)

from src.algorithms.gbfs_battery import (
    greedy_best_first_search_battery,
    DEFAULT_LAMBDA,
    DEFAULT_DETOUR_FACTOR,
    DEFAULT_MAX_EXPANSIONS
)
from src.algorithms.baseline_dijkstra import dijkstra_search

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

GRAPH_PATH = MEDAN_GRAPH_PATH
STATIONS_PATH = MEDAN_STATIONS_CONNECTED_PATH


def get_snap_distance(station):
    return float(station.get("snap_distance_m", 0) or 0)


def distance_search(graph, start, goal, battery_kwh=None):
    """Adapter GBFS heuristik jarak. battery_kwh diabaikan."""
    return greedy_best_first_search(graph, start, goal)


def make_battery_search(
    lam=DEFAULT_LAMBDA,
    detour_factor=DEFAULT_DETOUR_FACTOR,
    energy_consumption_kwh_per_km=None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS
):
    """
    Adapter GBFS battery-aware. Simulator mengirim baterai dalam kWh,
    sedangkan gbfs_battery meminta persentase, jadi dikonversi di sini.
    Anggaran satu leg digunakan bersama pencarian kandidat SPKLU dan
    direset setelah pencarian langsung mencapai batas atau SPKLU dipilih.
    """
    if max_expansions < 1:
        raise ValueError("max_expansions harus lebih besar dari 0.")

    remaining_expansions = max_expansions

    def battery_search(graph, start, goal, battery_kwh):
        nonlocal remaining_expansions

        if remaining_expansions < 1:
            return {
                "success": False,
                "route": [],
                "distance_km": 0,
                "energy_used_kwh": 0,
                "remaining_battery_kwh": battery_kwh,
                "expanded": 0,
                "max_expansions_reached": True,
                "failure_reason": "batas_simpul"
            }

        percentage = battery_kwh / BATTERY_CAPACITY * 100

        result = greedy_best_first_search_battery(
            graph,
            start,
            goal,
            percentage,
            lam=lam,
            detour_factor=detour_factor,
            energy_consumption_kwh_per_km=(
                energy_consumption_kwh_per_km
            ),
            max_expansions=remaining_expansions
        )
        remaining_expansions -= result["expanded"]
        return result

    def reset_expansion_budget():
        nonlocal remaining_expansions
        remaining_expansions = max_expansions

    setattr(
        battery_search,
        "reset_expansion_budget",
        reset_expansion_budget
    )
    return battery_search


def simulate_ev_trip(
    graph,
    stations,
    start_node,
    goal_node,
    initial_battery_percentage,
    target_charge_percentage=80,
    start_snap_distance_m: float = 0.0,
    goal_snap_distance_m: float = 0.0,
    search_fn=distance_search,
    reserve_percentage=0,
    algorithm_name="GBFS Distance",
    energy_consumption_kwh_per_km=None,
    allow_start_charging=False
):
    """
    search_fn(graph, start, goal, battery_kwh) harus mengembalikan
    dict berisi: success, route, distance_km, expanded.

    reserve_percentage: cadangan baterai minimum yang tidak boleh
    terpakai (disarankan 10 untuk eksperimen).
    """
    clear_road_distance_cache()

    current_node = int(start_node)
    goal_node = int(goal_node)

    battery = (
        BATTERY_CAPACITY
        * initial_battery_percentage / 100
    )

    target_charge = (
        BATTERY_CAPACITY
        * target_charge_percentage / 100
    )

    reserve = (
        BATTERY_CAPACITY
        * reserve_percentage / 100
    )

    total_distance = 0
    total_energy_used = 0
    total_charged = 0
    total_expanded = 0
    comparison_time = 0.0
    search_time = 0.0
    search_limit_reached = False
    direct_search_limit_reached = False
    station_search_limit_reached = False
    station_search_attempted = False
    station_lower_bound_rejected = False
    start_charge_attempted = False
    start_charge_used = False
    start_snap_pending = True

    charging_stops = []
    complete_route = [current_node]
    leg_comparisons = []

    visited_stations = set()
    current_station_snap_m = 0
    start_station_matches = stations[
        stations["access_status"] != "RESTRICTED"
    ]
    start_station_matches = start_station_matches[
        start_station_matches["nearest_node"].astype(int) == current_node
    ]
    start_station_index = (
        start_station_matches.index[0]
        if not start_station_matches.empty
        else None
    )

    def build_result(success, failure_reason=None):
        return {
            "algorithm": algorithm_name,
            "success": success,
            "failure_reason": (
                None if success else failure_reason
            ),
            "route": complete_route,
            "distance_km": total_distance,
            "energy_used_kwh": total_energy_used,
            "charged_kwh": total_charged,
            "remaining_battery_kwh": battery,
            "charging_stops": charging_stops,
            "leg_comparisons": leg_comparisons,
            "expanded": total_expanded,
            "computation_time_s": search_time,
            "comparison_time_s": comparison_time,
            "search_limit_reached": search_limit_reached,
            "direct_search_limit_reached": (
                direct_search_limit_reached
            ),
            "charged_at_start": start_charge_used
        }

    def reset_search_budget():
        reset_budget = getattr(
            search_fn,
            "reset_expansion_budget",
            None
        )
        if callable(reset_budget):
            reset_budget()

    def charge_at_start():
        nonlocal battery, total_charged, start_charge_used

        station = stations.loc[start_station_index]
        energy_added = max(0, target_charge - battery)
        if energy_added <= 0:
            return False

        battery += energy_added
        total_charged += energy_added
        start_charge_used = True
        visited_stations.add(start_station_index)
        distance_to_goal = haversine_distance(
            graph, current_node, goal_node
        )
        charging_stops.append({
            "name": station["name"],
            "nearest_node": int(station["nearest_node"]),
            "address": station.get("address", ""),
            "access_status": station["access_status"],
            "distance_to_goal_km": distance_to_goal,
            "progress_to_goal": False,
            "battery_before_kwh": battery - energy_added,
            "charged_kwh": energy_added,
            "battery_after_kwh": battery,
            "snap_distance_m": get_snap_distance(station),
            "is_start_station": True
        })
        return True

    def run_search(start, goal, battery_kwh):
        nonlocal search_limit_reached, search_time

        search_started = time.perf_counter()
        result = search_fn(
            graph,
            start,
            goal,
            battery_kwh
        )
        search_time += time.perf_counter() - search_started
        search_limit_reached = (
            search_limit_reached
            or result.get("max_expansions_reached", False)
        )
        return result

    def record_leg(
        leg_result,
        from_node,
        to_node,
        leg_distance,
        source_snap_m,
        destination_snap_m,
        first_leg,
        available_energy
    ):
        nonlocal comparison_time

        comparison_started = time.perf_counter()
        baseline = dijkstra_search(
            graph,
            from_node,
            to_node
        )
        comparison_time += time.perf_counter() - comparison_started

        baseline_distance = 0.0
        baseline_feasible = False
        if baseline["success"]:
            baseline_distance = (
                baseline["distance_km"]
                + source_snap_m / 1000
                + destination_snap_m / 1000
            )
            if first_leg:
                baseline_distance += start_snap_distance_m / 1000
            baseline_feasible = (
                calculate_energy(
                    baseline_distance,
                    energy_consumption_kwh_per_km
                ) <= available_energy
            )

        leg_comparisons.append({
            "from_node": from_node,
            "to_node": to_node,
            "algorithm_distance_km": leg_distance,
            "dijkstra_distance_km": baseline_distance,
            "route_stretch": (
                leg_distance / baseline_distance
                if baseline_distance > 0
                else None
            ),
            "algorithm_feasible": True,
            "dijkstra_feasible": baseline_feasible,
            "algorithm_expanded": leg_result["expanded"],
            "dijkstra_expanded": baseline["expanded"]
        })

    # SPKLU berstatus RESTRICTED tidak boleh digunakan.
    candidates = stations[
        stations["access_status"] != "RESTRICTED"
    ]

    while True:

        # Jarak garis lurus adalah batas bawah untuk jarak rute jalan.
        direct_lower_bound = (
            haversine_distance(graph, current_node, goal_node)
            + current_station_snap_m / 1000
            + goal_snap_distance_m / 1000
        )
        if start_snap_pending:
            direct_lower_bound += start_snap_distance_m / 1000

        direct_result = None
        if calculate_energy(
            direct_lower_bound,
            energy_consumption_kwh_per_km
        ) <= battery - reserve:
            direct_result = run_search(
                current_node,
                goal_node,
                max(battery - reserve, 0)
            )
            total_expanded += direct_result["expanded"]
            if direct_result.get("max_expansions_reached", False):
                direct_search_limit_reached = True
                reset_search_budget()

        if direct_result is not None and direct_result["success"]:

            destination_distance = (
                direct_result["distance_km"]
                + current_station_snap_m / 1000
                + goal_snap_distance_m / 1000
            )

            # Jarak lokasi awal ke simpul jalan hanya sekali,
            # yaitu pada leg pertama.
            if start_snap_pending:
                destination_distance += (
                    start_snap_distance_m / 1000
                )

            destination_energy = calculate_energy(
                destination_distance,
                energy_consumption_kwh_per_km
            )

            if destination_energy <= battery - reserve:
                record_leg(
                    direct_result,
                    current_node,
                    goal_node,
                    destination_distance,
                    current_station_snap_m,
                    goal_snap_distance_m,
                    start_snap_pending,
                    battery - reserve
                )

                battery -= destination_energy

                total_distance += destination_distance
                total_energy_used += destination_energy
                start_snap_pending = False

                complete_route.extend(
                    direct_result["route"][1:]
                )

                return build_result(True)

        if (
            allow_start_charging
            and not start_charge_attempted
            and start_station_index is not None
        ):
            start_charge_attempted = True
            if charge_at_start():
                reset_search_budget()
                continue

        # Urutkan kandidat SPKLU berdasarkan kedekatan
        # garis lurus terhadap tujuan.
        candidate_list = []
        current_distance_to_goal = haversine_distance(
            graph,
            current_node,
            goal_node
        )

        for index, station in candidates.iterrows():

            if index in visited_stations:
                continue

            station_node = int(station["nearest_node"])

            if station_node == current_node:
                continue

            distance_to_goal = haversine_distance(
                graph,
                station_node,
                goal_node
            )

            makes_progress = (
                distance_to_goal < current_distance_to_goal - 0.01
            )
            candidate_list.append(
                (not makes_progress, distance_to_goal, index)
            )

        candidate_list.sort()

        selected_station = None
        selected_result = None
        selected_distance = None
        selected_distance_to_goal = 0.0

        # Cari SPKLU yang masih dapat dijangkau.
        for _, distance_to_goal, index in candidate_list:

            station = stations.loc[index]
            station_node = int(station["nearest_node"])

            direct_lower_bound = (
                haversine_distance(
                    graph,
                    current_node,
                    station_node
                )
                + current_station_snap_m / 1000
                + get_snap_distance(station) / 1000
            )
            if start_snap_pending:
                direct_lower_bound += start_snap_distance_m / 1000
            if calculate_energy(
                direct_lower_bound,
                energy_consumption_kwh_per_km
            ) > battery - reserve:
                station_lower_bound_rejected = True
                continue

            station_search_attempted = True
            leg_result = run_search(
                current_node,
                station_node,
                max(battery - reserve, 0)
            )

            total_expanded += leg_result["expanded"]
            if leg_result.get("max_expansions_reached", False):
                station_search_limit_reached = True

            if not leg_result["success"]:
                continue

            leg_distance = (
                leg_result["distance_km"]
                + current_station_snap_m / 1000
                + get_snap_distance(station) / 1000
            )

            # Jarak dari lokasi awal ke simpul jalan.
            if start_snap_pending:
                leg_distance += start_snap_distance_m / 1000

            required_energy = calculate_energy(
                leg_distance,
                energy_consumption_kwh_per_km
            )

            if required_energy <= battery - reserve:

                selected_station = index
                selected_result = leg_result
                selected_distance = leg_distance
                selected_distance_to_goal = distance_to_goal

                break

        # Tidak ada SPKLU yang dapat dijangkau.
        if selected_station is None:
            if station_search_limit_reached:
                failure_reason = "batas_simpul"
            elif (
                direct_result is not None
                and not direct_result["success"]
                and not station_search_attempted
                and not station_lower_bound_rejected
            ):
                failure_reason = "tidak_ada_rute_layak"
            else:
                failure_reason = "spklu_tidak_terjangkau"
            return build_result(False, failure_reason)

        # Kendaraan menuju SPKLU terpilih.
        station = stations.loc[selected_station]
        reset_search_budget()

        energy_used = calculate_energy(
            selected_distance,
            energy_consumption_kwh_per_km
        )

        record_leg(
            selected_result,
            current_node,
            int(station["nearest_node"]),
            selected_distance,
            current_station_snap_m,
            get_snap_distance(station),
            start_snap_pending,
            battery - reserve
        )

        battery -= energy_used

        total_distance += selected_distance
        total_energy_used += energy_used
        start_snap_pending = False

        complete_route.extend(
            selected_result["route"][1:]
        )

        visited_stations.add(selected_station)

        # Simulasi pengisian baterai.
        battery_before_charge = battery

        energy_added = max(
            0,
            target_charge - battery
        )

        battery += energy_added
        total_charged += energy_added

        charging_stops.append({
            "name": station["name"],
            "nearest_node": int(station["nearest_node"]),
            "address": station.get("address", ""),
            "access_status": station["access_status"],
            "distance_to_goal_km": selected_distance_to_goal,
            "progress_to_goal": (
                selected_distance_to_goal
                < current_distance_to_goal - 0.01
            ),
            "battery_before_kwh": battery_before_charge,
            "charged_kwh": energy_added,
            "battery_after_kwh": battery,
            "snap_distance_m": get_snap_distance(station)
        })

        current_node = int(station["nearest_node"])
        current_station_snap_m = get_snap_distance(station)


def print_result(result):

    print(f"\n[{result['algorithm']}]")

    if result["success"]:
        print("Status: PERJALANAN BERHASIL")
    else:
        print("Status: PERJALANAN TIDAK DAPAT DISELESAIKAN")

    print(f"Total jarak: {result['distance_km']:.2f} km")
    print(f"Energi terpakai: {result['energy_used_kwh']:.2f} kWh")
    print(f"Energi pengisian: {result['charged_kwh']:.2f} kWh")
    print(f"Sisa baterai: {result['remaining_battery_kwh']:.2f} kWh")
    print(f"Jumlah pemberhentian: {len(result['charging_stops'])}")
    print(f"Waktu komputasi: {result['computation_time_s']:.3f} s")

    for stop in result["charging_stops"]:
        print(
            f"- {stop['name']}: "
            f"mengisi {stop['charged_kwh']:.2f} kWh "
            f"({stop['access_status']})"
        )


if __name__ == "__main__":

    import osmnx as ox

    from src.algorithms.baseline_dijkstra import simulate_baseline

    parser = argparse.ArgumentParser(
        description="Simulasikan rute EV pada graf Medan atau regional."
    )
    parser.add_argument(
        "--profile",
        choices=("medan", "regional"),
        default="medan"
    )
    arguments = parser.parse_args()

    graph_path = (
        REGIONAL_GRAPH_PATH
        if arguments.profile == "regional"
        else MEDAN_GRAPH_PATH
    )
    stations_path = (
        REGIONAL_STATIONS_CONNECTED_PATH
        if arguments.profile == "regional"
        else MEDAN_STATIONS_CONNECTED_PATH
    )

    print("SIMULASI PERJALANAN KENDARAAN LISTRIK")

    graph = ox.load_graphml(graph_path)

    stations = gpd.read_file(
        stations_path
    ).reset_index(drop=True)

    start_node = int(stations.iloc[0]["nearest_node"])
    goal_node = int(stations.iloc[1]["nearest_node"])

    for battery_pct in (80, 50, 20):

        print(f"\n===== BATERAI AWAL {battery_pct}% =====")

        print_result(
            simulate_baseline(
                graph,
                start_node,
                goal_node,
                battery_pct,
                reserve_percentage=10
            )
        )

        print_result(
            simulate_ev_trip(
                graph=graph,
                stations=stations,
                start_node=start_node,
                goal_node=goal_node,
                initial_battery_percentage=battery_pct,
                target_charge_percentage=80,
                search_fn=distance_search,
                algorithm_name="GBFS Distance",
                reserve_percentage=10
            )
        )

        print_result(
            simulate_ev_trip(
                graph=graph,
                stations=stations,
                start_node=start_node,
                goal_node=goal_node,
                initial_battery_percentage=battery_pct,
                target_charge_percentage=80,
                search_fn=make_battery_search(),
                algorithm_name="GBFS Battery-Aware",
                reserve_percentage=10
            )
        )
