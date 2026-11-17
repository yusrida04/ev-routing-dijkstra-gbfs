import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.baseline_dijkstra import (
    dijkstra_search,
    simulate_baseline
)
from src.algorithms.gbfs_battery import DEFAULT_DETOUR_FACTOR
from src.algorithms.gbfs_distance import (
    get_simple_road_graph,
    haversine_distance
)
from src.project_paths import (
    EXPERIMENTS_OUTPUT_DIR,
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_CONNECTED_PATH,
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH
)
from src.simulator import (
    distance_search,
    make_battery_search,
    simulate_ev_trip
)

BATTERY_LEVELS = (80, 50, 20)
ENERGY_CONSUMPTIONS = (0.1218, 0.15, 0.18)
LAMBDA_VALUES = (1, 5, 10, 20)
DISTANCE_STRATA = {
    "near": lambda distance_km: distance_km < 20,
    "medium": lambda distance_km: 20 <= distance_km <= 80,
    "far": lambda distance_km: distance_km > 80
}


def station_region_case(station):
    location = " ".join(
        str(station.get(field, ""))
        for field in ("city", "address")
    ).casefold()
    return "samosir_origin" if "samosir" in location else "non_samosir"


def build_pair_buckets(graph, starts, goals, region_case):
    road_graph = get_simple_road_graph(graph)
    buckets = {name: [] for name in DISTANCE_STRATA}

    for start in starts:
        start_node = int(start["nearest_node"])
        distances = nx.single_source_dijkstra_path_length(
            road_graph,
            start_node,
            weight="length"
        )
        for goal in goals:
            goal_node = int(goal["nearest_node"])
            if goal_node == start_node or goal_node not in distances:
                continue

            road_distance_km = distances[goal_node] / 1000
            stratum = next(
                (
                    name for name, predicate in DISTANCE_STRATA.items()
                    if predicate(road_distance_km)
                ),
                None
            )
            if stratum is None:
                continue

            buckets[stratum].append((
                start,
                goal,
                road_distance_km,
                haversine_distance(graph, start_node, goal_node),
                region_case
            ))

    return buckets


def sample_balanced_origins(bucket, count, rng, origin_counts):
    by_origin = {}
    for pair in bucket:
        origin_node = int(pair[0]["nearest_node"])
        by_origin.setdefault(origin_node, []).append(pair)

    selected = []
    while len(selected) < count:
        available_origins = [
            node for node, pairs in by_origin.items() if pairs
        ]
        if not available_origins:
            raise ValueError("Tidak cukup pasangan untuk menyeimbangkan asal.")

        lowest_count = min(
            origin_counts.get(node, 0) for node in available_origins
        )
        origins = [
            node for node in available_origins
            if origin_counts.get(node, 0) == lowest_count
        ]
        origin_node = rng.choice(origins)
        pair_position = rng.randrange(len(by_origin[origin_node]))
        pair = by_origin[origin_node].pop(pair_position)
        origin_counts[origin_node] = origin_counts.get(origin_node, 0) + 1
        selected.append(pair)

    return selected


def choose_pairs(graph, stations, pair_count, seed):
    eligible = stations[
        stations["access_status"] != "RESTRICTED"
    ].reset_index(drop=True)

    unique_stations = {}
    for _, station in eligible.iterrows():
        unique_stations.setdefault(
            int(station["nearest_node"]),
            station
        )

    stations_by_case = {"non_samosir": [], "samosir_origin": []}
    for station in unique_stations.values():
        stations_by_case[station_region_case(station)].append(station)

    rng = random.Random(seed)
    regular_buckets = build_pair_buckets(
        graph,
        stations_by_case["non_samosir"],
        stations_by_case["non_samosir"],
        "non_samosir"
    )
    if any(not regular_buckets[name] for name in DISTANCE_STRATA):
        counts = ", ".join(
            f"{name}={len(pairs)}"
            for name, pairs in regular_buckets.items()
        )
        raise ValueError(
            "Data non-Samosir tidak mencakup semua strata jarak jalan "
            f"({counts})."
        )

    per_stratum, remainder = divmod(pair_count, len(DISTANCE_STRATA))
    origin_counts = {}
    selected_regular = []
    for index, (stratum, bucket) in enumerate(regular_buckets.items()):
        count = per_stratum + (index < remainder)
        if len(bucket) < count:
            raise ValueError(
                f"Strata jalan '{stratum}' hanya memiliki {len(bucket)} "
                f"pasangan non-Samosir; diminta {count}."
            )
        selected_regular.extend(
            (pair, stratum)
            for pair in sample_balanced_origins(
                bucket,
                count,
                rng,
                origin_counts
            )
        )

    samosir_starts = stations_by_case["samosir_origin"]
    non_samosir_goals = stations_by_case["non_samosir"]
    if samosir_starts and non_samosir_goals:
        samosir_buckets = build_pair_buckets(
            graph,
            samosir_starts,
            non_samosir_goals,
            "samosir_origin"
        )
        samosir_pairs = [
            (
                start,
                goal,
                stratum,
                road_distance_km,
                straight_line_distance_km,
                region_case
            )
            for stratum, bucket in samosir_buckets.items()
            for (
                start,
                goal,
                road_distance_km,
                straight_line_distance_km,
                region_case
            ) in bucket
        ]
        selected_samosir = rng.sample(
            samosir_pairs,
            min(pair_count, len(samosir_pairs))
        )
    else:
        selected_samosir = []

    def format_pair(pair, stratum):
        (
            start,
            goal,
            road_distance_km,
            straight_line_distance_km,
            region_case
        ) = pair
        return (
            start,
            goal,
            stratum,
            road_distance_km,
            straight_line_distance_km,
            region_case
        )

    selected_pairs = [
        format_pair(pair, stratum)
        for pair, stratum in selected_regular
    ] + selected_samosir
    rng.shuffle(selected_pairs)
    return selected_pairs


def result_row(
    pair_index,
    start,
    goal,
    battery_percentage,
    consumption,
    reserve_percentage,
    lam,
    repeat,
    distance_stratum,
    road_distance_km,
    straight_line_distance_km,
    region_case,
    result,
    allow_start_charging=False
):
    return {
        "pair_index": pair_index,
        "start_name": start["name"],
        "goal_name": goal["name"],
        "start_node": int(start["nearest_node"]),
        "goal_node": int(goal["nearest_node"]),
        "battery_percentage": battery_percentage,
        "energy_consumption_kwh_per_km": consumption,
        "reserve_percentage": reserve_percentage,
        "lambda": lam,
        "region_case": region_case,
        "distance_stratum": distance_stratum,
        "road_distance_km": road_distance_km,
        "straight_line_distance_km": straight_line_distance_km,
        "repeat": repeat,
        "algorithm": result["algorithm"],
        "success": result["success"],
        "distance_km": result["distance_km"],
        "energy_used_kwh": result["energy_used_kwh"],
        "charged_kwh": result["charged_kwh"],
        "charging_stops": len(result["charging_stops"]),
        "expanded": result["expanded"],
        "computation_time_s": result["computation_time_s"],
        "comparison_time_s": result.get("comparison_time_s", 0.0),
        "direct_search_limit_reached": result.get(
            "direct_search_limit_reached",
            False
        ),
        "charged_at_start": result.get("charged_at_start", False),
        "start_charging_assumption": (
            "not_applicable"
            if result["algorithm"] == "Dijkstra (tanpa pengisian)"
            else (
                "only_if_needed"
                if allow_start_charging else "disabled"
            )
        ),
        "failure_reason": result.get("failure_reason"),
        "search_limit_reached": result.get(
            "search_limit_reached",
            result.get("max_expansions_reached", False)
        )
    }


def summarize_rows(rows):
    groups = {}
    dijkstra_spklu_by_pair = {}
    for row in rows:
        if row["algorithm"] == "Dijkstra + SPKLU":
            run_key = (
                row["pair_index"],
                row.get("region_case", "non_samosir"),
                row["battery_percentage"],
                row["energy_consumption_kwh_per_km"],
                row["reserve_percentage"],
                row.get("start_charging_assumption", "disabled")
            )
            dijkstra_spklu_by_pair[run_key] = row

        key = (
            row["algorithm"],
            row.get("region_case", "non_samosir"),
            row.get("distance_stratum", ""),
            row.get("start_charging_assumption", "disabled"),
            row["battery_percentage"],
            row["energy_consumption_kwh_per_km"],
            row["reserve_percentage"],
            row["lambda"]
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in groups.items():
        successful = [row for row in group if row["success"]]
        times = [
            row["computation_time_s"] for row in successful
            if row.get("computation_time_s") not in (None, "")
        ]
        distances = [row["distance_km"] for row in successful]
        energies = [
            row["energy_used_kwh"] for row in successful
        ]
        expanded = [row["expanded"] for row in group]
        distance_deltas = []

        for row in successful:
            run_key = (
                row["pair_index"],
                row.get("region_case", "non_samosir"),
                row["battery_percentage"],
                row["energy_consumption_kwh_per_km"],
                row["reserve_percentage"],
                row.get("start_charging_assumption", "disabled")
            )
            baseline = dijkstra_spklu_by_pair.get(run_key)
            if baseline is not None and baseline["success"]:
                distance_deltas.append(
                    row["distance_km"] - baseline["distance_km"]
                )

        summary.append({
            "algorithm": key[0],
            "region_case": key[1],
            "distance_stratum": key[2],
            "start_charging_assumption": key[3],
            "battery_percentage": key[4],
            "energy_consumption_kwh_per_km": key[5],
            "reserve_percentage": key[6],
            "lambda": key[7],
            "runs": len(group),
            "n_success": len(successful),
            "success_rate": len(successful) / len(group),
            "n_direct_search_limit": sum(
                row.get("direct_search_limit_reached", False)
                for row in group
            ),
            "mean_computation_time_s": (
                statistics.mean(times) if times else None
            ),
            "std_computation_time_s": (
                statistics.stdev(times) if len(times) > 1 else (
                    0.0 if times else None
                )
            ),
            "mean_distance_km": (
                statistics.mean(distances) if distances else None
            ),
            "std_distance_km": (
                statistics.stdev(distances)
                if len(distances) > 1 else (
                    0.0 if distances else None
                )
            ),
            "mean_energy_used_kwh": (
                statistics.mean(energies) if energies else None
            ),
            "std_energy_used_kwh": (
                statistics.stdev(energies)
                if len(energies) > 1 else (
                    0.0 if energies else None
                )
            ),
            "n_distance_comparisons": len(distance_deltas),
            "mean_distance_delta_vs_dijkstra_spklu_km": (
                statistics.mean(distance_deltas)
                if distance_deltas else None
            ),
            "std_distance_delta_vs_dijkstra_spklu_km": (
                statistics.stdev(distance_deltas)
                if len(distance_deltas) > 1 else (
                    0.0 if distance_deltas else None
                )
            ),
            "mean_expanded": statistics.mean(expanded),
            "std_expanded": (
                statistics.stdev(expanded)
                if len(expanded) > 1 else 0.0
            )
        })

    return summary


def timing_summary_rows(quality_rows, timing_rows):
    timing_groups = {}
    for row in timing_rows:
        key = (
            row["algorithm"],
            row.get("region_case", "non_samosir"),
            row["distance_stratum"],
            row.get("start_charging_assumption", "disabled"),
            row["battery_percentage"],
            row["energy_consumption_kwh_per_km"],
            row["reserve_percentage"],
            row["lambda"]
        )
        timing_groups.setdefault(key, []).append(row)

    summaries = summarize_rows(quality_rows)
    for summary in summaries:
        key = (
            summary["algorithm"],
            summary["region_case"],
            summary["distance_stratum"],
            summary["start_charging_assumption"],
            summary["battery_percentage"],
            summary["energy_consumption_kwh_per_km"],
            summary["reserve_percentage"],
            summary["lambda"]
        )
        group = timing_groups.get(key, [])
        successful_times = [
            row["computation_time_s"] for row in group if row["success"]
        ]
        summary["n_timing_runs"] = len(group)
        summary["mean_computation_time_s"] = (
            statistics.mean(successful_times)
            if successful_times else None
        )
        summary["std_computation_time_s"] = (
            statistics.stdev(successful_times)
            if len(successful_times) > 1 else (
                0.0 if successful_times else None
            )
        )
    return summaries


def choose_timing_pair_indices(pairs, timing_pair_count, seed):
    if timing_pair_count < 0:
        raise ValueError("Jumlah pasangan untuk pengukuran waktu >= 0.")

    buckets = {}
    for pair_index, pair in enumerate(pairs, start=1):
        region_case = pair[5] if len(pair) > 5 else "non_samosir"
        key = (region_case, pair[2])
        buckets.setdefault(key, []).append(pair_index)

    selected_count = min(timing_pair_count, len(pairs))
    rng = random.Random(seed)
    selected = set()
    while len(selected) < selected_count:
        available_groups = [
            group for group, pair_indices in buckets.items()
            if pair_indices
        ]
        if not available_groups:
            break
        group_counts = {
            group: sum(
                pair_index in selected for pair_index in buckets[group]
            )
            for group in available_groups
        }
        least_sampled = min(group_counts.values())
        groups = [
            group for group, count in group_counts.items()
            if count == least_sampled
        ]
        group = rng.choice(groups)
        pair_index = rng.choice(buckets[group])
        buckets[group].remove(pair_index)
        selected.add(pair_index)
    return selected


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError(f"Tidak ada hasil untuk ditulis ke {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                fieldnames if fieldnames is not None
                else list(rows[0].keys())
            )
        )
        writer.writeheader()
        writer.writerows(rows)


def experiment_algorithms(consumption):
    algorithms = [
        ("Dijkstra + SPKLU", dijkstra_search, ""),
        ("GBFS Distance", distance_search, "")
    ]
    algorithms.extend(
        (
            "GBFS Battery-Aware",
            make_battery_search(
                lam=lam,
                detour_factor=DEFAULT_DETOUR_FACTOR,
                energy_consumption_kwh_per_km=consumption
            ),
            lam
        )
        for lam in LAMBDA_VALUES
    )
    return algorithms


def timing_row(
    pair_index,
    region_case,
    distance_stratum,
    road_distance_km,
    battery_percentage,
    consumption,
    reserve_percentage,
    lam,
    repeat,
    result,
    allow_start_charging=False
):
    return {
        "pair_index": pair_index,
        "region_case": region_case,
        "distance_stratum": distance_stratum,
        "road_distance_km": road_distance_km,
        "battery_percentage": battery_percentage,
        "energy_consumption_kwh_per_km": consumption,
        "reserve_percentage": reserve_percentage,
        "lambda": lam,
        "repeat": repeat,
        "algorithm": result["algorithm"],
        "success": result["success"],
        "computation_time_s": result["computation_time_s"],
        "direct_search_limit_reached": result.get(
            "direct_search_limit_reached",
            False
        ),
        "charged_at_start": result.get("charged_at_start", False),
        "start_charging_assumption": (
            "not_applicable"
            if result["algorithm"] == "Dijkstra (tanpa pengisian)"
            else (
                "only_if_needed"
                if allow_start_charging else "disabled"
            )
        )
    }


def run_experiments(
    pair_count=30,
    repeats=3,
    timing_pair_count=6,
    reserve_percentage=10,
    seed=42,
    profile="regional",
    output_dir=EXPERIMENTS_OUTPUT_DIR,
    allow_start_charging=False
):
    if profile == "regional":
        graph_path = REGIONAL_GRAPH_PATH
        stations_path = REGIONAL_STATIONS_CONNECTED_PATH
    else:
        graph_path = MEDAN_GRAPH_PATH
        stations_path = MEDAN_STATIONS_CONNECTED_PATH

    if not graph_path.exists():
        raise FileNotFoundError(f"Graf tidak ditemukan: {graph_path}")
    if not stations_path.exists():
        raise FileNotFoundError(
            f"Data SPKLU tidak ditemukan: {stations_path}"
        )
    if pair_count < 30:
        raise ValueError("Minimal 30 pasangan non-Samosir harus diuji.")
    if not 3 <= repeats <= 5:
        raise ValueError("Pengulangan waktu harus antara 3 dan 5.")
    if timing_pair_count < 0:
        raise ValueError("Jumlah pasangan untuk pengukuran waktu >= 0.")
    if not 0 <= reserve_percentage <= 100:
        raise ValueError("Cadangan baterai harus di antara 0 dan 100%.")

    graph = ox.load_graphml(graph_path)
    stations = gpd.read_file(stations_path).reset_index(drop=True)
    pairs = choose_pairs(graph, stations, pair_count, seed)
    timing_pair_indices = choose_timing_pair_indices(
        pairs,
        timing_pair_count,
        seed
    )

    quality_rows = []
    timing_rows = []
    for pair_index, pair in enumerate(pairs, start=1):
        (
            start,
            goal,
            distance_stratum,
            road_distance_km,
            straight_line_distance_km,
            region_case
        ) = pair
        start_node = int(start["nearest_node"])
        goal_node = int(goal["nearest_node"])
        if start_node == goal_node:
            continue

        for battery_percentage in BATTERY_LEVELS:
            for consumption in ENERGY_CONSUMPTIONS:
                baseline = simulate_baseline(
                    graph,
                    start_node,
                    goal_node,
                    battery_percentage,
                    start_snap_distance_m=float(
                        start.get("snap_distance_m", 0) or 0
                    ),
                    goal_snap_distance_m=float(
                        goal.get("snap_distance_m", 0) or 0
                    ),
                    reserve_percentage=reserve_percentage,
                    energy_consumption_kwh_per_km=consumption
                )
                row = result_row(
                    pair_index,
                    start,
                    goal,
                    battery_percentage,
                    consumption,
                    reserve_percentage,
                    "",
                    1,
                    distance_stratum,
                    road_distance_km,
                    straight_line_distance_km,
                    region_case,
                    baseline,
                    allow_start_charging
                )
                row["computation_time_s"] = ""
                row["comparison_time_s"] = ""
                quality_rows.append(row)

                for algorithm_name, search_fn, lam in (
                    experiment_algorithms(consumption)
                ):
                    result = simulate_ev_trip(
                        graph=graph,
                        stations=stations,
                        start_node=start_node,
                        goal_node=goal_node,
                        initial_battery_percentage=battery_percentage,
                        target_charge_percentage=80,
                        start_snap_distance_m=float(
                            start.get("snap_distance_m", 0) or 0
                        ),
                        goal_snap_distance_m=float(
                            goal.get("snap_distance_m", 0) or 0
                        ),
                        search_fn=search_fn,
                        reserve_percentage=reserve_percentage,
                        algorithm_name=algorithm_name,
                        energy_consumption_kwh_per_km=consumption,
                        allow_start_charging=allow_start_charging
                    )
                    row = result_row(
                        pair_index,
                        start,
                        goal,
                        battery_percentage,
                        consumption,
                        reserve_percentage,
                        lam,
                        1,
                        distance_stratum,
                        road_distance_km,
                        straight_line_distance_km,
                        region_case,
                        result,
                        allow_start_charging
                    )
                    row["computation_time_s"] = ""
                    row["comparison_time_s"] = ""
                    quality_rows.append(row)

                if pair_index not in timing_pair_indices:
                    continue

                for repeat in range(1, repeats + 1):
                    baseline = simulate_baseline(
                        graph,
                        start_node,
                        goal_node,
                        battery_percentage,
                        start_snap_distance_m=float(
                            start.get("snap_distance_m", 0) or 0
                        ),
                        goal_snap_distance_m=float(
                            goal.get("snap_distance_m", 0) or 0
                        ),
                        reserve_percentage=reserve_percentage,
                        energy_consumption_kwh_per_km=consumption
                    )
                    timing_rows.append(timing_row(
                        pair_index,
                        region_case,
                        distance_stratum,
                        road_distance_km,
                        battery_percentage,
                        consumption,
                        reserve_percentage,
                        "",
                        repeat,
                        baseline,
                        allow_start_charging
                    ))

                    for algorithm_name, search_fn, lam in (
                        experiment_algorithms(consumption)
                    ):
                        result = simulate_ev_trip(
                            graph=graph,
                            stations=stations,
                            start_node=start_node,
                            goal_node=goal_node,
                            initial_battery_percentage=battery_percentage,
                            target_charge_percentage=80,
                            start_snap_distance_m=float(
                                start.get("snap_distance_m", 0) or 0
                            ),
                            goal_snap_distance_m=float(
                                goal.get("snap_distance_m", 0) or 0
                            ),
                            search_fn=search_fn,
                            reserve_percentage=reserve_percentage,
                            algorithm_name=algorithm_name,
                            energy_consumption_kwh_per_km=consumption,
                            allow_start_charging=allow_start_charging
                        )
                        timing_rows.append(timing_row(
                            pair_index,
                            region_case,
                            distance_stratum,
                            road_distance_km,
                            battery_percentage,
                            consumption,
                            reserve_percentage,
                            lam,
                            repeat,
                            result,
                            allow_start_charging
                        ))

        print(
            f"Pasangan {pair_index}/{len(pairs)} selesai: "
            f"{start['name']} -> {goal['name']}"
        )

    if not quality_rows:
        raise ValueError("Tidak ada pasangan node yang dapat diuji.")

    profile_suffix = profile.lower()
    run_path = output_dir / f"experiment_{profile_suffix}_runs.csv"
    timing_path = output_dir / f"experiment_{profile_suffix}_timing.csv"
    summary_path = output_dir / f"experiment_{profile_suffix}_summary.csv"
    write_csv(run_path, quality_rows)
    write_csv(
        timing_path,
        timing_rows,
        fieldnames=[
            "pair_index",
            "region_case",
            "distance_stratum",
            "road_distance_km",
            "battery_percentage",
            "energy_consumption_kwh_per_km",
            "reserve_percentage",
            "lambda",
            "repeat",
            "algorithm",
            "success",
            "computation_time_s",
            "direct_search_limit_reached",
            "charged_at_start",
            "start_charging_assumption"
        ]
    )
    write_csv(
        summary_path,
        timing_summary_rows(quality_rows, timing_rows)
    )

    print(f"\nEvaluasi kualitas per-pasangan: {run_path}")
    print(f"Pengukuran waktu berulang: {timing_path}")
    print(f"Ringkasan mean/std: {summary_path}")
    return run_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eksperimen perbandingan rute EV dengan CSV."
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=30,
        help="Pasangan non-Samosir minimum; kohor Samosir ditambahkan."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Pengulangan waktu untuk subset pasangan (3-5)."
    )
    parser.add_argument(
        "--timing-pairs",
        type=int,
        default=6,
        help="Jumlah pasangan kecil untuk pengukuran waktu."
    )
    parser.add_argument(
        "--allow-start-charging",
        action="store_true",
        help="Izinkan top-up di SPKLU awal jika rute langsung tidak layak."
    )
    parser.add_argument("--reserve", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=("medan", "regional"),
        default="regional"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_OUTPUT_DIR
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiments(
        pair_count=arguments.pairs,
        repeats=arguments.repeats,
        timing_pair_count=arguments.timing_pairs,
        reserve_percentage=arguments.reserve,
        seed=arguments.seed,
        profile=arguments.profile,
        output_dir=arguments.output_dir,
        allow_start_charging=arguments.allow_start_charging
    )
