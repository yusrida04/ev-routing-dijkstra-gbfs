import csv
import tempfile
import unittest
from math import asin, radians, sin
from pathlib import Path
from time import sleep
from unittest.mock import patch

import networkx as nx
import pandas as pd

import src.run_experiments as experiment_runner
from src.algorithms.baseline_dijkstra import (
    dijkstra_search,
    simulate_baseline
)
from src.algorithms.gbfs_battery import (
    battery_aware_heuristic,
    greedy_best_first_search_battery
)
from src.algorithms.gbfs_distance import (
    clear_road_distance_cache,
    get_road_distances_to_goal
)
from src.run_experiments import (
    choose_pairs,
    choose_timing_pair_indices,
    summarize_rows,
    timing_summary_rows
)
from src.simulator import make_battery_search, simulate_ev_trip
from src.utils.energy import BATTERY_CAPACITY


class RouteComparisonTests(unittest.TestCase):
    def test_dijkstra_reports_distance_and_expanded_nodes(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 3, length=1000)
        graph.add_edge(0, 2, length=5000)
        graph.add_edge(2, 3, length=5000)

        result = dijkstra_search(graph, 0, 3)

        self.assertTrue(result["success"])
        self.assertEqual(result["route"], [0, 1, 3])
        self.assertEqual(result["distance_km"], 2)
        self.assertGreater(result["expanded"], 0)

    def test_baseline_is_labeled_as_no_charging(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, length=1000)

        result = simulate_baseline(
            graph,
            0,
            1,
            initial_battery_percentage=80,
            energy_consumption_kwh_per_km=0.1218
        )

        self.assertEqual(result["algorithm"], "Dijkstra (tanpa pengisian)")

    def test_collapsed_parallel_edges_preserve_reverse_road_distances(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, length=10)
        graph.add_edge(0, 1, length=5)
        graph.add_edge(1, 2, length=7)
        graph.add_edge(0, 2, length=20)

        clear_road_distance_cache()
        distances = get_road_distances_to_goal(graph, 2)
        clear_road_distance_cache()

        self.assertEqual(distances, {2: 0, 1: 7, 0: 12})

    def test_battery_heuristic_uses_straight_line_energy_estimate(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_edge(0, 1, length=2000)

        heuristic = battery_aware_heuristic(
            graph,
            0,
            1,
            usable_battery=0.05,
            lam=10,
            detour_factor=1.3,
            remaining_distance_km=2,
            energy_consumption_kwh_per_km=0.2
        )

        straight_line_km = 6371 * 2 * asin(sin(radians(0.01) / 2))
        estimated_energy = straight_line_km * 1.3 * 0.2
        expected = 2 + 10 * max(0, estimated_energy - 0.05)
        self.assertAlmostEqual(heuristic, expected, places=6)

    def test_charge_stops_at_target_and_records_leg_comparison(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.225, y=0.0)
        graph.add_node(2, x=0.45, y=0.0)
        graph.add_edge(0, 1, length=25000)
        graph.add_edge(1, 2, length=25000)
        stations = pd.DataFrame([{
            "name": "Charger",
            "address": "",
            "nearest_node": 1,
            "access_status": "UNKNOWN",
            "snap_distance_m": 0
        }])

        result = simulate_ev_trip(
            graph=graph,
            stations=stations,
            start_node=0,
            goal_node=2,
            initial_battery_percentage=20,
            target_charge_percentage=80,
            search_fn=dijkstra_search,
            reserve_percentage=10,
            algorithm_name="Dijkstra test",
            energy_consumption_kwh_per_km=0.1218
        )

        target_kwh = BATTERY_CAPACITY * 0.8
        self.assertTrue(result["success"])
        self.assertEqual(len(result["charging_stops"]), 1)
        self.assertTrue(
            result["charging_stops"][0]["progress_to_goal"]
        )
        self.assertLessEqual(
            result["charging_stops"][0]["battery_after_kwh"],
            target_kwh
        )
        self.assertEqual(len(result["leg_comparisons"]), 2)
        self.assertAlmostEqual(
            result["leg_comparisons"][0]["route_stretch"],
            1.0
        )

    def test_battery_search_stops_at_expansion_limit(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_node(2, x=0.02, y=0.0)
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 2, length=1000)

        result = greedy_best_first_search_battery(
            graph,
            0,
            2,
            initial_battery_percentage=80,
            max_expansions=1
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["expanded"], 1)
        self.assertTrue(result["max_expansions_reached"])
        self.assertEqual(result["failure_reason"], "batas_simpul")

    def test_battery_search_fails_before_reverse_search_if_out_of_range(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=1.0, y=0.0)

        with patch(
            "src.algorithms.gbfs_battery.get_road_distances_to_goal"
        ) as distance_search:
            result = greedy_best_first_search_battery(
                graph,
                0,
                1,
                initial_battery_percentage=10,
                energy_consumption_kwh_per_km=0.2
            )

        distance_search.assert_not_called()
        self.assertFalse(result["success"])
        self.assertEqual(result["expanded"], 0)
        self.assertEqual(
            result["failure_reason"],
            "tidak_ada_rute_layak"
        )

    def test_battery_search_prunes_neighbor_outside_remaining_range(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=1.0, y=0.0)
        graph.add_node(2, x=0.05, y=0.0)
        graph.add_node(3, x=0.1, y=0.0)
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 3, length=1000)
        graph.add_edge(0, 2, length=1000)
        graph.add_edge(2, 3, length=1000)

        result = greedy_best_first_search_battery(
            graph,
            0,
            3,
            initial_battery_percentage=10,
            lam=0,
            energy_consumption_kwh_per_km=0.1218
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["route"], [0, 2, 3])

    def test_battery_search_reuses_goal_distances_without_changing_route(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_node(2, x=0.02, y=0.0)
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 2, length=1000)

        from src.algorithms.gbfs_distance import haversine_distance

        with patch(
            "src.algorithms.gbfs_battery.haversine_distance",
            wraps=haversine_distance
        ) as distance:
            result = greedy_best_first_search_battery(
                graph,
                0,
                2,
                initial_battery_percentage=80,
                energy_consumption_kwh_per_km=0.1218
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["route"], [0, 1, 2])
        self.assertEqual(distance.call_count, 3)

    def test_battery_expansion_budget_covers_all_trip_searches(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_node(2, x=0.02, y=0.0)
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 2, length=1000)
        stations = pd.DataFrame([{
            "name": "Charger",
            "nearest_node": 1,
            "access_status": "UNKNOWN",
            "snap_distance_m": 0
        }])

        result = simulate_ev_trip(
            graph=graph,
            stations=stations,
            start_node=0,
            goal_node=2,
            initial_battery_percentage=80,
            search_fn=make_battery_search(max_expansions=1)
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["expanded"], 2)
        self.assertTrue(result["search_limit_reached"])
        self.assertTrue(result["direct_search_limit_reached"])
        self.assertEqual(result["failure_reason"], "batas_simpul")

    def test_direct_expansion_limit_falls_back_to_station_search(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_node(2, x=0.02, y=0.0)
        graph.add_edge(0, 1, length=1000)
        graph.add_edge(1, 2, length=1000)
        stations = pd.DataFrame([{
            "name": "Charger",
            "nearest_node": 1,
            "access_status": "UNKNOWN",
            "snap_distance_m": 0
        }])

        result = simulate_ev_trip(
            graph=graph,
            stations=stations,
            start_node=0,
            goal_node=2,
            initial_battery_percentage=80,
            search_fn=make_battery_search(max_expansions=2)
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["route"], [0, 1, 2])
        self.assertTrue(result["direct_search_limit_reached"])
        self.assertEqual(len(result["charging_stops"]), 1)

    def test_lower_bound_skips_impossible_targets_for_any_search_fn(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=1.0, y=0.0)
        graph.add_node(2, x=2.0, y=0.0)
        stations = pd.DataFrame([{
            "name": "Too far",
            "nearest_node": 2,
            "access_status": "UNKNOWN",
            "snap_distance_m": 0
        }])
        calls = []

        def search_fn(*args):
            calls.append(args)
            return {
                "success": False,
                "route": [],
                "distance_km": 0,
                "expanded": 0
            }

        result = simulate_ev_trip(
            graph=graph,
            stations=stations,
            start_node=0,
            goal_node=1,
            initial_battery_percentage=10,
            search_fn=search_fn
        )

        self.assertFalse(result["success"])
        self.assertEqual(calls, [])
        self.assertEqual(result["expanded"], 0)
        self.assertEqual(
            result["failure_reason"],
            "spklu_tidak_terjangkau"
        )

    def test_computation_time_excludes_leg_comparison(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_edge(0, 1, length=1000)
        stations = pd.DataFrame(columns=[
            "name",
            "nearest_node",
            "access_status",
            "snap_distance_m"
        ])

        def search_fn(*args):
            sleep(0.02)
            return {
                "success": True,
                "route": [0, 1],
                "distance_km": 1,
                "expanded": 2
            }

        def slow_comparison(*args):
            sleep(0.05)
            return dijkstra_search(*args)

        with patch("src.simulator.dijkstra_search", slow_comparison):
            result = simulate_ev_trip(
                graph=graph,
                stations=stations,
                start_node=0,
                goal_node=1,
                initial_battery_percentage=80,
                search_fn=search_fn
            )

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["computation_time_s"], 0.02)
        self.assertLess(result["computation_time_s"], 0.05)
        self.assertGreaterEqual(result["comparison_time_s"], 0.05)

    def test_experiment_pairs_cover_all_distance_strata(self):
        graph = nx.MultiDiGraph()
        longitudes = (0.0, 0.01, 0.1, 0.4, 1.2, 1.3)
        for node, longitude in enumerate(longitudes):
            graph.add_node(node, x=longitude, y=0.0)
        for start in range(len(longitudes)):
            for goal in range(len(longitudes)):
                if start != goal:
                    road_distance_m = (
                        abs(longitudes[goal] - longitudes[start]) * 222_000
                    )
                    graph.add_edge(
                        start,
                        goal,
                        length=road_distance_m
                    )
        stations = pd.DataFrame([
            {
                "name": f"Station {node}",
                "nearest_node": node,
                "access_status": "UNKNOWN"
            }
            for node in range(len(longitudes))
        ])

        pairs = choose_pairs(graph, stations, pair_count=3, seed=7)

        self.assertEqual(
            {pair[2] for pair in pairs},
            {"near", "medium", "far"}
        )
        self.assertTrue(
            all(pair[3] > pair[4] for pair in pairs)
        )
        self.assertEqual(
            len({int(pair[0]["nearest_node"]) for pair in pairs}),
            len(pairs)
        )

    def test_samosir_gets_a_separate_additional_cohort(self):
        graph = nx.MultiDiGraph()
        longitudes = (0.0, 0.01, 0.1, 0.4, 1.2, 1.3, 1.4)
        for node, longitude in enumerate(longitudes):
            graph.add_node(node, x=longitude, y=0.0)
        for start in range(len(longitudes)):
            for goal in range(len(longitudes)):
                if start != goal:
                    graph.add_edge(
                        start,
                        goal,
                        length=abs(longitudes[goal] - longitudes[start])
                        * 222_000
                    )
        stations = pd.DataFrame([
            {
                "name": "SPKLU Samosir" if node == 0 else f"Station {node}",
                "address": (
                    "Kabupaten Samosir"
                    if node == 0 else "Sumatera Utara"
                ),
                "nearest_node": node,
                "access_status": "UNKNOWN"
            }
            for node in range(len(longitudes))
        ])

        pairs = choose_pairs(graph, stations, pair_count=3, seed=13)

        samosir = [
            pair for pair in pairs
            if pair[5] == "samosir_origin"
        ]
        regular = [
            pair for pair in pairs
            if pair[5] == "non_samosir"
        ]
        self.assertEqual(len(regular), 3)
        self.assertEqual(len(samosir), 3)
        self.assertEqual(
            {int(pair[0]["nearest_node"]) for pair in samosir},
            {0}
        )

    def test_start_station_charges_only_when_direct_route_is_infeasible(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.45, y=0.0)
        graph.add_edge(0, 1, length=50000)
        stations = pd.DataFrame([{
            "name": "Start charger",
            "nearest_node": 0,
            "access_status": "UNKNOWN",
            "snap_distance_m": 0
        }])

        result = simulate_ev_trip(
            graph=graph,
            stations=stations,
            start_node=0,
            goal_node=1,
            initial_battery_percentage=20,
            target_charge_percentage=80,
            search_fn=dijkstra_search,
            reserve_percentage=10,
            energy_consumption_kwh_per_km=0.1218,
            allow_start_charging=True
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["charged_at_start"])
        self.assertEqual(len(result["charging_stops"]), 1)
        self.assertTrue(result["charging_stops"][0]["is_start_station"])

    def test_summary_means_and_paired_deltas_use_successful_runs(self):
        def row(
            algorithm,
            pair_index,
            success,
            distance,
            energy,
            computation_time
        ):
            return {
                "algorithm": algorithm,
                "pair_index": pair_index,
                "battery_percentage": 50,
                "energy_consumption_kwh_per_km": 0.15,
                "reserve_percentage": 10,
                "lambda": 10 if algorithm == "GBFS Battery-Aware" else "",
                "region_case": "non_samosir",
                "start_charging_assumption": "disabled",
                "distance_stratum": "near",
                "repeat": 1,
                "success": success,
                "distance_km": distance,
                "energy_used_kwh": energy,
                "computation_time_s": computation_time,
                "expanded": 100
            }

        rows = [
            row("Dijkstra + SPKLU", 1, True, 10, 1, 0.2),
            row("Dijkstra + SPKLU", 2, False, 0, 0, 0.3),
            row("Dijkstra + SPKLU", 3, False, 0, 0, 0.4),
            row("GBFS Battery-Aware", 1, True, 12, 1.2, 0.5),
            row("GBFS Battery-Aware", 2, False, 900, 90, 9),
            row("GBFS Battery-Aware", 3, True, 20, 2, 0.7)
        ]

        summary = summarize_rows(rows)
        battery_summary = next(
            item for item in summary
            if item["algorithm"] == "GBFS Battery-Aware"
        )

        self.assertEqual(battery_summary["n_success"], 2)
        self.assertEqual(battery_summary["success_rate"], 2 / 3)
        self.assertEqual(battery_summary["mean_distance_km"], 16)
        self.assertEqual(battery_summary["mean_energy_used_kwh"], 1.6)
        self.assertAlmostEqual(
            battery_summary["mean_computation_time_s"],
            0.6
        )
        self.assertEqual(battery_summary["n_distance_comparisons"], 1)
        self.assertEqual(
            battery_summary[
                "mean_distance_delta_vs_dijkstra_spklu_km"
            ],
            2
        )

    def test_summary_separates_distance_strata_and_uses_timing_subset(self):
        def row(algorithm, stratum, pair_index, distance):
            return {
                "algorithm": algorithm,
                "pair_index": pair_index,
                "battery_percentage": 50,
                "energy_consumption_kwh_per_km": 0.15,
                "reserve_percentage": 10,
                "lambda": "",
                "distance_stratum": stratum,
                "success": True,
                "distance_km": distance,
                "energy_used_kwh": 1,
                "computation_time_s": "",
                "expanded": 10
            }

        quality_rows = [
            row("Dijkstra + SPKLU", "near", 1, 10),
            row("GBFS Distance", "near", 1, 12),
            row("Dijkstra + SPKLU", "far", 2, 100),
            row("GBFS Distance", "far", 2, 110)
        ]
        timing_rows = [{
            "algorithm": "GBFS Distance",
            "region_case": "non_samosir",
            "distance_stratum": "far",
            "start_charging_assumption": "disabled",
            "battery_percentage": 50,
            "energy_consumption_kwh_per_km": 0.15,
            "reserve_percentage": 10,
            "lambda": "",
            "success": True,
            "computation_time_s": value
        } for value in (1.0, 3.0)]

        summaries = timing_summary_rows(quality_rows, timing_rows)
        far_summary = next(
            item for item in summaries
            if item["algorithm"] == "GBFS Distance"
            and item["distance_stratum"] == "far"
        )
        near_summary = next(
            item for item in summaries
            if item["algorithm"] == "GBFS Distance"
            and item["distance_stratum"] == "near"
        )

        self.assertEqual(far_summary["runs"], 1)
        self.assertEqual(far_summary["n_timing_runs"], 2)
        self.assertEqual(far_summary["mean_computation_time_s"], 2.0)
        self.assertIsNone(near_summary["mean_computation_time_s"])
        self.assertEqual(len(choose_timing_pair_indices(
            [(None, None, "near", 10), (None, None, "far", 100)],
            timing_pair_count=1,
            seed=1
        )), 1)

    def test_experiments_separate_quality_from_subset_timing(self):
        def result(algorithm, elapsed=0.1):
            return {
                "algorithm": algorithm,
                "success": True,
                "distance_km": 10,
                "energy_used_kwh": 1,
                "charged_kwh": 0,
                "charging_stops": [],
                "expanded": 2,
                "computation_time_s": elapsed,
                "comparison_time_s": 0,
                "failure_reason": None,
                "search_limit_reached": False
            }

        starts = [
            pd.Series({
                "name": f"Start {i}",
                "nearest_node": i * 2,
                "snap_distance_m": 0
            })
            for i in range(2)
        ]
        goals = [
            pd.Series({
                "name": f"Goal {i}",
                "nearest_node": i * 2 + 1,
                "snap_distance_m": 0
            })
            for i in range(2)
        ]
        pairs = [
            (starts[0], goals[0], "near", 10, 8, "non_samosir"),
            (starts[1], goals[1], "far", 100, 90, "non_samosir")
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "graph.graphml"
            stations_path = Path(temp_dir) / "stations.geojson"
            graph_path.touch()
            stations_path.touch()
            output_dir = Path(temp_dir) / "results"

            with (
                patch.object(experiment_runner, "REGIONAL_GRAPH_PATH", graph_path),
                patch.object(
                    experiment_runner,
                    "REGIONAL_STATIONS_CONNECTED_PATH",
                    stations_path
                ),
                patch.object(
                    experiment_runner.ox,
                    "load_graphml",
                    return_value=nx.MultiDiGraph()
                ),
                patch.object(
                    experiment_runner.gpd,
                    "read_file",
                    return_value=pd.DataFrame()
                ),
                patch.object(
                    experiment_runner,
                    "choose_pairs",
                    return_value=pairs
                ),
                patch.object(experiment_runner, "BATTERY_LEVELS", (50,)),
                patch.object(experiment_runner, "ENERGY_CONSUMPTIONS", (0.15,)),
                patch.object(experiment_runner, "LAMBDA_VALUES", (10,)),
                patch.object(
                    experiment_runner,
                    "simulate_baseline",
                    return_value=result("Dijkstra (tanpa pengisian)")
                ),
                patch.object(
                    experiment_runner,
                    "simulate_ev_trip",
                    side_effect=lambda **kwargs: result(
                        kwargs["algorithm_name"]
                    )
                )
            ):
                run_path, summary_path = experiment_runner.run_experiments(
                    pair_count=30,
                    repeats=3,
                    timing_pair_count=1,
                    output_dir=output_dir
                )

            with run_path.open(encoding="utf-8-sig") as file:
                quality_rows = list(csv.DictReader(file))
            timing_path = output_dir / "experiment_regional_timing.csv"
            with timing_path.open(encoding="utf-8-sig") as file:
                timing_rows = list(csv.DictReader(file))
            with summary_path.open(encoding="utf-8-sig") as file:
                summaries = list(csv.DictReader(file))

        self.assertEqual(len(quality_rows), 8)
        self.assertTrue(all(not row["computation_time_s"] for row in quality_rows))
        self.assertEqual(len(timing_rows), 12)
        self.assertEqual(len({row["pair_index"] for row in timing_rows}), 1)
        self.assertEqual(
            {row["distance_stratum"] for row in summaries},
            {"near", "far"}
        )


if __name__ == "__main__":
    unittest.main()
