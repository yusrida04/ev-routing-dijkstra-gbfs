# EV Route Comparison

The project compares a Dijkstra baseline, GBFS Distance, and GBFS
Battery-Aware for the BYD Atto 3. Medan and intercity data are kept as
separate profiles so each charging-station `nearest_node` matches its graph.

## Regional SPKLU data

Build the intercity graph first if it does not exist, then map SPKLU records
from the local `data/raw/spklu_data.json` snapshot:

```powershell
.\.venv\Scripts\python.exe src\build_regional_graph.py
.\.venv\Scripts\python.exe src\build_regional_stations.py
```

The builder keeps locations inside the regional graph bounds and writes
separate regional directory, classified, and connected GeoJSON files. Snap
distances are retained in `snap_distance_m`; locations are not filtered by
distance to a road.

## Run the simulator and dashboard

```powershell
.\.venv\Scripts\python.exe src\simulator.py --profile medan
.\.venv\Scripts\python.exe src\simulator.py --profile regional
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard profile selector switches graph and SPKLU data together. It
includes the GBFS heuristic, lambda, energy consumption, reserve, and charging
target controls, plus Dijkstra and per-leg comparison tables.

## Run experiments

The default experiment samples at least 30 non-Samosir pairs and adds up to 30
Samosir-origin pairs as a separate cohort. Non-Samosir origins are balanced
across the selected pairs; Samosir is identified from station address/city
metadata and is not mixed into that origin balance. Distance strata use
Dijkstra road distance: near (<20 km), medium (20-80 km), and far (>80 km).
Quality is evaluated once per pair. By default, six pairs balanced across
cohort and available strata are repeated three times for timing. Use
`--timing-pairs N` to change the subset and `--repeats 3` through `--repeats 5`
to select the repetition count.

The comparison includes Dijkstra (tanpa pengisian), Dijkstra + SPKLU, GBFS
Distance, and GBFS Battery-Aware. Search time measures only calls to the search
function; per-leg Dijkstra comparisons are excluded. The summary groups by
`region_case` and road `distance_stratum`; distance deltas use Dijkstra +
SPKLU as the primary comparator and only matching successful pairs. The
quality and timing CSVs record `direct_search_limit_reached` separately from
overall `search_limit_reached`.

Battery-Aware has a 100,000-expansion budget for the current leg's direct and
station searches. When the direct search hits the limit, that result is
recorded as a direct-search limit and the expansion budget is renewed so the
SPKLU search can continue. If `--allow-start-charging` is enabled, a usable
SPKLU at the start is topped up to the target only when the direct route is
not feasible; this assumption and whether it was used are recorded in the
CSV. It is disabled by default. Straight-line estimates are reused within
each search, and reverse-Dijkstra uses a simple road graph with the minimum
parallel-edge length, preserving the shortest-distance heuristic. Quality,
timing, and summary CSVs are written to `data/processed/experiments`.

Days              : 0
Hours             : 1
Minutes           : 25
Seconds           : 7
Milliseconds      : 715
Ticks             : 51077152824
TotalDays         : 0.0591170750277778
TotalHours        : 1.41880980066667
TotalMinutes      : 85.12858804
TotalSeconds      : 5107.7152824
TotalMilliseconds : 5107715.2824

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py
.\.venv\Scripts\python.exe src\run_experiments.py --allow-start-charging
```