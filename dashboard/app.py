import sys
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium

from streamlit_folium import st_folium

from src.algorithms.baseline_dijkstra import (
    dijkstra_search,
    simulate_baseline
)
from src.simulator import (
    distance_search,
    make_battery_search,
    simulate_ev_trip
)
from src.utils.energy import BATTERY_CAPACITY
from src.project_paths import (
    MEDAN_GRAPH_PATH,
    MEDAN_BOUNDARY_PATH,
    MEDAN_STATIONS_CONNECTED_PATH,
    REGIONAL_GRAPH_PATH,
    REGIONAL_STATIONS_CONNECTED_PATH
)


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Electric Vehicle Route Optimizer",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Electric Vehicle Route Optimizer")

st.write(
    "Optimasi rute kendaraan listrik berbasis "
    "Dijkstra dan Greedy Best-First Search."
)

profile_name = st.sidebar.selectbox(
    "Graf dan data SPKLU",
    ["Medan", "Regional (Antarkota)"]
)
if profile_name == "Regional (Antarkota)":
    selected_graph_path = REGIONAL_GRAPH_PATH
    selected_stations_path = REGIONAL_STATIONS_CONNECTED_PATH
    selected_boundary_path = None
else:
    selected_graph_path = MEDAN_GRAPH_PATH
    selected_stations_path = MEDAN_STATIONS_CONNECTED_PATH
    selected_boundary_path = MEDAN_BOUNDARY_PATH


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_resource
def load_graph(path):
    return ox.load_graphml(path)


@st.cache_data
def load_stations(path):
    return gpd.read_file(
        path
    ).reset_index(drop=True)


@st.cache_data
def load_boundary(path):
    if path and Path(path).exists():
        return gpd.read_file(path).to_crs("EPSG:4326")

    return None


try:
    graph = load_graph(str(selected_graph_path))
    stations = load_stations(str(selected_stations_path))
    boundary = load_boundary(
        str(selected_boundary_path) if selected_boundary_path else ""
    )
    graph_bounds = (
        min(float(data["x"]) for _, data in graph.nodes(data=True)),
        min(float(data["y"]) for _, data in graph.nodes(data=True)),
        max(float(data["x"]) for _, data in graph.nodes(data=True)),
        max(float(data["y"]) for _, data in graph.nodes(data=True))
    )

except Exception as error:
    st.error(f"Gagal memuat data: {error}")
    st.stop()


if stations.empty:
    st.error("Data SPKLU tidak ditemukan.")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "start_location" not in st.session_state:
    st.session_state.start_location = None

if "goal_location" not in st.session_state:
    st.session_state.goal_location = None

if "start_node" not in st.session_state:
    st.session_state.start_node = None

if "goal_node" not in st.session_state:
    st.session_state.goal_node = None

if "start_snap" not in st.session_state:
    st.session_state.start_snap = 0

if "goal_snap" not in st.session_state:
    st.session_state.goal_snap = 0

if "map_version" not in st.session_state:
    st.session_state.map_version = 0

if "results" not in st.session_state:
    st.session_state.results = None

if "active_profile" not in st.session_state:
    st.session_state.active_profile = profile_name
elif st.session_state.active_profile != profile_name:
    st.session_state.active_profile = profile_name
    st.session_state.start_location = None
    st.session_state.goal_location = None
    st.session_state.start_node = None
    st.session_state.goal_node = None
    st.session_state.start_snap = 0
    st.session_state.goal_snap = 0
    st.session_state.results = None
    st.session_state.map_version += 1


# ==========================================
# FUNGSI BANTU
# ==========================================

def haversine_coordinates(lat1, lon1, lat2, lon2):

    radius = 6371000

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(delta_lon / 2) ** 2
    )

    return 2 * radius * atan2(
        sqrt(a),
        sqrt(1 - a)
    )


def get_nearest_node(lat, lon):

    node = ox.distance.nearest_nodes(
        graph,
        X=lon,
        Y=lat
    )

    node_data = graph.nodes[node]

    distance = haversine_coordinates(
        lat,
        lon,
        node_data["y"],
        node_data["x"]
    )

    return int(node), distance


def is_inside_graph(lat, lon):
    west, south, east, north = graph_bounds
    return west <= lon <= east and south <= lat <= north


def add_station_markers(map_object):

    for _, station in stations.iterrows():

        if station.geometry is None:
            continue

        lat = station.geometry.y
        lon = station.geometry.x

        status = station.get("access_status", "UNKNOWN")

        color = (
            "red"
            if status == "RESTRICTED"
            else "orange"
        )

        popup = f"""
        <b>{station.get('name', 'SPKLU')}</b><br>
        {station.get('address', '')}<br>
        Status akses: {status}<br>
        Catatan: {station.get('access_note', '')}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(map_object)

    legend = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px;">
      <b>Legenda SPKLU</b><br>
      <span style="color: orange;">●</span> Akses belum terverifikasi<br>
      <span style="color: red;">●</span> Akses terbatas<br>
      <span style="color: blue;">●</span> SPKLU dipilih
    </div>
    """
    map_object.get_root().html.add_child(
        folium.Element(legend)
    )


def create_base_map():

    west, south, east, north = graph_bounds
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        control_scale=True
    )

    if boundary is not None:
        folium.GeoJson(
            boundary,
            name="Batas Kota Medan",
            style_function=lambda feature: {
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.03
            }
        ).add_to(map_object)

    add_station_markers(map_object)

    return map_object


def add_selected_locations(map_object):

    if st.session_state.start_location:

        lat, lon = st.session_state.start_location

        folium.Marker(
            [lat, lon],
            tooltip="Lokasi Awal",
            popup="Lokasi Awal",
            icon=folium.Icon(
                color="green",
                icon="play"
            )
        ).add_to(map_object)

    if st.session_state.goal_location:

        lat, lon = st.session_state.goal_location

        folium.Marker(
            [lat, lon],
            tooltip="Lokasi Tujuan",
            popup="Lokasi Tujuan",
            icon=folium.Icon(
                color="red",
                icon="flag"
            )
        ).add_to(map_object)


def create_route_map(route, charging_stops):

    map_object = create_base_map()

    route_coordinates = []

    for node in route:

        data = graph.nodes[node]

        route_coordinates.append([
            data["y"],
            data["x"]
        ])

    if route_coordinates:

        folium.PolyLine(
            route_coordinates,
            color="blue",
            weight=5,
            opacity=0.8,
            tooltip="Rute kendaraan listrik"
        ).add_to(map_object)

    add_selected_locations(map_object)

    # Marker lokasi SPKLU yang digunakan untuk pengisian.
    for stop in charging_stops:

        station = stations[
            stations["nearest_node"].astype(int)
            == int(stop["nearest_node"])
        ]

        if station.empty:
            continue

        point = station.iloc[0].geometry

        folium.Marker(
            [point.y, point.x],
            tooltip=f"SPKLU: {stop['name']}",
            popup=(
                f"<b>{stop['name']}</b><br>"
                f"Pengisian: {stop['charged_kwh']:.2f} kWh<br>"
                f"Status akses: {stop['access_status']}"
            ),
            icon=folium.Icon(
                color="blue",
                icon="bolt",
                prefix="fa"
            )
        ).add_to(map_object)

    return map_object


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Pengaturan Kendaraan")

st.sidebar.subheader("BYD Atto 3")

st.sidebar.write(
    f"Kapasitas baterai: {BATTERY_CAPACITY:.2f} kWh"
)

energy_consumption = st.sidebar.select_slider(
    "Konsumsi energi (kWh/km)",
    options=[0.1218, 0.15, 0.18],
    value=0.1218,
    format_func=lambda value: f"{value:.4f}"
)

battery_percentage = st.sidebar.select_slider(
    "Kondisi baterai awal",
    options=[1, 5, 10, 20, 50, 80, 100],
    value=80,
    format_func=lambda value: f"{value}%"
)

st.sidebar.caption(
    "0,1218 adalah estimasi NEDC; nilai lebih tinggi "
    "mensimulasikan konsumsi yang lebih berat."
)

selected_heuristic = st.sidebar.selectbox(
    "Heuristik GBFS utama",
    ["GBFS Distance", "GBFS Battery-Aware"]
)

lambda_value = st.sidebar.select_slider(
    "Lambda (km/kWh)",
    options=[1, 5, 10, 20],
    value=10,
    disabled=selected_heuristic != "GBFS Battery-Aware"
)

reserve_percentage = st.sidebar.slider(
    "Cadangan baterai minimum (%)",
    min_value=0,
    max_value=30,
    value=10
)

target_charge_percentage = st.sidebar.slider(
    "Target pengisian SPKLU (%)",
    min_value=20,
    max_value=100,
    value=80
)

if target_charge_percentage <= reserve_percentage:
    st.sidebar.warning(
        "Target pengisian harus lebih tinggi dari cadangan baterai."
    )


# ==========================================
# PEMILIHAN LOKASI
# ==========================================

st.subheader("📍 Pilih Lokasi Perjalanan")

st.write(
    "Pilih jenis lokasi yang ingin ditentukan, "
    "kemudian klik titik pada peta."
)

selection_mode = st.radio(
    "Lokasi yang ingin dipilih:",
    ["Lokasi Awal", "Lokasi Tujuan"],
    horizontal=True
)

if selection_mode == "Lokasi Awal":
    st.info("Klik peta untuk menentukan lokasi awal.")
else:
    st.info("Klik peta untuk menentukan lokasi tujuan.")


# Buat peta pemilihan lokasi.
selection_map = create_base_map()

add_selected_locations(selection_map)

map_result = st_folium(
    selection_map,
    width=None,
    height=550,
    key=f"location_picker_{st.session_state.map_version}"
)


# ==========================================
# MEMPROSES KLIK PETA
# ==========================================

if map_result and map_result.get("last_clicked"):

    clicked = map_result["last_clicked"]

    lat = float(clicked["lat"])
    lon = float(clicked["lng"])

    if not is_inside_graph(lat, lon):

        st.warning(
            "Lokasi yang dipilih berada di luar cakupan graf jalan terpilih."
        )

    else:

        node, snap_distance = get_nearest_node(
            lat,
            lon
        )

        if selection_mode == "Lokasi Awal":

            st.session_state.start_location = (lat, lon)
            st.session_state.start_node = node
            st.session_state.start_snap = snap_distance

        else:

            st.session_state.goal_location = (lat, lon)
            st.session_state.goal_node = node
            st.session_state.goal_snap = snap_distance

        # Hapus hasil lama karena lokasi berubah.
        st.session_state.results = None

        # Ganti key peta agar klik sebelumnya tidak diproses ulang.
        st.session_state.map_version += 1

        st.rerun()


# ==========================================
# INFORMASI LOKASI
# ==========================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🟢 Lokasi Awal")

    if st.session_state.start_location:

        lat, lon = st.session_state.start_location

        st.success(
            f"Koordinat: {lat:.6f}, {lon:.6f}"
        )

        st.caption(
            f"Jarak ke simpul jalan terdekat: "
            f"{st.session_state.start_snap:.2f} meter"
        )

    else:
        st.warning("Lokasi awal belum dipilih.")


with col2:

    st.markdown("### 🔴 Lokasi Tujuan")

    if st.session_state.goal_location:

        lat, lon = st.session_state.goal_location

        st.success(
            f"Koordinat: {lat:.6f}, {lon:.6f}"
        )

        st.caption(
            f"Jarak ke simpul jalan terdekat: "
            f"{st.session_state.goal_snap:.2f} meter"
        )

    else:
        st.warning("Lokasi tujuan belum dipilih.")


if st.button("Hapus Lokasi", use_container_width=True):

    st.session_state.start_location = None
    st.session_state.goal_location = None

    st.session_state.start_node = None
    st.session_state.goal_node = None

    st.session_state.start_snap = 0
    st.session_state.goal_snap = 0

    st.session_state.results = None
    st.session_state.map_version += 1

    st.rerun()


# ==========================================
# PENCARIAN RUTE
# ==========================================

st.divider()

if st.button(
    "🔍 Cari Rute",
    type="primary",
    use_container_width=True,
    disabled=target_charge_percentage <= reserve_percentage
):

    if (
        st.session_state.start_node is None
        or st.session_state.goal_node is None
    ):

        st.warning(
            "Silakan pilih lokasi awal dan lokasi tujuan terlebih dahulu."
        )

    elif st.session_state.start_node == st.session_state.goal_node:

        st.warning(
            "Lokasi awal dan tujuan terlalu berdekatan "
            "karena terhubung ke simpul jalan yang sama. "
            "Silakan pilih titik yang berbeda."
        )

    else:

        start_node = st.session_state.start_node
        goal_node = st.session_state.goal_node

        with st.spinner("Sedang mencari rute terbaik..."):
            simulation_arguments = {
                "graph": graph,
                "stations": stations,
                "start_node": start_node,
                "goal_node": goal_node,
                "initial_battery_percentage": battery_percentage,
                "target_charge_percentage": target_charge_percentage,
                "start_snap_distance_m": st.session_state.start_snap,
                "goal_snap_distance_m": st.session_state.goal_snap,
                "reserve_percentage": reserve_percentage,
                "energy_consumption_kwh_per_km": energy_consumption
            }

            baseline_result = simulate_baseline(
                graph=graph,
                start_node=start_node,
                goal_node=goal_node,
                initial_battery_percentage=battery_percentage,
                start_snap_distance_m=st.session_state.start_snap,
                goal_snap_distance_m=st.session_state.goal_snap,
                reserve_percentage=reserve_percentage,
                energy_consumption_kwh_per_km=energy_consumption
            )
            dijkstra_charging_result = simulate_ev_trip(
                **simulation_arguments,
                search_fn=dijkstra_search,
                algorithm_name="Dijkstra + SPKLU"
            )
            distance_result = simulate_ev_trip(
                **simulation_arguments,
                search_fn=distance_search,
                algorithm_name="GBFS Distance"
            )
            battery_result = simulate_ev_trip(
                **simulation_arguments,
                search_fn=make_battery_search(
                    lam=lambda_value,
                    energy_consumption_kwh_per_km=energy_consumption
                ),
                algorithm_name="GBFS Battery-Aware"
            )

        st.session_state.results = {
            "baseline": baseline_result,
            "dijkstra_charging": dijkstra_charging_result,
            "distance": distance_result,
            "battery": battery_result,
            "selected_heuristic": selected_heuristic
        }


# ==========================================
# HASIL PENCARIAN
# ==========================================

if st.session_state.results:

    results = st.session_state.results

    distance_result = results["distance"]
    battery_result = results["battery"]
    simulation_result = (
        battery_result
        if results["selected_heuristic"] == "GBFS Battery-Aware"
        else distance_result
    )
    baseline_result = results["baseline"]
    dijkstra_charging_result = results["dijkstra_charging"]

    st.subheader("📊 Hasil Pencarian Rute")

    if distance_result["success"]:

        distance_total = distance_result["distance_km"]
        distance_energy = distance_result["energy_used_kwh"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total jarak — GBFS Distance",
            f"{distance_total:.2f} km"
        )

        col2.metric(
            "Estimasi Energi",
            f"{distance_energy:.2f} kWh"
        )

        col3.metric(
            "Simpul Dikembangkan",
            distance_result["expanded"]
        )
        st.caption(
            f"Waktu komputasi: "
            f"{distance_result['computation_time_s']:.3f} detik"
        )

    else:

        st.error(
            "GBFS Distance-Based tidak menemukan rute."
        )


    st.divider()

    if battery_result["success"]:

        battery_distance = battery_result["distance_km"]
        battery_energy = battery_result["energy_used_kwh"]
        remaining_battery = battery_result["remaining_battery_kwh"]

        remaining_percentage = (
            remaining_battery / BATTERY_CAPACITY
        ) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total jarak — GBFS Battery-Aware",
            f"{battery_distance:.2f} km"
        )

        col2.metric(
            "Estimasi Energi Terpakai",
            f"{battery_energy:.2f} kWh"
        )

        col3.metric(
            "Sisa Baterai",
            f"{remaining_percentage:.2f}%"
        )
        st.caption(
            f"Waktu komputasi: "
            f"{battery_result['computation_time_s']:.3f} detik "
            f"| lambda={lambda_value}"
        )

    else:

        st.error(
            "GBFS Battery-Aware tidak menemukan rute."
        )


    st.divider()

    st.subheader(
        f"🔌 Simulasi Pengisian — {results['selected_heuristic']}"
    )

    if simulation_result["success"]:

        remaining_percentage = (
            simulation_result["remaining_battery_kwh"]
            / BATTERY_CAPACITY
        ) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Jarak",
            f"{simulation_result['distance_km']:.2f} km"
        )

        col2.metric(
            "Energi Terpakai",
            f"{simulation_result['energy_used_kwh']:.2f} kWh"
        )

        col3.metric(
            "Energi Pengisian",
            f"{simulation_result['charged_kwh']:.2f} kWh"
        )

        col4.metric(
            "Sisa Baterai",
            f"{remaining_percentage:.2f}%"
        )

        st.success(
            "Perjalanan berhasil disimulasikan."
        )

        if simulation_result["charging_stops"]:

            st.markdown("#### Daftar Pemberhentian SPKLU")

            charging_data = []

            for index, stop in enumerate(
                simulation_result["charging_stops"],
                start=1
            ):

                charging_data.append({
                    "Pemberhentian": index,
                    "Nama SPKLU": stop["name"],
                    "Status Akses": stop["access_status"],
                    "Mendekati tujuan": (
                        "Ya" if stop["progress_to_goal"] else "Tidak"
                    ),
                    "Jarak ke tujuan (km)": round(
                        stop["distance_to_goal_km"], 2
                    ),
                    "Energi Ditambahkan (kWh)": round(
                        stop["charged_kwh"], 2
                    ),
                    "Baterai Setelah Pengisian (kWh)": round(
                        stop["battery_after_kwh"], 2
                    )
                })

            st.dataframe(
                charging_data,
                use_container_width=True,
                hide_index=True
            )

            if any(
                stop["access_status"] == "UNKNOWN"
                for stop in simulation_result["charging_stops"]
            ):

                st.warning(
                    "Terdapat SPKLU berstatus UNKNOWN. "
                    "Akses publiknya belum terverifikasi, "
                    "sehingga pengguna perlu memastikan akses sebelum berkunjung."
                )

        else:

            st.info(
                "Kendaraan dapat mencapai tujuan tanpa "
                "melakukan pengisian ulang."
            )

    else:

        st.error(
            "Perjalanan tidak dapat diselesaikan dengan "
            "kondisi baterai dan SPKLU yang tersedia."
        )

        st.write(
            f"Sisa baterai: "
            f"{simulation_result['remaining_battery_kwh']:.2f} kWh"
        )

        st.write(
            f"Jumlah pemberhentian: "
            f"{len(simulation_result['charging_stops'])}"
        )

    if simulation_result.get("search_limit_reached", False):
        st.warning(
            "Batas 100.000 ekspansi tercapai pada setidaknya satu "
            "subpencarian Battery-Aware; rute akhir masih dapat ditemukan "
            "melalui pencarian kandidat lain."
        )


    # ======================================
    # TABEL PERBANDINGAN
    # ======================================

    st.divider()

    st.subheader("⚖️ Perbandingan Algoritma")

    comparison = []

    for result in (
        baseline_result,
        dijkstra_charging_result,
        distance_result,
        battery_result
    ):
        comparison.append({
            "Algoritma": result["algorithm"],
            "Status": "Berhasil" if result["success"] else "Gagal",
            "Jarak (km)": round(result["distance_km"], 2),
            "Energi isi (kWh)": round(result["charged_kwh"], 2),
            "Jumlah SPKLU": len(result["charging_stops"]),
            "Simpul dikembangkan": result["expanded"],
            "Waktu (detik)": round(
                result["computation_time_s"], 3
            ),
            "Batas ekspansi": result.get(
                "search_limit_reached",
                result.get("max_expansions_reached", False)
            )
        })

    if comparison:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        f"#### Perbandingan per leg — {simulation_result['algorithm']}"
    )
    if simulation_result["leg_comparisons"]:
        leg_rows = []
        for leg_number, leg in enumerate(
            simulation_result["leg_comparisons"],
            start=1
        ):
            leg_rows.append({
                "Leg": leg_number,
                "Algoritma": f"{leg['algorithm_distance_km']:.2f} km",
                "Dijkstra": f"{leg['dijkstra_distance_km']:.2f} km",
                "Rasio jarak": (
                    f"{leg['route_stretch']:.3f}"
                    if leg["route_stretch"] is not None
                    else "—"
                ),
                "Layak (algoritma)": leg["algorithm_feasible"],
                "Layak (Dijkstra)": leg["dijkstra_feasible"],
                "Expanded (algoritma)": leg["algorithm_expanded"],
                "Expanded (Dijkstra)": leg["dijkstra_expanded"]
            })
        st.dataframe(
            leg_rows,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Tidak ada leg yang berhasil diselesaikan untuk dibandingkan.")


    # ======================================
    # PETA HASIL RUTE
    # ======================================

    st.divider()

    st.subheader("🗺️ Visualisasi Rute")

    if simulation_result["success"]:

        selected_route = simulation_result["route"]

    elif battery_result["success"]:

        selected_route = battery_result["route"]

    elif distance_result["success"]:

        selected_route = distance_result["route"]

    else:

        selected_route = []

    if selected_route:

        route_map = create_route_map(
            selected_route,
            simulation_result["charging_stops"]
        )

        st_folium(
            route_map,
            width=None,
            height=550,
            returned_objects=[],
            key="route_result_map"
        )

    st.caption(
        "Catatan: Hasil memakai graf dan data SPKLU yang sesuai dengan "
        f"profil {profile_name}. "
        "Model belum memperhitungkan kondisi lalu lintas, "
        "kemiringan jalan, antrean SPKLU, dan karakteristik "
        "kurva pengisian baterai. "
        "Cadangan baterai diperlakukan sebagai batas minimum. GBFS merupakan metode heuristik "
        "dan tidak menjamin rute optimal."
    )


else:

    st.info(
        "Tentukan lokasi awal dan tujuan pada peta, "
        "atur kondisi baterai, kemudian tekan tombol Cari Rute."
    )
