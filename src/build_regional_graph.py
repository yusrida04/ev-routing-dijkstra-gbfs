"""
Membangun graf jalan regional (Medan + sekitarnya) untuk skenario
rute antarkota. Hasil disimpan di data/raw/roads_antarkota.graphml.

Strategi:
1. Wilayah luas dibagi menjadi petak-petak kecil (jalan utama saja),
   diunduh satu per satu dengan percobaan ulang dan jeda otomatis,
   karena server publik Overpass sering penuh.
2. Dalam batas Kota Medan: semua jalan yang bisa dilalui mobil.
3. Semua graf digabung SEBELUM disederhanakan, supaya simpul di
   perbatasan petak dan perbatasan Medan tersambung dengan benar.

Petak yang sudah berhasil diunduh disimpan di folder cache/, sehingga
menjalankan ulang skrip tidak mengunduh petak yang sama lagi.

Jalankan dari folder utama proyek:
    python src/build_regional_graph.py
"""

import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import requests
from osmnx import _errors as ox_errors
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_BOUNDARY_PATH,
    REGIONAL_GRAPH_PATH
)

OUTPUT_PATH = REGIONAL_GRAPH_PATH

# (barat, selatan, timur, utara) dalam derajat.
# Mencakup Binjai, Berastagi, Parapat, Pematangsiantar, Tebing Tinggi.
REGION_BOUNDS = (98.35, 2.55, 99.30, 3.85)

# Wilayah dibagi menjadi TILE_ROWS x TILE_COLS petak.
TILE_ROWS = 3
TILE_COLS = 3

# Hapus "|tertiary" jika unduhan terlalu berat atau memori kurang.
MAJOR_ROADS = "motorway|trunk|primary|secondary|tertiary"

REGION_FILTER = (
    f'["highway"~"^({MAJOR_ROADS})(_link)?$"]'
    '["area"!~"yes"]["access"!~"private"]'
)

# Server Overpass bergantian dipakai jika satu server sibuk.
# Daftar mirror bisa berubah; server yang mati akan dilewati.
OVERPASS_SERVERS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]

MAX_ATTEMPTS = 8
FIRST_WAIT_SECONDS = 20
MAX_WAIT_SECONDS = 300

# Titik uji (lintang, bujur), perkiraan pusat kota.
CHECK_POINTS = {
    "Medan": (3.5896, 98.6738),
    "Berastagi": (3.1904, 98.5114),
    "Parapat": (2.6683, 98.9367),
    "Pematangsiantar": (2.9595, 99.0687),
    "Tebing Tinggi": (3.3280, 99.1625),
}

# Galat jaringan/server yang layak dicoba ulang.
RETRYABLE_ERRORS = (
    requests.exceptions.RequestException,
    getattr(
        ox_errors,
        "ResponseStatusCodeError",
        requests.exceptions.RequestException
    ),
)

# Galat "petak tidak berisi jalan sama sekali"; bukan kegagalan.
EMPTY_RESPONSE_ERROR = getattr(
    ox_errors, "InsufficientResponseError", ()
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_overpass_server(base_url):
    """Nama dan format pengaturan berbeda antar versi OSMnx."""
    current = (
        getattr(ox.settings, "overpass_url", None)
        or getattr(ox.settings, "overpass_endpoint", "")
    )

    suffix = (
        "/interpreter"
        if str(current).rstrip("/").endswith("/interpreter")
        else ""
    )

    url = base_url.rstrip("/") + suffix

    ox.settings.overpass_url = url
    ox.settings.overpass_endpoint = url


def with_retry(description, download):
    """
    Menjalankan fungsi unduhan dengan percobaan ulang, jeda yang
    makin lama, dan pergantian server. Mengembalikan None jika
    area memang tidak berisi jalan.
    """
    wait = FIRST_WAIT_SECONDS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        server = OVERPASS_SERVERS[
            (attempt - 1) % len(OVERPASS_SERVERS)
        ]
        set_overpass_server(server)

        try:
            return download()

        except EMPTY_RESPONSE_ERROR:
            log(f"    {description}: tidak ada jalan di area ini, "
                "dilewati")
            return None

        except RETRYABLE_ERRORS as error:
            reason = str(error).strip().splitlines()[0][:110]
            log(f"    GAGAL {description} "
                f"(percobaan {attempt}/{MAX_ATTEMPTS}): {reason}")

            if attempt == MAX_ATTEMPTS:
                raise

            log(f"    menunggu {wait} detik, lalu coba server lain...")
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT_SECONDS)


def get_medan_polygon():
    if not MEDAN_BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"{MEDAN_BOUNDARY_PATH} tidak ditemukan. "
            "Jalankan skrip dari folder utama proyek."
        )

    geometry = gpd.read_file(MEDAN_BOUNDARY_PATH).geometry

    if hasattr(geometry, "union_all"):
        return geometry.union_all()

    return geometry.unary_union


def make_tiles():
    west, south, east, north = REGION_BOUNDS

    width = (east - west) / TILE_COLS
    height = (north - south) / TILE_ROWS

    tiles = []

    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tiles.append(
                box(
                    west + col * width,
                    south + row * height,
                    west + (col + 1) * width,
                    south + (row + 1) * height
                )
            )

    return tiles


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    value = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(value))


def download_region():
    tiles = make_tiles()
    combined = None

    for number, polygon in enumerate(tiles, start=1):
        log(f"    petak {number}/{len(tiles)} ...")

        # truncate_by_edge=True menjaga jalan yang melintasi batas
        # petak, sehingga petak yang bersebelahan nyambung saat digabung.
        tile_graph = with_retry(
            f"petak {number}",
            lambda polygon=polygon: ox.graph_from_polygon(
                polygon,
                custom_filter=REGION_FILTER,
                simplify=False,
                retain_all=True,
                truncate_by_edge=True
            )
        )

        if tile_graph is None:
            continue

        log(f"      node: {len(tile_graph.nodes)} | "
            f"edge: {len(tile_graph.edges)}")

        combined = (
            tile_graph
            if combined is None
            else nx.compose(combined, tile_graph)
        )

    if combined is None:
        raise RuntimeError("Tidak ada data jalan yang berhasil diunduh.")

    return combined


def build_graph():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    # Nama pengaturan berbeda antar versi OSMnx; atur keduanya.
    ox.settings.timeout = 900
    ox.settings.requests_timeout = 900

    log("1/4 Mengunduh jalan utama wilayah regional per petak "
        "(bisa beberapa menit)...")
    region_graph = download_region()
    log(f"    total wilayah, node: {len(region_graph.nodes)} | "
        f"edge: {len(region_graph.edges)}")

    log("2/4 Mengunduh semua jalan di dalam Kota Medan...")
    medan_polygon = get_medan_polygon()
    medan_graph = with_retry(
        "Kota Medan",
        lambda: ox.graph_from_polygon(
            medan_polygon,
            network_type="drive",
            simplify=False,
            retain_all=True
        )
    )
    log(f"    node: {len(medan_graph.nodes)} | "
        f"edge: {len(medan_graph.edges)}")

    log("3/4 Menggabungkan dan menyederhanakan graf...")
    graph = nx.compose(region_graph, medan_graph)
    graph = ox.simplify_graph(graph)

    # Pertahankan hanya komponen yang saling terhubung terbesar,
    # supaya tidak ada jalan buntu terisolasi.
    largest = max(nx.strongly_connected_components(graph), key=len)
    before = len(graph.nodes)
    graph = graph.subgraph(largest).copy()
    log(f"    node akhir: {len(graph.nodes)} | "
        f"edge akhir: {len(graph.edges)} "
        f"(dibuang {before - len(graph.nodes)} node terisolasi)")

    log(f"4/4 Menyimpan ke {OUTPUT_PATH} ...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=OUTPUT_PATH)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    log(f"    selesai, ukuran file {size_mb:.1f} MB")

    return graph


def verify_graph(graph):
    """Cek wajar: jarak jalan antar kota vs garis lurus."""

    node_ids = list(graph.nodes)
    ys = np.array([graph.nodes[n]["y"] for n in node_ids])
    xs = np.array([graph.nodes[n]["x"] for n in node_ids])

    def nearest(lat, lon):
        dy = ys - lat
        dx = (xs - lon) * math.cos(math.radians(lat))
        return node_ids[int(np.argmin(dx * dx + dy * dy))]

    nodes = {
        name: nearest(lat, lon)
        for name, (lat, lon) in CHECK_POINTS.items()
    }

    print("\nPENGECEKAN JARAK DARI MEDAN")
    print(f"{'Tujuan':<18}{'Garis lurus':>13}{'Jarak jalan':>14}"
          f"{'Rasio':>8}")

    medan_lat, medan_lon = CHECK_POINTS["Medan"]

    for name, (lat, lon) in CHECK_POINTS.items():
        if name == "Medan":
            continue

        straight = haversine_km(medan_lat, medan_lon, lat, lon)

        try:
            road_m = nx.shortest_path_length(
                graph, nodes["Medan"], nodes[name], weight="length"
            )
            road_km = road_m / 1000
            print(f"{name:<18}{straight:>10.1f} km"
                  f"{road_km:>11.1f} km{road_km / straight:>8.2f}")
        except nx.NetworkXNoPath:
            print(f"{name:<18}{straight:>10.1f} km"
                  f"{'TIDAK ADA JALUR':>25}")

    print(
        "\nRasio wajar sekitar 1,1-1,6. Jauh di atas itu atau "
        "'TIDAK ADA JALUR' berarti ada jalan yang terputus."
    )


if __name__ == "__main__":
    graph = build_graph()
    verify_graph(graph)
