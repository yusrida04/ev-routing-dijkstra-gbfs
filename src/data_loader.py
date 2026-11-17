import sys
import osmnx as ox
import json
import geopandas as gpd

from urllib.request import Request, urlopen
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (
    MEDAN_BOUNDARY_PATH,
    MEDAN_GRAPH_PATH,
    MEDAN_STATIONS_DIRECTORY_PATH,
    SPKLU_SOURCE_PATH
)


def download_medan_roads():
    print("Memulai pengunduhan jaringan jalan Kota Medan...")

    # Lokasi penyimpanan data
    output_path = MEDAN_GRAPH_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mengaktifkan cache agar data bisa digunakan kembali
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # Mengunduh jaringan jalan Kota Medan
    graph = ox.graph_from_place(
        "Kota Medan, Sumatera Utara, Indonesia",
        network_type="drive",
        simplify=True
    )

    # Menyimpan jaringan jalan
    ox.save_graphml(graph, filepath=output_path)

    # Menampilkan informasi hasil pengunduhan
    nodes = len(graph.nodes)
    edges = len(graph.edges)

    print("\nPENGUNDUHAN BERHASIL!")
    print(f"Jumlah persimpangan: {nodes}")
    print(f"Jumlah ruas jalan: {edges}")
    print(f"Lokasi penyimpanan: {output_path}")

def download_charging_stations():
    print("Mengunduh lokasi SPKLU Kota Medan...")

    output_path = Path("data/raw/charging_stations.geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mencari lokasi SPKLU dan titik perangkat pengisian
    tags = {
        "amenity": "charging_station",
        "man_made": "charge_point"
    }

    try:
        stations = ox.features_from_place(
            "Kota Medan, Sumatera Utara, Indonesia",
            tags=tags
        )

    except ox._errors.InsufficientResponseError:
        print("Data SPKLU tidak ditemukan di OpenStreetMap.")
        print("Kita perlu menggunakan sumber data SPKLU alternatif.")
        return

    # Mengubah geometri menjadi titik koordinat
    stations = stations.to_crs(epsg=32647)
    stations.geometry = stations.geometry.representative_point()
    stations = stations.to_crs(epsg=4326)

    # Menyimpan ke GeoJSON
    stations.to_file(output_path, driver="GeoJSON")

    print("\nPENGUNDUHAN SPKLU BERHASIL!")
    print(f"Jumlah data ditemukan: {len(stations)}")
    print(f"Lokasi penyimpanan: {output_path}")
    
def download_medan_boundary():
    print("Mengunduh batas wilayah Kota Medan...")

    output_path = MEDAN_BOUNDARY_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boundary = ox.geocode_to_gdf(
        "Kota Medan, Sumatera Utara, Indonesia"
    )

    boundary.to_file(output_path, driver="GeoJSON")

    print("\nBATAS WILAYAH BERHASIL DIUNDUH!")
    print(f"Lokasi penyimpanan: {output_path}")
    

def download_medan_spklu_directory():
    print("Membaca data SPKLU dari file JSON...")

    json_path = SPKLU_SOURCE_PATH
    output_path = MEDAN_STATIONS_DIRECTORY_PATH

    # Membaca JSON lokal
    with open(json_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    rows = []

    # Mengambil data yang berada di Kota Medan
    for item in dataset["data"]:
        if len(item) < 8:
            continue

        name = item[0]
        address = item[1]
        city = item[2]

        if city.strip() != "Kota Medan":
            continue

        latitude = float(item[3])
        longitude = float(item[4])
        power = float(item[5]) if item[5] is not None else None
        operator = item[7]

        rows.append({
            "name": name,
            "address": address,
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
            "power_kw": power,
            "operator": operator,
            "source": "spklu.web.id"
        })

    if not rows:
        print("Tidak ditemukan data SPKLU Kota Medan.")
        return

    stations = gpd.GeoDataFrame(
        rows,
        geometry=[
            Point(row["longitude"], row["latitude"])
            for row in rows
        ],
        crs="EPSG:4326"
    )

    # Memeriksa batas wilayah Kota Medan
    boundary = gpd.read_file(
        MEDAN_BOUNDARY_PATH
    ).to_crs("EPSG:4326")

    medan_area = unary_union(boundary.geometry)

    stations = stations[
        stations.geometry.apply(medan_area.covers)
    ].copy()

    # Menyimpan hasil
    stations.to_file(output_path, driver="GeoJSON")

    print("\nDATA SPKLU KOTA MEDAN BERHASIL DIAMBIL!")
    print(f"Jumlah data dari katalog: {len(rows)}")
    print(f"Jumlah data dalam batas Medan: {len(stations)}")
    print(f"Lokasi penyimpanan: {output_path}")

    
if __name__ == "__main__":
    download_medan_spklu_directory()