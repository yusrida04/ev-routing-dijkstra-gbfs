
import geopandas as gpd
import pandas as pd

# Membaca data SPKLU
catalog = gpd.read_file(
    "data/raw/charging_stations_directory.geojson"
)

osm = gpd.read_file(
    "data/raw/charging_stations.geojson"
)

# Memberikan identitas sumber data
catalog["source"] = "Katalog"
catalog["station_name"] = catalog["name"]

osm["source"] = "OpenStreetMap"
osm["station_name"] = osm["id"].apply(
    lambda x: f"OSM ID {x}"
)

# Menggabungkan kedua data
stations = pd.concat(
    [
        catalog[["station_name", "source", "geometry"]],
        osm[["station_name", "source", "geometry"]]
    ],
    ignore_index=True
)

stations = gpd.GeoDataFrame(
    stations,
    geometry="geometry",
    crs="EPSG:4326"
)

# Mengubah koordinat ke meter
stations = stations.to_crs("EPSG:32647")

print("\nPEMERIKSAAN LOKASI BERDEKATAN")
print("Batas pemeriksaan: 100 meter\n")

found = False

for i in range(len(stations)):
    for j in range(i + 1, len(stations)):
        distance = stations.geometry.iloc[i].distance(
            stations.geometry.iloc[j]
        )

        if distance <= 100:
            found = True

            print(f"Jarak: {distance:.2f} meter")
            print(
                f"  Lokasi 1: {stations.station_name.iloc[i]}"
            )
            print(
                f"  Sumber 1: {stations.source.iloc[i]}"
            )
            print(
                f"  Lokasi 2: {stations.station_name.iloc[j]}"
            )
            print(
                f"  Sumber 2: {stations.source.iloc[j]}"
            )
            print("-" * 50)

if not found:
    print("Tidak ditemukan lokasi dalam radius 100 meter.")
