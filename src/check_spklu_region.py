import json
from collections import Counter
from pathlib import Path

JSON_PATH = Path("data/raw/spklu_data.json")

# (barat, selatan, timur, utara); sama dengan graf antarkota.
BOUNDS = (98.35, 2.55, 99.30, 3.85)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


with open(JSON_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

rows = [item for item in dataset["data"] if len(item) >= 8]

print(f"Total baris katalog: {len(rows)}")

west, south, east, north = BOUNDS

inside = []

for item in rows:
    lat = to_float(item[3])
    lon = to_float(item[4])

    if lat is None or lon is None:
        continue

    if south <= lat <= north and west <= lon <= east:
        inside.append(item)

print(f"Di dalam kotak wilayah antarkota: {len(inside)}")

print("\nJumlah per kota/kabupaten:")

for city, count in Counter(
    str(item[2]).strip() for item in inside
).most_common():
    print(f"  {count:>3}  {city}")

print("\nDaftar SPKLU di LUAR Kota Medan:")

outside = [
    item for item in inside
    if str(item[2]).strip() != "Kota Medan"
]

if not outside:
    print("  (tidak ada)")

for item in outside:
    print(
        f"  {item[0]} | {item[2]} | {item[3]}, {item[4]} "
        f"| {item[5]} kW | {item[7]}"
    )