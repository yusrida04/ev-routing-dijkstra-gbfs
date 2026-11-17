from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MEDAN_GRAPH_PATH = RAW_DATA_DIR / "roads.graphml"
REGIONAL_GRAPH_PATH = RAW_DATA_DIR / "roads_antarkota.graphml"
MEDAN_BOUNDARY_PATH = RAW_DATA_DIR / "medan_boundary.geojson"
SPKLU_SOURCE_PATH = RAW_DATA_DIR / "spklu_data.json"

MEDAN_STATIONS_DIRECTORY_PATH = (
    RAW_DATA_DIR / "charging_stations_directory.geojson"
)
MEDAN_STATIONS_CLASSIFIED_PATH = (
    PROCESSED_DATA_DIR / "charging_stations_classified.geojson"
)
MEDAN_STATIONS_CONNECTED_PATH = (
    PROCESSED_DATA_DIR / "charging_stations_connected.geojson"
)

REGIONAL_STATIONS_DIRECTORY_PATH = (
    RAW_DATA_DIR / "charging_stations_regional_directory.geojson"
)
REGIONAL_STATIONS_CLASSIFIED_PATH = (
    PROCESSED_DATA_DIR / "charging_stations_regional_classified.geojson"
)
REGIONAL_STATIONS_CONNECTED_PATH = (
    PROCESSED_DATA_DIR / "charging_stations_regional_connected.geojson"
)

EXPERIMENTS_OUTPUT_DIR = PROCESSED_DATA_DIR / "experiments"
