"""Central configuration for the AYACUT pipeline. Edit values here, not in scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------- input data
RAW_KML = PROJECT_ROOT / "canal_network.kml"

# Attribute filters applied to the national canal layer
FILTER_STATE = "Tamil Nadu"
FILTER_BASIN = "Cauvery"

# Cauvery delta area of interest, WGS84 (minx, miny, maxx, maxy).
# Extends west of 79.0 so the network's head reaches near the
# Grand Anicut are included.
AOI_BBOX = (78.70, 10.00, 79.95, 11.40)

# ---------------------------------------------------------------- graph build
CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32644"  # UTM zone 44N

# Endpoints / T-junctions closer than this (metres) are considered connected.
SNAP_TOLERANCE_M = 25.0

# Floating canal components are linked to the main network with a virtual
# connector edge (is_virtual=1) if within this distance (metres). Connectors
# stand in for unmapped offtake/head channels; every one is logged.
ATTACH_TOLERANCE_M = 2000.0

# Segments shorter than this after splitting are merged away (metres).
MIN_SEGMENT_M = 5.0

# Warn if the undirected graph has more components than this.
MAX_EXPECTED_COMPONENTS = 5

# BFS orientation source: Grand Anicut (Kallanai), lat/lon WGS84.
SOURCE_LATLON = (10.8306, 78.8181)

# ---------------------------------------------------------------- earth engine
EE_PROJECT = "canal-intelligence"

# Sample a feature point every N metres along each canal segment.
SAMPLE_SPACING_M = 200.0

# Season windows for the Cauvery delta (Mettur release ~mid-June; Samba
# paddy season runs Aug-Jan; Feb-May is the dry/closure period).
WET_WINDOW = ("2025-08-01", "2026-02-01")
DRY_WINDOW = ("2026-02-01", "2026-06-01")

# Sentinel-2 Cloud Score+ threshold: keep pixels with cs_cdf >= this.
CSP_CLEAR_THRESHOLD = 0.60

# NDVI control ring around the channel (metres).
RING_INNER_M = 30
RING_OUTER_M = 60

# Points per getInfo request (EE payload limit safety).
EE_CHUNK_SIZE = 400

# ---------------------------------------------------------------- dynamic world
# Radius of the corridor Dynamic World is averaged over, metres. Sized to the
# ~10-15 m positional accuracy of the source canal geometry (measured against
# sub-metre imagery), so the sample always contains the channel even where the
# centreline is offset or the canal runs beside a road.
DW_CORRIDOR_M = 20

# Dynamic World class probability above which a cell counts as cropland.
# DW gives a probability per class; 0.30 on the 'crops' band is deliberately
# permissive because delta paddy is often classed partly as flooded_vegetation.
DW_CROP_THRESHOLD = 0.30

# ---------------------------------------------------------------- outputs
RIVERS_GPKG = DATA_DIR / "rivers.gpkg"      # OSM river connectors (step 0)
CANALS_GPKG = DATA_DIR / "canals.gpkg"      # filtered raw canals (AOI subset)
SEGMENTS_GPKG = DATA_DIR / "segments.gpkg"  # noded, oriented edges w/ seg_id
GRAPH_PKL = DATA_DIR / "graph.pkl"          # pickled networkx DiGraph
FEATURES_CSV = DATA_DIR / "features.csv"    # per-point S2 features (step 2)
DW_FEATURES_CSV = DATA_DIR / "dw_features.csv"  # Dynamic World features (step 2b)
LABELS_CSV = DATA_DIR / "labels.csv"        # hand labels (step 3)
CHIPS_DIR = DATA_DIR / "chips"              # Sentinel-2 chips (what the model sees)
CHIPS_HR_DIR = DATA_DIR / "chips_hr"        # sub-metre imagery (what you label from)
MODEL_PKL = DATA_DIR / "condition_model.pkl"        # trained classifier (step 4)
POINT_PRED_CSV = DATA_DIR / "point_predictions.csv"
SEGMENT_HEALTH_CSV = DATA_DIR / "segment_health.csv"

# Classes that indicate impaired conveyance (used to build the risk score).
IMPAIRED_CLASSES = ["choked", "encroached"]

# ---------------------------------------------------------------- command area
# ESA WorldCover 10 m: class 40 = cropland.
CROPLAND_GRID_M = 200          # resolution the cropland grid is fetched at
MAX_SERVICE_DIST_M = 2000      # farthest a field can be from its serving canal

# Finer canals serve fields directly; a field next to both a minor and a main
# canal is commanded by the minor. Lower number = wins the allocation.
SERVICE_PRIORITY = {
    "Sub Sub Minor": 0, "Sub Minor": 1, "Minor": 2, "Water Course": 2,
    "Distributary": 3, "Branch Canal": 4, "Main Canal": 5, "Feeder": 6,
}

# priority = command_area_ha * (1 - health) * (1 + BETWEENNESS_ALPHA * bc_norm)
BETWEENNESS_ALPHA = 1.0

# Health a segment is assumed to reach after desilting. Not 1.0 — clearing a
# channel restores conveyance, it does not make the canal new.
DESILT_TARGET_HEALTH = 0.95

def aoi_utm_bounds():
    """Exact UTM 44N bounding box of the AOI.

    The WGS84 rectangle maps to a curved quadrilateral in UTM, so transforming
    only the two corner points understates the extent. Densify the boundary
    first, then take its bounds, and snap outward to whole grid cells so the
    downloaded raster aligns predictably.
    """
    import geopandas as gpd
    from shapely.geometry import box

    poly = gpd.GeoSeries([box(*AOI_BBOX)], crs=CRS_WGS84)
    poly = poly.segmentize(0.01).to_crs(CRS_UTM)
    x0, y0, x1, y1 = poly.total_bounds
    g = CROPLAND_GRID_M
    import math
    return (math.floor(x0 / g) * g, math.floor(y0 / g) * g,
            math.ceil(x1 / g) * g, math.ceil(y1 / g) * g)


CROPLAND_NPZ = DATA_DIR / "cropland_grid.npz"
CELL_ALLOC_NPZ = DATA_DIR / "cell_allocation.npz"
PADDY_NPZ = DATA_DIR / "paddy_grid.npz"
RANKED_GPKG = DATA_DIR / "ranked_segments.gpkg"

# ---------------------------------------------------------------- labelling
LABEL_CLASSES = ["flowing", "dry", "choked", "encroached", "unclear"]
N_LABEL_POINTS = 300
CHIP_BUFFER_M = 150      # half-width of the satellite chip around a point
CHIP_PIXELS = 256        # Sentinel-2 chip (10 m native — upsampling adds nothing)
CHIP_HR_PIXELS = 512     # sub-metre chip, where extra pixels are real detail
LABEL_SEED = 42
