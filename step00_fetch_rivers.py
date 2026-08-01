"""Step 0 (optional): fetch river centerlines for the AOI from OSM Overpass.

The canal layer has canals only, but the delta's canal systems join through the
rivers (Cauvery, Vennar, Vettar, ...). Rivers go into the graph as connectors
(is_river=1) to keep it connected, and are excluded from desilting ranking.

Run:  python step00_fetch_rivers.py     ->  data/rivers.gpkg
"""

import sys

import geopandas as gpd
import requests
from shapely.geometry import LineString

import config

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json][timeout:180];
way["waterway"~"^(river|canal)$"]({s},{w},{n},{e});
out geom;
"""


def main():
    minx, miny, maxx, maxy = config.AOI_BBOX
    q = QUERY.format(s=miny, w=minx, n=maxy, e=maxx)
    print("Querying Overpass for rivers/canals in AOI ...")
    r = requests.post(
        OVERPASS_URL,
        data={"data": q},
        headers={"User-Agent": "AYACUT-hackathon/0.1 (canal network research)"},
        timeout=300,
    )
    r.raise_for_status()
    elements = r.json()["elements"]
    print(f"  {len(elements)} ways returned")

    rows = []
    for el in elements:
        coords = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id": el["id"],
            "name": tags.get("name"),
            "waterway": tags.get("waterway"),
            "geometry": LineString(coords),
        })
    gdf = gpd.GeoDataFrame(rows, crs=config.CRS_WGS84)
    print(f"  {len(gdf)} lines "
          f"({(gdf['waterway'] == 'river').sum()} river, {(gdf['waterway'] == 'canal').sum()} canal)")

    config.DATA_DIR.mkdir(exist_ok=True)
    gdf.to_file(config.RIVERS_GPKG, driver="GPKG")
    print(f"  wrote {config.RIVERS_GPKG}")


if __name__ == "__main__":
    sys.exit(main())
