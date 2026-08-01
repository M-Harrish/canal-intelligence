"""Step 5: command area + criticality ranking -> data/ranked_segments.gpkg

Three ingredients per segment:

  command_area_ha   cropland commanded by this segment AND everything
                    downstream of it in the oriented graph
  health_score      from step 4 (1 = surface signature of a clear channel)
  betweenness       edge betweenness centrality — how much of the network's
                    source-to-field routing passes through this segment

  priority = command_area_ha * (1 - health_score) * (1 + alpha * bc_norm)

Command area allocation
-----------------------
Cropland comes from ESA WorldCover 10 m (class 40), fetched once and cached.
Each cell goes to exactly ONE segment — the finest canal within
MAX_SERVICE_DIST_M — so areas never double count. That is how an ayacut works:
a field is served by the minor beside it, not the main canal 2 km away.

Run:
  python step05_rank_segments.py fetch   # download cropland grid (once)
  python step05_rank_segments.py         # rank
"""

import pickle
import sys
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Point

import config

warnings.filterwarnings("ignore")


def fetch_cropland():
    """Download an ESA WorldCover cropland-fraction grid for the AOI as NPY."""
    import io
    import urllib.request

    import ee
    ee.Initialize(project=config.EE_PROJECT)

    # request in UTM so the array's georeferencing is exact, not inferred
    x0, y0, x1, y1 = config.aoi_utm_bounds()
    aoi = ee.Geometry.Rectangle([x0, y0, x1, y1], proj="EPSG:32644",
                                geodesic=False)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    cropland = wc.eq(40).rename("crop")

    # let EE pyramid the 10 m mask down to the request scale — an explicit
    # reproject() blows past the size limit over an AOI this large
    frac = cropland.toFloat()

    url = frac.getDownloadURL({
        "region": aoi, "scale": config.CROPLAND_GRID_M,
        "crs": "EPSG:32644", "format": "NPY",
    })
    print("downloading cropland grid ...")
    raw = urllib.request.urlopen(url).read()
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    grid = np.nan_to_num(arr["crop"].astype("float32"))
    print(f"  grid {grid.shape}, cropland fraction mean {grid.mean():.3f}")
    np.savez(config.CROPLAND_NPZ, grid=grid, bounds=np.array([x0, y0, x1, y1]))
    print(f"  wrote {config.CROPLAND_NPZ}")


def load_cropland_cells():
    """-> (Nx2 UTM coords, N cropland hectares per cell) for non-zero cells."""
    z = np.load(config.CROPLAND_NPZ)
    grid, (x0, y0, x1, y1) = z["grid"], z["bounds"]
    ny, nx_ = grid.shape

    # EE returns rows north-to-south; use cell centres, not edges
    g = config.CROPLAND_GRID_M
    xs = x0 + (np.arange(nx_) + 0.5) * g
    ys = y1 - (np.arange(ny) + 0.5) * g
    xx, yy = np.meshgrid(xs, ys)

    m = grid > 0.05
    cell_ha = (config.CROPLAND_GRID_M ** 2) / 10_000.0
    coords = np.column_stack([xx[m], yy[m]])
    areas = grid[m] * cell_ha
    print(f"{len(coords)} cropland cells, {areas.sum():,.0f} ha total in AOI")
    return coords, areas


def allocate_command_area(segs, coords, areas):
    """Assign each cropland cell to the finest canal within reach."""
    order = segs["can_type"].map(config.SERVICE_PRIORITY).fillna(9).values
    geoms = list(segs.geometry)

    best_rank = np.full(len(coords), 99, dtype=np.int16)
    best_seg = np.full(len(coords), -1, dtype=np.int32)
    best_dist = np.full(len(coords), np.inf)

    tree = STRtree(geoms)
    pts = [Point(c) for c in coords]
    # candidate (cell, segment) pairs within service distance
    pairs = tree.query(pts, predicate="dwithin", distance=config.MAX_SERVICE_DIST_M)
    print(f"  {pairs.shape[1]:,} cell-segment candidate pairs")

    for ci, si in zip(pairs[0], pairs[1]):
        d = geoms[si].distance(pts[ci])
        r = order[si]
        if (r, d) < (best_rank[ci], best_dist[ci]):
            best_rank[ci], best_seg[ci], best_dist[ci] = r, si, d

    local = np.zeros(len(segs))
    served = best_seg >= 0
    np.add.at(local, best_seg[served], areas[served])
    print(f"  {served.sum():,} of {len(coords):,} cells allocated "
          f"({areas[served].sum():,.0f} ha commanded)")

    # step 7 reuses this to break a segment's command area down by crop type
    np.savez(
        config.CELL_ALLOC_NPZ,
        coords=coords[served],
        areas=areas[served],
        seg_id=np.array(segs["seg_id"].values)[best_seg[served]],
    )
    print(f"  wrote {config.CELL_ALLOC_NPZ}")
    return local


def downstream_area(D, segs, local_ha):
    """A segment's command area: its own local area plus everything
    downstream of it."""
    idx = {sid: i for i, sid in enumerate(segs["seg_id"])}
    edge_of = {d["seg_id"]: (u, v) for u, v, d in D.edges(data=True)}

    # cache descendants per node so we don't recompute for every edge
    cache = {}

    def desc(n):
        if n not in cache:
            cache[n] = nx.descendants(D, n) | {n}
        return cache[n]

    out = np.zeros(len(segs))
    for sid, i in idx.items():
        if sid not in edge_of:
            out[i] = local_ha[i]
            continue
        _, v = edge_of[sid]
        reach = desc(v)
        total = local_ha[i]
        for u2, v2, d2 in D.edges(data=True):
            if u2 in reach and d2["seg_id"] in idx and d2["seg_id"] != sid:
                total += local_ha[idx[d2["seg_id"]]]
        out[i] = total
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        return fetch_cropland()

    segs = gpd.read_file(config.SEGMENTS_GPKG)
    canals = segs[(segs["is_river"] == 0) & (segs["can_type"] != "Connector")].copy()
    canals = canals.reset_index(drop=True)
    print(f"{len(canals)} canal segments to rank")

    if not config.CROPLAND_NPZ.exists():
        print("no cropland grid — run: python step05_rank_segments.py fetch")
        return 1
    coords, areas = load_cropland_cells()
    local_ha = allocate_command_area(canals, coords, areas)
    canals["local_area_ha"] = local_ha

    with open(config.GRAPH_PKL, "rb") as f:
        D = pickle.load(f)
    canals["command_area_ha"] = downstream_area(D, canals, local_ha)

    print("computing edge betweenness ...")
    bc = nx.edge_betweenness_centrality(D, weight="length_m")
    bc_by_seg = {d["seg_id"]: bc.get((u, v), 0.0) for u, v, d in D.edges(data=True)}
    canals["betweenness"] = canals["seg_id"].map(bc_by_seg).fillna(0.0)
    mx = canals["betweenness"].max()
    canals["bc_norm"] = canals["betweenness"] / mx if mx > 0 else 0.0

    health = pd.read_csv(config.SEGMENT_HEALTH_CSV)
    canals = canals.merge(
        health[["seg_id", "health_score", "dominant_class", "n_points", "method"]],
        on="seg_id", how="left",
    )
    missing = canals["health_score"].isna().sum()
    if missing:
        print(f"  {missing} segments have no health score — using median")
        canals["health_score"] = canals["health_score"].fillna(
            canals["health_score"].median())

    canals["centrality_weight"] = 1 + config.BETWEENNESS_ALPHA * canals["bc_norm"]
    canals["priority"] = (canals["command_area_ha"]
                          * (1 - canals["health_score"])
                          * canals["centrality_weight"])
    mx = canals["priority"].max()
    canals["priority_norm"] = canals["priority"] / mx if mx > 0 else 0.0
    canals = canals.sort_values("priority", ascending=False).reset_index(drop=True)
    canals["rank"] = np.arange(1, len(canals) + 1)

    canals.to_file(config.RANKED_GPKG, driver="GPKG")
    print(f"\nwrote {config.RANKED_GPKG}")
    print(f"method: {canals['method'].iloc[0]}")
    cols = ["rank", "can_name", "can_type", "prj_name", "length_m",
            "command_area_ha", "health_score", "bc_norm", "priority"]
    print("\ntop 15 segments by priority:")
    print(canals[cols].head(15).to_string(index=False,
          formatters={"length_m": "{:.0f}".format,
                      "command_area_ha": "{:,.0f}".format,
                      "health_score": "{:.2f}".format,
                      "bc_norm": "{:.2f}".format,
                      "priority": "{:,.0f}".format}))


if __name__ == "__main__":
    sys.exit(main())
