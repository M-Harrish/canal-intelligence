

import pickle
import sys
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

import assumptions
import config
import crop_economics


def segment_district(row):
    """District of the segment being simulated, for the crop-mix lookup."""
    if not len(row):
        return crop_economics.DEFAULT_DISTRICT
    return crop_economics.district_for_segment(
        row["seg_id"].iloc[0], row["prj_name"].iloc[0])


_PRJ_LOOKUP = {}


def _prj_lookup(segs):
    """seg_id -> project name, built once. Step 8 calls simulate() per segment,
    so a linear scan per call would be quadratic."""
    global _PRJ_LOOKUP
    if not _PRJ_LOOKUP:
        _PRJ_LOOKUP = dict(zip(segs["seg_id"], segs["prj_name"]))
    return _PRJ_LOOKUP

warnings.filterwarnings("ignore")


def fetch_paddy_grid():
    """Wet-season maximum MNDWI over the AOI, on the cropland grid."""
    import io
    import urllib.request

    import ee
    ee.Initialize(project=config.EE_PROJECT)

    x0, y0, x1, y1 = config.aoi_utm_bounds()
    aoi = ee.Geometry.Rectangle([x0, y0, x1, y1], proj="EPSG:32644",
                                geodesic=False)
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(*config.WET_WINDOW)
          .linkCollection(csp, ["cs_cdf"])
          .map(lambda im: im.updateMask(
              im.select("cs_cdf").gte(config.CSP_CLEAR_THRESHOLD))))
    mndwi_max = (s2.map(lambda im: im.normalizedDifference(["B3", "B11"]))
                 .max().rename("mndwi_max").toFloat())

    url = mndwi_max.getDownloadURL({
        "region": aoi, "scale": config.CROPLAND_GRID_M,
        "crs": "EPSG:32644", "format": "NPY",
    })
    print("downloading wet-season MNDWI grid ...")
    arr = np.load(io.BytesIO(urllib.request.urlopen(url).read()),
                  allow_pickle=False)
    grid = np.nan_to_num(arr["mndwi_max"].astype("float32"), nan=-1.0)
    np.savez(config.PADDY_NPZ, grid=grid)
    print(f"  grid {grid.shape}; {(grid > 0).mean()*100:.1f}% of cells "
          f"show standing water in the wet season")
    print(f"  wrote {config.PADDY_NPZ}")


def load_cells_with_crop():
    """Allocated cropland cells, each tagged with an inferred crop class."""
    z = np.load(config.CELL_ALLOC_NPZ, allow_pickle=True)
    coords, areas, seg_ids = z["coords"], z["areas"], z["seg_id"]

    crop = np.full(len(coords), "cropland (unclassified)", dtype=object)
    if config.PADDY_NPZ.exists():
        c = np.load(config.CROPLAND_NPZ)
        x0, y0, x1, y1 = c["bounds"]
        ny, nx_ = c["grid"].shape
        g = config.CROPLAND_GRID_M
        col = np.clip(((coords[:, 0] - x0) / g - 0.5).round().astype(int),
                      0, nx_ - 1)
        row = np.clip(((y1 - coords[:, 1]) / g - 0.5).round().astype(int),
                      0, ny - 1)
        mndwi = np.load(config.PADDY_NPZ)["grid"][row, col]
        crop = np.where(mndwi > 0,
                        "paddy-like (inferred)",
                        "other cropland (inferred)").astype(object)
    else:
        print("  (no paddy grid — run `fetch` for a crop-class breakdown)")
    return pd.DataFrame({"seg_id": seg_ids, "area_ha": areas, "crop": crop})


def simulate(D, seg_id, cells, segs):
    """Remove one segment; report what falls off the network."""
    edge = next(((u, v) for u, v, d in D.edges(data=True)
                 if d["seg_id"] == seg_id), None)
    if edge is None:
        print(f"segment {seg_id} not found in the graph")
        return None

    sources = [n for n, d in D.nodes(data=True) if d.get("is_source")]
    before = set()
    for s in sources:
        before |= nx.descendants(D, s) | {s}

    H = D.copy()
    H.remove_edge(*edge)
    after = set()
    for s in sources:
        after |= nx.descendants(H, s) | {s}

    lost_nodes = before - after
    lost_segs = {d["seg_id"] for u, v, d in D.edges(data=True)
                 if u in lost_nodes or d["seg_id"] == seg_id}

    lost_cells = cells[cells["seg_id"].isin(lost_segs)]
    by_crop = lost_cells.groupby("crop")["area_ha"].sum().sort_values(
        ascending=False)
    total_ha = lost_cells["area_ha"].sum()

    # The downstream set usually spans several districts, so the blocked
    # segment's own district alone understates who loses supply.
    prj = _prj_lookup(segs)
    by_district = {}
    for sid, ha in lost_cells.groupby("seg_id")["area_ha"].sum().items():
        d = crop_economics.district_for_segment(sid, prj.get(sid))
        by_district[d] = by_district.get(d, 0.0) + float(ha)
    by_district = dict(sorted(by_district.items(), key=lambda kv: -kv[1]))

    row = segs[segs["seg_id"] == seg_id]
    name = (row["can_name"].iloc[0] if len(row) else "?") or "(unnamed)"
    ctype = row["can_type"].iloc[0] if len(row) else "?"
    length_km = row["length_m"].iloc[0] / 1000 if len(row) else 0

    # Revenue uses the district's REPORTED crop mix, not "everything is paddy".
    # A district running 20% cane earns far more per hectare than one running
    # 90% rice, so equal command area is not equal value.
    district = segment_district(row)
    rev_per_ha = crop_economics.revenue_per_ha(district)
    revenue = total_ha * rev_per_ha
    repair = assumptions.cost_to_desilt(length_km * 1000, ctype)
    mix = crop_economics.crop_mix(district)
    water_mm = crop_economics.water_demand_mm(district)
    irr_dep = crop_economics.irrigation_dependency(district)

    print("\n" + "=" * 72)
    print(f"FAILURE SIMULATION — {seg_id}")
    print(f"  {name} ({ctype}, {length_km:.1f} km)")
    print("=" * 72)
    print(f"  segments cut off from source : {len(lost_segs)}")
    print(f"  command area lost            : {total_ha:,.0f} ha")
    for crop, ha in by_crop.items():
        print(f"      {crop:32s} {ha:10,.0f} ha")
    if by_district:
        print("  area lost by district        :")
        for d, ha in by_district.items():
            print(f"      {d.title():32s} {ha:10,.0f} ha")
    print(f"\n  district (crop mix basis)    : {district.title()}")
    top = ", ".join(f"{c} {s*100:.0f}%" for c, s in mix.head(3).items())
    print(f"    reported cropping pattern  : {top}")
    print(f"    crop water requirement     : {water_mm:,.0f} mm/ha/yr, "
          f"rainfall meets {(1-irr_dep)*100:.0f}% of it")
    print(f"\n  gross revenue at risk        : Rs {revenue:,.0f} "
          f"({revenue/1e7:,.2f} crore)")
    print(f"    at Rs {rev_per_ha:,.0f}/ha/year, area-weighted over the "
          f"district crop mix")
    print(f"    (paddy priced at MSP {assumptions.PADDY_MSP_PER_QUINTAL.value:,}"
          f"/quintal [sourced]; other crop prices are ASSUMED)")
    print(f"  cost to desilt this segment  : Rs {repair:,.0f}  "
          f"[{assumptions.DESILT_COST_PER_KM.status}]")
    if repair > 0:
        print(f"  benefit-cost ratio           : {revenue/repair:,.1f}x")
    print("\n  Revenue is GROSS at MSP, single season, and assumes total loss "
          "of\n  supply to the cut-off area. It is an order-of-magnitude "
          "figure for\n  prioritisation, not a compensation calculation.")

    return {
        "seg_id": seg_id, "can_name": name, "can_type": ctype,
        "length_km": length_km, "segments_cut_off": len(lost_segs),
        "area_lost_ha": total_ha, "revenue_at_risk_inr": revenue,
        "desilt_cost_inr": repair,
        "benefit_cost_ratio": revenue / repair if repair else np.nan,
        "district": district, "revenue_per_ha": rev_per_ha,
        "water_demand_mm": water_mm, "irrigation_dependency": irr_dep,
        "crop_mix_top3": "; ".join(f"{c} {s*100:.0f}%"
                                   for c, s in mix.head(3).items()),
        # Consumed by the web app: which reaches go dry, and where they are.
        # Dropped before the CSV is written — they are structured, not scalar.
        "lost_seg_ids": sorted(lost_segs),
        "area_by_district": by_district,
        **{f"ha_{k}": v for k, v in by_crop.items()},
    }


def main():
    args = sys.argv[1:]
    if args and args[0] == "fetch":
        return fetch_paddy_grid()

    if not config.CELL_ALLOC_NPZ.exists():
        print("no cell allocation — run step05_rank_segments.py first")
        return 1

    with open(config.GRAPH_PKL, "rb") as f:
        D = pickle.load(f)
    segs = gpd.read_file(config.RANKED_GPKG)
    cells = load_cells_with_crop()

    if args and args[0] == "--top":
        n = int(args[1]) if len(args) > 1 else 5
        targets = segs.sort_values("rank")["seg_id"].head(n).tolist()
    elif args:
        targets = args
    else:
        targets = segs.sort_values("rank")["seg_id"].head(3).tolist()

    results = [r for r in (simulate(D, s, cells, segs) for s in targets) if r]
    if results:
        out = pd.DataFrame(results).drop(
            columns=["lost_seg_ids", "area_by_district"], errors="ignore")
        path = config.DATA_DIR / "failure_simulations.csv"
        out.to_csv(path, index=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
