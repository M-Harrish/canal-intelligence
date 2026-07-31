"""Step 2: Sentinel-2 features for every canal sample point -> data/features.csv

For each real canal segment (rivers and virtual connectors excluded), sample a
point every ~200 m. For a wet-season and a dry-season window, extract:

  ndvi_chan   NDVI on the channel centerline (10 m pixel)
  ndvi_ring   mean NDVI in a 30-60 m ring beside the channel (cropland control)
  ndvi_diff   ndvi_chan - ndvi_ring  <- key signal: vegetation IN the channel
              relative to the surrounding fields is a surface signature of
              impaired conveyance (NOT a direct measurement of bed siltation)
  mndwi       median MNDWI on the channel
  water_frac  fraction of clear observations where MNDWI > 0 (water present)
  nobs        clear observations count (QA)

Restartable: already-extracted point_ids are skipped on re-run.

Run:  python step02_extract_features.py
"""

import sys
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")
import ee  # noqa: E402


def build_sample_points():
    gdf = gpd.read_file(config.SEGMENTS_GPKG)
    gdf = gdf[(gdf["is_river"] == 0) & (gdf["can_type"] != "Connector")]
    print(f"{len(gdf)} canal segments, {gdf.length_m.sum()/1000:.0f} km")

    rows = []
    for _, seg in gdf.iterrows():
        n = max(1, int(seg.length_m // config.SAMPLE_SPACING_M))
        # midpoints of n equal reaches -> no duplicate points at junctions
        for k in range(n):
            d = (k + 0.5) * seg.length_m / n
            pt = seg.geometry.interpolate(d)
            rows.append({
                "point_id": f"{seg.seg_id}_{k:03d}",
                "seg_id": seg.seg_id,
                "chainage_m": round(d, 1),
                "can_type": seg.can_type,
                "prj_name": seg.prj_name,
                "geometry": pt,
            })
    pts = gpd.GeoDataFrame(rows, crs=config.CRS_UTM).to_crs(config.CRS_WGS84)
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y
    print(f"{len(pts)} sample points at {config.SAMPLE_SPACING_M:.0f} m spacing")
    return pts


def season_image(start, end, aoi):
    """Cloud-masked S2 stack -> single multiband image for one season."""
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .linkCollection(csp, ["cs_cdf"])
        .map(lambda im: im.updateMask(im.select("cs_cdf").gte(config.CSP_CLEAR_THRESHOLD)))
    )

    def add_indices(im):
        ndvi = im.normalizedDifference(["B8", "B4"]).rename("ndvi")
        mndwi = im.normalizedDifference(["B3", "B11"]).rename("mndwi")
        water = mndwi.gt(0).rename("water")
        return im.addBands([ndvi, mndwi, water])

    s2 = s2.map(add_indices)
    ndvi_med = s2.select("ndvi").median().rename("ndvi")
    mndwi_med = s2.select("mndwi").median().rename("mndwi")
    water_frac = s2.select("water").mean().rename("water_frac")
    nobs = s2.select("water").count().rename("nobs")

    # Control ring as raster math instead of per-point buffer polygons.
    # focal_mean over a disc of radius r gives the mean of that disc, so the
    # mean over the ring between r_in and r_out is the area-weighted
    # difference of the two discs. This keeps the whole extraction to one
    # reduceRegions call per season, which is far cheaper than building a
    # buffer/difference geometry for every sample point.
    r_in, r_out = config.RING_INNER_M, config.RING_OUTER_M
    disc_in = ndvi_med.focal_mean(radius=r_in, units="meters")
    disc_out = ndvi_med.focal_mean(radius=r_out, units="meters")
    a_in, a_out = r_in ** 2, r_out ** 2
    ring = (disc_out.multiply(a_out).subtract(disc_in.multiply(a_in))
            .divide(a_out - a_in).rename("ndvi_ring"))

    return ndvi_med.addBands([mndwi_med, water_frac, nobs, ring])


def extract_chunk(chunk, images):
    """One getInfo round-trip for a chunk of points; returns list of dicts."""
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
        for r in chunk.itertuples()
    ])
    merged = {}
    for season, img in images.items():
        sampled = img.reduceRegions(fc, ee.Reducer.first(), scale=10)
        for f in sampled.getInfo()["features"]:
            p = f["properties"]
            row = merged.setdefault(p["point_id"], {})
            row[f"{season}_ndvi_chan"] = p.get("ndvi")
            row[f"{season}_ndvi_ring"] = p.get("ndvi_ring")
            for k in ("mndwi", "water_frac", "nobs"):
                row[f"{season}_{k}"] = p.get(k)

    out = []
    for pid, row in merged.items():
        for season in images:
            c, r = row.get(f"{season}_ndvi_chan"), row.get(f"{season}_ndvi_ring")
            row[f"{season}_ndvi_diff"] = None if c is None or r is None else c - r
        out.append({"point_id": pid, **row})
    return out


def main():
    ee.Initialize(project=config.EE_PROJECT)
    pts = build_sample_points()

    done = set()
    if config.FEATURES_CSV.exists():
        done = set(pd.read_csv(config.FEATURES_CSV)["point_id"])
        print(f"resuming: {len(done)} points already extracted")
    todo = pts[~pts["point_id"].isin(done)]
    if todo.empty:
        print("all points already extracted")
        return

    minx, miny, maxx, maxy = config.AOI_BBOX
    aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    images = {
        "wet": season_image(*config.WET_WINDOW, aoi),
        "dry": season_image(*config.DRY_WINDOW, aoi),
    }

    meta = pts.drop(columns="geometry").set_index("point_id")
    n_chunks = int(np.ceil(len(todo) / config.EE_CHUNK_SIZE))
    for i in range(n_chunks):
        chunk = todo.iloc[i * config.EE_CHUNK_SIZE:(i + 1) * config.EE_CHUNK_SIZE]
        t0 = time.time()
        for attempt in range(3):
            try:
                rows = extract_chunk(chunk, images)
                break
            except ee.EEException as e:
                print(f"  chunk {i}: attempt {attempt + 1} failed ({e}); retrying")
                time.sleep(20 * (attempt + 1))
        else:
            print(f"  chunk {i}: giving up — rerun the script to resume")
            return 1
        df = pd.DataFrame(rows).set_index("point_id").join(meta).reset_index()
        header = not config.FEATURES_CSV.exists()
        df.to_csv(config.FEATURES_CSV, mode="a", header=header, index=False)
        print(f"  chunk {i + 1}/{n_chunks}: {len(df)} points in {time.time() - t0:.0f}s")

    print(f"wrote {config.FEATURES_CSV}")


if __name__ == "__main__":
    sys.exit(main())
