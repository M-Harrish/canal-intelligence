"""Step 2b: Dynamic World features per sample point -> data/dw_features.csv

Dynamic World (GOOGLE/DYNAMICWORLD/V1) is a 10 m near-real-time land cover
product giving a PROBABILITY for each of 9 classes per Sentinel-2 scene. Three
of those classes map almost directly onto the labels being collected in step 3:

  built              -> "encroached"  (structures over the alignment)
  trees + shrub      -> "choked"      (woody growth in the section)
  water              -> "flowing"     (standing/moving water in the section)

That is why Dynamic World is worth adding on top of raw NDVI/MNDWI: the model
no longer has to learn "green in the channel" from scratch from index values,
it gets a purpose-built classifier's opinion as an input feature. NDVI cannot
tell a tree from a paddy crop; Dynamic World can.

Sampled over a 20 m corridor, not a single pixel, because the source canal
geometry is only accurate to ~10-15 m (verified against sub-metre imagery) and
delta canals frequently run alongside a road. A single-pixel sample lands on
bank, road or field a good fraction of the time; a corridor mean always
contains the channel.

Restartable: already-extracted point_ids are skipped.
Run:  python step02b_dynamic_world.py
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

import config
import step02_extract_features as s2m

warnings.filterwarnings("ignore")
import ee  # noqa: E402

DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare"]


def dw_season(start, end, aoi):
    """Mean Dynamic World class probabilities over a season, as a corridor mean."""
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi)
          .filterDate(start, end)
          .select(DW_BANDS))
    mean = dw.mean()
    # Corridor mean absorbs the ~10-15 m positional error of the canal
    # centreline; see module docstring.
    corridor = mean.focal_mean(radius=config.DW_CORRIDOR_M, units="meters")
    n = dw.select("water").count().rename("dw_nobs")
    return corridor.addBands(n)


def extract_chunk(chunk, images):
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
            for b in DW_BANDS:
                row[f"{season}_dw_{b}"] = p.get(b)
            row[f"{season}_dw_nobs"] = p.get("dw_nobs")
    return [{"point_id": pid, **row} for pid, row in merged.items()]


def main():
    ee.Initialize(project=config.EE_PROJECT)
    pts = s2m.build_sample_points()

    done = set()
    if config.DW_FEATURES_CSV.exists():
        done = set(pd.read_csv(config.DW_FEATURES_CSV)["point_id"])
        print(f"resuming: {len(done)} points already extracted")
    todo = pts[~pts["point_id"].isin(done)]
    if todo.empty:
        print("all points already extracted")
        return 0

    minx, miny, maxx, maxy = config.AOI_BBOX
    aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    images = {
        "wet": dw_season(*config.WET_WINDOW, aoi),
        "dry": dw_season(*config.DRY_WINDOW, aoi),
    }

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
            print(f"  chunk {i}: giving up — rerun to resume")
            return 1
        df = pd.DataFrame(rows)
        header = not config.DW_FEATURES_CSV.exists()
        df.to_csv(config.DW_FEATURES_CSV, mode="a", header=header, index=False)
        print(f"  chunk {i + 1}/{n_chunks}: {len(df)} points in {time.time() - t0:.0f}s")

    print(f"wrote {config.DW_FEATURES_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
