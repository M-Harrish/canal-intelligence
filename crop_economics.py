"""District crop mix, water demand and revenue, from real TN statistics.

Replaces the earlier single-crop assumption ("everything is paddy") with the
actual reported cropping pattern of each delta district.

Inputs (all in data/, all real published figures):
  tn_crop_production.csv    district x crop x season sown area, TN 2017
  crop_water_requirement.csv  per-crop water demand and duration
  tn_rainfall.csv           district rainfall 2023-24 actual vs normal

What it gives the pipeline:
  crop_mix(district)        share of cropped area by crop
  revenue_per_ha(district)  area-weighted INR/ha/year across the real mix
  water_demand_mm(district) area-weighted crop water requirement
  rainfall_deficit(district) how far below normal rain the district ran

Why this matters for canal prioritisation: a hectare of sugarcane is worth
several times a hectare of paddy and needs roughly twice the water, so two
canals commanding equal area are NOT equally valuable. And a district running
a big rainfall deficit depends on canal supply more than one that is not.

PRICES: only paddy MSP is sourced (see assumptions.py). Every other crop price
here is an ASSUMED order-of-magnitude figure and is tagged as such wherever it
surfaces. Replace from the CACP price schedule before operational use.
"""

import pandas as pd

import config

# ---------------------------------------------------------------- prices
# INR per tonne of harvested produce. Paddy is the one sourced figure
# (CCEA Kharif 2025-26, PIB 2131983 — see assumptions.PADDY_MSP_PER_QUINTAL).
# The rest are ASSUMED: indicative of MSP order of magnitude only.
CROP_PRICE_PER_TONNE = {
    "Rice":                 23_690,   # SOURCED via assumptions.py (2369/qtl)
    "Jowar":                36_990,
    "Bajra":                27_750,
    "Maize":                24_000,
    "Ragi":                 48_860,
    "Arhar/Tur":            80_000,
    "Other Pulses":         78_000,
    "Groundnut":            72_630,
    "Sunflower":            77_210,
    "Sesamum":              98_460,
    "Other Oilseeds":       75_000,
    "Cotton":               77_100,
    "Sugarcane":             3_550,   # FRP basis; cane is priced per tonne cane
    "Tobacco":              60_000,
    "Fodder Crops":          3_000,
    "Other Non Food Crops": 20_000,
    "Other Food Crops":     20_000,
    "Soyabean":             53_280,
    "Gram":                 58_000,
}
PRICE_STATUS = {"Rice": "SOURCED"}  # everything else -> ASSUMED

# Tonnes of harvested produce per hectare.
# Paddy is DERIVED (assumptions.PADDY_YIELD_T_PER_HA). Others ASSUMED.
CROP_YIELD_T_PER_HA = {
    "Rice":                  3.45,
    "Jowar":                 1.20,
    "Bajra":                 1.50,
    "Maize":                 5.50,
    "Ragi":                  2.20,
    "Arhar/Tur":             0.90,
    "Other Pulses":          0.70,
    "Groundnut":             2.20,
    "Sunflower":             1.20,
    "Sesamum":               0.60,
    "Other Oilseeds":        1.20,
    "Cotton":                0.55,
    "Sugarcane":           100.00,
    "Tobacco":               1.50,
    "Fodder Crops":         30.00,
    "Other Non Food Crops":  2.00,
    "Other Food Crops":      2.00,
    "Soyabean":              1.10,
    "Gram":                  1.00,
}

# Crop name in the production statistics -> crop name in the water table.
CROP_TO_WATER_KEY = {
    "Rice": "rice", "Jowar": "sorghum", "Bajra": "pearl millet",
    "Maize": "maize", "Ragi": "ragi", "Arhar/Tur": "redgram",
    "Other Pulses": "blackgram", "Groundnut": "groundnut",
    "Sunflower": "sunflower", "Sesamum": "gingely",
    "Other Oilseeds": "castor", "Cotton": "cotton",
    "Sugarcane": "sugarcane", "Tobacco": "tobacco",
    "Fodder Crops": "fodder cholam", "Soyabean": "soyabean",
    "Gram": "bengalgram",
}

# Districts the AOI actually covers, in rough order of how much of the mapped
# network sits in each. Used when a segment has no district attribute.
DEFAULT_DISTRICT = "THANJAVUR"

_cache = {}


def _load():
    if _cache:
        return _cache
    prod = pd.read_csv(config.DATA_DIR / "tn_crop_production.csv")
    water = pd.read_csv(config.DATA_DIR / "crop_water_requirement.csv")
    rain = pd.read_csv(config.DATA_DIR / "tn_rainfall.csv")
    _cache.update(prod=prod, water=water.set_index("Crop"), rain=rain.set_index("District"))
    return _cache


def crop_mix(district=DEFAULT_DISTRICT):
    """Share of district cropped area by crop, summed across seasons.

    Seasons are summed rather than averaged: a district that grows rice in
    both Winter and Summer really does devote that much cropped area to rice
    over the year, and that is what water demand and revenue scale with.
    """
    prod = _load()["prod"]
    d = prod[prod["District"] == district.upper()]
    if d.empty:
        d = prod[prod["District"] == DEFAULT_DISTRICT]
    by_crop = d.groupby("Crop")["Area_ha"].sum()
    total = by_crop.sum()
    return (by_crop / total).sort_values(ascending=False) if total else by_crop


def revenue_per_ha(district=DEFAULT_DISTRICT):
    """Area-weighted gross revenue, INR per cropped hectare per year."""
    mix = crop_mix(district)
    total = 0.0
    for crop, share in mix.items():
        price = CROP_PRICE_PER_TONNE.get(crop)
        yld = CROP_YIELD_T_PER_HA.get(crop)
        if price and yld:
            total += share * price * yld
    return total


def water_demand_mm(district=DEFAULT_DISTRICT):
    """Area-weighted crop water requirement, mm per cropped hectare per year.

    Uses the midpoint of the published min-max range for each crop.
    """
    mix = crop_mix(district)
    water = _load()["water"]
    total = weight = 0.0
    for crop, share in mix.items():
        key = CROP_TO_WATER_KEY.get(crop)
        if key and key in water.index:
            row = water.loc[key]
            total += share * (row["Water_min_mm"] + row["Water_max_mm"]) / 2
            weight += share
    return total / weight if weight else 0.0


def rainfall_deficit(district=DEFAULT_DISTRICT):
    """Fraction below normal rainfall (0.18 = 18% short). Negative = surplus."""
    rain = _load()["rain"]
    key = district.upper()
    if key not in rain.index:
        key = DEFAULT_DISTRICT
    return -float(rain.loc[key, "Dev_pct"]) / 100.0


def irrigation_dependency(district=DEFAULT_DISTRICT):
    """How much of crop water demand rainfall cannot cover, 0-1.

    Canal supply matters most where crops need more water than the sky
    delivers. This is a coarse annual water balance, not a soil-moisture
    model: it ignores timing, which is exactly what makes a canal valuable.
    Treat it as a relative weight between districts, not an absolute deficit.
    """
    rain = _load()["rain"]
    key = district.upper() if district.upper() in rain.index else DEFAULT_DISTRICT
    actual = float(rain.loc[key, "Actual_mm"])
    demand = water_demand_mm(district)
    if demand <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - actual / demand))


def build_segment_districts():
    """Assign a district to every canal segment by spatial join, cached to CSV.

    Uses FAO GAUL level-2 admin boundaries in Earth Engine rather than guessing
    from the canal's project name, because several delta systems (Cauvery Delta
    System, Lower Colleron) span more than one district and their crop mixes
    differ materially. Run once; the CSV is reused thereafter.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import geopandas as gpd
    import ee

    out = config.DATA_DIR / "segment_district.csv"
    ee.Initialize(project=config.EE_PROJECT)

    segs = gpd.read_file(config.SEGMENTS_GPKG).to_crs(config.CRS_WGS84)
    cent = segs.geometry.centroid
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")

    rows, batch = [], 300
    for i in range(0, len(segs), batch):
        sl = segs.iloc[i:i + batch]
        cs = cent.iloc[i:i + batch]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([p.x, p.y]), {"seg_id": s})
            for s, p in zip(sl["seg_id"], cs)
        ])
        joined = gaul.filterBounds(fc.geometry()).map(
            lambda f: f.set("n", f.get("ADM2_NAME")))
        sampled = fc.map(
            lambda f: f.set("district", ee.Algorithms.If(
                joined.filterBounds(f.geometry()).size().gt(0),
                joined.filterBounds(f.geometry()).first().get("ADM2_NAME"),
                "UNKNOWN")))
        for f in sampled.getInfo()["features"]:
            p = f["properties"]
            rows.append({"seg_id": p["seg_id"],
                         "district": str(p.get("district", "UNKNOWN")).upper()})
        print(f"  {min(i + batch, len(segs))}/{len(segs)} segments")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df["district"].value_counts().to_string())
    return df


# Fallback used when no spatial join has been run: the irrigation project a
# canal belongs to is a decent proxy for its district, and is already in the
# source data. Approximate — prefer build_segment_districts().
PROJECT_TO_DISTRICT = {
    "Pullambadi Canal": "TIRUCHIRAPPALLI",
    "Kattalai": "KARUR",
    "Cauvery Mettur": "TIRUCHIRAPPALLI",
    "Cauvery Delta System": "THANJAVUR",
    "Lower Colleron Anicut System": "NAGAPATTINAM",
    "Nandhiyar Channel": "THANJAVUR",
    "Pelandhurai Anicut System": "TIRUCHIRAPPALLI",
}

# FAO GAUL spells several TN districts differently from the state crop
# statistics, and the AOI clips a little of Puducherry (Karaikal), which has no
# row in the TN tables. Map both onto the nearest district we have data for.
DISTRICT_ALIASES = {
    "TIRUCHCHIRAPPALLI": "TIRUCHIRAPPALLI",
    "TIRUCHIRAPALLI": "TIRUCHIRAPPALLI",
    "THIRUVARUR": "THIRUVARUR",
    "TIRUVARUR": "THIRUVARUR",
    "NAGAPPATTINAM": "NAGAPATTINAM",
    "MAYILADUTHURAI": "NAGAPATTINAM",
    # Karaikal (Puducherry UT) is enclosed by the Nagapattinam delta and grows
    # the same crops; no separate TN statistics row exists for it.
    "KARAIKAL": "NAGAPATTINAM",
}


def normalise_district(name):
    if not name:
        return DEFAULT_DISTRICT
    n = str(name).strip().upper()
    return DISTRICT_ALIASES.get(n, n)


_DISTRICT_LOOKUP = {}


def district_for_segment(seg_id, prj_name=None):
    """District of a segment: spatial join if available, else project name."""
    global _DISTRICT_LOOKUP
    if not _DISTRICT_LOOKUP:
        path = config.DATA_DIR / "segment_district.csv"
        if path.exists():
            d = pd.read_csv(path)
            _DISTRICT_LOOKUP = dict(zip(d["seg_id"], d["district"]))
        else:
            _DISTRICT_LOOKUP = {"__empty__": True}

    known = set(_load()["prod"]["District"].unique())
    d = normalise_district(_DISTRICT_LOOKUP.get(seg_id))
    if d in known:
        return d
    d = normalise_district(PROJECT_TO_DISTRICT.get(prj_name))
    return d if d in known else DEFAULT_DISTRICT


def summary_table():
    prod = _load()["prod"]
    rows = []
    for d in sorted(prod["District"].unique()):
        mix = crop_mix(d)
        rows.append({
            "district": d,
            "top_crop": mix.index[0],
            "top_share": round(float(mix.iloc[0]), 3),
            "rice_share": round(float(mix.get("Rice", 0)), 3),
            "cane_share": round(float(mix.get("Sugarcane", 0)), 3),
            "revenue_inr_ha": round(revenue_per_ha(d)),
            "water_mm": round(water_demand_mm(d)),
            "rain_mm": float(_load()["rain"].loc[d, "Actual_mm"])
            if d in _load()["rain"].index else None,
            "irr_dependency": round(irrigation_dependency(d), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "districts":
        build_segment_districts()
        sys.exit(0)
    print("=" * 78)
    print("DISTRICT CROP ECONOMICS  (TN crop statistics 2017, rainfall 2023-24)")
    print("=" * 78)
    print(summary_table().to_string(index=False))
    print()
    print("Crop mix, Thanjavur (share of cropped area):")
    print((crop_mix("THANJAVUR") * 100).round(1).head(8).to_string())
    print()
    print("!! Only the paddy price is SOURCED (CCEA Kharif 2025-26). Every other")
    print("   crop price and yield here is an ASSUMED order-of-magnitude value.")
    print("   Replace from the CACP schedule before any operational use.")
