# AYACUT

Ranks irrigation canal segments in the Cauvery delta by **desilting priority**,
so a limited budget goes to the canals that protect the most farmland — not the
canals that generate the most complaints.

## What it claims, and what it doesn't

The model detects **surface signatures of impaired conveyance**: vegetation
growing in a channel relative to the cropland beside it, and the absence of
water when water should be flowing. Both are visible to Sentinel-2.

It does **not** detect siltation. Bed-level sediment is under the water surface
and optical satellites cannot see it. A low health score means "this reach looks
like it isn't conveying properly, send someone to check" — not "this reach has
N cubic metres of silt". Every output carries that caveat.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0 | `step00_fetch_rivers.py` | `data/rivers.gpkg` — OSM river connectors |
| 1 | `step01_build_graph.py` | `data/segments.gpkg`, `data/graph.pkl` |
| 2 | `step02_extract_features.py` | `data/features.csv` — Sentinel-2 features |
| 3 | `step03_label_points.py` | `data/labels.csv` — hand labels |
| 4 | `step04_train_classifier.py` | `data/segment_health.csv` |
| 5 | `step05_rank_segments.py` | `data/ranked_segments.gpkg` |
| 6 | `step06_allocate_budget.py` | `data/work_plan.gpkg` |
| 7 | `step07_simulate_failure.py` | `data/failure_simulations.csv` |
| 8 | `step08_build_map.py` | `ayacut_map.html` |

```bash
pip install -r requirements.txt
earthengine authenticate

python step00_fetch_rivers.py
python step01_build_graph.py
python step02_extract_features.py

python step03_label_points.py fetch      # download ~300 chips
python step03_label_points.py label      # label them in the browser
python step04_train_classifier.py        # or: ... provisional

python step05_rank_segments.py fetch     # cropland grid, once
python step05_rank_segments.py
python step06_allocate_budget.py --budget 50000000
python step07_simulate_failure.py fetch  # crop grid, once
python step07_simulate_failure.py --top 5
python step08_build_map.py chips         # segment chips, once
python step08_build_map.py
```

Every stage is restartable and every stage writes a file the next stage reads,
so you can stop anywhere and still have something to show.

## How the priority score works

```
priority = command_area_ha × (1 − health_score) × (1 + α × betweenness_norm)
```

- **command_area_ha** — cropland served by this segment *and everything
  downstream of it*. Each cropland cell is allocated to exactly one segment (the
  finest-order canal within 2 km), so areas never double count.
- **health_score** — from the classifier, 1 = channel looks clear.
- **betweenness** — edge betweenness centrality; how much of the network's
  source-to-field routing passes through this segment.

A big main canal in good condition scores low. A minor serving 40 ha scores low.
A distributary that is visibly choked and sits upstream of 4,000 ha scores high.

## Why the classifier is a random forest

Small labelled dataset (~300 points) and every prediction has to be explainable
to a reviewer. The script prints per-class precision/recall, a confusion matrix,
and both Gini and permutation feature importances.

Validation splits are **grouped by segment** — points on the same canal are
spatially autocorrelated, so a plain random split leaks and reports an
accuracy that won't survive contact with new canals.

## Numbers and where they come from

`assumptions.py` holds every rupee figure, each tagged `SOURCED`, `DERIVED`,
or `ASSUMED`, with a citation. Run `python assumptions.py` to print them.

- **Desilting cost ₹2,39,644/km** — `DERIVED` from ₹53 crore for 2,211.63 km of
  river and canal desilting in Thanjavur, Tiruvarur and Nagapattinam (PWD
  2019-20, reported in Deccan Chronicle, 18 Aug 2019). A programme average in
  exactly these districts, not a schedule rate, and not inflation-adjusted.
- **Paddy MSP ₹2,369/quintal** — `SOURCED`, CCEA Kharif 2025-26 (PIB 2131983).
- **Paddy yield 3.45 t/ha** — `DERIVED` from Tamil Nadu rice yield 2.31 t/ha
  (2023-24) and the 100 kg paddy → 67 kg rice conversion norm.
- **Cost multipliers by canal order** — `ASSUMED`. No public breakdown found.
- **Cropping intensity 1.0** — `ASSUMED`, deliberately conservative; the delta
  often grows two seasons.

Replace the `ASSUMED` values with the PWD schedule of rates before this informs
an actual tender.

## Known limitations

- Flow direction comes from BFS over the network topology, not from elevation.
  Correct for tree-like irrigation networks; arbitrary where the graph loops.
- The canal layer omits rivers, so canal systems are stitched together with OSM
  river centerlines plus 32 short virtual connectors (`is_virtual=1`). Each is
  logged. They represent unmapped offtakes, not verified structures.
- `water_frac` is weak for narrow canals: a 10 m Sentinel-2 pixel over a 5 m
  channel is mostly bank. NDVI difference carries most of the signal.
- Crop classes are inferred from wet-season standing water, not a crop survey.
- Revenue figures are gross at MSP and assume total supply loss. They are for
  ranking, not compensation.
