# AYACUT

Ranks irrigation canal segments in the Cauvery delta by **desilting priority**,
so a limited budget goes to the canals that protect the most farmland — not the
canals that generate the most complaints. Output is one self-contained
`ayacut_app.html`.

**What it detects:** surface signatures of impaired conveyance — vegetation
growing in a channel relative to the cropland beside it, and channels that stay
dry when they should be carrying water. Both visible to Sentinel-2.

**What it doesn't:** siltation. Bed sediment is under the water surface and
optical satellites cannot see it. A low health score means "this reach looks like
it isn't conveying properly, send someone to check" — not "this reach has N cubic
metres of silt". Every output carries that caveat.

## Setup

Needs Python 3.10+ and a Google Earth Engine account.

```bash
pip install -r requirements.txt
earthengine authenticate
```

Two things before the first run:

1. Set `EE_PROJECT` in [config.py](config.py) to your own Earth Engine project —
   it currently points at ours.
2. Drop `canal_network.kml` in the repo root. It's the national canal layer,
   183 MB, too big to commit. Step 1 filters it to Tamil Nadu / Cauvery and
   expects the attributes `prj_name`, `can_name`, `can_type`, `basin`, `state`,
   `length_km`.

## Run

```bash
python run_all.py
```

Runs every stage in order and skips whatever is already cached in `data/`. Most
of the wall time is Earth Engine downloads. Use `--from 5` to restart partway,
`--force` to ignore caches.

Labelling is interactive, so `run_all.py` never does it for you. Without
`data/labels.csv`, step 4 falls back to a provisional heuristic and the pipeline
still completes — every output is then marked unvalidated. For a real model:

```bash
python step03_label_points.py fetch    # download ~300 chips, once
python step03_label_points.py label    # label them in the browser
python step04_train_classifier.py      # retrain on the labels
python step08_build_app.py             # rebuild the app
```

See [TRAINING.md](TRAINING.md) for what to label and how to read the metrics.

Then open `ayacut_app.html` in a browser. No server needed.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0 | `step00_fetch_rivers.py` | `data/rivers.gpkg` — OSM river connectors |
| 1 | `step01_build_graph.py` | `data/segments.gpkg`, `data/graph.pkl` |
| 2 | `step02_extract_features.py` | `data/features.csv` — Sentinel-2 features |
| 2b | `step02b_dynamic_world.py` | `data/dw_features.csv` — land-cover features |
| 3 | `step03_label_points.py` | `data/labels.csv` — hand labels |
| 4 | `step04_train_classifier.py` | `data/segment_health.csv` |
| 5 | `step05_rank_segments.py` | `data/ranked_segments.gpkg` |
| 6 | `step06_allocate_budget.py` | `data/work_plan.gpkg` |
| 7 | `step07_simulate_failure.py` | `data/failure_simulations.csv` |
| 8 | `step08_build_app.py` | `ayacut_app.html` |

Every stage is restartable and writes a file the next one reads, so you can stop
anywhere and still have something to show.

## How the priority score works

```
priority = command_area_ha × (1 − health_score) × (1 + α × betweenness_norm)
```

- **command_area_ha** — cropland served by this segment *and everything
  downstream of it*. Each cropland cell is allocated to exactly one segment (the
  finest-order canal within 2 km), so areas never double count.
- **health_score** — from the classifier, 1 = channel looks clear.
- **betweenness** — how much of the network's source-to-field routing passes
  through this segment.

A big main canal in good condition scores low. A minor serving 40 ha scores low.
A distributary that is visibly choked and sits upstream of 4,000 ha scores high.

The classifier is a random forest: ~300 labelled points is far too few for a
neural net, and every prediction has to be explainable to the engineer who acts
on it. Validation splits are **grouped by segment** — points on one canal are
spatially autocorrelated, so a plain random split leaks and reports an accuracy
that won't survive contact with new canals.

## Numbers and where they come from

`assumptions.py` holds every rupee figure, each tagged `SOURCED`, `DERIVED` or
`ASSUMED` with a citation. Run `python assumptions.py` to print them.

- **Desilting cost ₹2,39,644/km** — `DERIVED` from ₹53 crore for 2,211.63 km in
  Thanjavur, Tiruvarur and Nagapattinam (PWD 2019-20, Deccan Chronicle, 18 Aug
  2019). A programme average in exactly these districts, not a schedule rate,
  not inflation-adjusted.
- **Paddy MSP ₹2,369/quintal** — `SOURCED`, CCEA Kharif 2025-26 (PIB 2131983).
- **Paddy yield 3.45 t/ha** — `DERIVED` from TN rice yield 2.31 t/ha (2023-24)
  and the 100 kg paddy → 67 kg rice conversion.
- **Cost multipliers by canal order** — `ASSUMED`. No public breakdown found.
- **Cropping intensity 1.0** — `ASSUMED`, deliberately conservative; the delta
  often grows two seasons.

Replace the `ASSUMED` values with the PWD schedule of rates before this informs
an actual tender.

## Known limitations

- Flow direction comes from BFS over network topology, not elevation. Correct
  for tree-like irrigation networks, arbitrary where the graph loops.
- The canal layer omits rivers, so systems are stitched together with OSM river
  centerlines plus 32 short virtual connectors (`is_virtual=1`). Each is logged.
  They stand for unmapped offtakes, not verified structures.
- `water_frac` is weak for narrow canals: a 10 m Sentinel-2 pixel over a 5 m
  channel is mostly bank. NDVI difference carries most of the signal.
- Crop classes are inferred from wet-season standing water, not a crop survey.
- Revenue is gross at MSP and assumes total supply loss. For ranking, not
  compensation.
