"""Run the whole AYACUT pipeline in order.

  python run_all.py               # full run, skipping stages already cached
  python run_all.py --from 5      # rerun from step 5 onward
  python run_all.py --force       # ignore caches, redo everything

Labelling (step 3) is interactive and never runs automatically. Without
data/labels.csv, step 4 falls back to the provisional heuristic.
"""

import argparse
import subprocess
import sys
import time

import config

PY = sys.executable

# (step number, label, command, output that means "already done")
STAGES = [
    (0, "OSM river connectors", ["step00_fetch_rivers.py"], config.RIVERS_GPKG),
    (1, "Build canal graph", ["step01_build_graph.py"], config.SEGMENTS_GPKG),
    (2, "Sentinel-2 features", ["step02_extract_features.py"], config.FEATURES_CSV),
    (2, "Dynamic World features", ["step02b_dynamic_world.py"], config.DW_FEATURES_CSV),
    (2, "Segment districts", ["crop_economics.py", "districts"],
     config.DATA_DIR / "segment_district.csv"),
    (4, "Condition model", None, config.SEGMENT_HEALTH_CSV),
    (5, "Cropland grid", ["step05_rank_segments.py", "fetch"], config.CROPLAND_NPZ),
    (5, "Rank segments", ["step05_rank_segments.py"], config.RANKED_GPKG),
    (6, "Budget allocation", ["step06_allocate_budget.py"],
     config.DATA_DIR / "budget_comparison.csv"),
    (7, "Crop class grid", ["step07_simulate_failure.py", "fetch"], config.PADDY_NPZ),
    (7, "Failure simulations", ["step07_simulate_failure.py", "--top", "5"],
     config.DATA_DIR / "failure_simulations.csv"),
    # Thumbnails are no longer embedded in the app — unreadable at 10 m and 80%
    # of the file size. `step08_build_app.py chips` still fetches them on demand.
    (8, "Build web app", ["step08_build_app.py"],
     config.PROJECT_ROOT / "ayacut_app.html"),
]


def model_stage_cmd():
    if config.LABELS_CSV.exists():
        return ["step04_train_classifier.py"]
    print("  no data/labels.csv — using the PROVISIONAL heuristic.")
    print("  Run `python step03_label_points.py fetch` then `label` for a "
          "trained model.")
    return ["step04_train_classifier.py", "provisional"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    for num, label, cmd, output in STAGES:
        if num < args.start:
            continue
        header = f"[step {num}] {label}"
        if output is not None and output.exists() and not args.force:
            print(f"{header}: cached ({output.name})")
            continue
        print(f"\n{'=' * 68}\n{header}\n{'=' * 68}")
        cmd = model_stage_cmd() if cmd is None else cmd
        t0 = time.time()
        r = subprocess.run([PY] + cmd, cwd=config.PROJECT_ROOT)
        if r.returncode != 0:
            print(f"\n{header} FAILED (exit {r.returncode}) — stopping.")
            return r.returncode
        print(f"-- {label} done in {time.time() - t0:.0f}s")

    print(f"\nPipeline complete in {time.time() - t_all:.0f}s")
    print(f"Open {config.PROJECT_ROOT / 'ayacut_app.html'}")


if __name__ == "__main__":
    sys.exit(main())
