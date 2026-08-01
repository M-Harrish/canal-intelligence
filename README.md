<<<<<<< HEAD
# canal-intelligence
AI based canal intelligence
=======

## Setup

Python 3.10+ and an Earth Engine account.

```bash
pip install -r requirements.txt
earthengine authenticate
```

Point `EE_PROJECT` in `config.py` at your own Earth Engine project, and put
`canal_network.kml` (183 MB, not committed) in the repo root.

```bash
python run_all.py
```

Skips whatever is already cached in `data/`. Labelling is interactive — without
`data/labels.csv` step 4 uses a provisional heuristic and the run still finishes.

Open `ayacut_app.html` when it's done.
>>>>>>> base-code
