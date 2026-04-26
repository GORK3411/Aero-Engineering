# AeroML3 — Oceanic Flight Trajectory Reconstruction

Reconstructs aircraft positions during oceanic ADS-B gaps using three methods:
a great-circle baseline, an anchored Kalman smoother, and a BiGRU deep-learning model.
Ground truth is ADS-C waypoints (held out during reconstruction, used only for evaluation).

---

## Results at a Glance

| Method | Mean Error | Median Error | vs Baseline |
|---|---|---|---|
| Great-circle baseline | 140.3 km | 109.8 km | — |
| Kalman smoother | 111.3 km | 88.2 km | **−20.7 %** |
| BiGRU model | see notebook | see notebook | see notebook |

Evaluated on 291 held-out test flights. The Kalman smoother is validated-tuned and
always connects exactly to the ADS-B endpoints (no floating trajectories).

---

## Project Layout

```
AeroML3/
├── notebook/                   ← main pipeline notebooks (run these in order)
│   ├── 01_ingest.ipynb
│   ├── 02_clean.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_step4_dataset.ipynb
│   ├── 04_model .ipynb
│   ├── 05a_step5_kalman_evaluation.ipynb
│   ├── 05_evaluate .ipynb
│   ├── 06_analytics.ipynb
│   ├── 07_serve .ipynb         ← visualization & GeoJSON export
│   ├── 08_monitoring.ipynb
│   └── step5_kalman_aeroml3.py ← Kalman smoother implementation
│
├── noel_part/                  ← ETL, BiGRU training, reconstruction helpers
│   ├── etl.ipynb
│   ├── model_training.ipynb
│   ├── visualization.ipynb
│   ├── reconstruction.py
│   ├── cleaned_data_final/     ← cleaned per-flight parquet files (pre-built)
│   └── models/
│       └── BiGRU.pth           ← trained model weights
│
├── artifacts/
│   ├── step4_ml_dataset/       ← sequence dataset (train / val / test)
│   └── step5_kalman/           ← Kalman evaluation results
│       ├── test_summary.json
│       ├── test_predictions.npz
│       └── per_flight_metrics_test.csv
│
├── raw_data/                   ← raw ADS-B / ADS-C parquet files
├── outputs/                    ← GeoJSON exports for geojson.io
└── requirements.txt
```

---

## Prerequisites

- **Python 3.10**
- **Jupyter** (JupyterLab or classic Notebook)
- A GPU is optional — the BiGRU model runs on CPU as well

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages used: `torch`, `numpy`, `pandas`, `pyproj`, `scipy`, `scikit-learn`,
`matplotlib`, `pyarrow`, `joblib`.

---

## Quick Start — Just the Visualization (recommended for reviewers)

Everything is pre-built. To jump straight to the interactive map:

1. Open **`notebook/07_serve .ipynb`** in Jupyter.
2. Run **Kernel → Restart Kernel and Run All Cells**.
3. After Cell 5 finishes, a file appears at:
   ```
   outputs/<FLIGHT_NAME>_final_comparison.geojson
   ```
4. Go to **[geojson.io](https://geojson.io)** and drag that file onto the map.

You will see:

| Colour | Meaning |
|---|---|
| Grey | ADS-B context (30 min before and after the gap) |
| Yellow | ADS-C ground truth (never used during reconstruction) |
| Red | Baseline — constant-speed great-circle interpolation |
| Cyan | Kalman smoother — anchored RTS, tuned on validation set |
| Green | BiGRU deep-learning model |

Cell 6 (optional) lists the 30 flights with the largest Kalman improvement.
Copy any `flight_name` value into Cell 2 and re-run to view a different flight.

---

## Full Pipeline (step by step)

Run the notebooks **in the order below**. Each step saves its output to `artifacts/`
so you can re-run any later step independently without re-running earlier ones.

### Step 0 — ETL (already done, skip if cleaned data exists)

```
noel_part/etl.ipynb
```

Reads raw ADS-B / ADS-C parquets from `raw_data/`, aligns timestamps,
and writes per-flight folders to `noel_part/cleaned_data_final/`.
Each folder contains three files: `adsb_before.parquet`, `adsc.parquet`, `adsb_after.parquet`.

> Only needed if you want to regenerate from scratch. The `cleaned_data_final/` folder
> is already included.

---

### Step 1 — Ingest

```
notebook/01_ingest.ipynb
```

Validates the raw data inventory and prints a summary of available flights and date ranges.

---

### Step 2 — Clean

```
notebook/02_clean.ipynb
```

Applies quality filters (duplicate removal, timestamp coercion, coordinate range checks)
and reports data quality statistics.

---

### Step 3 — Baseline

```
notebook/03_baseline.ipynb
```

Establishes the great-circle interpolation baseline.
Computes per-flight errors against ADS-C ground truth and saves summary statistics.

---

### Step 4 — Build ML Dataset

```
notebook/04_step4_dataset.ipynb
```

Converts the cleaned flights into fixed-length numpy sequences (train / val / test split)
suitable for the BiGRU model and Kalman smoother.

Output: `artifacts/step4_ml_dataset/dataset/sequences_{train,val,test}.npz`

---

### Step 4b — Train BiGRU (optional, model is pre-trained)

```
noel_part/model_training.ipynb
```

Trains the BiGRU trajectory model. The trained weights are already saved at
`noel_part/models/BiGRU.pth` — skip this step unless you want to retrain.

---

### Step 5 — Kalman Smoother Evaluation

```
notebook/05a_step5_kalman_evaluation.ipynb
```

Runs a grid search over Kalman noise parameters on the validation set,
then evaluates the best configuration on the test set.

**Key results saved to `artifacts/step5_kalman/`:**
- `test_summary.json` — aggregate metrics
- `per_flight_metrics_test.csv` — per-flight breakdown
- `test_predictions.npz` — lat/lon predictions for all test flights

To re-run the full tuning (takes ~5 minutes):
```python
RUN_KALMAN  = True    # top of the notebook
TUNING_GRID = 'compact'
```

To just load existing results without re-running:
```python
RUN_KALMAN = False
```

---

### Step 5b — BiGRU Evaluation

```
notebook/05_evaluate .ipynb
```

Evaluates the trained BiGRU model on the test set and compares it against
the baseline and Kalman smoother.

---

### Step 6 — Analytics

```
notebook/06_analytics.ipynb
```

Generates summary plots: error distributions, improvement histograms,
error vs. gap position (tau), and per-flight scatter plots.

---

### Step 7 — Visualization (interactive globe)

```
notebook/07_serve .ipynb
```

Reconstructs a chosen flight with all three methods and exports a GeoJSON
file for [geojson.io](https://geojson.io).

**How to use:**

1. (Optional) Run **Cell 6** first to see a ranked table of the best flights to visualize.
2. Paste a `flight_name` from that table into **Cell 2** (`FLIGHT_NAME = "..."`).
3. Run **Cells 1 → 5** in order.
4. Cell 3 prints an anchor verification line — both gaps should be 0.0 m:
   ```
   Kalman : 849 steps  start_gap=0.0m  end_gap=0.0m
   ```
5. Drag the file from `outputs/` to [geojson.io](https://geojson.io).

> **Important:** always use **Kernel → Restart Kernel and Run All Cells** when
> opening this notebook. This ensures the latest Kalman code is loaded — Jupyter
> caches imported modules and will silently use an old version otherwise.

---

### Step 8 — Monitoring

```
notebook/08_monitoring.ipynb
```

Generates a monitoring report (`outputs/monitoring_report.json`) with data
freshness checks and pipeline health statistics.

---

## How the Kalman Smoother Works

The smoother operates in **along-track / cross-track coordinates** relative to the
great-circle path between the gap endpoints.

1. **Initialisation** — state set to position `(0, 0)` (= before anchor) with velocity
   estimated from the last ADS-B heading and speed.
2. **Forward pass** — constant-velocity prediction with a single terminal measurement
   at `t1` (the after anchor), uncertainty tuned to 200 km on the validation set.
3. **RTS backward pass** — smooths the trajectory so it is consistent with both
   the initial heading and the terminal constraint.
4. **Hard anchoring** — after smoothing, the boundary position states are clamped
   to `(0, 0)` and `(total_dist, 0)` so the output line always starts and ends
   exactly at the ADS-B endpoints.
5. **Coordinate conversion** — `path_state_to_latlon` maps along/cross-track metres
   back to WGS-84 latitude/longitude.

---

## File Formats

| File | Description |
|---|---|
| `adsb_before.parquet` | ADS-B track before the oceanic gap |
| `adsb_after.parquet` | ADS-B track after the oceanic gap |
| `adsc.parquet` | ADS-C waypoints (ground truth, held out) |
| `sequences_*.npz` | Numpy arrays: `adsc_tau`, `adsc_targets`, `before_anchor_lat/lon`, … |
| `test_predictions.npz` | Kalman predictions aligned with `sequences_test.npz` |
| `*_final_comparison.geojson` | GeoJSON export for geojson.io |

---

## Troubleshooting

**"Flight not found" error in Cell 2 of notebook 07**
Run Cell 6 first to get a valid list of flight names, then copy one into Cell 2.

**Kalman line still looks disconnected on the map**
Go to **Kernel → Restart Kernel and Run All Cells**. Jupyter caches imported
modules; a fresh kernel always loads the latest code.

**`FileNotFoundError` for `sequences_test.npz`**
Run `notebook/04_step4_dataset.ipynb` first to build the ML dataset.

**`FileNotFoundError` for `BiGRU.pth`**
The model weights must be in `noel_part/models/BiGRU.pth`.
Either run `noel_part/model_training.ipynb` or restore the file from the archive.

**CUDA / GPU warnings**
Harmless — the pipeline falls back to CPU automatically.

---

## Authors

Marko Sharb — pipeline design, Kalman smoother, evaluation framework  
Noel — ETL, BiGRU model training, reconstruction helpers
