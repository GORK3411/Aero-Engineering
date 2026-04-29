# AeroML3 — Oceanic Flight Trajectory Reconstruction

Reconstructs aircraft positions during oceanic ADS-B gaps using three methods:
a great-circle baseline, an anchored Kalman smoother, and a GRU deep-learning model.
Ground truth is ADS-C waypoints (held out during reconstruction, used only for evaluation).

---

## Results at a Glance

| Method | Mean Error | Median Error | P90 Error | vs Baseline |
|---|---|---|---|---|
| Great-circle baseline | 140.3 km | 109.8 km | 259.0 km | — |
| Kalman smoother | 111.3 km | 88.2 km | 215.9 km | **−20.7 %** |
| GRU v2 | 92.6 km | 68.7 km | 157.8 km | **−33.8 %** |

Evaluated on 291 held-out test flights (1,540 train / 315 val / 291 test).  
The GRU v2 model has 976,706 parameters, was trained on Google Colab (GPU), and weights are stored at `step5_v2/best_model_v2.pt`.

---

## Project Layout

```
AeroML3/
├── notebook/                       ← main pipeline (run in order)
│   ├── 01_ingest.ipynb
│   ├── 02_clean.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_step4_dataset.ipynb
│   ├── 04_model .ipynb             ← GRU model overview
│   ├── 05a_step5_kalman_evaluation.ipynb
│   ├── 05_evaluate .ipynb          ← GRU v2 evaluation
│   ├── 06_analytics.ipynb          ← route analytics & CO2 impact
│   ├── 07_serve .ipynb             ← visualization & GeoJSON export
│   ├── demo.ipynb
│   ├── step3_baseline.py
│   ├── step4_build_ml_dataset_aeroml3.py
│   ├── step5_kalman_aeroml3.py
│   ├── step5_train_gru_v2.py       ← GRU v2 training script (Colab)
│   ├── step5_evaluate.py
│   └── step6_analytics.py
│
├── noel_part/                      ← ETL, early BiGRU exploration
│   ├── etl.ipynb                   ← raw data processing
│   ├── model_training.ipynb        ← early BiGRU (superseded by GRU v2)
│   ├── visualization.ipynb
│   ├── cleaned_data_final/         ← cleaned per-flight parquet files
│   ├── scalers/                    ← normalization scalers (early approach)
│   └── display.py
│
├── step5_v2/                       ← GRU v2 model & results (active)
│   ├── best_model_v2.pt            ← trained model weights
│   ├── test_summary.json
│   ├── test_predictions.npz
│   └── training_history.json
│
├── step5/                          ← GRU v1 (superseded)
│   └── best_model.pt
│
├── artifacts/
│   ├── step4_ml_dataset/           ← sequence dataset (train / val / test)
│   │   └── dataset/
│   │       ├── sequences_train.npz
│   │       ├── sequences_val.npz
│   │       ├── sequences_test.npz
│   │       └── normalization_stats.json
│   └── step5_kalman/               ← Kalman evaluation results
│       ├── test_summary.json
│       ├── test_predictions.npz
│       ├── per_flight_metrics_test.csv
│       └── val_tuning_results.csv
│
├── raw_data/                       ← raw ADS-B / ADS-C parquet files
├── outputs/                        ← GeoJSON exports & analytics plots
└── requirements.txt
```

---

## Prerequisites

- **Python 3.10**
- **Jupyter** (JupyterLab or classic Notebook)
- A GPU is optional for inference — the GRU model runs on CPU as well.
  Training was done on Google Colab (GPU required).

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `numpy`, `pandas`, `pyproj`, `scipy`, `scikit-learn`,
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
| Green | GRU v2 — learned trajectory model |

Cell 6 (optional) lists the 30 flights with the largest improvement.
Copy any `flight_name` value into Cell 2 and re-run to view a different flight.

---

## Full Pipeline (step by step)

Run the notebooks **in the order below**. Each step saves its output to `artifacts/`
or `step5_v2/` so later steps can be re-run independently.

### Step 0 — ETL (already done, skip if cleaned data exists)

```
noel_part/etl.ipynb
```

Reads raw ADS-B / ADS-C parquets from `raw_data/`, aligns timestamps,
and writes per-flight folders to `noel_part/cleaned_data_final/`.
Each folder contains up to three files: `adsb_before_clean.parquet`, `adsc_clean.parquet`, `adsb_after_clean.parquet`.

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

Converts the cleaned flights into fixed-length numpy sequences (1,540 train / 315 val / 291 test).

Output: `artifacts/step4_ml_dataset/dataset/sequences_{train,val,test}.npz` and `normalization_stats.json`.

---

### Step 4b — Train GRU v2 (optional, model is pre-trained)

```
notebook/step5_train_gru_v2.py   ← run on Google Colab (GPU)
```

Trains the GRU v2 trajectory model. Architecture highlights:
- Both before/after encoders bidirectional
- Attention pooling over all GRU output steps
- Smoothness loss + noise injection during training
- 976,706 parameters, 80 epochs, hidden size 128

Weights are already saved at `step5_v2/best_model_v2.pt` — skip this step unless you want to retrain.

---

### Step 5 — Kalman Smoother Evaluation

```
notebook/05a_step5_kalman_evaluation.ipynb
```

Runs a grid search over Kalman noise parameters on the validation set,
then evaluates the best configuration on the test set.

Best params: `measurement_std=200 km`, `accel_along=0.005 m/s²`, `accel_cross=4.0 m/s²`.

**Results saved to `artifacts/step5_kalman/`:**
- `test_summary.json` — aggregate metrics
- `per_flight_metrics_test.csv` — per-flight breakdown
- `val_tuning_results.csv` — full grid search results

To re-run tuning (takes ~5 minutes):
```python
RUN_KALMAN  = True
TUNING_GRID = 'compact'
```

To just load existing results without re-running:
```python
RUN_KALMAN = False
```

---

### Step 5b — GRU v2 Evaluation

```
notebook/05_evaluate .ipynb
```

Evaluates the trained GRU v2 model on the test set and compares it against
the baseline and Kalman smoother. Results saved to `outputs/evaluation_results.csv`.

---

### Step 6 — Route Analytics

```
notebook/06_analytics.ipynb
```

Quantifies the downstream impact of better reconstruction across all 2,149 flights:

1. **Route distance** — how much distance raw ADS-B misses by having no ocean coverage
2. **CO2 proxy** — how much fuel/emissions are unaccounted for without gap filling
3. **Lateral deviation** — how far each method deviates from the great-circle path

Key finding: raw ADS-B (no ocean coverage) misses ~3,700 km per flight on average.
Reconstruction with GRU v2 closes this gap and captures jet-stream routing deviations.

Output: `outputs/analytics_results.csv` and `outputs/analytics_comparison.png`.

---

### Step 7 — Visualization (interactive map)

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
> opening this notebook. Jupyter caches imported modules and will silently use
> an old version of the Kalman code otherwise.

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
4. **Hard anchoring** — boundary position states are clamped to `(0, 0)` and
   `(total_dist, 0)` so the output always starts and ends exactly at the ADS-B endpoints.
5. **Coordinate conversion** — `path_state_to_latlon` maps along/cross-track metres
   back to WGS-84 latitude/longitude.

---

## How the GRU v2 Model Works

The model takes the ADS-B context on both sides of the gap and predicts gap waypoints
as a correction on top of the great-circle baseline.

1. **Before encoder** — bidirectional GRU over the last N ADS-B points before the gap.
2. **After encoder** — bidirectional GRU over the first N ADS-B points after the gap.
3. **Attention pooling** — weighted sum over all encoder output steps.
4. **Decoder** — produces `(Δlat, Δlon)` residuals at each interpolated timestep τ.
5. **Output** — baseline lat/lon + residuals, always anchored to the gap endpoints.

Input features: latitude, longitude, velocity, heading (sin/cos), altitude — all normalized.

---

## File Formats

| File | Description |
|---|---|
| `adsb_before_clean.parquet` | ADS-B track before the oceanic gap |
| `adsb_after_clean.parquet` | ADS-B track after the oceanic gap |
| `adsc_clean.parquet` | ADS-C waypoints (ground truth, held out) |
| `sequences_*.npz` | Numpy arrays: `adsc_tau`, `adsc_targets`, `before_anchor_lat/lon`, … |
| `best_model_v2.pt` | GRU v2 PyTorch state dict |
| `test_predictions.npz` | Per-flight predictions aligned with `sequences_test.npz` |
| `*_final_comparison.geojson` | GeoJSON export for geojson.io |

---

## Troubleshooting

**"Flight not found" error in Cell 2 of notebook 07**  
Run Cell 6 first to get a valid list of flight names, then copy one into Cell 2.

**Kalman line looks disconnected on the map**  
Use **Kernel → Restart Kernel and Run All Cells**. Jupyter caches imported modules;
a fresh kernel always loads the latest Kalman code.

**`FileNotFoundError` for `sequences_test.npz`**  
Run `notebook/04_step4_dataset.ipynb` first to build the ML dataset.

**`FileNotFoundError` for `best_model_v2.pt`**  
The model weights must be at `step5_v2/best_model_v2.pt`.
Either run `notebook/step5_train_gru_v2.py` on Colab or restore the file from the archive.

**CUDA / GPU warnings**  
Harmless — the pipeline falls back to CPU automatically for inference.

---

## Authors

Marko Sharb — pipeline design, Kalman smoother, GRU v2 integration, evaluation framework  
Noel — ETL, early BiGRU exploration, reconstruction helpers
