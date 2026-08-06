# Weather–Wildfire Model: Dataset Preparation

This notebook (`WeatherWildfireModel_PrepareDataSet.ipynb`) builds the training dataset for a wildfire risk model focused on Alaska. It joins historical wildfire records with nearby weather station observations, engineers pre-fire weather features, and computes a composite weather-driven fire risk score for each fire.

## What it does

1. **Loads raw data** from two folders of CSV files:
   - `./WeatherDataFromAcrossAlaska` — daily weather station observations
   - `./WildFireDataFromAcrossAlaska` — historical wildfire incident records
2. **Scores each fire** on a 0–100 `WEATHER_FIRE_RISK_SCORE`, combining ignition cause, seasonality, terrain aspect, elevation, slope, and fire spread rate.
3. **Matches each fire to the nearest weather station** using Haversine distance, then pulls that station's observations from the 7, 30, and 90 days *before* the fire's discovery date (strictly before, to avoid data leakage).
4. **Engineers weather features** — drying trends, drought accumulation, and extreme-event counts (heat, wind, low humidity) — for each fire.
5. **Filters and exports** a final training set (fires discovered between 1990-05-01 and 2012-08-30) to `filtered_training_data_selectedColumn_Category_together.csv`.
6. **Visualizes** fire locations on an interactive map and plots a correlation heatmap of the engineered features.

## Requirements

- Python 3.13 (developed with 3.13.7)
- Packages: `pandas`, `numpy`, `folium`, `seaborn`, `matplotlib`, `IPython`

```bash
pip install pandas numpy folium seaborn matplotlib ipython
```

## Expected input data

Place raw CSVs in two sibling folders next to the notebook:

- **`WeatherDataFromAcrossAlaska/*.csv`** — must include: `Latitude`, `Longitude`, `Date`, `Max Temperature`, `Min Temperature`, `Precipitation`, `Wind`, `Relative Humidity`, `Solar`.
- **`WildFireDataFromAcrossAlaska/*.csv`** — must include: `LATITUDE`, `LONGITUDE`, `DISCOVERYDATETIME`, `OUTDATE`, `CONTROLDATETIME`, `GENERALCAUSE`, `ORIGINASPECT`, `ORIGINELEVATION`, `ORIGINSLOPE`, `ESTIMATEDTOTALACRES`.

> **Note:** some source weather CSVs are known to have trailing commas and two-digit years. The first cell contains (commented-out) `perl` one-liners to fix both issues before loading.

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1–3 | Data-cleanup notes, imports, display options |
| 4 | Target location (Fairbanks, AK) and search radius (250 mi) constants |
| 5 | `readAllCSVFiles()` — bulk-load all CSVs in a folder into one DataFrame |
| 6 | `calculate_haversine_distance()` — great-circle distance between two coordinates |
| 7 | `find_nearest_weather_coordinate()` — cross-joins fires with weather stations to find the closest one per fire |
| 8 | `filter_data_within_max_radius()` — restricts records to within a radius of a target point |
| 9 | `calculate_fire_weather_risk()` — computes the six risk sub-scores and composite `WEATHER_FIRE_RISK_SCORE` |
| 10 | Weather feature engineering: `_get_weather_window()` plus 7-day, 30-day, 90-day, and extreme-event feature functions, assembled by `build_weather_features()` |
| 11 | `create_map()` — Folium map of fire locations with hoverable tooltips |
| 12 | `plot_correlation_heatmap()` — correlation heatmap for selected columns |
| 13–17 | Load data, compute risk scores, find nearest stations, parse dates |
| 18–19 | Build the full feature set, filter to the 1990–2012 window, export to CSV |
| 20 | Render the fire-location map (`FireSpots.html`) |
| 21–22 | Define columns to exclude from model training and plot their correlation heatmap |

## Key outputs

- **`filtered_training_data_selectedColumn_Category_together.csv`** — the final training dataset: one row per wildfire event with weather features, risk sub-scores, and `WEATHER_FIRE_RISK_SCORE` / `WEATHER_RISK_CATEGORY` as potential targets.
- **`FireSpots.html`** — interactive map of fire locations.

## Risk score components (`WEATHER_FIRE_RISK_SCORE`, 0–100)

| Component | Range | Basis |
|---|---|---|
| Ignition–weather link | 0–15 | Lightning-caused fires score highest; human-caused fires lowest |
| Seasonal climatology | 0–20 | Peaks June–July per Alaska's fire season |
| Solar aspect exposure | 0–15 | South/southwest-facing origins dry out fastest |
| Elevation / fuel-moisture band | 0–15 | Peaks at 1,500–2,500 ft (interior black-spruce belt) |
| Slope–wind interaction | 0–10 | Steeper terrain amplifies wind-driven spread |
| Fire spread rate | 0–25 | Acres burned per day; falls back to a size-only score if duration is unresolvable |

Fires are also bucketed into `WEATHER_RISK_CATEGORY`: Very Low, Low, Moderate, High, Extreme.

## Weather feature windows

For each fire, features are computed only from weather data strictly **before** its discovery date:

- **7-day**: short-term drying (temps, humidity, precip, wind, solar, dry-day count) and extreme-event counts (heat/wind/low-RH days)
- **30-day**: medium-term drought signal, including a precipitation anomaly vs. climatological mean
- **90-day**: seasonal drought build-up (cumulative temp, precip, humidity, solar)

## Known limitations / things to check before using downstream

- `build_weather_features()` loops row-by-row over every fire (`iterrows`), which will be slow on large datasets.
- `find_nearest_weather_coordinate()` (Cell 7) computes `NEAREST_WEATHER_DIST` via a full cross join but does not return the reduced/argmin DataFrame — check that it returns the expected columns before relying on it downstream.
- Several CSV columns (`GENERALCAUSE`, `ORIGINASPECT`, `ORIGINELEVATION`, `ORIGINSLOPE`) are matched against hard-coded value maps; unrecognized or differently formatted values silently fall back to a default mid-range score.
- `EXCLUDE_COLS` (Cell 21) is defined but not applied to `filtered_training_data_selectedColumn` before it's exported — confirm it's actually used wherever the model-training script consumes this CSV.

---
# Weather Wildfire Model: Build DNN Models

This notebook (`WeatherWildfireModel_BuildDNNModels.ipynb`) trains deep-learning models — a PyTorch feedforward network (DNN) and TabNet — to predict wildfire risk from the engineered weather/fire feature set, for both a classification task (risk category) and a regression task (risk score).

## What it does

1. **Loads the prepared training CSV** (`filtered_training_data_selectedColumn_Category_together.csv`, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`).
2. Builds two task-specific views of the data:
   - **Classification**: predict `WEATHER_RISK_CATEGORY` (Very Low/Low/Moderate/High/Extreme), with `WEATHER_FIRE_RISK_SCORE` dropped from the inputs to avoid leakage.
   - **Regression**: predict `WEATHER_FIRE_RISK_SCORE`, with `WEATHER_RISK_CATEGORY` dropped from the inputs.
3. **Splits each task** 70/30 train/test (stratified for classification), keeping `LATITUDE`, `LONGITUDE`, `NEAREST_WEATHER_LAT`, `NEAREST_WEATHER_LON`, and `DISCOVERYDATETIME` out of the model inputs but carrying their test-split values through for reference in the results.
4. **Trains two model types per task**:
   - A small **DNN** (PyTorch `Linear(128) → ReLU → Dropout → Linear(64) → ReLU → Dropout → Linear(output)`), trained with Adam + cross-entropy (classification) or MSE (regression).
   - **TabNet** (via `pytorch-tabnet`), a tabular-specific deep learning architecture, using its built-in classifier/regressor.
5. **Evaluates and compares** both model types per task — accuracy/precision/recall/F1 for classification, R²/MAE/RMSE/percentage-accuracy for regression — and saves results and comparison tables to CSV.
6. **Visualizes predictions on a map**, plotting actual-vs-predicted results by location.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`, `folium`, `torch`, `pytorch-tabnet`

```bash
pip install pandas numpy scikit-learn folium torch pytorch-tabnet
```

Both `torch` and `pytorch-tabnet` are optional at import time — if either is missing, the corresponding model is skipped with a printed note rather than raising an error.

## Expected input data

`filtered_training_data_selectedColumn_Category_together.csv` — the exact output of `WeatherWildfireModel_PrepareDataSet.ipynb`, containing the engineered weather features plus `WEATHER_FIRE_RISK_SCORE` and `WEATHER_RISK_CATEGORY`.

## Configuration (Cells 2–4)

| Setting | Default | Meaning |
|---|---|---|
| `TARGET_LAT` / `TARGET_LON` / `MAX_RADIUS` | Fairbanks, AK / 250 mi | Used for map centering |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `DATA_PATH` | `filtered_training_data_selectedColumn_Category_together.csv` | Input CSV |
| `CLASSIFICATION_TARGET` | `WEATHER_RISK_CATEGORY` | Classification label |
| `REGRESSION_TARGET` | `WEATHER_FIRE_RISK_SCORE` | Regression label |
| `EXCLUDE_COLS` | lat/lon/station-coords/date columns + both targets | Columns kept for tracking but excluded from model inputs |

## Data leakage note

`WEATHER_RISK_CATEGORY` is a deterministic binning of `WEATHER_FIRE_RISK_SCORE`. Each task drops the *other* task's target column from its input features before training, so neither model can simply read the answer off the other target.

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring: pipeline overview, leakage note |
| 2 | Imports (`torch` and `pytorch-tabnet` wrapped in try/except) |
| 3 | Map target location/radius constants |
| 4 | Run configuration constants (paths, targets, random state) |
| 5 | `EXCLUDE_COLS` definition |
| 6 | `SimpleMLP` — the PyTorch feedforward network architecture |
| 7 | `DNNWrapper` / `TabNetWrapper` — give both model types a uniform `.predict()` interface |
| 8 | `load_data()` — reads the CSV |
| 9 | `split_data()` — 70/30 split (stratified for classification), separates excluded/tracking columns |
| 10 | `train_dnn()` / `train_tabnet()` — training functions for each model type |
| 11 | `test_model()` — runs predictions, returns actual-vs-predicted DataFrame with tracking columns reattached |
| 12 | `create_map()` — Folium map of predictions with hoverable tooltips |
| 13 | `evaluate_models()` — computes classification or regression metrics per model |
| 14 | `compare_models()` — prints and optionally saves the comparison table |
| 15 | Loads data; builds classification/regression input views |
| 16 | Splits data for both tasks |
| 17 | Trains DNN + TabNet for classification |
| 18 | Tests classification models, saves results CSVs, plots prediction maps |
| 19 | Evaluates + compares classification models |
| 20 | Trains DNN + TabNet for regression |
| 21 | Tests regression models, saves results CSVs, plots prediction maps |
| 22 | Evaluates + compares regression models |

## Key outputs

- **`clf_results_dnn.csv`**, **`clf_results_tabnet.csv`** — per-record actual vs. predicted risk category, with tracking columns.
- **`reg_results_dnn.csv`**, **`reg_results_tabnet.csv`** — per-record actual vs. predicted risk score, with tracking columns.
- **`dnn_classification_model_comparison_DNN.csv`** — DNN vs. TabNet classification metrics.
- **`regression_model_comparison_DNN.csv`** — DNN vs. TabNet regression metrics.
- **`dnn_predicted_vs_actual.html`**, **`tabnet_predicted_vs_actual.html`** — interactive maps of predictions (note: regenerated for both classification and regression runs under the same filenames — see limitations below).

## Known limitations / things to check before using downstream

- **Map output filenames collide across tasks**: `dnn_predicted_vs_actual.html` and `tabnet_predicted_vs_actual.html` are written in both Cell 18 (classification) and Cell 21 (regression) with the same `save_path`, so the regression run's maps overwrite the classification run's maps. Rename one set (e.g. add a `_clf`/`_reg` suffix) if you need both.
- The DNN uses a fixed architecture and fixed 100 epochs with no early stopping or validation-loss monitoring — unlike TabNet, which supports `patience`-based early stopping natively.
- `evaluate_models()`'s regression "Percentage Accuracy (100-MAPE)" can go negative or become misleading if actual values are at or near zero, since MAPE is undefined/unstable near zero.
- Model inputs are not explicitly checked for `NaN`s before training; any missing values in the weather feature columns (e.g. from short weather windows near the start of the station record) will propagate into `StandardScaler`/PyTorch and likely raise errors or silently produce `NaN` predictions.
- As in the earlier notebooks, `EXCLUDE_COLS` still contains commented-out lines showing which columns were experimented with — worth reviewing if you plan to change the feature set.

---

# Weather Wildfire Model: Build Zero-Shot Model (TabFM)

This notebook (`WeatherWildfireModel_BuildZeroShotModel.ipynb`) evaluates **TabFM**, a zero-shot tabular foundation model from Google Research, on the same wildfire risk classification and regression tasks as the DNN notebook — but without any gradient-based training on this dataset.

## What it does

1. **Loads the prepared training CSV** (`filtered_training_data_selectedColumn_Category_together.csv`, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`).
2. Builds the same two task-specific views as the DNN notebook:
   - **Classification**: predict `WEATHER_RISK_CATEGORY`, with `WEATHER_FIRE_RISK_SCORE` dropped from inputs.
   - **Regression**: predict `WEATHER_FIRE_RISK_SCORE`, with `WEATHER_RISK_CATEGORY` dropped from inputs.
3. **Splits each task** 70/30 train/test (stratified for classification), excluding `LATITUDE`, `LONGITUDE`, `NEAREST_WEATHER_LAT`, `NEAREST_WEATHER_LON`, and `DISCOVERYDATETIME` from model inputs while carrying their test-split values through for reference.
4. **Fits TabFM** for each task. Unlike the DNN/TabNet notebook, TabFM does not learn weights via gradient descent — `.fit()` prepares encoders/scalers and stores the training data as "context," and predictions at inference time come from in-context learning over that context (true zero-shot/few-shot tabular inference).
5. **Evaluates** each task with the same metrics as the DNN notebook (accuracy/precision/recall/F1 for classification; R²/MAE/RMSE/percentage-accuracy for regression) and saves results to CSV.
6. **Plots predictions on a map** via `create_map()`.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`
- **TabFM** — not on PyPI; install from source:

```bash
pip install pandas numpy scikit-learn
git clone https://github.com/google-research/tabfm.git
cd tabfm && pip install -e .[pytorch]
```

If `tabfm` isn't importable, `train_tabfm()` prints an install note and returns `None`, and downstream cells skip that task's results.

## Expected input data

`filtered_training_data_selectedColumn_Category_together.csv` — same input as the DNN notebook, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`.

## Configuration (Cells 2–3)

| Setting | Default | Meaning |
|---|---|---|
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `DATA_PATH` | `filtered_training_data_selectedColumn_Category_together.csv` | Input CSV |
| `CLASSIFICATION_TARGET` | `WEATHER_RISK_CATEGORY` | Classification label |
| `REGRESSION_TARGET` | `WEATHER_FIRE_RISK_SCORE` | Regression label |
| `EXCLUDE_COLS` | lat/lon/station-coords/date columns | Columns kept for tracking but excluded from model inputs |

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring: TabFM overview, leakage note, install instructions |
| 2 | Imports (`tabfm` wrapped in try/except) |
| 3 | Run configuration constants |
| 4 | `EXCLUDE_COLS` definition |
| 5 | `load_data()` — reads the CSV |
| 6 | `split_data()` — 70/30 split (stratified for classification), separates excluded/tracking columns |
| 7 | `train_tabfm()` — fits TabFM as a zero-shot classifier or regressor |
| 8 | `test_model()` — runs predictions, returns actual-vs-predicted DataFrame with tracking columns reattached |
| 9 | `evaluate_models()` — computes classification or regression metrics |
| 10 | `compare_models()` — prints and optionally saves the comparison table |
| 11 | Loads data |
| 12 | Builds classification/regression input views |
| 13 | Splits data for both tasks |
| 14 | Trains TabFM for classification |
| 15 | Tests classification model, saves results CSV, plots prediction map |
| 16 | Evaluates + compares classification results |
| 17 | Trains TabFM for regression |
| 18 | Tests regression model, saves results CSV, plots prediction map |
| 19 | Evaluates + compares regression results |

## Key outputs

- **`clf_results_tabfm.csv`** — per-record actual vs. predicted risk category, with tracking columns.
- **`reg_results_tabfm.csv`** — per-record actual vs. predicted risk score, with tracking columns.
- **`classification_model_comparison_ZeroshotModels.csv`** — TabFM classification metrics.
- **`regression_model_comparison_Zeroshot.csv`** — TabFM regression metrics.
- **`tabfm_predicted_vs_actual.html`** — interactive map of predictions (see limitation below — only ever reflects the classification results as currently written).

## Known limitations / things to check before using downstream

- **`create_map()` is called (Cells 14 and 17) but never defined in this notebook.** It must be defined/imported from elsewhere (e.g. by having run the DNN notebook's definition in the same kernel session) or these cells will raise a `NameError`. Copy the `create_map()` function from `WeatherWildfireModel_BuildDNNModels.ipynb` into this notebook if running it standalone.
- **Cell 17 (regression testing) plots the wrong DataFrame**: it calls `create_map(tabfm_clf_results_df, ...)` instead of `tabfm_reg_results_df`, and saves to the same `tabfm_predicted_vs_actual.html` path used in Cell 14 — so the regression cell overwrites the classification map with a re-plot of the classification results rather than showing regression predictions. Fix by changing the argument to `tabfm_reg_results_df` and using a distinct output filename.
- TabFM's `.fit()` doesn't train weights on this data — it stores training rows as in-context examples, so prediction speed and memory use scale with training-set size in a way gradient-trained models don't; this can be slow/heavy for large datasets.
- As with the DNN notebook, inputs aren't explicitly checked for `NaN`s before fitting; missing values in weather feature columns could cause errors or degrade TabFM's in-context predictions.
- `EXCLUDE_COLS` here does *not* comment out `WEATHER_FIRE_RISK_SCORE`/`WEATHER_RISK_CATEGORY` the way the DNN notebook's list does — leakage protection instead relies entirely on the separate `df_for_classification`/`df_for_regression` drop in Cell 11, so keep that step intact if you modify the pipeline.

---

# Weather Wildfire Model: Build ML Models (Random Forest / XGBoost / SVM)

This notebook (`1_WeatherWildfireModel_BuildMLModels.ipynb`) trains and compares three classical machine-learning models — Random Forest, XGBoost, and SVM — on the same wildfire risk classification and regression tasks used in the DNN and TabFM notebooks. Unlike those, it is fully self-contained: it defines its own `create_map()` helper rather than depending on another notebook's kernel state.

## What it does

1. **Loads the prepared training CSV** (`filtered_training_data_selectedColumn_Category_together.csv`, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`).
2. Builds the same two task-specific views used elsewhere in this project:
   - **Classification**: predict `WEATHER_RISK_CATEGORY`, with `WEATHER_FIRE_RISK_SCORE` dropped from inputs.
   - **Regression**: predict `WEATHER_FIRE_RISK_SCORE`, with `WEATHER_RISK_CATEGORY` dropped from inputs.
3. **Splits each task** 70/30 train/test (stratified for classification), excluding `LATITUDE`, `LONGITUDE`, `NEAREST_WEATHER_LAT`, `NEAREST_WEATHER_LON`, and `DISCOVERYDATETIME` from model inputs while carrying their test-split values through for classification results.
4. **Trains three models per task**:
   - **Random Forest** (`RandomForestClassifier`/`Regressor`, 300 trees, `class_weight="balanced"` for classification).
   - **XGBoost** (`XGBClassifier`/`Regressor`, 300 trees, depth 6), wrapped in `XGBWrapper` to transparently handle the integer label-encoding XGBoost needs internally for classification. Skipped with a printed note if `xgboost` isn't installed.
   - **SVM** (`SVC`/`SVR` with an RBF kernel), wrapped in a `Pipeline` with `StandardScaler` since SVMs are distance-based.
5. **Evaluates and compares** all three per task — accuracy/precision/recall/F1 (macro and weighted) for classification, R²/MAE/RMSE/percentage-accuracy for regression — saving results and comparison tables to CSV.
6. **Plots predictions on a map** via its own built-in `create_map()`.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`, `folium`, `IPython`, `xgboost`

```bash
pip install pandas numpy scikit-learn folium ipython xgboost
```

`xgboost` is optional at import time — if missing, `train_xgboost()` prints an install note and returns `None`, and XGBoost is skipped in the comparison.

## Expected input data

`filtered_training_data_selectedColumn_Category_together.csv` — same input as the DNN and TabFM notebooks, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`.

## Configuration (Cells 2–4)

| Setting | Default | Meaning |
|---|---|---|
| `TARGET_LAT` / `TARGET_LON` / `MAX_RADIUS` | Fairbanks, AK / 250 mi | Used for map centering |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `DATA_PATH` | `filtered_training_data_selectedColumn_Category_together.csv` | Input CSV |
| `CLASSIFICATION_TARGET` | `WEATHER_RISK_CATEGORY` | Classification label |
| `REGRESSION_TARGET` | `WEATHER_FIRE_RISK_SCORE` | Regression label |
| `EXCLUDE_COLS` | lat/lon/station-coords/date columns + both targets | Columns kept for tracking but excluded from model inputs |

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring: pipeline overview, requirements, leakage note |
| 2 | Imports (`xgboost` wrapped in try/except) |
| 3 | Map target location/radius constants |
| 4 | Run configuration constants |
| 5 | `EXCLUDE_COLS` definition |
| 6 | `XGBWrapper` — normalizes XGBoost's `.predict()` output back to original label space |
| 7 | `load_data()` — reads the CSV |
| 8 | `split_data()` — 70/30 split (stratified for classification), separates excluded/tracking columns |
| 9 | `train_random_forest()` / `train_xgboost()` / `train_svm()` — training functions for each model type |
| 10 | `test_model()` — runs predictions, returns actual-vs-predicted DataFrame with tracking columns reattached |
| 11 | `create_map()` — Folium map of predictions with hoverable tooltips (self-contained, unlike the TabFM notebook) |
| 12 | `evaluate_models()` — computes classification or regression metrics per model |
| 13 | `compare_models()` — prints and optionally saves the comparison table |
| 14 | Loads data; builds classification/regression input views |
| 15 | Splits data for both tasks |
| 16 | Trains Random Forest, XGBoost, SVM for classification |
| 17 | Tests classification models, saves results CSVs, plots prediction maps |
| 18 | Evaluates + compares classification models |
| 19 | Trains Random Forest, XGBoost, SVM for regression |
| 20 | Tests regression models, saves results CSVs, plots prediction maps |
| 21 | Evaluates + compares regression models |

## Key outputs

- **`clf_results_randomforest.csv`**, **`clf_results_xgboost.csv`**, **`clf_results_svm.csv`** — per-record actual vs. predicted risk category, with tracking columns.
- **`reg_results_randomforest.csv`**, **`reg_results_xgboost.csv`**, **`reg_results_svm.csv`** — per-record actual vs. predicted risk score.
- **`ml_classification_model_comparison_TraditionalML.csv`** — Random Forest vs. XGBoost vs. SVM classification metrics.
- **`regression_model_comparison_TraditionalML.csv`** — Random Forest vs. XGBoost vs. SVM regression metrics.
- **`randomforest_predicted_vs_actual.html`**, **`xgboost_predicted_vs_actual.html`**, **`svm_predicted_vs_actual.html`** — interactive prediction maps (see limitations below).

## Known limitations / things to check before using downstream

- **Regression map cells will likely error**: `test_model()` is called for the regression task (Cell 19) *without* `tracking_df`, so `reg_results[...]` has no `LATITUDE`/`LONGITUDE` columns — but Cell 19 still calls `create_map(reg_results[...], ...)` using `create_map()`'s default `lat_col='LATITUDE'`/`lon_col='LONGITUDE'`. This will raise a `KeyError` unless `tracking_df=tracking_r_test` is added to the regression `test_model()` calls.
- **Map output filenames collide across tasks**, same issue as the DNN notebook: `randomforest_predicted_vs_actual.html`, `xgboost_predicted_vs_actual.html`, and `svm_predicted_vs_actual.html` are written in both the classification cell (17) and the regression cell (20) with the same paths, so whichever runs second overwrites the first's maps.
- `evaluate_models()` here does **not** guard against an empty `results_dict` the way the TabFM notebook's version does — if all models fail to train, this will raise a `KeyError` on `pd.DataFrame(rows).set_index("Model")` instead of printing a friendly message.
- SVM training scales poorly with dataset size (its RBF-kernel implementation is roughly quadratic in the number of training rows), so it may be noticeably slower than Random Forest/XGBoost on large training sets.
- As elsewhere, inputs aren't explicitly checked for `NaN`s before training; missing values in weather feature columns could raise errors (Random Forest/XGBoost handle some missingness, but SVM via `StandardScaler` does not).
- 
---

# Weather Wildfire Model: Build DNN Models

This notebook (`WeatherWildfireModel_BuildDNNModels.ipynb`) trains deep-learning models — a PyTorch feedforward network (DNN) and TabNet — to predict wildfire risk from the engineered weather/fire feature set, for both a classification task (risk category) and a regression task (risk score).

## What it does

1. **Loads the prepared training CSV** (`filtered_training_data_selectedColumn_Category_together.csv`, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`).
2. Builds two task-specific views of the data:
   - **Classification**: predict `WEATHER_RISK_CATEGORY` (Very Low/Low/Moderate/High/Extreme), with `WEATHER_FIRE_RISK_SCORE` dropped from the inputs to avoid leakage.
   - **Regression**: predict `WEATHER_FIRE_RISK_SCORE`, with `WEATHER_RISK_CATEGORY` dropped from the inputs.
3. **Splits each task** 70/30 train/test (stratified for classification), keeping `LATITUDE`, `LONGITUDE`, `NEAREST_WEATHER_LAT`, `NEAREST_WEATHER_LON`, and `DISCOVERYDATETIME` out of the model inputs but carrying their test-split values through for reference in the results.
4. **Trains two model types per task**:
   - A small **DNN** (PyTorch `Linear(128) → ReLU → Dropout → Linear(64) → ReLU → Dropout → Linear(output)`), trained with Adam + cross-entropy (classification) or MSE (regression).
   - **TabNet** (via `pytorch-tabnet`), a tabular-specific deep learning architecture, using its built-in classifier/regressor.
5. **Evaluates and compares** both model types per task — accuracy/precision/recall/F1 for classification, R²/MAE/RMSE/percentage-accuracy for regression — and saves results and comparison tables to CSV.
6. **Visualizes predictions on a map**, plotting actual-vs-predicted results by location.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`, `folium`, `torch`, `pytorch-tabnet`

```bash
pip install pandas numpy scikit-learn folium torch pytorch-tabnet
```

Both `torch` and `pytorch-tabnet` are optional at import time — if either is missing, the corresponding model is skipped with a printed note rather than raising an error.

## Expected input data

`filtered_training_data_selectedColumn_Category_together.csv` — the exact output of `WeatherWildfireModel_PrepareDataSet.ipynb`, containing the engineered weather features plus `WEATHER_FIRE_RISK_SCORE` and `WEATHER_RISK_CATEGORY`.

## Configuration (Cells 2–4)

| Setting | Default | Meaning |
|---|---|---|
| `TARGET_LAT` / `TARGET_LON` / `MAX_RADIUS` | Fairbanks, AK / 250 mi | Used for map centering |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `DATA_PATH` | `filtered_training_data_selectedColumn_Category_together.csv` | Input CSV |
| `CLASSIFICATION_TARGET` | `WEATHER_RISK_CATEGORY` | Classification label |
| `REGRESSION_TARGET` | `WEATHER_FIRE_RISK_SCORE` | Regression label |
| `EXCLUDE_COLS` | lat/lon/station-coords/date columns + both targets | Columns kept for tracking but excluded from model inputs |

## Data leakage note

`WEATHER_RISK_CATEGORY` is a deterministic binning of `WEATHER_FIRE_RISK_SCORE`. Each task drops the *other* task's target column from its input features before training, so neither model can simply read the answer off the other target.

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring: pipeline overview, leakage note |
| 2 | Imports (`torch` and `pytorch-tabnet` wrapped in try/except) |
| 3 | Map target location/radius constants |
| 4 | Run configuration constants (paths, targets, random state) |
| 5 | `EXCLUDE_COLS` definition |
| 6 | `SimpleMLP` — the PyTorch feedforward network architecture |
| 7 | `DNNWrapper` / `TabNetWrapper` — give both model types a uniform `.predict()` interface |
| 8 | `load_data()` — reads the CSV |
| 9 | `split_data()` — 70/30 split (stratified for classification), separates excluded/tracking columns |
| 10 | `train_dnn()` / `train_tabnet()` — training functions for each model type |
| 11 | `test_model()` — runs predictions, returns actual-vs-predicted DataFrame with tracking columns reattached |
| 12 | `create_map()` — Folium map of predictions with hoverable tooltips |
| 13 | `evaluate_models()` — computes classification or regression metrics per model |
| 14 | `compare_models()` — prints and optionally saves the comparison table |
| 15 | Loads data; builds classification/regression input views |
| 16 | Splits data for both tasks |
| 17 | Trains DNN + TabNet for classification |
| 18 | Tests classification models, saves results CSVs, plots prediction maps |
| 19 | Evaluates + compares classification models |
| 20 | Trains DNN + TabNet for regression |
| 21 | Tests regression models, saves results CSVs, plots prediction maps |
| 22 | Evaluates + compares regression models |

## Key outputs

- **`clf_results_dnn.csv`**, **`clf_results_tabnet.csv`** — per-record actual vs. predicted risk category, with tracking columns.
- **`reg_results_dnn.csv`**, **`reg_results_tabnet.csv`** — per-record actual vs. predicted risk score, with tracking columns.
- **`dnn_classification_model_comparison_DNN.csv`** — DNN vs. TabNet classification metrics.
- **`regression_model_comparison_DNN.csv`** — DNN vs. TabNet regression metrics.
- **`dnn_predicted_vs_actual.html`**, **`tabnet_predicted_vs_actual.html`** — interactive maps of predictions (note: regenerated for both classification and regression runs under the same filenames — see limitations below).

## Known limitations / things to check before using downstream

- **Map output filenames collide across tasks**: `dnn_predicted_vs_actual.html` and `tabnet_predicted_vs_actual.html` are written in both Cell 18 (classification) and Cell 21 (regression) with the same `save_path`, so the regression run's maps overwrite the classification run's maps. Rename one set (e.g. add a `_clf`/`_reg` suffix) if you need both.
- The DNN uses a fixed architecture and fixed 100 epochs with no early stopping or validation-loss monitoring — unlike TabNet, which supports `patience`-based early stopping natively.
- `evaluate_models()`'s regression "Percentage Accuracy (100-MAPE)" can go negative or become misleading if actual values are at or near zero, since MAPE is undefined/unstable near zero.
- Model inputs are not explicitly checked for `NaN`s before training; any missing values in the weather feature columns (e.g. from short weather windows near the start of the station record) will propagate into `StandardScaler`/PyTorch and likely raise errors or silently produce `NaN` predictions.
- As in the earlier notebooks, `EXCLUDE_COLS` still contains commented-out lines showing which columns were experimented with — worth reviewing if you plan to change the feature set.

---

# Weather Wildfire Model: Build Zero-Shot Model (TabFM)

This notebook (`WeatherWildfireModel_BuildZeroShotModel.ipynb`) evaluates **TabFM**, a zero-shot tabular foundation model from Google Research, on the same wildfire risk classification and regression tasks as the DNN notebook — but without any gradient-based training on this dataset.

## What it does

1. **Loads the prepared training CSV** (`filtered_training_data_selectedColumn_Category_together.csv`, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`).
2. Builds the same two task-specific views as the DNN notebook:
   - **Classification**: predict `WEATHER_RISK_CATEGORY`, with `WEATHER_FIRE_RISK_SCORE` dropped from inputs.
   - **Regression**: predict `WEATHER_FIRE_RISK_SCORE`, with `WEATHER_RISK_CATEGORY` dropped from inputs.
3. **Splits each task** 70/30 train/test (stratified for classification), excluding `LATITUDE`, `LONGITUDE`, `NEAREST_WEATHER_LAT`, `NEAREST_WEATHER_LON`, and `DISCOVERYDATETIME` from model inputs while carrying their test-split values through for reference.
4. **Fits TabFM** for each task. Unlike the DNN/TabNet notebook, TabFM does not learn weights via gradient descent — `.fit()` prepares encoders/scalers and stores the training data as "context," and predictions at inference time come from in-context learning over that context (true zero-shot/few-shot tabular inference).
5. **Evaluates** each task with the same metrics as the DNN notebook (accuracy/precision/recall/F1 for classification; R²/MAE/RMSE/percentage-accuracy for regression) and saves results to CSV.
6. **Plots predictions on a map** via `create_map()`.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`
- **TabFM** — not on PyPI; install from source:

```bash
pip install pandas numpy scikit-learn
git clone https://github.com/google-research/tabfm.git
cd tabfm && pip install -e .[pytorch]
```

If `tabfm` isn't importable, `train_tabfm()` prints an install note and returns `None`, and downstream cells skip that task's results.

## Expected input data

`filtered_training_data_selectedColumn_Category_together.csv` — same input as the DNN notebook, produced by `WeatherWildfireModel_PrepareDataSet.ipynb`.

## Configuration (Cells 2–3)

| Setting | Default | Meaning |
|---|---|---|
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `DATA_PATH` | `filtered_training_data_selectedColumn_Category_together.csv` | Input CSV |
| `CLASSIFICATION_TARGET` | `WEATHER_RISK_CATEGORY` | Classification label |
| `REGRESSION_TARGET` | `WEATHER_FIRE_RISK_SCORE` | Regression label |
| `EXCLUDE_COLS` | lat/lon/station-coords/date columns | Columns kept for tracking but excluded from model inputs |

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring: TabFM overview, leakage note, install instructions |
| 2 | Imports (`tabfm` wrapped in try/except) |
| 3 | Run configuration constants |
| 4 | `EXCLUDE_COLS` definition |
| 5 | `load_data()` — reads the CSV |
| 6 | `split_data()` — 70/30 split (stratified for classification), separates excluded/tracking columns |
| 7 | `train_tabfm()` — fits TabFM as a zero-shot classifier or regressor |
| 8 | `test_model()` — runs predictions, returns actual-vs-predicted DataFrame with tracking columns reattached |
| 9 | `evaluate_models()` — computes classification or regression metrics |
| 10 | `compare_models()` — prints and optionally saves the comparison table |
| 11 | Loads data |
| 12 | Builds classification/regression input views |
| 13 | Splits data for both tasks |
| 14 | Trains TabFM for classification |
| 15 | Tests classification model, saves results CSV, plots prediction map |
| 16 | Evaluates + compares classification results |
| 17 | Trains TabFM for regression |
| 18 | Tests regression model, saves results CSV, plots prediction map |
| 19 | Evaluates + compares regression results |

## Key outputs

- **`clf_results_tabfm.csv`** — per-record actual vs. predicted risk category, with tracking columns.
- **`reg_results_tabfm.csv`** — per-record actual vs. predicted risk score, with tracking columns.
- **`classification_model_comparison_ZeroshotModels.csv`** — TabFM classification metrics.
- **`regression_model_comparison_Zeroshot.csv`** — TabFM regression metrics.
- **`tabfm_predicted_vs_actual.html`** — interactive map of predictions (see limitation below — only ever reflects the classification results as currently written).

## Known limitations / things to check before using downstream

- **`create_map()` is called (Cells 14 and 17) but never defined in this notebook.** It must be defined/imported from elsewhere (e.g. by having run the DNN notebook's definition in the same kernel session) or these cells will raise a `NameError`. Copy the `create_map()` function from `WeatherWildfireModel_BuildDNNModels.ipynb` into this notebook if running it standalone.
- **Cell 17 (regression testing) plots the wrong DataFrame**: it calls `create_map(tabfm_clf_results_df, ...)` instead of `tabfm_reg_results_df`, and saves to the same `tabfm_predicted_vs_actual.html` path used in Cell 14 — so the regression cell overwrites the classification map with a re-plot of the classification results rather than showing regression predictions. Fix by changing the argument to `tabfm_reg_results_df` and using a distinct output filename.
- TabFM's `.fit()` doesn't train weights on this data — it stores training rows as in-context examples, so prediction speed and memory use scale with training-set size in a way gradient-trained models don't; this can be slow/heavy for large datasets.
- As with the DNN notebook, inputs aren't explicitly checked for `NaN`s before fitting; missing values in weather feature columns could cause errors or degrade TabFM's in-context predictions.
- `EXCLUDE_COLS` here does *not* comment out `WEATHER_FIRE_RISK_SCORE`/`WEATHER_RISK_CATEGORY` the way the DNN notebook's list does — leakage protection instead relies entirely on the separate `df_for_classification`/`df_for_regression` drop in Cell 11, so keep that step intact if you modify the pipeline.

---

# Weather Model: Build ML Models

This notebook (`WeatherModel_BuildMLModels.ipynb`) trains and compares two multivariate time-series forecasting models — VARMAX ("ARIMA") and per-variable SARIMAX ("SARIMA") — to predict daily weather parameters at a given location, then uses the fitted models to forecast weather for a new date and coordinate.

## What it does

1. **Loads weather data** from `./WeatherDataFromAcrossAlaska/`, one row per station-day.
2. For each distinct `(Latitude, Longitude)` location in the data:
   - Builds a continuous daily time series (gaps filled via time-based interpolation).
   - Splits it chronologically into 70% train / 30% test (no shuffling, since it's a time series).
   - Fits a **VARMAX** model jointly over all six weather variables (captures cross-correlations, e.g. temperature vs. humidity).
   - Fits six independent **SARIMAX** models, one per weather variable.
   - Forecasts across the test period and computes MAE, RMSE, and MAPE for both approaches, saving a per-location comparison CSV.
3. **Predicts weather for a new date/location** by finding the nearest known coordinate, using its fitted model, and forecasting forward from the last known date.
4. Optionally takes **interactive input** (date, latitude, longitude, model choice) via `predictFromUserInput()`.

## Modeling approach

- **ARIMA (VARMAX)** — one model per location, jointly modeling `Max Temperature`, `Min Temperature`, `Precipitation`, `Wind`, `Relative Humidity`, and `Solar` as a single vector time series. This is the true multivariate approach: it can capture relationships between variables (e.g. rising temperature with falling humidity).
- **SARIMA (SARIMAX)** — one independent model per variable per location. Each captures its own seasonality but not cross-variable correlation.
- **Latitude/longitude are not used as regressors** inside a given location's model, since they're constant within that series (and therefore collinear with the intercept). Instead, they're used to select *which* location's trained model to apply.

> **Note on seasonality:** the data is daily, so a true annual cycle needs `seasonal_order` period = 365, but fitting SARIMAX with a 365-day seasonal period on 30+ years of daily data is impractically slow with `statsmodels`. `SEASONAL_PERIOD` defaults to **7 (weekly)** here purely for a runnable demo — for real annual-cycle accuracy, either set it to 365 (expect a long fit) or resample the data to weekly/monthly frequency first.

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `scikit-learn`, `statsmodels`

```bash
pip install pandas numpy scikit-learn statsmodels
```

If `statsmodels` isn't installed, the notebook prints a warning and skips model creation rather than failing outright.

## Expected input data

`./WeatherDataFromAcrossAlaska/*.csv`, each with at least: `Date`, `Latitude`, `Longitude`, `Max Temperature`, `Min Temperature`, `Precipitation`, `Wind`, `Relative Humidity`, `Solar`.

## Configuration (Cells 2–3)

| Setting | Default | Meaning |
|---|---|---|
| `DATA_FOLDER` | `./WeatherDataFromAcrossAlaska/` | Input folder |
| `TARGET_COLS` | the 6 weather variables | Variables to forecast |
| `ARIMA_ORDER` | `(2, 1, 2)` | (p, d, q) for VARMAX |
| `SARIMA_ORDER` | `(1, 1, 1)` | (p, d, q) for each SARIMAX model |
| `SEASONAL_PERIOD` | `7` | Seasonal cycle length (weekly demo default; use 365 for annual) |
| `SARIMA_SEASONAL_ORDER` | `(1, 1, 1, 7)` | (P, D, Q, s) for SARIMAX |

## Notebook structure

| Cell(s) | Purpose |
|---|---|
| 1 | Module docstring explaining the modeling approach and caveats |
| 2 | Imports (`statsmodels` wrapped in a try/except so the notebook degrades gracefully if it's missing) |
| 3 | Data/column configuration constants |
| 4 | Model order/seasonality constants |
| 5 | `loadData()` — reads and concatenates all CSVs in the data folder |
| 6 | `_prepareLocationSeries()` — extracts one location's series and fills gaps to a continuous daily index |
| 7 | `splitData()` — chronological 70/30 train/test split |
| 8 | `createARIMAModel()` / `createSARIMAModel()` — build (unfitted) VARMAX and SARIMAX models |
| 9 | `trainModel()` — fits either model type |
| 10 | `testModel()` — forecasts over the test period, returns actual vs. predicted per variable |
| 11 | `_evaluateForecast()` — computes MAE, RMSE, MAPE per variable |
| 12 | `compareModelAccuracy()` — side-by-side ARIMA vs. SARIMA comparison table |
| 13 | `_findNearestLocation()` — nearest-neighbor coordinate lookup (Euclidean, not Haversine) |
| 14 | `predictNewData()` — forecasts weather for a new date/location using a fitted model |
| 15 | `predictFromUserInput()` — interactive CLI-style prediction (prompts for date/lat/lon/model) |
| 16 | Loads the weather data (`df`) |
| 17–18 | Initialize empty dicts to hold fitted models per location |
| 19 | Training/evaluation loop over locations |
| 20 | Runs `predictFromUserInput()` |

## Key outputs

- **`model_comparison_lat{lat}_lon{lon}.csv`** — one file per location, comparing ARIMA vs. SARIMA MAE/RMSE/MAPE for each weather variable.
- Two dictionaries of fitted models keyed by `(lat, lon)`: `arima_fitted_by_location`, `sarima_fitted_by_location`.
- An interactive single-row weather prediction printed to output (or returned as a DataFrame) when `predictFromUserInput()` is run.

## Known limitations / things to check before running

- **Cell 19 (the training loop) looks incomplete as saved**: it starts with `for lat, lon in locations:` and ends with a bare `return arima_fitted_by_location, sarima_fitted_by_location, df`, but there's no enclosing `def ...():` in the notebook and `locations` is never defined beforehand. This cell likely needs to be wrapped in a function definition (e.g. `def trainAndEvaluateAllLocations(df, locations): ...`) with `locations` set from `df[[LAT_COL, LON_COL]].drop_duplicates()` before it will run.
- **Cell 20** calls `predictFromUserInput(full_df, arima_models, sarima_models)`, but none of `full_df`, `arima_models`, or `sarima_models` are defined anywhere in the notebook — these appear to be leftover names from a different version of the training-loop function and need to be reconciled with the actual variable names (`df`, `arima_fitted_by_location`, `sarima_fitted_by_location`).
- `_findNearestLocation()` uses plain Euclidean distance on raw lat/lon degrees, unlike the Haversine distance used in the dataset-prep notebook — fine for a coarse nearest-neighbor lookup, but not a true geographic distance.
- Fitting VARMAX/SARIMAX per location can be slow with many locations or long histories; runtime scales with both the number of unique `(lat, lon)` pairs and the length of each series.
- `predictNewData()` only supports forecasting *forward* from the last known date (or returning an exact historical row) — it doesn't interpolate within gaps in the historical range.

